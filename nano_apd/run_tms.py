"""Train a TMS target model and decompose it with APD, using the paper's hyperparameters.

Usage:
    python -m nano_apd.run_tms --variant 5-2
    python -m nano_apd.run_tms --variant 40-10

Variants (from the reference tms_topk_config.yaml):
    5-2:   5 features, 2 hidden, 12 instances; C=5,  batch topk 0.211, lr 3e-2 constant
    40-10: 40 features, 10 hidden, 3 instances; C=40, batch topk 2.0,  lr 1e-3 cosine
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nano_apd.apd import APDConfig, calc_recon_mse, calc_topk_mask, calc_grad_attributions, optimize
from nano_apd.models import SparseFeatureDataset, TMSAPDModel, TMSConfig, TMSModel, train_tms

OUT_ROOT = Path(__file__).parent / "out"

TMS_VARIANTS = {
    "5-2": dict(
        tms=TMSConfig(n_instances=12, n_features=5, n_hidden=2, feature_probability=0.05,
                      batch_size=1024, steps=5000, seed=0),
        apd=APDConfig(C=5, topk=0.211, batch_size=2048, steps=20_000, lr=3e-2, seed=0,
                      lr_schedule="constant", lr_warmup_pct=0.05, param_match_coeff=1.0,
                      topk_recon_coeff=1.0, schatten_pnorm=1.0, schatten_coeff=0.7,
                      print_freq=1000),
    ),
    "40-10": dict(
        tms=TMSConfig(n_instances=3, n_features=40, n_hidden=10, feature_probability=0.05,
                      batch_size=2048, steps=2000, seed=0),
        apd=APDConfig(C=40, topk=2.0, batch_size=2048, steps=20_000, lr=1e-3, seed=0,
                      lr_schedule="cosine", lr_warmup_pct=0.05, param_match_coeff=1.0,
                      topk_recon_coeff=10.0, schatten_pnorm=0.9, schatten_coeff=15.0,
                      print_freq=1000),
    ),
}


def get_target(tms_config: TMSConfig, device: str) -> TMSModel:
    name = (f"tms_f{tms_config.n_features}_h{tms_config.n_hidden}_i{tms_config.n_instances}"
            f"_p{tms_config.feature_probability}_seed{tms_config.seed}")
    ckpt = OUT_ROOT / "targets" / f"{name}.pth"
    model = TMSModel(tms_config).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        print(f"loaded target from {ckpt}", flush=True)
    else:
        model = train_tms(tms_config, device)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"saved target to {ckpt}", flush=True)
    return model


def eval_mmcs_ml2r(target: TMSModel, apd_model: TMSAPDModel) -> dict:
    """MMCS / ML2R as defined in the APD paper (eqs. in section 3.1).

    Paper definition: for each input feature j, compare the j-th column of W (the
    feature's n_hidden-dim embedding, W[j, :] in our [n_features, n_hidden] layout)
    with the j-th column of each component P_c; MMCS = mean_j max_c cos, ML2R =
    mean_j ||P_mcs(j)[j, :]|| / ||W[j, :]||.

    We additionally report a stricter whole-matrix variant ("rank1_*"): cosine between
    the flattened component matrix and the matrix that is zero except for row j = W[j, :].
    This also penalizes components carrying weight on rows of *other* features.
    """
    W = target.linear1.weight.detach()             # [i, f, h]
    P = apd_model.linear1.component_weights.detach()  # [i, C, f, h]
    n_inst, n_feat, _ = W.shape

    # Paper (column-wise) metric
    cos_col = einops.einsum(F.normalize(W, dim=-1), F.normalize(P, dim=-1),
                            "i f h, i C f h -> i f C")
    max_cos, best_c = cos_col.max(dim=-1)          # [i, f]
    P_col_norms = P.norm(dim=-1)                   # [i, C, f]
    best_norms = P_col_norms.gather(
        1, einops.rearrange(best_c, "i f -> i 1 f")).squeeze(1)  # [i, f]
    l2_ratio = best_norms / W.norm(dim=-1)

    # Stricter whole-matrix variant
    gt = torch.zeros(n_inst, n_feat, n_feat, W.shape[-1], device=W.device)
    for f in range(n_feat):
        gt[:, f, f, :] = W[:, f, :]
    gt_flat = einops.rearrange(gt, "i f a b -> i f (a b)")
    P_flat = einops.rearrange(P, "i C a b -> i C (a b)")
    cos_mat = einops.einsum(F.normalize(gt_flat, dim=-1), F.normalize(P_flat, dim=-1),
                            "i f d, i C d -> i f C")
    rank1_max_cos, rank1_best_c = cos_mat.max(dim=-1)

    return {
        "mmcs_per_instance": max_cos.mean(dim=-1).tolist(),
        "mmcs": max_cos.mean().item(),
        "mmcs_std_over_instances": max_cos.mean(dim=-1).std().item(),
        "ml2r_per_instance": l2_ratio.mean(dim=-1).tolist(),
        "ml2r": l2_ratio.mean().item(),
        "ml2r_std_over_instances": l2_ratio.mean(dim=-1).std().item(),
        "rank1_mmcs_per_instance": rank1_max_cos.mean(dim=-1).tolist(),
        "rank1_mmcs": rank1_max_cos.mean().item(),
        "best_component_per_feature": best_c.tolist(),
    }


def plot_components(target: TMSModel, apd_model: TMSAPDModel, out_dir: Path) -> None:
    """2D polygon plots (only meaningful for n_hidden=2): target W rows and each
    component's linear1 rows, first instance."""
    if target.config.n_hidden != 2:
        return
    W = target.linear1.weight.detach().cpu()[0]                    # [f, 2]
    P = apd_model.linear1.component_weights.detach().cpu()[0]      # [C, f, 2]
    C = P.shape[0]
    fig, axs = plt.subplots(1, C + 1, figsize=(2.2 * (C + 1), 2.4))
    for ax, mat, title in zip(
        axs, [W] + [P[c] for c in range(C)],
        ["target"] + [f"component {c}" for c in range(C)], strict=True,
    ):
        m = mat.numpy()
        ax.scatter(m[:, 0], m[:, 1], s=8)
        for row in m:
            ax.plot([0, row[0]], [0, row[1]], lw=1)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=8)
        ax.tick_params(labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(out_dir / "components_polygon.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(TMS_VARIANTS), default="5-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the APD seed (target model seed is unchanged)")
    args = parser.parse_args()

    variant = TMS_VARIANTS[args.variant]
    tms_config: TMSConfig = variant["tms"]
    apd_config: APDConfig = variant["apd"]
    if args.seed is not None:
        apd_config = replace(apd_config, seed=args.seed)
    device = args.device
    seed_suffix = f"_seed{apd_config.seed}" if apd_config.seed != 0 else ""
    out_dir = OUT_ROOT / f"tms_{args.variant}{seed_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = get_target(tms_config, device)

    torch.manual_seed(apd_config.seed)
    apd_model = TMSAPDModel(tms_config, C=apd_config.C, m=apd_config.m).to(device)
    # As in the reference: copy the target's trained bias and freeze it (train_bias: false).
    apd_model.b_final.data[:] = target.b_final.data.clone()
    apd_model.b_final.requires_grad = False

    dataset = SparseFeatureDataset(tms_config.n_instances, tms_config.n_features,
                                   tms_config.feature_probability, device, value_range=(0.0, 1.0))
    optimize(
        model=apd_model, target_model=target, config=apd_config,
        generate_batch=dataset.generate_batch, param_names=["linear1", "linear2"],
        device=device, out_dir=out_dir,
    )

    # Final evaluation on a fresh batch
    summary = eval_mmcs_ml2r(target, apd_model)
    batch = dataset.generate_batch(apd_config.batch_size)
    target_out, target_cache = target(batch)
    attributions = calc_grad_attributions(
        target_out=target_out,
        pre_weight_acts={n: target_cache[n]["pre"] for n in ["linear1", "linear2"]},
        post_weight_acts={n: target_cache[n]["post"] for n in ["linear1", "linear2"]},
        component_weights=apd_model.component_weights(),
        C=apd_config.C,
    )
    topk_mask = calc_topk_mask(attributions, apd_config.topk, batch_topk=True)
    out_topk, _ = apd_model(batch, topk_mask=topk_mask)
    summary["final_topk_recon"] = calc_recon_mse(out_topk, target_out).tolist()
    out_full, _ = apd_model(batch)
    summary["final_out_recon"] = calc_recon_mse(out_full, target_out).tolist()

    plot_components(target, apd_model, out_dir)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "best_component_per_feature"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
