"""Train a residual-MLP target model and decompose it with APD (paper hyperparameters).

Usage:
    python -m nano_apd.run_resid --n_layers 2   # toy model of cross-layer distributed
                                                # representations (the one we care about)
    python -m nano_apd.run_resid --n_layers 1   # toy model of compressed computation

Paper setup: 100 features, d_embed=1000, 50 MLP neurons total (25 per layer in the
2-layer case), W_E fixed random unit-norm rows, W_U = W_E^T, labels y = ReLU(x) + x,
feature probability 0.01, inputs uniform in [-1, 1].
APD (2-layer config from resid_mlp_topk_config.yaml): C=200, batch topk 1.28 at batch
size 256, param_match 1.0, topk_recon 2.0, act_recon 1.0, schatten p=0.9 coeff 7,
lr 1e-3 cosine, 10k steps, init_scale 2.0, unit_norm_matrices false.
"""

import argparse
import json
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from nano_apd.apd import APDConfig, calc_grad_attributions, calc_recon_mse, calc_topk_mask, optimize
from nano_apd.models import (
    ResidMLPAPDModel,
    ResidMLPConfig,
    ResidMLPModel,
    SparseFeatureDataset,
    train_resid_mlp,
)

OUT_ROOT = Path(__file__).parent / "out"


def apd_config_for(n_layers: int) -> APDConfig:
    if n_layers == 2:
        return APDConfig(C=200, topk=1.28, batch_size=256, steps=10_000, lr=1e-3, seed=0,
                         lr_schedule="cosine", lr_warmup_pct=0.01, param_match_coeff=1.0,
                         topk_recon_coeff=2.0, act_recon_coeff=1.0,
                         schatten_pnorm=0.9, schatten_coeff=7.0,
                         unit_norm_matrices=False, print_freq=500,
                         extra={"init_scale": 2.0})
    # 1-layer (compressed computation) reference config; unit_norm_matrices is true there.
    return APDConfig(C=130, topk=1.28, batch_size=256, steps=10_000, lr=1e-3, seed=0,
                     lr_schedule="cosine", lr_warmup_pct=0.01, param_match_coeff=1.0,
                     topk_recon_coeff=1.0, act_recon_coeff=1.0,
                     schatten_pnorm=0.9, schatten_coeff=10.0,
                     unit_norm_matrices=True, print_freq=500,
                     extra={"init_scale": 2.0})


def get_target(config: ResidMLPConfig, device: str) -> ResidMLPModel:
    name = (f"resid_f{config.n_features}_e{config.d_embed}_m{config.d_mlp}"
            f"_l{config.n_layers}_p{config.feature_probability}_seed{config.seed}")
    ckpt = OUT_ROOT / "targets" / f"{name}.pth"
    torch.manual_seed(config.seed)  # so W_E is reproducible whether loading or training
    model = ResidMLPModel(config).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        print(f"loaded target from {ckpt}", flush=True)
    else:
        model = train_resid_mlp(config, device)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"saved target to {ckpt}", flush=True)
    return model


def diag_relu_conns(W_E: torch.Tensor, W_U: torch.Tensor,
                    w_in: dict[str, torch.Tensor], w_out: dict[str, torch.Tensor],
                    n_layers: int, per_component: bool) -> torch.Tensor:
    """Neuron contributions per input feature (reference plotting.py:
    spd_calculate_diag_relu_conns): conns[f, neuron] = (W_E[f,:] @ W_in)[neuron] *
    (W_out @ W_U[:,f])[neuron], concatenated across layers.

    Returns [i, F, d_mlp_total] for the target (per_component=False) or
    [i, C, F, d_mlp_total] per component (per_component=True).
    """
    conns = []
    for l in range(n_layers):
        Wi, Wo = w_in[f"layers.{l}.mlp_in"], w_out[f"layers.{l}.mlp_out"]
        if per_component:
            in_conns = einops.einsum(W_E, Wi, "i f e, i C e m -> i C f m")
            out_conns = einops.einsum(Wo, W_U, "i C m e, i e f -> i C m f")
            conns.append(in_conns * einops.rearrange(out_conns, "i C m f -> i C f m"))
        else:
            in_conns = einops.einsum(W_E, Wi, "i f e, i e m -> i f m")
            out_conns = einops.einsum(Wo, W_U, "i m e, i e f -> i m f")
            conns.append(in_conns * einops.rearrange(out_conns, "i m f -> i f m"))
    return torch.cat(conns, dim=-1)


def evaluate(target: ResidMLPModel, apd_model: ResidMLPAPDModel, cutoff: float = 4e-2) -> dict:
    """Component/feature analysis on the ReLU connection strengths.

    - mmcs_conns: mean over features of max-over-components cosine similarity between the
      component's neuron-contribution vector and the target's (1.0 = each feature's
      cross-layer computation is recovered by some single component).
    - component counts at the reference cutoff 4e-2: dead / mono / duo / poly-semantic.
    """
    cfg = target.config
    tgt = diag_relu_conns(target.W_E, target.W_U,
                          {n: w for n, w in target.weights().items() if "mlp_in" in n},
                          {n: w for n, w in target.weights().items() if "mlp_out" in n},
                          cfg.n_layers, per_component=False).detach()      # [i, F, M]
    comp = diag_relu_conns(apd_model.W_E, apd_model.W_U,
                           {n: w for n, w in apd_model.component_weights().items() if "mlp_in" in n},
                           {n: w for n, w in apd_model.component_weights().items() if "mlp_out" in n},
                           cfg.n_layers, per_component=True).detach()      # [i, C, F, M]

    cos = einops.einsum(F.normalize(tgt, dim=-1), F.normalize(comp, dim=-1),
                        "i f m, i C f m -> i f C")
    max_cos, best_c = cos.max(dim=-1)                                      # [i, F]

    active = comp.max(dim=-1).values > cutoff                              # [i, C, F]
    n_features_per_component = active.sum(dim=-1)                          # [i, C]
    counts = {
        "dead": (n_features_per_component == 0).sum(dim=-1).tolist(),
        "monosemantic": (n_features_per_component == 1).sum(dim=-1).tolist(),
        "duosemantic": (n_features_per_component == 2).sum(dim=-1).tolist(),
        "polysemantic": (n_features_per_component > 2).sum(dim=-1).tolist(),
    }
    features_covered = (active.any(dim=1)).sum(dim=-1).tolist()            # per instance
    n_unique_best = [len(set(best_c[i].tolist())) for i in range(best_c.shape[0])]
    return {
        "mmcs_conns": max_cos.mean().item(),
        "min_feature_max_cos": max_cos.min().item(),
        "component_counts_at_cutoff": counts,
        "features_covered_at_cutoff": features_covered,
        "n_unique_best_components": n_unique_best,
        "best_component_per_feature": best_c[0].tolist(),
        "_target_conns": tgt, "_comp_conns": comp,
    }


def plot_conns(tgt: torch.Tensor, comp: torch.Tensor, best_c: list[int], out_dir: Path,
               n_show: int = 10) -> None:
    """Paper-style figure: target neuron contributions for the first n_show features (top)
    vs the best-matching component's contributions (bottom). First instance."""
    t = tgt[0, :n_show].cpu().numpy()
    b = torch.stack([comp[0, best_c[f], f] for f in range(n_show)]).cpu().numpy()
    vmax = max(abs(t).max(), abs(b).max())
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, mat, title in zip(axs, [t, b], ["target model", "best APD component per feature"],
                              strict=True):
        im = ax.matshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel("feature"); ax.set_title(title, fontsize=9)
    axs[1].set_xlabel("neuron (layers concatenated)")
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.savefig(out_dir / "neuron_contributions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    d_mlp_per_layer = 50 // args.n_layers
    resid_config = ResidMLPConfig(
        n_instances=1, n_features=100, d_embed=1000, d_mlp=d_mlp_per_layer,
        n_layers=args.n_layers, feature_probability=0.01, batch_size=2048, steps=10_000, seed=0,
    )
    apd_config = apd_config_for(args.n_layers)
    device = args.device
    out_dir = OUT_ROOT / f"resid_{args.n_layers}layer"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = get_target(resid_config, device)

    torch.manual_seed(apd_config.seed)
    apd_model = ResidMLPAPDModel(resid_config, C=apd_config.C, m=apd_config.m,
                                 init_scale=apd_config.extra["init_scale"]).to(device)
    # As in the reference: copy the target's embedding matrices and freeze them.
    apd_model.W_E.data[:] = target.W_E.data.clone()
    apd_model.W_U.data[:] = target.W_U.data.clone()

    param_names = target.param_names()
    dataset = SparseFeatureDataset(resid_config.n_instances, resid_config.n_features,
                                   resid_config.feature_probability, device,
                                   value_range=(-1.0, 1.0))
    # act_recon in the reference compares post-ReLU activations of the mlp_in layers only.
    act_recon_transform = lambda acts: {k: F.relu(v) for k, v in acts.items() if "mlp_in" in k}

    optimize(
        model=apd_model, target_model=target, config=apd_config,
        generate_batch=dataset.generate_batch, param_names=param_names,
        device=device, out_dir=out_dir, act_recon_transform=act_recon_transform,
    )

    summary = evaluate(target, apd_model)
    tgt_conns = summary.pop("_target_conns")
    comp_conns = summary.pop("_comp_conns")
    plot_conns(tgt_conns, comp_conns, summary["best_component_per_feature"], out_dir)

    batch = dataset.generate_batch(apd_config.batch_size)
    target_out, target_cache = target(batch)
    attributions = calc_grad_attributions(
        target_out=target_out,
        pre_weight_acts={n: target_cache[n]["pre"] for n in param_names},
        post_weight_acts={n: target_cache[n]["post"] for n in param_names},
        component_weights=apd_model.component_weights(),
        C=apd_config.C,
    )
    topk_mask = calc_topk_mask(attributions, apd_config.topk, batch_topk=True)
    out_topk, _ = apd_model(batch, topk_mask=topk_mask)
    summary["final_topk_recon"] = calc_recon_mse(out_topk, target_out).tolist()
    out_full, _ = apd_model(batch)
    summary["final_out_recon"] = calc_recon_mse(out_full, target_out).tolist()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "best_component_per_feature"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
