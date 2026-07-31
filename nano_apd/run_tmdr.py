"""Run original APD on the repo's TMDR benchmark target (toy model of distributed
representations) and evaluate with the metrics used elsewhere in this repo
(nano_param_decomp/apd_mask.py:feature_recovery_resid): separation (injectivity),
coverage (active_frac), cross_layer, keep_only error.

Target: /tmp/toy/resid_2l_apd_dmlp40.pt — nano_param_decomp.toy_models.ResidMLP with
100 features, d_embed=256, 20 MLP neurons per layer x 2 layers, in_proj bias, W_E fixed
unit-norm rows, unembed W_E^T; trained on inputs U(0,1) active w.p. 0.01 with labels
y = x + ReLU(x). This differs from the APD paper's 2-layer model (d_embed=1000,
25 neurons/layer, no bias, inputs U(-1,1)).

Metric adaptation (stated explicitly since APD has no trained gate): apd_mask.py assigns
feature i to the component with highest mean causal-importance gate over 128
single-feature probes and calls it "active" if that mean gate > 0.5. Here assignment is
by highest mean gradient attribution over the same probes, and a feature is "active" if
the assigned component is the per-probe argmax on >50% of probes. separation, cross_layer
(weight-norm span > 0.1 across layers), keep_only and keep_only_off are computed exactly
as in apd_mask.py.

    python -m nano_apd.run_tmdr
"""

import argparse
import json
import os
from pathlib import Path

import einops
import torch
import torch.nn.functional as F

from nano_apd.apd import APDConfig, calc_recon_mse, calc_topk_mask, optimize
from nano_apd.attributions import compute_attributions
from nano_apd.greedy import greedy_add_mask, greedy_prune_mask
from nano_apd.models import ResidMLPAPDModel, ResidMLPConfig, ResidMLPModel, SparseFeatureDataset

OUT_ROOT = Path(__file__).parent / "out"
TARGET_CKPT = "/tmp/toy/resid_2l_apd_dmlp40.pt"


def load_tmdr_target(device: str) -> ResidMLPModel:
    """Load the benchmark checkpoint (nn.Linear convention, [d_out, d_in]) into our
    einsum-convention model ([d_in, d_out]) and verify equivalence against the original
    class on random batches."""
    state = torch.load(TARGET_CKPT, weights_only=True, map_location=device)
    n_layers = 2
    d_mlp, d_embed = state["blocks.0.in_proj.weight"].shape
    n_features = state["W_E"].shape[0]
    config = ResidMLPConfig(
        n_instances=1, n_features=n_features, d_embed=d_embed, d_mlp=d_mlp,
        n_layers=n_layers, feature_probability=0.01, batch_size=2048, steps=0, in_bias=True,
    )
    model = ResidMLPModel(config).to(device)
    model.W_E.data[0] = state["W_E"]
    model.W_U.data[0] = state["W_E"].t()
    for l in range(n_layers):
        model.mlp_in[l].weight.data[0] = state[f"blocks.{l}.in_proj.weight"].t()
        model.mlp_out[l].weight.data[0] = state[f"blocks.{l}.out_proj.weight"].t()
        model.bias1[l].data[0] = state[f"blocks.{l}.in_proj.bias"]
        model.bias1[l].requires_grad = False

    # Verify against the original implementation
    import sys
    sys.path.insert(0, "/workspace/param-decomp")
    from nano_param_decomp.toy_models import ResidMLP as RepoResidMLP
    repo_model = RepoResidMLP(n_features, d_embed, d_mlp * n_layers, n_layers, seed=0).to(device)
    repo_model.load_state_dict(state)
    x = torch.rand(64, n_features, device=device) * (torch.rand(64, n_features, device=device) < 0.05)
    ours, _ = model(x.unsqueeze(1))
    theirs = repo_model(x)
    max_diff = (ours[:, 0] - theirs).abs().max().item()
    assert max_diff < 1e-4, f"port mismatch: max|diff|={max_diff}"
    print(f"target port verified: max|out diff| vs repo model = {max_diff:.2e}", flush=True)
    return model


@torch.no_grad()
def _component_layer_norms(apd_model: ResidMLPAPDModel) -> torch.Tensor:
    """[C, n_layers] weight norm of each component per layer (mlp_in + mlp_out)."""
    n_layers = apd_model.config.n_layers
    cw = apd_model.component_weights()
    norms = torch.zeros(apd_model.C, n_layers, device=apd_model.W_E.device)
    for l in range(n_layers):
        for proj in ("mlp_in", "mlp_out"):
            norms[:, l] += cw[f"layers.{l}.{proj}"][0].pow(2).sum(dim=(-2, -1))
    return norms.sqrt()


def eval_tmdr(target: ResidMLPModel, apd_model: ResidMLPAPDModel, device: str,
              n_probe: int = 128, attribution_type: str = "gradient",
              value_range: tuple[float, float] = (0.0, 1.0)) -> dict:
    nf = target.config.n_features
    C = apd_model.C
    param_names = target.param_names()
    lo, hi = value_range

    def probe_values(n):
        return torch.rand(n, device=device) * (hi - lo) + lo

    # pass 1: mean attribution per feature over single-feature probes -> assignment
    # (assignment uses the same attribution method as the run under evaluation)
    A = torch.zeros(nf, C, device=device)
    argmax_agreement = torch.zeros(nf, device=device)
    for i in range(nf):
        x = torch.zeros(n_probe, 1, nf, device=device)
        x[:, 0, i] = probe_values(n_probe)
        target_out, cache = target(x)
        attr = compute_attributions(
            attribution_type, target_model=target, apd_model=apd_model, batch=x,
            target_out=target_out, target_cache=cache, param_names=param_names, C=C,
        )[:, 0].detach()                               # [n_probe, C]
        A[i] = attr.mean(dim=0)
        assigned_i = A[i].argmax()
        argmax_agreement[i] = (attr.argmax(dim=-1) == assigned_i).float().mean()
    assigned = A.argmax(dim=1)                         # [nf]
    active = argmax_agreement > 0.5
    n_active = int(active.sum())
    purity = (A.gather(1, assigned[:, None]).squeeze(1) / (A.sum(dim=1) + 1e-8))[active]

    separation = assigned[active].unique().numel() / n_active if n_active else 0.0
    coverage = n_active / nf

    layer_norms = _component_layer_norms(apd_model)    # [C, 2]
    a_norms = layer_norms[assigned[active]]
    span = a_norms.min(dim=1).values / (a_norms.max(dim=1).values + 1e-8)
    cross_layer = (span > 0.1).float().mean().item() if n_active else 0.0

    # causal sufficiency: keep ONLY the assigned component, recon feature i's own output dim
    keep_only, keep_off = [], []
    with torch.no_grad():
        for i in range(nf):
            if not bool(active[i]):
                continue
            x = torch.zeros(n_probe, 1, nf, device=device)
            x[:, 0, i] = probe_values(n_probe)
            tgt, _ = target(x)
            var_i = tgt[:, 0, i].var() + 1e-8
            mask = torch.zeros(n_probe, 1, C, device=device)
            mask[:, 0, assigned[i]] = 1.0
            pred, _ = apd_model(x, topk_mask=mask)
            keep_only.append((F.mse_loss(pred[:, 0, i], tgt[:, 0, i]) / var_i).item())
            off, _ = apd_model(x, topk_mask=torch.zeros(n_probe, 1, C, device=device))
            keep_off.append((F.mse_loss(off[:, 0, i], tgt[:, 0, i]) / var_i).item())

    return {
        "separation": separation,
        "coverage": coverage,
        "cross_layer": cross_layer,
        "keep_only": sum(keep_only) / len(keep_only) if keep_only else float("nan"),
        "keep_only_off": sum(keep_off) / len(keep_off) if keep_off else float("nan"),
        "purity_mean_attr_share": purity.mean().item() if n_active else 0.0,
        "n_active": n_active,
        "n_unique_assigned": assigned[active].unique().numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--C", type=int, default=int(os.environ.get("C", "200")))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--topk", type=float, default=1.28)
    parser.add_argument("--topk_recon", type=float, default=2.0)
    parser.add_argument("--act_recon", type=float, default=1.0)
    parser.add_argument("--schatten", type=float, default=7.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unit_norm", action="store_true")
    parser.add_argument("--attribution", default="gradient",
                        choices=["gradient", "gim", "ig", "relp", "ifr"])
    parser.add_argument("--selection", default="batch_topk",
                        choices=["batch_topk", "greedy_add", "greedy_prune",
                                 "greedy_prune_iter"])
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="per-sample MSE threshold for the greedy selections")
    parser.add_argument("--tag", default="", help="suffix for the output dir")
    args = parser.parse_args()
    device = args.device
    name = f"tmdr_C{args.C}"
    if args.attribution != "gradient":
        name += f"_{args.attribution}"
    if args.selection != "batch_topk":
        name += f"_{args.selection}_eps{args.eps:g}"
    out_dir = OUT_ROOT / f"{name}{('_' + args.tag) if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = load_tmdr_target(device)
    config = target.config

    # APD paper 2-layer config, adapted only where the target differs (dims; data U(0,1)).
    apd_config = APDConfig(C=args.C, topk=None if args.selection != "batch_topk" else args.topk,
                           batch_size=256, steps=args.steps, lr=args.lr,
                           seed=0, lr_schedule="cosine", lr_warmup_pct=0.01,
                           param_match_coeff=1.0, topk_recon_coeff=args.topk_recon,
                           act_recon_coeff=args.act_recon,
                           schatten_pnorm=0.9, schatten_coeff=args.schatten,
                           unit_norm_matrices=args.unit_norm,
                           attribution_type=args.attribution, selection=args.selection,
                           eps=args.eps, print_freq=500, extra={"init_scale": 2.0})

    torch.manual_seed(apd_config.seed)
    apd_model = ResidMLPAPDModel(config, C=apd_config.C, m=apd_config.m,
                                 init_scale=apd_config.extra["init_scale"]).to(device)
    apd_model.W_E.data[:] = target.W_E.data.clone()
    apd_model.W_U.data[:] = target.W_U.data.clone()
    for l in range(config.n_layers):
        apd_model.bias1[l].data[:] = target.bias1[l].data.clone()

    param_names = target.param_names()
    dataset = SparseFeatureDataset(1, config.n_features, config.feature_probability, device,
                                   value_range=(0.0, 1.0))
    # act_recon on the actual hidden activations relu(post + bias). (The reference code
    # relu's the pre-bias post acts, but its paper models have no bias; with a bias the
    # real activation includes it, and it's identical on both sides since the bias is
    # frozen and shared.)
    biases = [apd_model.bias1[l].data for l in range(config.n_layers)]
    def act_recon_transform(acts: dict) -> dict:
        return {k: F.relu(v + biases[int(k.split(".")[1])])
                for k, v in acts.items() if "mlp_in" in k}

    optimize(
        model=apd_model, target_model=target, config=apd_config,
        generate_batch=dataset.generate_batch, param_names=param_names,
        device=device, out_dir=out_dir, act_recon_transform=act_recon_transform,
    )

    summary = eval_tmdr(target, apd_model, device, attribution_type=args.attribution)

    # Final sparse-forward reconstruction, using the run's own selection procedure
    batch = dataset.generate_batch(apd_config.batch_size)
    target_out, cache = target(batch)
    attr = compute_attributions(
        args.attribution, target_model=target, apd_model=apd_model, batch=batch,
        target_out=target_out, target_cache=cache, param_names=param_names, C=apd_config.C,
    )
    if args.selection == "batch_topk":
        mask = calc_topk_mask(attr, apd_config.topk, batch_topk=True)
    elif args.selection == "greedy_add":
        mask = greedy_add_mask(apd_model, batch, target_out, C=apd_config.C, eps=apd_config.eps)
    elif args.selection == "greedy_prune_iter":
        from nano_apd.greedy import greedy_prune_iter_mask
        mask = greedy_prune_iter_mask(apd_model, batch, target_out, C=apd_config.C,
                                      eps=apd_config.eps)
    else:
        mask = greedy_prune_mask(apd_model, batch, target_out, attr, eps=apd_config.eps)
    out_topk, _ = apd_model(batch, topk_mask=mask)
    summary["final_topk_recon"] = calc_recon_mse(out_topk, target_out).tolist()
    summary["final_mask_l0"] = mask.float().sum(dim=-1).mean().item()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\ntable row — APD ({args.attribution} attribution, {args.selection}): "
          f"separation {summary['separation']:.2f}, coverage {summary['coverage']:.2f}, "
          f"cross-layer {summary['cross_layer']:.2f}, keep-only {summary['keep_only']:.3f} "
          f"(off baseline {summary['keep_only_off']:.3f})", flush=True)


if __name__ == "__main__":
    main()
