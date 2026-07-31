"""Audit the TMDR benchmark measurement itself: recompute separation / cross-layer /
keep-only under four different feature->component assignment rules, to decouple
"the decomposition is bad" from "the assignment step of the metric is bad".

Assignment rules:
  attr_raw   argmax over components of mean gradient attribution on single-feature
             probes (what run_tmdr.py's eval uses).
  attr_norm  same, but each component's attribution profile over features is first
             divided by its max — removes global-scale artifacts where one
             large-attribution component wins argmax for every feature.
  conns      argmax cosine between the feature's neuron-contribution vector
             (diag_relu_conns) and each component's — weight-based, no attribution.
  oracle     per feature, the component whose SOLO forward pass best reconstructs the
             feature's own output dim (keep-only error argmin) — upper bound on
             single-component recoverability, independent of any attribution.

For each rule we report: separation (unique assigned / n_features), cross_layer
(weight-norm span > 0.1 of assigned components), keep_only (mean normalized error using
that rule's assignment).

    python -m nano_apd.audit_tmdr --runs tmdr_C200 tmdr_C130_unitnorm1L ...
"""

import argparse
import json

import einops
import torch
import torch.nn.functional as F

from nano_apd.apd import calc_grad_attributions
from nano_apd.models import ResidMLPAPDModel
from nano_apd.run_resid import diag_relu_conns
from nano_apd.run_tmdr import OUT_ROOT, _component_layer_norms, load_tmdr_target


@torch.no_grad()
def keep_only_matrix(target, apd_model, device: str, n_probe: int = 64) -> torch.Tensor:
    """err[i, c] = normalized recon error of feature i's own output dim with ONLY
    component c active, averaged over single-feature probes. [nf, C]"""
    nf = target.config.n_features
    C = apd_model.C
    x = torch.zeros(nf, n_probe, 1, nf, device=device)
    for i in range(nf):
        x[i, :, 0, i] = torch.rand(n_probe, device=device)
    x_flat = einops.rearrange(x, "f p i nf -> (f p) i nf")
    tgt, _ = target(x_flat)
    own = einops.rearrange(tgt, "(f p) i nf -> f p i nf", f=nf)
    own_dim = torch.stack([own[i, :, 0, i] for i in range(nf)])       # [nf, n_probe]
    var = own_dim.var(dim=1) + 1e-8                                    # [nf]

    err = torch.zeros(nf, C, device=device)
    for c in range(C):
        mask = torch.zeros(x_flat.shape[0], 1, C, device=device)
        mask[:, 0, c] = 1.0
        pred, _ = apd_model(x_flat, topk_mask=mask)
        pred_own = einops.rearrange(pred, "(f p) i nf -> f p i nf", f=nf)
        pred_dim = torch.stack([pred_own[i, :, 0, i] for i in range(nf)])
        err[:, c] = ((pred_dim - own_dim) ** 2).mean(dim=1) / var
    return err


def attribution_profile(target, apd_model, device: str, n_probe: int = 64) -> torch.Tensor:
    """A[i, c] = mean gradient attribution of component c on feature-i probes. [nf, C]"""
    nf = target.config.n_features
    C = apd_model.C
    param_names = target.param_names()
    A = torch.zeros(nf, C, device=device)
    for i in range(nf):
        x = torch.zeros(n_probe, 1, nf, device=device)
        x[:, 0, i] = torch.rand(n_probe, device=device)
        out, cache = target(x)
        attr = calc_grad_attributions(
            target_out=out,
            pre_weight_acts={n: cache[n]["pre"] for n in param_names},
            post_weight_acts={n: cache[n]["post"] for n in param_names},
            component_weights=apd_model.component_weights(), C=C,
        )
        A[i] = attr[:, 0].mean(dim=0).detach()
    return A


def metrics_for_assignment(assigned: torch.Tensor, layer_norms: torch.Tensor,
                           ko_matrix: torch.Tensor) -> dict:
    nf = assigned.shape[0]
    a_norms = layer_norms[assigned]
    span = a_norms.min(dim=1).values / (a_norms.max(dim=1).values + 1e-8)
    return {
        "separation": assigned.unique().numel() / nf,
        "cross_layer": (span > 0.1).float().mean().item(),
        "keep_only": ko_matrix[torch.arange(nf), assigned].mean().item(),
    }


def audit_run(run_name: str, device: str) -> None:
    run_dir = OUT_ROOT / run_name
    cfg = json.load(open(run_dir / "apd_config.json"))
    target = load_tmdr_target(device)
    apd_model = ResidMLPAPDModel(target.config, C=cfg["C"], m=cfg["m"]).to(device)
    apd_model.load_state_dict(
        torch.load(run_dir / "apd_model.pth", weights_only=True, map_location=device)
    )

    A = attribution_profile(target, apd_model, device)
    ko = keep_only_matrix(target, apd_model, device)
    layer_norms = _component_layer_norms(apd_model)

    tw, cw = target.weights(), apd_model.component_weights()
    tgt_conns = diag_relu_conns(target.W_E, target.W_U,
                                {n: w for n, w in tw.items() if "mlp_in" in n},
                                {n: w for n, w in tw.items() if "mlp_out" in n},
                                2, per_component=False).detach()
    comp_conns = diag_relu_conns(apd_model.W_E, apd_model.W_U,
                                 {n: w for n, w in cw.items() if "mlp_in" in n},
                                 {n: w for n, w in cw.items() if "mlp_out" in n},
                                 2, per_component=True).detach()
    cos = einops.einsum(F.normalize(tgt_conns, dim=-1), F.normalize(comp_conns, dim=-1),
                        "i f m, i C f m -> i f C")[0]

    assignments = {
        "attr_raw": A.argmax(dim=1),
        "attr_norm": (A / (A.max(dim=0, keepdim=True).values + 1e-12)).argmax(dim=1),
        "conns": cos.argmax(dim=1),
        "oracle": ko.argmin(dim=1),
    }
    print(f"\n=== {run_name}")
    print(f"{'rule':<10} {'separation':>10} {'cross_layer':>11} {'keep_only':>10}")
    for rule, assigned in assignments.items():
        m = metrics_for_assignment(assigned.cpu(), layer_norms.cpu(), ko.cpu())
        print(f"{rule:<10} {m['separation']:>10.2f} {m['cross_layer']:>11.2f} "
              f"{m['keep_only']:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for run_name in args.runs:
        audit_run(run_name, args.device)


if __name__ == "__main__":
    main()
