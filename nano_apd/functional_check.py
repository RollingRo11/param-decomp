"""Functional layer-half verification for a trained polar/APD run — the test that has
caught two counterfeit decompositions (metric-perfect components whose function lives
entirely in one layer). Now part of the standard eval.

For each feature's oracle-assigned component: keep-only error with the full component,
with only its layer-0 blocks, and with only its layer-1 blocks. A genuinely cross-layer
component needs both halves (both ablations hurt).

    python -m nano_apd.functional_check --run tmdr_polar_...
"""

import argparse
import json

import einops
import torch

from nano_apd.models import ResidMLPAPDModel
from nano_apd.run_tmdr import OUT_ROOT, load_tmdr_target


def functional_report(target, apd, device, n_probe=64, value_range=(0.0, 1.0)):
    nf, C = target.config.n_features, apd.C
    names = target.param_names()
    lo, hi = value_range
    x = torch.zeros(nf, n_probe, 1, nf, device=device)
    for i in range(nf):
        x[i, :, 0, i] = torch.rand(n_probe, device=device) * (hi - lo) + lo
    xf = einops.rearrange(x, "f p i nf -> (f p) i nf")
    tgt, _ = target(xf)
    own = einops.rearrange(tgt, "(f p) i nf -> f p i nf", f=nf)
    own_dim = torch.stack([own[i, :, 0, i] for i in range(nf)])
    var = own_dim.var(dim=1) + 1e-8
    cw = {n: apd.component_weights()[n].detach() for n in names}
    orig = {n: target.weights()[n].data.clone() for n in names}

    err_full = torch.zeros(nf, C, device=device)
    with torch.no_grad():
        for c in range(C):
            for n in names:
                target.weights()[n].data[0] = cw[n][0, c]
            out, _ = target(xf)
            po = einops.rearrange(out, "(f p) i nf -> f p i nf", f=nf)
            pd = torch.stack([po[i, :, 0, i] for i in range(nf)])
            err_full[:, c] = ((pd - own_dim) ** 2).mean(dim=1) / var
        for n in names:
            target.weights()[n].data[:] = orig[n]
    assigned = err_full.argmin(dim=1)
    e_full = err_full[torch.arange(nf), assigned]

    @torch.no_grad()
    def keeponly_layer(layer_keep):
        errs = torch.zeros(nf, device=device)
        for i in range(nf):
            for n in names:
                w = cw[n][0, assigned[i]].clone()
                if int(n.split(".")[1]) != layer_keep:
                    w.zero_()
                target.weights()[n].data[0] = w
            out, _ = target(x[i])
            errs[i] = ((out[:, 0, i] - own[i, :, 0, i]) ** 2).mean() / var[i]
        for n in names:
            target.weights()[n].data[:] = orig[n]
        return errs

    e_l0, e_l1 = keeponly_layer(0), keeponly_layer(1)
    both = (e_l0 > 2 * e_full + 0.02) & (e_l1 > 2 * e_full + 0.02)
    return {
        "separation_oracle": assigned.unique().numel() / nf,
        "keep_only_oracle": e_full.mean().item(),
        "functional_cross_layer": int(both.sum()),
        "keep_only_layer0_only": e_l0.mean().item(),
        "keep_only_layer1_only": e_l1.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--target", choices=["benchmark", "paper"], default="benchmark")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = json.load(open(OUT_ROOT / args.run / "apd_config.json"))
    if args.target == "paper":
        from nano_apd.models import ResidMLPConfig
        from nano_apd.run_resid import get_target
        rc = ResidMLPConfig(n_instances=1, n_features=100, d_embed=1000, d_mlp=25,
                            n_layers=2, feature_probability=0.01, batch_size=2048,
                            steps=10_000, seed=0)
        target = get_target(rc, args.device)
        value_range = (-1.0, 1.0)
    else:
        target = load_tmdr_target(args.device)
        value_range = (0.0, 1.0)
    apd = ResidMLPAPDModel(target.config, C=cfg["C"], m=cfg["m"]).to(args.device)
    apd.load_state_dict(torch.load(OUT_ROOT / args.run / "apd_model.pth",
                                   weights_only=True, map_location=args.device))
    print(json.dumps({"run": args.run,
                      **functional_report(target, apd, args.device,
                                          value_range=value_range)}, indent=2))


if __name__ == "__main__":
    main()
