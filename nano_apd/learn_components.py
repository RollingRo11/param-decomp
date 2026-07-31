"""Learn full components on the TMDR benchmark, warm-started from split-and-test v2
pieces (attribution-clustered, refined, ridge-solved — see split_v2.py / refine_v2.py).

Pipeline:
  1. build v2 pieces and run a few refinement rounds; keep the snapshot with the best
     separation (the pieces are rank-1 with a residual bucket — used ONLY as init);
  2. convert pieces to an APD component model (A/B factors via SVD, rank m per matrix,
     the residual bucket is spread equally over all components so nothing is left out);
  3. train with: faithfulness (components sum to the target weights), per-sample
     epsilon-sufficiency reconstruction (greedy_add selection — smallest component set
     whose masked forward reconstructs the sample below eps; no k anywhere), and
     activation reconstruction. The warm start gives the certifier real structure from
     step 0, which is what from-scratch epsilon-selection lacked.
  4. evaluate with the corrected benchmark metrics.

    python -m nano_apd.learn_components
"""

import argparse
import json

import einops
import torch
import torch.nn.functional as F

from nano_apd.apd import APDConfig, optimize
from nano_apd.models import ResidMLPAPDModel, SparseFeatureDataset
from nano_apd.run_tmdr import OUT_ROOT, eval_tmdr, load_tmdr_target
from nano_apd.split_test import kmeans2
from nano_apd.split_v2 import build_pieces, collect_data, evaluate, recursive_cluster
from nano_apd.refine_v2 import per_sample_piece_errors


def carve_init_pieces(target, device, n_samples, refine_rounds, lam, min_size=16):
    """split_v2 + refinement; returns the cw snapshot with best separation."""
    x, V, pres, gposts, names, shapes, slices = collect_data(target, device, n_samples)
    Vn = F.normalize(V, dim=1)
    groups = recursive_cluster(V, min_size=min_size)
    cw, _ = build_pieces(target, groups, V, pres, gposts, names, shapes, slices, lam=lam)
    best_cw, best_sep = cw, -1.0
    for rnd in range(refine_rounds):
        res = evaluate(target, cw, device, with_residual_backbone=False)
        print(f"carve round {rnd}: sep {res['separation_oracle']:.2f} "
              f"keep {res['keep_only_oracle']:.3f} groups {len(groups)}", flush=True)
        if res["separation_oracle"] > best_sep:
            best_sep, best_cw = res["separation_oracle"], cw
        err = per_sample_piece_errors(target, cw, x, names)
        assign = err.argmin(dim=1)
        new_groups = []
        for k in range(err.shape[1]):
            idx = (assign == k).nonzero().squeeze(1)
            if len(idx) < min_size:
                continue
            if len(idx) >= 2 * min_size:
                cl = kmeans2(Vn[idx])
                a, b = idx[~cl], idx[cl]
                if len(a) >= min_size and len(b) >= min_size:
                    new_groups.extend([a, b])
                    continue
            new_groups.append(idx)
        groups = new_groups
        cw, _ = build_pieces(target, groups, V, pres, gposts, names, shapes, slices, lam=lam)
    res = evaluate(target, cw, device, with_residual_backbone=False)
    if res["separation_oracle"] > best_sep:
        best_sep, best_cw = res["separation_oracle"], cw
    print(f"init snapshot: separation {best_sep:.2f}", flush=True)
    return best_cw


def init_apd_from_pieces(target, cw, m, device):
    """SVD each piece into rank-m A/B factors; spread the residual bucket equally."""
    K = next(iter(cw.values())).shape[1] - 1
    apd = ResidMLPAPDModel(target.config, C=K, m=m).to(device)
    apd.W_E.data[:] = target.W_E.data.clone()
    apd.W_U.data[:] = target.W_U.data.clone()
    for l in range(target.config.n_layers):
        apd.bias1[l].data[:] = target.bias1[l].data.clone()
    for n, comp in [(f"layers.{l}.{p}", None) for l in range(target.config.n_layers)
                    for p in ("mlp_in", "mlp_out")]:
        pieces = cw[n][0]                                     # [K+1, d_in, d_out]
        spread = pieces[:K] + pieces[K:K + 1] / K             # absorb residual equally
        U, S, Vh = torch.linalg.svd(spread, full_matrices=False)
        r = min(m, S.shape[-1])
        A = U[..., :r] * S[..., None, :r].sqrt()              # [K, d_in, r]
        B = S[..., :r, None].sqrt() * Vh[..., :r, :]          # [K, r, d_out]
        comp_mod = apd.mlp_in[int(n.split(".")[1])] if "mlp_in" in n else \
            apd.mlp_out[int(n.split(".")[1])]
        comp_mod.A.data.zero_(); comp_mod.B.data.zero_()
        comp_mod.A.data[0, :, :, :r] = A
        comp_mod.B.data[0, :, :r, :] = B
        # tiny noise on the unused ranks so they can be recruited if needed
        comp_mod.A.data[0, :, :, r:].normal_(0, 1e-3)
        comp_mod.B.data[0, :, r:, :].normal_(0, 1e-3)
    return apd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=16384)
    parser.add_argument("--refine_rounds", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--param_match", type=float, default=1.0)
    parser.add_argument("--topk_recon", type=float, default=2.0)
    parser.add_argument("--act_recon", type=float, default=1.0)
    parser.add_argument("--schatten", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    target = load_tmdr_target(device)
    cw = carve_init_pieces(target, device, args.n_samples, args.refine_rounds, args.lam)
    apd_model = init_apd_from_pieces(target, cw, m=args.m, device=device)

    config = APDConfig(
        C=apd_model.C, topk=None, batch_size=256, steps=args.steps, lr=args.lr,
        seed=args.seed, lr_schedule="cosine", lr_warmup_pct=0.01,
        param_match_coeff=args.param_match, topk_recon_coeff=args.topk_recon,
        act_recon_coeff=args.act_recon,
        schatten_coeff=args.schatten,
        schatten_pnorm=0.9 if args.schatten is not None else None,
        selection="greedy_add", eps=args.eps, attribution_type="gradient",
        print_freq=250, extra={"init": "split_v2_refined", "m": args.m},
    )
    param_names = target.param_names()
    dataset = SparseFeatureDataset(1, target.config.n_features, 0.01, device, (0.0, 1.0))
    act_recon_transform = None
    if args.act_recon is not None:
        biases = [apd_model.bias1[l].data for l in range(target.config.n_layers)]
        act_recon_transform = lambda acts: {
            k: F.relu(v + biases[int(k.split(".")[1])])
            for k, v in acts.items() if "mlp_in" in k
        }

    out_dir = OUT_ROOT / f"tmdr_learned_C{apd_model.C}{('_' + args.tag) if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    optimize(model=apd_model, target_model=target, config=config,
             generate_batch=dataset.generate_batch, param_names=param_names,
             device=device, out_dir=out_dir, act_recon_transform=act_recon_transform)

    summary = eval_tmdr(target, apd_model, device)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "best_component_per_feature"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
