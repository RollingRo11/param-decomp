"""Refinement rounds on top of split-and-test v2 (Lloyd-style, still training-free).

Each round:
  1. REASSIGN — for every sample, run each piece alone and assign the sample to the
     piece that best reconstructs the full model's output on it (reconstruction picks).
  2. SPLIT — every group additionally gets a provisional 2-means sub-split in
     attribution space (both halves kept if big enough). Reassignment can merge
     duplicates but can never split a blended group by itself; the provisional splits
     give it purer candidates to migrate to, and spurious splits dissolve next round.
  3. REBUILD — new pieces from the new groups (same ridge solve as split_v2).

    python -m nano_apd.refine_v2 [--rounds 5] [--filter_frac 0.0]
"""

import argparse
import json

import einops
import torch
import torch.nn.functional as F

from nano_apd.run_tmdr import OUT_ROOT, load_tmdr_target
from nano_apd.split_test import kmeans2
from nano_apd.split_v2 import build_pieces, collect_data, evaluate, recursive_cluster


@torch.no_grad()
def per_sample_piece_errors(target, cw, x_flat, names):
    """err[n, k] = MSE over output dims of (piece k alone) vs full model, per sample."""
    K = next(iter(cw.values())).shape[1] - 1
    orig = {n: target.weights()[n].data.clone() for n in names}
    y_full, _ = target(x_flat)
    errs = []
    for k in range(K):
        for n in names:
            target.weights()[n].data[0] = cw[n][0, k]
        y_k, _ = target(x_flat)
        errs.append(((y_k - y_full) ** 2).mean(dim=(1, 2)))
    for n in names:
        target.weights()[n].data[:] = orig[n]
    return torch.stack(errs, dim=1)                              # [N, K]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--filter_frac", type=float, default=0.0,
                        help="drop this fraction of worst-explained samples from rebuilding")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    target = load_tmdr_target(device)
    x, V, pres, gposts, names, shapes, slices = collect_data(target, device, args.n_samples)
    x_flat = x
    Vn = F.normalize(V, dim=1)
    print(f"{x.shape[0]} samples", flush=True)

    groups = recursive_cluster(V, min_size=args.min_size)
    cw, resid_rel = build_pieces(target, groups, V, pres, gposts, names, shapes, slices,
                                 lam=args.lam)
    history = []
    for rnd in range(args.rounds + 1):
        alone = evaluate(target, cw, device, with_residual_backbone=False)
        with_bb = evaluate(target, cw, device, with_residual_backbone=True)
        rec = {"round": rnd, "n_groups": len(groups),
               "sep": alone["separation_oracle"], "keep_alone": alone["keep_only_oracle"],
               "keep_bb": with_bb["keep_only_oracle"],
               "xlayer": alone["cross_layer_oracle"]}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if rnd == args.rounds:
            break

        # 1. reassign by reconstruction
        err = per_sample_piece_errors(target, cw, x_flat, names)   # [N, K]
        best_err, assign = err.min(dim=1)
        keep = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        if args.filter_frac > 0:
            cutoff = best_err.quantile(1 - args.filter_frac)
            keep = best_err <= cutoff

        # 2. rebuild groups; provisional 2-means sub-split of each
        new_groups = []
        for k in range(err.shape[1]):
            idx = ((assign == k) & keep).nonzero().squeeze(1)
            if len(idx) < args.min_size:
                continue
            if len(idx) >= 2 * args.min_size:
                cl = kmeans2(Vn[idx])
                a, b = idx[~cl], idx[cl]
                if len(a) >= args.min_size and len(b) >= args.min_size:
                    new_groups.extend([a, b])
                    continue
            new_groups.append(idx)
        groups = new_groups

        # 3. rebuild pieces
        cw, resid_rel = build_pieces(target, groups, V, pres, gposts, names, shapes,
                                     slices, lam=args.lam)

    out_dir = OUT_ROOT / f"tmdr_refine_v2_ff{args.filter_frac:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("final residual_rel:", {k: round(v, 3) for k, v in resid_rel.items()}, flush=True)


if __name__ == "__main__":
    main()
