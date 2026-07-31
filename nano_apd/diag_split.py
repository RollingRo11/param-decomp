"""Diagnose WHY recursive split-and-test failed on the TMDR benchmark.

Three isolated tests:
  A. representation ceiling — oracle coordinate partition: assign every coordinate to the
     feature with the largest mean |grad*param| on that feature's single-feature probes
     (the best case for ANY disjoint-coordinate-partition method), then evaluate it with
     the piece metrics. If this is bad, coordinate partitions fundamentally cannot
     express the mechanisms (they overlap in coordinates) and the search was never the
     issue. Also reports per-feature attribution overlap stats.
  B. search criterion — force 2-means splits unconditionally (no DL acceptance test)
     down to n_blocks_forced blocks, evaluate. Good result here + bad greedy result
     = the myopic DL acceptance was the blocker.
  C. clustering signal — on single-active-feature samples, does the first 2-means split
     align with feature identity? Reports mean per-feature coherence (1.0 = all of a
     feature's samples land in one cluster).

    python -m nano_apd.diag_split
"""

import argparse
import json

import einops
import torch
import torch.nn.functional as F

from nano_apd.models import SparseFeatureDataset
from nano_apd.run_tmdr import OUT_ROOT, load_tmdr_target
from nano_apd.split_test import (
    evaluate_pieces,
    blocks_to_component_weights,
    kmeans2,
    per_sample_grad_times_param,
)


def per_feature_attribution(target, device, n_probe=64):
    """mean |grad*param| per feature over single-feature probes. [nf, D]"""
    nf = target.config.n_features
    rows = []
    for i in range(nf):
        x = torch.zeros(n_probe, 1, nf, device=device)
        x[:, 0, i] = torch.rand(n_probe, device=device)
        V, names, shapes, slices = per_sample_grad_times_param(target, x)
        rows.append(V.abs().mean(dim=0))
    return torch.stack(rows), names, shapes, slices              # [nf, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_blocks_forced", type=int, default=128)
    args = parser.parse_args()
    device = args.device
    target = load_tmdr_target(device)
    nf = target.config.n_features

    # --- A: oracle coordinate partition + overlap stats
    A, names, shapes, slices = per_feature_attribution(target, device)
    An = F.normalize(A, dim=1)
    overlap = An @ An.t()
    off_diag = overlap[~torch.eye(nf, dtype=torch.bool, device=device)]
    # participation ratio of each feature's attribution vector (effective #coords)
    p = A / (A.sum(dim=1, keepdim=True) + 1e-12)
    part_ratio = 1.0 / (p.pow(2).sum(dim=1) + 1e-12)
    assign = A.argmax(dim=0)                                     # coord -> feature [D]
    blocks = [(assign == i) for i in range(nf)]
    n_empty = sum(1 for b in blocks if int(b.sum()) == 0)
    blocks = [b for b in blocks if int(b.sum()) > 0]
    cw = blocks_to_component_weights(blocks, target, shapes, slices)
    oracle_partition = evaluate_pieces(target, cw, device)
    print("A. per-feature attribution overlap: mean off-diag cosine "
          f"{off_diag.mean():.3f} (p90 {off_diag.quantile(0.9):.3f}); "
          f"participation ratio median {part_ratio.median():.0f} coords "
          f"(of {A.shape[1]})", flush=True)
    print(f"A. ORACLE coordinate partition ({len(blocks)} nonempty pieces, "
          f"{n_empty} features got no coords): {json.dumps(oracle_partition)}", flush=True)

    # --- B: forced splits, no DL acceptance
    dataset = SparseFeatureDataset(1, nf, 0.01, device, (0.0, 1.0))
    x = dataset.generate_batch(8192)
    x = x[(x.abs().sum(dim=(1, 2)) > 0)]
    V, _, _, _ = per_sample_grad_times_param(target, x)
    V_abs = V.abs()
    forced = [torch.ones(V.shape[1], dtype=torch.bool, device=device)]
    while len(forced) < args.n_blocks_forced:
        sizes = [int(b.sum()) for b in forced]
        bi = int(torch.tensor(sizes).argmax())
        b = forced[bi]
        if sizes[bi] < 16:
            break
        share = V_abs[:, b].sum(dim=1) / (V_abs.sum(dim=1) + 1e-12)
        users = share > 0.05
        if int(users.sum()) < 32:
            break
        cl = kmeans2(F.normalize(V[users][:, b], dim=1))
        if cl.all() or (~cl).all():
            break
        mean_a = V_abs[users][:, b][~cl].mean(dim=0)
        mean_b = V_abs[users][:, b][cl].mean(dim=0)
        to_b = mean_b > mean_a
        if int(to_b.sum()) == 0 or int((~to_b).sum()) == 0:
            break
        idxs = b.nonzero().squeeze(1)
        ca = torch.zeros_like(b); ca[idxs[~to_b]] = True
        cb = torch.zeros_like(b); cb[idxs[to_b]] = True
        forced = forced[:bi] + [ca, cb] + forced[bi + 1:]
    cw = blocks_to_component_weights(forced, target, shapes, slices)
    forced_eval = evaluate_pieces(target, cw, device)
    print(f"B. FORCED splits (no DL test, {len(forced)} blocks): "
          f"{json.dumps(forced_eval)}", flush=True)

    # --- C: first-split cluster coherence on single-active samples
    single = (x[:, 0] != 0).sum(dim=1) == 1
    feat_id = x[single, 0].argmax(dim=1)
    cl = kmeans2(F.normalize(V[single.nonzero().squeeze(1)], dim=1))
    coherence = []
    for i in range(nf):
        m = feat_id == i
        if int(m.sum()) >= 5:
            frac = cl[m].float().mean().item()
            coherence.append(max(frac, 1 - frac))
    coherence = torch.tensor(coherence)
    print(f"C. first-split feature coherence: mean {coherence.mean():.3f} "
          f"(1.0 = each feature's samples land wholly in one cluster; "
          f"0.5 = split ignores feature identity), n_features measured {len(coherence)}",
          flush=True)


if __name__ == "__main__":
    main()
