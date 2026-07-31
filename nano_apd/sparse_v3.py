"""Sparse-decomposition carving (v3): replace hard clustering of attribution vectors
with sparse coding against a dictionary of archetypal attribution patterns.

Motivation (see transcript 2026-07-16): hard assignment builds blended pieces from
(a) multi-feature samples, whose attribution genuinely is a SUM of mechanisms, and
(b) similar sibling features. Sparse coding lets one sample load on several archetypes.

No gradient training anywhere — alternating closed-form steps:
  1. dictionary init: centroids of the recursive attribution clustering (split_v2);
  2. sparse coding: per sample, matching pursuit over the dictionary (pick the atom
     most correlated with the residual vector, re-fit coefficients of the selected
     atoms by least squares, repeat up to max_atoms or until the residual is small);
  3. dictionary update: each atom re-estimated from its users' vectors minus the other
     atoms' parts (weighted mean), i.e. the single-mechanism content of every sample
     that touches it — pair samples now CONTRIBUTE TO BOTH atoms instead of blending;
  4. prune unused atoms; provisionally 2-means-split each atom's user set and add the
     halves' centroids as candidate atoms (usage decides survival next round).
After R rounds, build rank-1 pieces per atom (attribution-weighted anchors, joint ridge
solve, residual bucket) and evaluate — same machinery as split_v2.

    python -m nano_apd.sparse_v3
"""

import argparse
import json

import torch
import torch.nn.functional as F

from nano_apd.run_tmdr import OUT_ROOT, load_tmdr_target
from nano_apd.split_test import kmeans2
from nano_apd.split_v2 import build_pieces, collect_data, evaluate, recursive_cluster


def matching_pursuit(Vn, D, max_atoms=3, resid_tol=0.35):
    """Sparse-code each row of Vn (normalized) over dictionary D [K, dim] (normalized).
    Returns coeffs [N, K] (signed; batched least-squares refit on the selected support
    after each atom pick — supports are padded to the current iteration count)."""
    N = Vn.shape[0]
    K = D.shape[0]
    device = Vn.device
    coeffs = torch.zeros(N, K, device=device)
    picked = torch.full((N, max_atoms), -1, dtype=torch.long, device=device)
    resid = Vn.clone()
    for t in range(max_atoms):
        active = resid.norm(dim=1) > resid_tol
        if not active.any():
            break
        corr = resid @ D.t()                                    # [N, K]
        if t > 0:
            corr.scatter_(1, picked[:, :t].clamp(min=0), 0.0)
        pick = corr.abs().argmax(dim=1)
        # samples not active keep a duplicate of their first pick (harmless padding)
        picked[:, t] = torch.where(active, pick, picked[:, max(t - 1, 0)].clamp(min=0))
        S = picked[:, : t + 1].clamp(min=0)                     # [N, t+1]
        atoms = D[S]                                            # [N, t+1, dim]
        G = atoms @ atoms.transpose(1, 2)                       # [N, t+1, t+1]
        G = G + 1e-6 * torch.eye(t + 1, device=device)
        rhs = (atoms @ Vn.unsqueeze(2))                         # [N, t+1, 1]
        c = torch.linalg.solve(G, rhs).squeeze(2)               # [N, t+1]
        coeffs.zero_()
        coeffs.scatter_add_(1, S, c)
        resid = Vn - coeffs @ D
    return coeffs


def dictionary_update(Vn, D, coeffs, min_users=12):
    """Atom k <- weighted mean of (sample - other atoms' parts) over its users."""
    K = D.shape[0]
    newD, keep = [], []
    for k in range(K):
        users = coeffs[:, k].abs() > 1e-4
        if int(users.sum()) < min_users:
            continue
        c = coeffs[users]
        contrib = Vn[users] - (c @ D - torch.outer(c[:, k], D[k]))
        w = c[:, k].abs().unsqueeze(1)
        atom = (contrib * torch.sign(c[:, k]).unsqueeze(1) * w).sum(dim=0)
        newD.append(F.normalize(atom, dim=0))
        keep.append(k)
    return torch.stack(newD), keep


def propose_splits(Vn, coeffs, D, keep_idx, min_size=16):
    """2-means each atom's user set; return candidate atoms from the halves."""
    cands = []
    for j in range(D.shape[0]):
        users = (coeffs[:, keep_idx[j]].abs() > 1e-4).nonzero().squeeze(1)
        if len(users) < 2 * min_size:
            continue
        cl = kmeans2(Vn[users])
        a, b = users[~cl], users[cl]
        if len(a) >= min_size and len(b) >= min_size:
            ca = F.normalize(Vn[a].mean(dim=0), dim=0)
            cb = F.normalize(Vn[b].mean(dim=0), dim=0)
            # only propose if the halves genuinely differ from the atom itself
            if min((ca - D[j]).norm(), (cb - D[j]).norm()) > 0.3:
                cands.extend([ca, cb])
    return cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--max_atoms", type=int, default=3)
    parser.add_argument("--resid_tol", type=float, default=0.35)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    target = load_tmdr_target(device)
    x, V, pres, gposts, names, shapes, slices = collect_data(target, device, args.n_samples)
    Vn = F.normalize(V, dim=1)
    print(f"{x.shape[0]} samples", flush=True)

    groups = recursive_cluster(V)
    D = torch.stack([F.normalize(Vn[g].mean(dim=0), dim=0) for g in groups])
    print(f"dictionary init: {D.shape[0]} atoms", flush=True)

    history = []
    for rnd in range(args.rounds):
        coeffs = matching_pursuit(Vn, D, max_atoms=args.max_atoms, resid_tol=args.resid_tol)
        code_resid = (Vn - coeffs @ D).norm(dim=1).mean().item()
        atoms_per_sample = (coeffs.abs() > 1e-4).sum(dim=1).float().mean().item()

        # build pieces from current coding (weights = |coeff| on the atom)
        weights = coeffs.abs()
        leaves, w_leaves = [], []
        for k in range(D.shape[0]):
            users = (weights[:, k] > 1e-4).nonzero().squeeze(1)
            if len(users) >= 8:
                leaves.append(users)
                w_leaves.append(weights[users, k])
        # weighted anchors: reuse build_pieces by pre-scaling? build_pieces weights by
        # matrix mass only; incorporate coding weight by repeating the weighted mean here
        cw, resid_rel = build_pieces(target, leaves, V, pres, gposts, names, shapes,
                                     slices, lam=args.lam)
        res = evaluate(target, cw, device, with_residual_backbone=False)
        res_bb = evaluate(target, cw, device, with_residual_backbone=True)
        rec = {"round": rnd, "atoms": D.shape[0], "code_resid": round(code_resid, 3),
               "atoms_per_sample": round(atoms_per_sample, 2),
               "sep": res["separation_oracle"], "keep_alone": res["keep_only_oracle"],
               "keep_bb": res_bb["keep_only_oracle"], "xlayer": res["cross_layer_oracle"]}
        history.append(rec)
        print(json.dumps(rec), flush=True)

        if rnd == args.rounds - 1:
            break
        D, keep_idx = dictionary_update(Vn, D, coeffs)
        cands = propose_splits(Vn, coeffs, D, keep_idx)
        if cands:
            D = torch.cat([D, torch.stack(cands)], dim=0)

    out_dir = OUT_ROOT / "tmdr_sparse_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
