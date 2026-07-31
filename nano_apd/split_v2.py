"""Recursive split-and-test v2: value splits instead of coordinate splits.

v1 failed because "split" partitioned parameter COORDINATES, and this model's mechanisms
overlap in coordinates with signed cancellation (see diag_split.py). v2 keeps the same
propose/certify loop but a split now divides parameter VALUES: each leaf cluster gets a
signed rank-1 piece per matrix (u_k c_k^T), pieces overlap and are overcomplete, and the
joint least-squares remainder is kept as one explicit residual piece so that the pieces
sum to the target weights EXACTLY.

Pipeline:
  1. per-sample attribution vectors v(x) = grad_theta(||y||^2/2) * theta (as in v1);
  2. recursive 2-means on normalized signed v; accept a sub-split only if the size-
     weighted mean cosine-to-centroid of the children beats the parent's by `margin`
     ("children are used differently"); leaf count = discovered, no k;
  3. per leaf and per matrix: input direction u_k = normalized attribution-weighted mean
     of the matrix's input activations over the leaf's samples;
  4. per matrix: solve C = lstsq(U, W) jointly over all leaves -> pieces u_k c_k^T,
     plus residual piece W - U C (exact faithfulness);
  5. evaluate keep-only / separation / cross-layer for the leaf pieces, with and without
     the residual piece held always-active.

    python -m nano_apd.split_v2
"""

import argparse
import json

import einops
import torch
import torch.nn.functional as F

from nano_apd.models import SparseFeatureDataset
from nano_apd.run_tmdr import OUT_ROOT, load_tmdr_target
from nano_apd.split_test import kmeans2, per_sample_grad_times_param


def collect_data(target, device, n_samples):
    dataset = SparseFeatureDataset(1, target.config.n_features, 0.01, device, (0.0, 1.0))
    x = dataset.generate_batch(n_samples)
    x = x[(x.abs().sum(dim=(1, 2)) > 0)]
    V, names, shapes, slices = per_sample_grad_times_param(target, x)
    out, cache = target(x)
    s = 0.5 * (out ** 2).sum()
    posts = [cache[n]["post"] for n in names]
    grads = torch.autograd.grad(s, posts)
    pres = {n: cache[n]["pre"].detach()[:, 0] for n in names}     # [N, d_in] per matrix
    gposts = {n: g.detach()[:, 0] for n, g in zip(names, grads, strict=True)}  # [N, d_out]
    return x, V, pres, gposts, names, shapes, slices


def coherence(X):
    """Size-weighted mean cosine of rows to their centroid. X rows normalized."""
    c = F.normalize(X.mean(dim=0), dim=0)
    return (X @ c).mean().item()


def recursive_cluster(V, min_size=16, margin=0.02, max_leaves=256):
    """Recursive 2-means on normalized rows of V. Returns list of index tensors."""
    Xall = F.normalize(V, dim=1)
    leaves, queue = [], [torch.arange(V.shape[0], device=V.device)]
    while queue and len(leaves) + len(queue) < max_leaves:
        idx = queue.pop()
        X = Xall[idx]
        if len(idx) < 2 * min_size:
            leaves.append(idx)
            continue
        assign = kmeans2(X)
        a, b = idx[~assign], idx[assign]
        if len(a) < min_size or len(b) < min_size:
            leaves.append(idx)
            continue
        parent_coh = coherence(X)
        child_coh = (coherence(Xall[a]) * len(a) + coherence(Xall[b]) * len(b)) / len(idx)
        if child_coh > parent_coh + margin:
            queue.extend([a, b])
        else:
            leaves.append(idx)
    leaves.extend(queue)
    return leaves


def _ridge_lstsq(U, W, lam):
    """argmin_C ||U C - W||^2 + lam_rel * ||C||^2. U [d, K], W [d, m] -> [K, m]."""
    K = U.shape[1]
    G = U.t() @ U
    reg = lam * G.diagonal().mean() * torch.eye(K, device=U.device)
    return torch.linalg.solve(G + reg, U.t() @ W)


def build_pieces(target, leaves, V, pres, gposts, names, shapes, slices, lam=1e-3):
    """Rank-1 piece per leaf per matrix; the known factor is taken on whichever side
    lives in the wide embedding space (input side for mlp_in, OUTPUT side for mlp_out —
    its 20-dim neuron input space cannot host K independent directions and plain lstsq
    blows up with mutually-cancelling coefficients). Joint ridge solve for the free
    factors; the remainder is kept as one explicit residual piece (exact faithfulness)."""
    K = len(leaves)
    cw, resid_rel = {}, {}
    for n in names:
        lo, hi = slices[n]
        W = target.weights()[n].detach()[0]                        # [d_in, d_out]
        w_mass = V[:, lo:hi].abs().sum(dim=1)                      # [N] usage of this matrix
        anchor = pres[n] if "mlp_in" in n else gposts[n]           # [N, 256] embed side
        U = []
        for idx in leaves:
            u = (anchor[idx] * w_mass[idx].unsqueeze(1)).sum(dim=0)
            U.append(F.normalize(u, dim=0))
        U = torch.stack(U, dim=1)                                  # [256, K]
        if "mlp_in" in n:
            C = _ridge_lstsq(U, W, lam)                            # [K, d_out]
            pieces = einops.einsum(U, C, "e k, k m -> k e m")
        else:
            D = _ridge_lstsq(U, W.t(), lam)                        # [K, d_in]
            pieces = einops.einsum(D, U, "k m, e k -> k m e")
        residual = (W - pieces.sum(dim=0)).unsqueeze(0)
        resid_rel[n] = (residual.norm() / W.norm()).item()
        cw[n] = torch.cat([pieces, residual], dim=0).unsqueeze(0)  # [1, K+1, ...]
    return cw, resid_rel


@torch.no_grad()
def evaluate(target, cw, device, with_residual_backbone, n_probe=64):
    """Keep-only/separation/cross-layer over the K leaf pieces (residual piece is index
    K; if with_residual_backbone, it is always active in every masked forward)."""
    nf = target.config.n_features
    K = next(iter(cw.values())).shape[1] - 1
    names = target.param_names()
    orig = {n: target.weights()[n].data.clone() for n in names}

    x = torch.zeros(nf, n_probe, 1, nf, device=device)
    for i in range(nf):
        x[i, :, 0, i] = torch.rand(n_probe, device=device)
    x_flat = einops.rearrange(x, "f p i nf -> (f p) i nf")
    tgt, _ = target(x_flat)
    own = einops.rearrange(tgt, "(f p) i nf -> f p i nf", f=nf)
    own_dim = torch.stack([own[i, :, 0, i] for i in range(nf)])
    var = own_dim.var(dim=1) + 1e-8

    def run(sel):
        for n in names:
            target.weights()[n].data[0] = cw[n][0, sel].sum(dim=0)
        out, _ = target(x_flat)
        for n in names:
            target.weights()[n].data[:] = orig[n]
        return out

    err = torch.zeros(nf, K, device=device)
    for c in range(K):
        sel = torch.zeros(K + 1, dtype=torch.bool, device=device)
        sel[c] = True
        sel[K] = with_residual_backbone
        pred = run(sel)
        po = einops.rearrange(pred, "(f p) i nf -> f p i nf", f=nf)
        pd = torch.stack([po[i, :, 0, i] for i in range(nf)])
        err[:, c] = ((pd - own_dim) ** 2).mean(dim=1) / var

    assigned = err.argmin(dim=1)
    layer_norm = torch.zeros(K, 2, device=device)
    for n in names:
        l = int(n.split(".")[1])
        layer_norm[:, l] += cw[n][0, :K].pow(2).sum(dim=(-2, -1))
    a = layer_norm.sqrt()[assigned]
    span = a.min(dim=1).values / (a.max(dim=1).values + 1e-8)
    return {
        "separation_oracle": assigned.unique().numel() / nf,
        "cross_layer_oracle": (span > 0.1).float().mean().item(),
        "keep_only_oracle": err[torch.arange(nf), assigned].mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=16384)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--max_leaves", type=int, default=256)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    target = load_tmdr_target(device)
    x, V, pres, gposts, names, shapes, slices = collect_data(target, device, args.n_samples)
    print(f"{x.shape[0]} non-empty samples, V {tuple(V.shape)}", flush=True)

    leaves = recursive_cluster(V, min_size=args.min_size, margin=args.margin,
                               max_leaves=args.max_leaves)
    sizes = sorted(len(l) for l in leaves)
    print(f"{len(leaves)} leaf clusters discovered (sizes {sizes[:3]}...{sizes[-3:]})",
          flush=True)

    cw, resid_rel = build_pieces(target, leaves, V, pres, gposts, names, shapes, slices,
                                 lam=args.lam)
    summary = {
        "n_leaves": len(leaves),
        "faithfulness_residual_rel": {k: round(v, 4) for k, v in resid_rel.items()},
        "pieces_alone": evaluate(target, cw, device, with_residual_backbone=False),
        "pieces_plus_residual_backbone": evaluate(target, cw, device,
                                                  with_residual_backbone=True),
    }
    out_dir = OUT_ROOT / "tmdr_split_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
