"""Recursive split-and-test decomposition of the TMDR target (Rohan's idea, 2026-07-16).

Start from ONE block containing every decomposed parameter (both matrices of both MLP
layers). Repeatedly, for each leaf block:
  1. collect per-sample attribution vectors v(x) = grad_theta s(x) * theta restricted to
     the block's coordinates, with s(x) = ||y(x)||^2 / 2 (per-sample gradients come from
     cached activations x output-gradients, no per-sample autograd);
  2. cluster the (normalized, signed) vectors of the samples that use the block (2-means);
  3. split the block: each coordinate goes to the cluster with larger mean |v| there —
     children are DISJOINT coordinate partitions, so sum(children) == parent and
     faithfulness of the full decomposition is exact by construction, always;
  4. keep the split only if mean per-input description length improves:
     DL = E_x [ sum of size(b) over blocks b active on x ], where block b is active on x
     if its share of x's total attribution mass exceeds `active_frac`.
Recurse until no split improves DL (or max_blocks). The leaves are the components.

Everything is post-hoc on the trained target — no decomposition training.

    python -m nano_apd.split_test [--n_samples 8192] [--active_frac 0.01] [--max_blocks 256]
"""

import argparse
import json

import einops
import torch
import torch.nn.functional as F

from nano_apd.models import SparseFeatureDataset
from nano_apd.run_resid import diag_relu_conns
from nano_apd.run_tmdr import OUT_ROOT, load_tmdr_target


def per_sample_grad_times_param(target, x_flat):
    """v[n, :] = (d s / d theta)(x_n) * theta, s = ||y||^2/2, over all decomposed
    coordinates (concatenated per-matrix, flattened). Returns (V [N, D], names, shapes,
    slices) with per-matrix coordinate slices."""
    param_names = target.param_names()
    out, cache = target(x_flat)
    s = 0.5 * (out ** 2).sum()
    posts = [cache[n]["post"] for n in param_names]
    grads = torch.autograd.grad(s, posts)  # per-sample rows since posts are batched
    Vs, slices, shapes = [], {}, {}
    off = 0
    for n, g in zip(param_names, grads, strict=True):
        pre = cache[n]["pre"].detach()[:, 0]          # [N, d_in]
        gp = g.detach()[:, 0]                          # [N, d_out]
        W = target.weights()[n].detach()[0]            # [d_in, d_out]
        v = einops.einsum(pre, gp, "n a, n b -> n a b") * W
        Vs.append(einops.rearrange(v, "n a b -> n (a b)"))
        shapes[n] = tuple(W.shape)
        slices[n] = (off, off + W.numel())
        off += W.numel()
    return torch.cat(Vs, dim=1), param_names, shapes, slices


def description_length(V_abs, blocks, active_frac):
    """Mean per-input description length. blocks: list of bool coordinate masks [D]."""
    total = V_abs.sum(dim=1, keepdim=True) + 1e-12                  # [N, 1]
    dl = torch.zeros(V_abs.shape[0], device=V_abs.device)
    for b in blocks:
        share = V_abs[:, b].sum(dim=1, keepdim=True) / total        # [N, 1]
        active = (share > active_frac).squeeze(1)
        dl = dl + active.float() * b.sum()
    return dl.mean().item()


def kmeans2(X, iters=25, seed=0):
    """2-means with cosine geometry (rows pre-normalized). Returns bool assignment [N]."""
    g = torch.Generator(device=X.device.type if X.device.type == "cpu" else None)
    idx = torch.randperm(X.shape[0], generator=None)[:2]
    centers = X[idx].clone()
    assign = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
    for _ in range(iters):
        d = X @ centers.t()                                          # cosine sim [N, 2]
        new_assign = d[:, 1] > d[:, 0]
        if (new_assign == assign).all():
            break
        assign = new_assign
        for k in (0, 1):
            members = X[assign == bool(k)]
            if len(members):
                c = members.mean(dim=0)
                centers[k] = c / (c.norm() + 1e-12)
    return assign


def split_and_test(V, active_frac, max_blocks, min_block=8, min_samples=32, seed=0):
    """Recursive splitting. V [N, D] per-sample grad*param. Returns list of bool masks."""
    torch.manual_seed(seed)
    V_abs = V.abs()
    D = V.shape[1]
    blocks = [torch.ones(D, dtype=torch.bool, device=V.device)]
    dl = description_length(V_abs, blocks, active_frac)
    improved = True
    while improved and len(blocks) < max_blocks:
        improved = False
        for bi in range(len(blocks)):
            b = blocks[bi]
            if int(b.sum()) < 2 * min_block:
                continue
            total = V_abs.sum(dim=1) + 1e-12
            share = V_abs[:, b].sum(dim=1) / total
            users = share > active_frac
            if int(users.sum()) < min_samples:
                continue
            Xb = F.normalize(V[users][:, b], dim=1)
            assign = kmeans2(Xb)
            if assign.all() or (~assign).all():
                continue
            mean_a = V_abs[users][:, b][~assign].mean(dim=0)
            mean_b_ = V_abs[users][:, b][assign].mean(dim=0)
            to_b = mean_b_ > mean_a                                   # per-coordinate
            if int(to_b.sum()) < min_block or int((~to_b).sum()) < min_block:
                continue
            idxs = b.nonzero().squeeze(1)
            child_a = torch.zeros_like(b); child_a[idxs[~to_b]] = True
            child_b = torch.zeros_like(b); child_b[idxs[to_b]] = True
            candidate = blocks[:bi] + [child_a, child_b] + blocks[bi + 1:]
            cand_dl = description_length(V_abs, candidate, active_frac)
            if cand_dl < dl:
                blocks = candidate
                dl = cand_dl
                improved = True
                print(f"split kept: {len(blocks)} blocks, sizes "
                      f"{sorted(int(x.sum()) for x in blocks)[-6:]}..., DL {dl:.1f}", flush=True)
                break  # restart scan over the new block list
    return blocks, dl


def blocks_to_component_weights(blocks, target, shapes, slices):
    """Each block -> a component: the target weights masked to the block's coordinates.
    Returns dict name -> [1, C, d_in, d_out] (sums exactly to the target weights)."""
    C = len(blocks)
    cw = {}
    for n, (a, b) in shapes.items():
        lo, hi = slices[n]
        W = target.weights()[n].detach()[0]
        pieces = []
        for blk in blocks:
            m = blk[lo:hi].reshape(a, b).float()
            pieces.append(W * m)
        cw[n] = torch.stack(pieces).unsqueeze(0)                      # [1, C, a, b]
    return cw


@torch.no_grad()
def evaluate_pieces(target, cw, device, n_probe=64):
    """Oracle/conns metrics for piece components (masked forward = sum selected pieces)."""
    nf = target.config.n_features
    C = next(iter(cw.values())).shape[1]
    names = target.param_names()
    orig = {n: target.weights()[n].data.clone() for n in names}

    def forward_with(piece_sel):
        for n in names:
            w = einops.einsum(cw[n][0, piece_sel], "s a b -> a b") if piece_sel.any() else \
                torch.zeros_like(orig[n][0])
            target.weights()[n].data[0] = w
        out, _ = target(x_flat)
        for n in names:
            target.weights()[n].data[:] = orig[n]
        return out

    x = torch.zeros(nf, n_probe, 1, nf, device=device)
    for i in range(nf):
        x[i, :, 0, i] = torch.rand(n_probe, device=device)
    x_flat = einops.rearrange(x, "f p i nf -> (f p) i nf")
    tgt, _ = target(x_flat)
    own = einops.rearrange(tgt, "(f p) i nf -> f p i nf", f=nf)
    own_dim = torch.stack([own[i, :, 0, i] for i in range(nf)])
    var = own_dim.var(dim=1) + 1e-8

    err = torch.zeros(nf, C, device=device)
    for c in range(C):
        sel = torch.zeros(C, dtype=torch.bool, device=device); sel[c] = True
        pred = forward_with(sel)
        po = einops.rearrange(pred, "(f p) i nf -> f p i nf", f=nf)
        pd = torch.stack([po[i, :, 0, i] for i in range(nf)])
        err[:, c] = ((pd - own_dim) ** 2).mean(dim=1) / var
    assigned = err.argmin(dim=1)
    keep_only = err[torch.arange(nf), assigned].mean().item()

    # cross-layer span of assigned pieces
    layer_norm = torch.zeros(C, 2, device=device)
    for n in names:
        l = int(n.split(".")[1])
        layer_norm[:, l] += cw[n][0].pow(2).sum(dim=(-2, -1))
    layer_norm = layer_norm.sqrt()
    a = layer_norm[assigned]
    span = a.min(dim=1).values / (a.max(dim=1).values + 1e-8)
    return {
        "separation_oracle": assigned.unique().numel() / nf,
        "cross_layer_oracle": (span > 0.1).float().mean().item(),
        "keep_only_oracle": keep_only,
        "n_blocks": C,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=8192)
    parser.add_argument("--active_frac", type=float, default=0.01)
    parser.add_argument("--max_blocks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = args.device

    target = load_tmdr_target(device)
    dataset = SparseFeatureDataset(1, target.config.n_features, 0.01, device, (0.0, 1.0))
    x = dataset.generate_batch(args.n_samples)
    x = x[(x.abs().sum(dim=(1, 2)) > 0)]            # drop all-zero samples
    print(f"{x.shape[0]} non-empty samples", flush=True)

    V, names, shapes, slices = per_sample_grad_times_param(target, x)
    print(f"attribution matrix {tuple(V.shape)}", flush=True)

    blocks, dl = split_and_test(V, args.active_frac, args.max_blocks, seed=args.seed)
    cw = blocks_to_component_weights(blocks, target, shapes, slices)
    summary = evaluate_pieces(target, cw, device)
    summary["final_dl"] = dl
    summary["block_sizes"] = sorted(int(b.sum()) for b in blocks)

    out_dir = OUT_ROOT / f"tmdr_split_test_af{args.active_frac:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"blocks": [b.cpu() for b in blocks]}, out_dir / "blocks.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
