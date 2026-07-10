"""Pairwise causal-interaction structure of a trained AlgZoo decomposition.

For components i, j define the interaction as the second difference of ablation damage:

    I(i,j) = L({i,j}) - L({i}) - L({j}) + L({})

where L(S) = normalized MSE to the target output with the gates of S forced to 0 (others at CI).
  I ~ 0   -> i and j are causally independent modules (additive damage)
  I >> 0  -> super-additive: REDUNDANT pair (each masks the other's absence -- the blob-choir
             signature that rate-based losses cannot see)
  I << 0  -> sub-additive (overlapping damage; parts of the same broken pathway)

We ask: does interaction structure mirror the documented neuron groups (delay line / leave-one-out /
running max) better than co-activation or rates do? If yes, interaction is the right identity signal
for the dense regime, worth turning into a grouping step or a training loss.

Run:  CUDA_VISIBLE_DEVICES=1 python -m nano_param_decomp.alg_interaction
Env:  CKPT (default /tmp/algzoo/hc10_r1_hid.pt), TOPK (components analyzed), B
"""

import os
import sys

import torch
import torch.nn.functional as F

from .apd_mask import ApdConfig, clear_masks, install_banks, refresh_caches
from .apd_alg import AlgCI, UnrolledRNN, masked_forward_rnn
from .run import Config as VpdConfig

sys.path.insert(0, "/workspace/alg-zoo")


def load_alg_decomp(path: str, device: torch.device):
    from alg_zoo.handcrafted import handcrafted_2nd_argmax

    ck = torch.load(path, weights_only=False)
    cfg = ApdConfig(**ck["cfg"])
    ci_cfg = VpdConfig(**ck["ci_cfg"])
    seq_len = ci_cfg.seq_len
    model = UnrolledRNN.from_dist_rnn(handcrafted_2nd_argmax(seq_len)).to(device)
    banks = install_banks(model, cfg)
    model = model.to(device)
    for n, b in banks.items():
        b.load_state_dict(ck["banks"][n])
    ci = AlgCI(model.hidden_size, seq_len, cfg.n_components, ci_cfg).to(device)
    ci.load_state_dict(ck["ci"])
    return model, banks, ci, cfg


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.environ.get("CKPT", "/tmp/algzoo/hc10_r1_hid.pt")
    topk = int(os.environ.get("TOPK", "16"))
    B = int(os.environ.get("B", "8192"))
    model, banks, ci, cfg = load_alg_decomp(ckpt, device)
    T, C = model.seq_len, cfg.n_components

    x = torch.randn(B, T, device=device)
    clear_masks(banks)
    target = model(x)
    h_prev = model.trace_h_prev
    refresh_caches(banks)
    g_lower, g_upper = ci(x, h_prev)
    tvar = target.var() + 1e-8
    zeros = {n: torch.zeros(B, T, device=device) for n in banks}

    mean_g = g_upper.mean(dim=(0, 1))
    comps = mean_g.topk(topk).indices.tolist()

    def L(ablate: list[int]) -> float:
        gate = g_lower.clone()
        if ablate:
            gate[..., ablate] = 0.0
        pred = masked_forward_rnn(model, banks, x, gate, zeros)
        return (F.mse_loss(pred, target) / tvar).item()

    base = L([])
    singles = {c: L([c]) - base for c in comps}
    print(f"ckpt={ckpt}  base ci-masked mse={base:.4f}  top-{topk} comps by use", flush=True)
    print("single-ablation damage: " +
          " ".join(f"{c}:{singles[c]:.3f}" for c in comps), flush=True)

    K = len(comps)
    I = torch.zeros(K, K)
    for a in range(K):
        for b in range(a + 1, K):
            i, j = comps[a], comps[b]
            I[a, b] = I[b, a] = L([i, j]) - base - singles[i] - singles[j]

    # component -> documented group by write-mass (hc10 answer key)
    groups = {"delay": list(range(10)), "loo": list(range(10, 19)), "runmax": [19, 20, 21]}
    W = {n: b.materialized_weights() for n, b in banks.items()}
    labels = []
    for c in comps:
        nm = W["ih"][c].pow(2).sum(1) + W["hh"][c].pow(2).sum(1) + W["out"][c].pow(2).sum(0)
        nm = nm / (nm.sum() + 1e-12)
        gmass = {k: nm[v].sum().item() for k, v in groups.items()}
        labels.append(max(gmass, key=gmass.get))

    print("\npairwise interaction I(i,j) = L(both) - L(i) - L(j) + L(base)   [>0 = redundant pair]", flush=True)
    hdr = "        " + " ".join(f"{c:>6}" for c in comps)
    print(hdr, flush=True)
    for a in range(K):
        row = " ".join(f"{I[a, b]:>6.3f}" for b in range(K))
        print(f"{comps[a]:>4} {labels[a]:<4} {row}", flush=True)

    within, between = [], []
    for a in range(K):
        for b in range(a + 1, K):
            (within if labels[a] == labels[b] else between).append(I[a, b].item())
    t_within = sum(within) / max(1, len(within))
    t_between = sum(between) / max(1, len(between))
    print(f"\nmean interaction: same-group={t_within:.4f} (n={len(within)})  "
          f"cross-group={t_between:.4f} (n={len(between)})", flush=True)
    off = I[torch.triu(torch.ones_like(I), 1) > 0]
    print(f"|I| mean={off.abs().mean():.4f} max={off.abs().max():.4f}; "
          f"frac positive (redundant)={100 * (off > 0.01).float().mean():.0f}%", flush=True)
    print("ALG_INTERACTION DONE", flush=True)


if __name__ == "__main__":
    main()
