"""Which components activate on INDUCTION (repeated-context copying) — one, a few, or many?

Not circuit-recovery-in-one-component (retired): we ask which components are *induction-conditional*
and how the behavior is distributed across them.

Detection (token-identity controlled): sequences are [first_half, first_half]. For component c,
delta(c) = mean gate on second-copy positions minus mean gate on the SAME tokens at their first-copy
positions. A large positive delta means the component responds to "this token is a repeat with known
continuation", not to the token itself.

Causal check (set-wise dose-response): ablate the top-N delta components together, N = 1..64, and
measure induction copy accuracy (does the model predict the continuation it saw in the first copy?)
under the CI gate, vs ablating N random firing-rate-matched components. If induction is carried by k
components, accuracy should fall sharply around N ~ k for the targeted curve and stay flat for the
matched-random curve.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.induction_components
Env:  CKPT (default 100k best), B (default 128), TOPK_PRINT (default 10)
"""

import os

import torch
import torch.nn.functional as F

from .apd_lm import load_decomp, masked_forward
from .apd_mask import clear_masks, refresh_caches
from .pythia14m import load_pythia14m_target
from .run import induction_copy_acc

CKPT = os.environ.get("CKPT", "/tmp/pythia_compare/apd_c4096_100k.pt.best.pt")


@torch.no_grad()
def main() -> None:
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = int(os.environ.get("B", "128"))
    topk_print = int(os.environ.get("TOPK_PRINT", "10"))
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

    model = load_pythia14m_target().float()
    banks, ci, cfg, model = load_decomp(CKPT, model, device)
    C = cfg.n_components
    pool = torch.load("/tmp/pythia_compare/pool.pt", weights_only=True)[:B].to(device)
    L = 64
    first = pool[:, :L]
    seq = torch.cat([first, first], dim=1)  # [B, 2L]

    # gates on the repeated probe (chunked for memory)
    g_chunks = []
    for lo in range(0, B, 32):
        clear_masks(banks)
        model(seq[lo:lo + 32])
        acts = {n: b.last_input for n, b in banks.items()}
        gl, _ = ci(acts)
        g_chunks.append(gl)
    g = torch.cat(g_chunks, dim=0)  # [B, 2L, C]
    refresh_caches(banks)

    # token-identity-controlled delta: same token, copy-2 position minus copy-1 position.
    # positions 1..L-1 in each copy (skip pos 0 of copy 2: it continues copy 1, not a clean repeat).
    d = (g[:, L + 1:, :] - g[:, 1:L, :]).mean(dim=(0, 1))  # [C]
    rate = (g > 0.5).float().mean(dim=(0, 1))
    order = d.argsort(descending=True)

    print(f"ckpt={CKPT}\ninduction-conditional components (delta = gate on repeat - gate on same "
          f"token first time):", flush=True)
    kinds = {"query_key_value": "qkv", "attention.dense": "attnO",
             "dense_h_to_4h": "mlpUp", "dense_4h_to_h": "mlpDn"}
    W = {n: b.materialized_weights() for n, b in banks.items()}
    for c in order[:topk_print].tolist():
        km, lm = {}, {}
        for n, w in W.items():
            m = w[c].pow(2).sum().item()
            km[kinds[next(k for k in kinds if k in n)]] = km.get(kinds[next(k for k in kinds if k in n)], 0) + m
            lm[int(n.split(".")[2])] = lm.get(int(n.split(".")[2]), 0) + m
        tot = sum(km.values()) + 1e-12
        fp = " ".join(f"{k}:{v/tot:.2f}" for k, v in sorted(km.items(), key=lambda kv: -kv[1])[:2])
        lfp = " ".join(f"L{l}:{v/tot:.2f}" for l, v in sorted(lm.items(), key=lambda kv: -kv[1])[:2])
        print(f"  comp {c:>4}  delta={d[c]:+.3f}  overall rate={rate[c]:.3f}  {fp} | {lfp}", flush=True)
    n_pos = int((d > 0.05).sum().item())
    print(f"components with delta > 0.05: {n_pos}; > 0.01: {int((d > 0.01).sum().item())}", flush=True)

    # causal dose-response: ablate top-N delta comps vs rate-matched random comps
    zeros = {n: torch.zeros(*seq.shape, device=device) for n in banks}
    base_ci = induction_copy_acc(masked_forward(model, banks, seq, g, zeros), first, L)
    clear_masks(banks)
    base_unmasked = induction_copy_acc(model(seq), first, L)
    refresh_caches(banks)
    print(f"\ninduction copy accuracy: unmasked={base_unmasked:.3f}  ci-masked(all)={base_ci:.3f}", flush=True)

    gen = torch.Generator(device=device).manual_seed(0)

    def ablate(comps: torch.Tensor) -> float:
        gm = g.clone()
        gm[..., comps] = 0.0
        return induction_copy_acc(masked_forward(model, banks, seq, gm, zeros), first, L)

    def keep_only(comps: torch.Tensor) -> float:
        gm = torch.zeros_like(g)
        gm[..., comps] = g[..., comps]
        return induction_copy_acc(masked_forward(model, banks, seq, gm, zeros), first, L)

    print(f"\n{'N':>4} {'ablate top-N delta':>19} {'ablate rate-matched rand':>25} {'keep-only top-N':>16}", flush=True)
    for N in (1, 2, 4, 8, 16, 32, 64):
        top = order[:N]
        # rate-matched random: for each targeted comp, sample an untargeted comp with similar rate
        rand = []
        pool_c = order[200:].tolist()  # exclude anything near the top of the delta ranking
        rates_pool = rate[torch.tensor(pool_c, device=device)]
        for c in top.tolist():
            j = (rates_pool - rate[c]).abs().argmin().item()
            rand.append(pool_c.pop(j))
            rates_pool = rate[torch.tensor(pool_c, device=device)]
        acc_t = ablate(top)
        acc_r = ablate(torch.tensor(rand, device=device))
        acc_k = keep_only(top)
        print(f"{N:>4} {acc_t:>19.3f} {acc_r:>25.3f} {acc_k:>16.3f}", flush=True)
    print("\nINDUCTION_COMPONENTS DONE", flush=True)


if __name__ == "__main__":
    main()
