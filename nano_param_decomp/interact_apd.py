"""Interact with the learned APD-basis components of the attn-only-2l decomposition.

Loads a saved decomposition (apd_lm.load_decomp) and lets us poke individual whole-network components:
  - rank components by activity (mean CI) on real text and on an induction probe,
  - each component's weight-mass "fingerprint" over (layer, head) across the 8 q/k/v/o matrices,
  - causal probes: ablate a single component (gate->0) or keep-only-one (that gate 1, rest 0) and
    measure the effect on next-token KL and on induction copy.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.interact_apd
Env:  CKPT (default /tmp/attn2l_compare/apd_lm.pt), SEQ, TOPK
"""

import os

import torch
import torch.nn.functional as F
from torch import Tensor

from .apd_lm import load_decomp, masked_forward
from .apd_mask import clear_masks, refresh_caches
from .attn_only_2l import INDUCTION_HEAD, PREV_TOKEN_HEAD, load_attn_only_2l_target
from .run import induction_copy_acc, kl_logits


def head_fingerprint(bank_W: dict[str, Tensor], c: int, d_head: int) -> dict[tuple[int, int], float]:
    """Fraction of component c's total weight-norm on each (layer, head)."""
    mass: dict[tuple[int, int], float] = {}
    for n, w in bank_W.items():
        layer = 0 if "blocks.0" in n else 1
        proj = n.split(".")[-1]
        wc = w[c]  # [d_out, d_in]
        per = wc.pow(2).sum(dim=0) if proj == "o_proj" else wc.pow(2).sum(dim=1)  # heads on cols(o)/rows(qkv)
        nh = per.shape[0] // d_head
        for h in range(nh):
            m = per[h * d_head:(h + 1) * d_head].sum().item()
            mass[(layer, h)] = mass.get((layer, h), 0.0) + m
    tot = sum(mass.values()) + 1e-12
    return {k: v / tot for k, v in mass.items()}


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.environ.get("CKPT", "/tmp/attn2l_compare/apd_lm.pt")
    seq_len = int(os.environ.get("SEQ", "128"))
    topk = int(os.environ.get("TOPK", "8"))

    model = load_attn_only_2l_target().to(device)
    d_head = model.blocks[0].d_head
    banks, ci, cfg, model = load_decomp(ckpt, model, device)
    C = cfg.n_components
    pool = torch.load("/tmp/attn2l_compare/pool.pt", weights_only=True)[:, :seq_len].to(device)
    W = {n: b.materialized_weights() for n, b in banks.items()}
    print(f"loaded {C} components across {len(banks)} modules; heads {model.blocks[0].n_heads}, d_head {d_head}", flush=True)
    print(f"answer key: prev-token head {PREV_TOKEN_HEAD}, induction head {INDUCTION_HEAD}\n", flush=True)

    # --- activity on real text vs on an induction probe ---
    txt = pool[:16]
    clear_masks(banks); model(txt)
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g_txt, _ = ci(acts)                       # [B,S,C]
    act_txt = g_txt.mean(dim=(0, 1))          # [C]

    L = seq_len // 2
    first = pool[:16, :L]
    probe = torch.cat([first, first], dim=1)
    clear_masks(banks); model(probe)
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g_probe, _ = ci(acts)
    act_ind = g_probe[:, L:, :].mean(dim=(0, 1))   # activity on 2nd-copy (induction) positions

    print("=== most ACTIVE components on real text (mean CI) ===", flush=True)
    for c in act_txt.topk(topk).indices.tolist():
        fp = head_fingerprint(W, c, d_head)
        top_heads = sorted(fp.items(), key=lambda kv: -kv[1])[:3]
        hs = ", ".join(f"L{l}H{h}:{m:.2f}" for (l, h), m in top_heads)
        print(f"  comp {c:>3}  act_text={act_txt[c]:.3f}  act_induction={act_ind[c]:.3f}  top heads: {hs}", flush=True)

    print("\n=== components most active on INDUCTION (2nd-copy) positions ===", flush=True)
    ind_top = act_ind.topk(topk).indices.tolist()
    for c in ind_top:
        fp = head_fingerprint(W, c, d_head)
        prev_m = fp.get(PREV_TOKEN_HEAD, 0.0)
        ind_m = fp.get(INDUCTION_HEAD, 0.0)
        top_heads = sorted(fp.items(), key=lambda kv: -kv[1])[:3]
        hs = ", ".join(f"L{l}H{h}:{m:.2f}" for (l, h), m in top_heads)
        xl = "cross-layer" if (sum(m for (l, _), m in fp.items() if l == 0) > 0.1 and
                               sum(m for (l, _), m in fp.items() if l == 1) > 0.1) else "single-layer"
        print(f"  comp {c:>3}  act_ind={act_ind[c]:.3f}  mass@prevhead={prev_m:.2f} mass@indhead={ind_m:.2f}  "
              f"[{xl}]  top: {hs}", flush=True)

    # --- causal probes on the induction probe: ablate-one vs keep-only-one ---
    B, S = probe.shape
    zeros = {n: torch.zeros(B, S, device=device) for n in banks}
    clear_masks(banks)
    target = model(probe)
    refresh_caches(banks)
    g_lower, _ = ci(acts)  # note: acts currently from probe forward
    base_copy = induction_copy_acc(target, first, L)
    ci_all = masked_forward(model, banks, probe, g_lower, zeros)
    base_ci_copy = induction_copy_acc(ci_all, first, L)
    print(f"\ninduction copy: original={base_copy:.3f}  ci-masked(all comps)={base_ci_copy:.3f}", flush=True)

    print("\n=== causal effect of the top induction components (on the induction probe) ===", flush=True)
    for c in ind_top[:5]:
        # ablate just comp c (set its gate to 0, keep others at ci)
        gate_ab = g_lower.clone(); gate_ab[..., c] = 0.0
        ab = masked_forward(model, banks, probe, gate_ab, zeros)
        d_copy = induction_copy_acc(ab, first, L) - base_ci_copy
        d_kl = kl_logits(ab, target).item()
        # keep only comp c
        gate_ko = torch.zeros(B, S, C, device=device); gate_ko[..., c] = 1.0
        ko = masked_forward(model, banks, probe, gate_ko, zeros)
        ko_copy = induction_copy_acc(ko, first, L)
        print(f"  comp {c:>3}: ablate -> Δinduction={d_copy:+.3f}, KL_vs_orig={d_kl:.3f} | keep-only -> induction={ko_copy:.3f}", flush=True)
    print("INTERACT DONE", flush=True)


if __name__ == "__main__":
    main()
