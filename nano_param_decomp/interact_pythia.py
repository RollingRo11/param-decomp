"""Open-ended interaction with a Pythia-14M APD-basis decomposition: do components mean anything?

No target circuit. For each inspected component we report:
  - firing rate + selectivity (how rare and how strong its gate is)
  - the tokens it fires ON (top current-tokens at high-gate positions, decoded)
  - example contexts around its strongest firings (pool text, firing token marked with «»)
  - what it causally SUPPORTS: ablate the component everywhere and rank next-token predictions by
    how much their probability drops at the positions where it fired (top damaged predictions)
  - its module/layer weight fingerprint

Components are chosen two ways: most-used, and most-selective (rare but strong) — the latter are
usually the interpretable ones. Caveat: the pool is text sampled from pythia-14m itself
(in-distribution for the CI net, but 14M-model prose, so semi-coherent).

Run:  CUDA_VISIBLE_DEVICES=1 python -m nano_param_decomp.interact_pythia
Env:  CKPT (default the C=4096 best ckpt), NCOMP (per group), NCTX (examples per comp), B
"""

import os

import torch
import torch.nn.functional as F

from .apd_lm import load_decomp, masked_forward
from .apd_mask import clear_masks, refresh_caches
from .pythia14m import load_pythia14m_target

CKPT = os.environ.get("CKPT", "/tmp/pythia_compare/apd_c4096_imp3e3.pt.best.pt")


def ctx_str(tok, seq: list[int], pos: int, before: int = 10, after: int = 2) -> str:
    lo, hi = max(0, pos - before), min(len(seq), pos + after + 1)
    parts = []
    for i in range(lo, hi):
        t = tok.decode([seq[i]]).replace("\n", "\\n")
        parts.append(f"«{t}»" if i == pos else t)
    return "".join(parts)


@torch.no_grad()
def main() -> None:
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = int(os.environ.get("B", "256"))
    n_comp = int(os.environ.get("NCOMP", "6"))
    n_ctx = int(os.environ.get("NCTX", "5"))
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

    model = load_pythia14m_target().float()
    banks, ci, cfg, model = load_decomp(CKPT, model, device)
    C = cfg.n_components
    pool = torch.load("/tmp/pythia_compare/pool.pt", weights_only=True)[:B, :128].to(device)
    print(f"ckpt={CKPT}  C={C}  pool {tuple(pool.shape)}", flush=True)

    clear_masks(banks)
    target = model(pool)  # [B,S,V]
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g_lower, g_upper = ci(acts)  # [B,S,C]
    S = pool.shape[1]
    zeros = {n: torch.zeros(B, S, device=device) for n in banks}

    fire = (g_lower > 0.5).float()
    rate = fire.mean(dim=(0, 1))                      # [C] firing rate
    alive_any = int((g_lower.amax(dim=(0, 1)) > 0.5).sum().item())
    alive_soft = int((g_lower.amax(dim=(0, 1)) > 0.05).sum().item())
    print(f"alive components on this pool: {alive_any} fire hard (>0.5), {alive_soft} fire at all "
          f"(>0.05), of C={g_lower.shape[-1]}", flush=True)
    strength = g_upper.mean(dim=(0, 1))               # [C] mean gate
    peak = g_upper.amax(dim=(0, 1))                   # [C] max gate
    # selectivity: strong somewhere, rare overall
    selective = peak * (1 - rate).clamp(min=0) * (rate > 1e-4).float()
    top_used = strength.topk(n_comp).indices.tolist()
    top_sel = selective.topk(n_comp).indices.tolist()
    comps = list(dict.fromkeys(top_used + top_sel))  # dedup, keep order

    # module/layer fingerprint helper
    kinds = {"query_key_value": "qkv", "attention.dense": "attnO",
             "dense_h_to_4h": "mlpUp", "dense_4h_to_h": "mlpDn"}
    W = {n: b.materialized_weights() for n, b in banks.items()}

    for c in comps:
        r, st, pk = rate[c].item(), strength[c].item(), peak[c].item()
        group = "USED" if c in top_used else "SELECTIVE"
        # fingerprint
        km, lm = {}, {}
        for n, w in W.items():
            m = w[c].pow(2).sum().item()
            km[next(v for k, v in kinds.items() if k in n)] = km.get(
                next(v for k, v in kinds.items() if k in n), 0.0) + m
            layer = int(n.split(".")[2])
            lm[layer] = lm.get(layer, 0.0) + m
        totm = sum(km.values()) + 1e-12
        fp = " ".join(f"{k}:{v/totm:.2f}" for k, v in sorted(km.items(), key=lambda kv: -kv[1]))
        lfp = " ".join(f"L{l}:{v/totm:.2f}" for l, v in sorted(lm.items(), key=lambda kv: -kv[1])[:2])
        print(f"\n=== comp {c} [{group}] fire_rate={r:.4f} mean_gate={st:.3f} peak={pk:.2f} | {fp} | {lfp}",
              flush=True)
        if r < 1e-5:
            print("  (never fires on this pool)", flush=True)
            continue

        # where it fires
        gc = g_lower[..., c]
        flat = gc.flatten()
        top_pos = flat.topk(min(200, int((flat > 0.5).sum().item()) or 1)).indices
        bi, si = top_pos // S, top_pos % S
        # tokens it fires ON
        fired_toks = pool[bi, si].tolist()
        from collections import Counter
        common = Counter(tok.decode([t]).replace("\n", "\\n") for t in fired_toks).most_common(8)
        print("  fires on: " + ", ".join(f"'{t}'x{n}" for t, n in common), flush=True)

        # causal: ablate comp c, measure prob drops at its firing positions
        gate_ab = g_lower.clone()
        gate_ab[..., c] = 0.0
        pred_ab = masked_forward(model, banks, pool, gate_ab, zeros)
        pred_ci = masked_forward(model, banks, pool, g_lower, zeros)
        p_ci = F.softmax(pred_ci[bi, si], dim=-1)
        p_ab = F.softmax(pred_ab[bi, si], dim=-1)
        drop = p_ci - p_ab                              # [n_pos, V]
        mean_drop = drop.mean(0)
        top_supported = mean_drop.topk(8)
        sup = ", ".join(f"'{tok.decode([i]).strip() or repr(tok.decode([i]))}'({v:.3f})"
                        for v, i in zip(top_supported.values.tolist(), top_supported.indices.tolist()))
        kl_at_fire = F.kl_div(torch.log(p_ab + 1e-9), p_ci, reduction="batchmean").item()
        print(f"  ablation KL at firing positions={kl_at_fire:.3f}; predictions it supports: {sup}", flush=True)

        # example contexts
        seen_seqs = set()
        shown = 0
        for b_i, s_i in zip(bi.tolist(), si.tolist()):
            if b_i in seen_seqs or shown >= n_ctx:
                continue
            seen_seqs.add(b_i)
            print(f"    g={gc[b_i, s_i]:.2f}  ...{ctx_str(tok, pool[b_i].tolist(), s_i)}...", flush=True)
            shown += 1
    print("\nINTERACT_PYTHIA DONE", flush=True)


if __name__ == "__main__":
    main()
