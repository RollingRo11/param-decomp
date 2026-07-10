"""Automated component interpretation (auto-interp) for a Pythia decomposition, LLM-judged.

Two stages per component, mirroring the standard SAE auto-interp recipe but with our causal
evidence included:

  1. LABEL: the judge sees the component's evidence pack -- firing rate, the tokens it fires on,
     example contexts (firing token marked with «»), the next-token predictions that degrade when
     the component alone is ablated, and its layer/matrix fingerprint -- and returns a short label
     plus a monosemanticity score 1-5 with a one-line reason.
  2. DETECT (validation): the judge sees ONLY the label and 8 held-out marked contexts (4 where the
     component fires, 4 where it doesn't, shuffled) and predicts which ones fire. Balanced accuracy
     scores how *predictive* the label is -- labels that sound nice but don't discriminate score ~0.5.

API key: put ANTHROPIC_API_KEY=... in the repo root `.env` (never committed) or the environment.
Model: claude-haiku-4-5 by default (cheap, fast, good enough for label/judge work; escalate odd
cases by rerunning with AUTOINTERP_MODEL=claude-sonnet-4-6).

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.auto_interp
Env:  CKPT (default 100k best), NCOMP (default 40; 0 = all live comps), MIN_RATE (default 1e-4),
      AUTOINTERP_MODEL, WORKERS (default 4), OUT (default /tmp/auto_interp.jsonl)
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .apd_lm import load_decomp, masked_forward
from .apd_mask import clear_masks, refresh_caches
from .pythia14m import load_pythia14m_target

CKPT = os.environ.get("CKPT", "/tmp/pythia_compare/apd_c4096_100k.pt.best.pt")
MODEL = os.environ.get("AUTOINTERP_MODEL", "claude-haiku-4-5")

LABEL_PROMPT = """You are labeling one component from a decomposition of a small language model.
Evidence:
- fires on {rate_pct:.2f}% of token positions
- tokens it most often fires on: {fires_on}
- example contexts (the token where it fires is marked with «»):
{contexts}
- next-token predictions that get WORSE when this component alone is deleted: {supports}
- weight location: {fingerprint}

Reply with JSON only: {{"label": "<10 words max, describe when it fires and what it does>",
"monosemantic": <1-5, 5 = one crisp human-describable job, 1 = no discernible pattern>,
"reason": "<one sentence>"}}"""

DETECT_PROMPT = """A language-model component has this label: "{label}"
For each numbered snippet below, the token marked «» is a candidate firing position.
Predict for each whether this component fires there (true/false).
{snippets}

Reply with JSON only: {{"predictions": [true/false, ... {n} items]}}"""


def _load_env_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no ANTHROPIC_API_KEY in environment or repo-root .env")


def _judge(client, prompt: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            return json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception as e:  # rate limits, transient API errors, malformed JSON
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt + random.random())
    return {"error": "unreachable"}


def ctx_str(tok, seq: list[int], pos: int, before: int = 10, after: int = 2) -> str:
    lo, hi = max(0, pos - before), min(len(seq), pos + after + 1)
    return "".join((f"«{tok.decode([seq[i]])}»" if i == pos else tok.decode([seq[i]])).replace("\n", "\\n")
                   for i in range(lo, hi))


@torch.no_grad()
def build_evidence(device) -> tuple[list[dict], object]:
    """Returns (per-component evidence dicts, tokenizer). Split pool: first half for evidence,
    second half held out for the detection test."""
    from transformers import AutoTokenizer
    import torch.nn.functional as F
    from collections import Counter

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    model = load_pythia14m_target().float()
    banks, ci, cfg, model = load_decomp(CKPT, model, device)
    pool = torch.load("/tmp/pythia_compare/pool.pt", weights_only=True)[:512, :128].to(device)
    half = pool.shape[0] // 2
    n_comp = int(os.environ.get("NCOMP", "40"))
    min_rate = float(os.environ.get("MIN_RATE", "1e-4"))

    # gates over the full pool, computed in sequence-chunks (C=4096 gate tensors are ~1 GB each;
    # the masked forwards below are the real memory hazard and are restricted per-component)
    S = pool.shape[1]
    g_chunks = []
    for lo in range(0, pool.shape[0], 128):
        clear_masks(banks)
        model(pool[lo:lo + 128])
        acts = {n: b.last_input for n, b in banks.items()}
        gl, _ = ci(acts)
        g_chunks.append(gl)
    g_lower = torch.cat(g_chunks, dim=0)
    del g_chunks
    refresh_caches(banks)
    g_upper = g_lower  # for usage ordering, lower/upper differ only outside [0,1]; lower suffices
    rate = (g_lower[:half] > 0.5).float().mean(dim=(0, 1))
    alive = torch.nonzero((rate > min_rate)).squeeze(1)
    # order by usage; NCOMP=0 -> all alive
    order = alive[g_upper[:half].mean(dim=(0, 1))[alive].argsort(descending=True)]
    comps = order.tolist() if n_comp == 0 else order[:n_comp].tolist()
    print(f"{len(alive)} components above rate {min_rate}; judging {len(comps)}", flush=True)

    kinds = {"query_key_value": "qkv", "attention.dense": "attn-out",
             "dense_h_to_4h": "mlp-in", "dense_4h_to_h": "mlp-out"}
    W = {n: b.materialized_weights() for n, b in banks.items()}

    out = []
    for c in comps:
        gc = g_lower[..., c]
        # evidence from first half
        ev_pos = (gc[:half] > 0.5).nonzero()
        if len(ev_pos) == 0:
            continue
        top = gc[:half].flatten().topk(min(60, len(ev_pos))).indices
        bi, si = top // S, top % S
        fires_on = Counter(tok.decode([pool[b, s].item()]).replace("\n", "\\n")
                           for b, s in zip(bi.tolist(), si.tolist())).most_common(6)
        seen, contexts = set(), []
        for b, s in zip(bi.tolist(), si.tolist()):
            if b in seen or len(contexts) >= 6:
                continue
            seen.add(b); contexts.append("  " + ctx_str(tok, pool[b].tolist(), s))
        # ablation support: run masked forwards ONLY on this component's evidence sequences
        ub = torch.unique(bi)
        sub = pool[ub]
        row = {b.item(): i for i, b in enumerate(ub)}
        bi_s = torch.tensor([row[b.item()] for b in bi], device=device)
        g_sub = g_lower[ub]
        zeros_sub = {n: torch.zeros(*sub.shape, device=device) for n in banks}
        pred_ci = masked_forward(model, banks, sub, g_sub, zeros_sub)
        gate_ab = g_sub.clone(); gate_ab[..., c] = 0.0
        pred_ab = masked_forward(model, banks, sub, gate_ab, zeros_sub)
        p_ci = F.softmax(pred_ci[bi_s, si], -1); p_ab = F.softmax(pred_ab[bi_s, si], -1)
        sup_v, sup_i = (p_ci - p_ab).mean(0).topk(6)
        supports = ", ".join(f"'{tok.decode([i]).strip() or repr(tok.decode([i]))}'"
                             for v, i in zip(sup_v.tolist(), sup_i.tolist()) if v > 1e-4) or "(none clear)"
        km = Counter()
        for n, w in W.items():
            km[kinds[next(k for k in kinds if k in n)]] += w[c].pow(2).sum().item()
        tot = sum(km.values()) + 1e-12
        fingerprint = ", ".join(f"{k} {v/tot:.0%}" for k, v in km.most_common(2))
        # held-out detection snippets from second half: 4 firing, 4 non-firing
        pos2 = (gc[half:] > 0.5).nonzero()
        neg2 = (gc[half:] < 0.05).nonzero()
        det = []
        if len(pos2) >= 4 and len(neg2) >= 4:
            picks = [(pos2[torch.randint(len(pos2), (1,))].squeeze(0), True) for _ in range(4)] + \
                    [(neg2[torch.randint(len(neg2), (1,))].squeeze(0), False) for _ in range(4)]
            random.shuffle(picks)
            for p, lab in picks:
                b2, s2 = p[0].item(), p[1].item()
                det.append((ctx_str(tok, pool[half + b2].tolist(), s2), lab))
        out.append({"comp": c, "rate": rate[c].item(),
                    "fires_on": ", ".join(f"'{t}'x{n}" for t, n in fires_on),
                    "contexts": "\n".join(contexts), "supports": supports,
                    "fingerprint": fingerprint, "detect": det})
    return out, tok


def main() -> None:
    import anthropic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = anthropic.Anthropic(api_key=_load_env_key())
    evidence, _tok = build_evidence(device)
    est_in = len(evidence) * 2 * 900  # ~900 tokens per call, 2 calls per comp
    print(f"model={MODEL}; ~{est_in/1e6:.2f}M input tokens estimated", flush=True)

    def one(ev: dict) -> dict:
        lab = _judge(client, LABEL_PROMPT.format(rate_pct=100 * ev["rate"], fires_on=ev["fires_on"],
                                                 contexts=ev["contexts"], supports=ev["supports"],
                                                 fingerprint=ev["fingerprint"]))
        rec = {"comp": ev["comp"], "rate": ev["rate"], **{k: lab.get(k) for k in ("label", "monosemantic", "reason", "error")}}
        if ev["detect"] and "label" in lab:
            snips = "\n".join(f"{i+1}. {s}" for i, (s, _) in enumerate(ev["detect"]))
            det = _judge(client, DETECT_PROMPT.format(label=lab["label"], snippets=snips, n=len(ev["detect"])))
            preds = det.get("predictions")
            if isinstance(preds, list) and len(preds) == len(ev["detect"]):
                truth = [lab_ for _, lab_ in ev["detect"]]
                rec["detection_acc"] = sum(p == t for p, t in zip(preds, truth)) / len(truth)
        return rec

    workers = int(os.environ.get("WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, evidence))

    out_path = os.environ.get("OUT", "/tmp/auto_interp.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in results if r.get("monosemantic")]
    det = [r["detection_acc"] for r in results if "detection_acc" in r]
    print(f"\nlabeled {len(ok)}/{len(results)}; mean monosemanticity "
          f"{sum(r['monosemantic'] for r in ok)/max(1,len(ok)):.2f}/5; "
          f"mean detection accuracy {sum(det)/max(1,len(det)):.2f} (n={len(det)}; 0.5=chance)", flush=True)
    for r in sorted(ok, key=lambda r: -(r.get("monosemantic") or 0))[:12]:
        print(f"  comp {r['comp']:>4} [{r.get('monosemantic')}/5, det={r.get('detection_acc', float('nan')):.2f}] {r.get('label')}", flush=True)
    print(f"full results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
