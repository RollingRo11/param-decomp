"""Merge evidence shards into decoded per-component example files for auto-interp.

Each component gets: usage stats, top unique-context activations, and a few
mid-strength activations. Context: 35 tokens before the firing token, the firing
token marked «like this», and 8 tokens after (the component's attribution is about
predicting the token AFTER the marked one, so the right context matters).

Writes n_chunks files evidence_chunk_XX.md plus evidence_all.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/param-decomp")
from nano_param_decomp.pythia14m import MODEL_ID  # noqa: E402

CTX_BEFORE, CTX_AFTER = 35, 8


def render(tok, idx, b, t, T):
    lo, hi = max(0, t - CTX_BEFORE), min(T, t + 1 + CTX_AFTER)
    pre = tok.decode(idx[b, lo:t].tolist()).replace("\n", "\\n")
    cur = tok.decode(idx[b, t:t + 1].tolist()).replace("\n", "\\n")
    post = tok.decode(idx[b, t + 1:hi].tolist()).replace("\n", "\\n")
    return f"{pre}«{cur}»{post}", cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_chunks", type=int, default=16)
    parser.add_argument("--n_top", type=int, default=10)
    parser.add_argument("--n_mid", type=int, default=4)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or MODEL_ID)

    shards = [torch.load(s, weights_only=True) for s in args.shards]
    C = shards[0]["vals"].shape[1]
    n_tok = sum(s["n_tok"] for s in shards)
    top1 = sum(s["top1_counts"] for s in shards)

    records = []
    for c in range(C):
        cands = []          # (gate, shard, b, t)
        for si, s in enumerate(shards):
            T = s["T"]
            for v, p in zip(s["vals"][:, c].tolist(), s["pos"][:, c].tolist()):
                cands.append((v, si, p // T, p % T))
        cands.sort(key=lambda x: -x[0])

        seen, top_ex = set(), []
        for v, si, b, t in cands:
            if len(top_ex) >= args.n_top:
                break
            text, cur = render(tok, shards[si]["idx"], b, t, shards[si]["T"])
            key = (cur, text.split("«")[0][-25:])
            if key in seen:
                continue
            seen.add(key)
            top_ex.append({"gate": round(v, 3), "text": text, "token": cur})
        # mid-strength: sample from the lower half of the candidate list
        mid_ex = []
        lower = cands[len(cands) // 2::17]
        for v, si, b, t in lower[:args.n_mid]:
            text, cur = render(tok, shards[si]["idx"], b, t, shards[si]["T"])
            mid_ex.append({"gate": round(v, 3), "text": text, "token": cur})

        records.append({
            "component": c,
            "top1_share_pct": round(100 * top1[c].item() / n_tok, 3),
            "peak_gate": round(cands[0][0], 3),
            "top_examples": top_ex, "mid_examples": mid_ex,
        })

    with open(out_dir / "evidence_all.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    per = (C + args.n_chunks - 1) // args.n_chunks
    for k in range(args.n_chunks):
        chunk = records[k * per:(k + 1) * per]
        with open(out_dir / f"evidence_chunk_{k:02d}.md", "w") as f:
            for r in chunk:
                f.write(f"## component {r['component']}  "
                        f"(top-choice on {r['top1_share_pct']}% of tokens, "
                        f"peak gate {r['peak_gate']})\n")
                f.write("TOP ACTIVATIONS:\n")
                for e in r["top_examples"]:
                    f.write(f"  [{e['gate']}] {e['text']}\n")
                f.write("MID-STRENGTH ACTIVATIONS:\n")
                for e in r["mid_examples"]:
                    f.write(f"  [{e['gate']}] {e['text']}\n")
                f.write("\n")
    print(f"wrote {args.n_chunks} chunks + evidence_all.jsonl to {out_dir}")


if __name__ == "__main__":
    main()
