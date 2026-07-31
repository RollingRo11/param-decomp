"""Inspect what components of a polar_lm decomposition fire on.

For each selected component, prints the top-gate token occurrences with context:
    ...preceding tokens [FIRING TOKEN] following token...   gate=0.93

Firing = the per-token gate for that component (attribution / max-attribution).
Runs on held-out self-generated sequences (the training distribution) plus a few
hard-coded natural-text paragraphs (Pythia-14M was trained on the Pile, so real
English is in-distribution for the target even though our pool is self-generated).

    python -m nano_apd.inspect_lm --run pythia14m_polar_C1024_v3_phased
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/param-decomp")
from nano_param_decomp.pythia14m import MODEL_ID, generate_pool, load_pythia14m_target  # noqa: E402
from nano_apd.polar_lm import MODULES, Runner, build_banks, compute_attributions  # noqa: E402

OUT_ROOT = Path(__file__).parent / "out"

NATURAL_TEXT = [
    "The cat sat on the mat. The dog sat on the mat. The cat and the dog "
    "looked at each other. Then the cat jumped over the dog and ran out of "
    "the door into the garden, where the birds were singing in the trees.",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. "
    "Neil Armstrong and Buzz Aldrin spent about two hours outside the "
    "spacecraft, collecting samples and taking photographs. The mission was "
    "a major milestone in the history of space exploration.",
    "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    result = 0\n"
    "    for i in range(b):\n        result = add(result, a)\n    return result\n",
    "The results are shown in Table 3. We observe that the proposed method "
    "achieves an accuracy of 94.2%, compared to 91.7% for the baseline. "
    "However, the difference is not statistically significant (p = 0.08).",
]


def gather_attributions(runner, target, idx):
    logits_t, tcache = runner.target_pass(idx)
    s = F.cross_entropy(logits_t[:, :-1].flatten(0, 1), idx[:, 1:].flatten(),
                        reduction="sum")
    posts = [tcache[p]["post"] for p in MODULES]
    gposts = {p: g.detach() for p, g in zip(
        MODULES, torch.autograd.grad(s, posts), strict=True)}
    A = compute_attributions(runner.banks, tcache, gposts).detach()
    return A


def show_component(c, A, g, idx, tok, n_show, min_ctx=12):
    """Print the top-gate occurrences of component c with decoded context."""
    B, T, _ = g.shape
    vals, flat = g[..., c].flatten().sort(descending=True)
    shown, seen_ctx = 0, set()
    lines = []
    for v, f in zip(vals.tolist(), flat.tolist()):
        if shown >= n_show or v < 0.01:
            break
        b, t = divmod(f, T)
        lo = max(0, t - min_ctx)
        pre = tok.decode(idx[b, lo:t].tolist())
        cur = tok.decode(idx[b, t:t + 1].tolist())
        nxt = tok.decode(idx[b, t + 1:t + 2].tolist()) if t + 1 < T else ""
        key = (cur, pre[-20:])
        if key in seen_ctx:  # skip near-duplicate contexts
            continue
        seen_ctx.add(key)
        pre = pre.replace("\n", "\\n")
        cur = cur.replace("\n", "\\n")
        nxt = nxt.replace("\n", "\\n")
        lines.append(f"    gate={v:4.2f}  ...{pre}[{cur}]{nxt}...")
        shown += 1
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_seqs", type=int, default=192)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_components", type=int, default=16)
    parser.add_argument("--n_show", type=int, default=8)
    args = parser.parse_args()
    device = args.device

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    cfg = json.load(open(OUT_ROOT / args.run / "config.json"))
    target = load_pythia14m_target().float().to(device)
    for p in target.parameters():
        p.requires_grad_(True)
    banks = build_banks(target, cfg["C"], cfg["m"]).to(device)
    banks.load_state_dict(torch.load(OUT_ROOT / args.run / "banks.pt",
                                     weights_only=True, map_location=device))
    runner = Runner(target, banks, cfg.get("tau", 1.0))

    # held-out generated pool + natural text, one big batch of positions
    idx = generate_pool(target, args.n_seqs, args.seq_len, torch.device(device),
                        seed=999).to(device)
    nat = [tok(t, return_tensors="pt").input_ids[0, :args.seq_len] for t in NATURAL_TEXT]
    nat = torch.stack([F.pad(n, (0, args.seq_len - n.shape[0]),
                             value=tok.eos_token_id or 0) for n in nat]).to(device)
    nat_len = [min(len(tok(t).input_ids), args.seq_len) for t in NATURAL_TEXT]
    idx = torch.cat([idx, nat], dim=0)

    A = gather_attributions(runner, target, idx)
    g = (A / (A.amax(-1, keepdim=True) + 1e-12)) ** cfg.get("tau", 1.0)
    # mask out pad positions of the natural-text rows
    for i, L in enumerate(nat_len):
        g[args.n_seqs + i, L:] = 0.0

    C = cfg["C"]
    # rank components by how often they are a token's top gate (specialists first),
    # then also show some mid-usage ones
    top1 = g.argmax(-1).flatten()
    top1_counts = torch.bincount(top1, minlength=C).float()
    mass = A.flatten(0, 1).sum(0)
    order = top1_counts.argsort(descending=True)

    n_tok = g.shape[0] * g.shape[1]
    print(f"pool: {g.shape[0]} seqs x {g.shape[1]} tokens = {n_tok} positions")
    print(f"components that are top-1 for at least one token: "
          f"{(top1_counts > 0).sum().item()}/{C}")
    print(f"mean gates > 0.01 per token: {(g > 0.01).float().sum(-1).mean().item():.1f}")
    print()

    half = args.n_components // 2
    mid_start = int((top1_counts > 0).sum().item() * 0.4)
    chosen = order[:half].tolist() + order[mid_start:mid_start + half].tolist()
    for rank_label, c in zip(
            [f"top1-rank {i}" for i in range(half)]
            + [f"top1-rank {mid_start + i}" for i in range(half)], chosen):
        share = top1_counts[c].item() / n_tok
        print(f"== component {c}  ({rank_label}; top-gate on {share:5.2%} of tokens, "
              f"attr mass share {mass[c].item() / mass.sum().item():5.2%})")
        for line in show_component(c, A, g, idx, tok, args.n_show):
            print(line)
        print()


if __name__ == "__main__":
    main()
