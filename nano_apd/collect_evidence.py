"""Collect auto-interp evidence for every component of a polar_lm decomposition.

Shardable across GPUs: each shard scores a disjoint pool of held-out sequences and
saves per-component top activating positions (with the raw token ids), to be merged
and decoded by merge_evidence.py.

    python -m nano_apd.collect_evidence --run ... --device cuda:0 --seed 999  --out ev0.pt
    python -m nano_apd.collect_evidence --run ... --device cuda:1 --seed 1001 --out ev1.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/param-decomp")
from nano_param_decomp.pythia14m import generate_pool, load_pythia14m_target  # noqa: E402
from nano_apd.polar_lm import MODULES, Runner, build_banks, compute_attributions  # noqa: E402

OUT_ROOT = Path(__file__).parent / "out"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_seqs", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--top_per_comp", type=int, default=200)
    parser.add_argument("--target", choices=["pythia14m","pile4l"], default="pythia14m")
    parser.add_argument("--data", choices=["gen", "pile"], default="gen",
                        help="pythia14m only: score model-generated pool text (gen, "
                             "matches training) or real Pile text (pile; same NeoX "
                             "tokenizer, faster at scale and more legible)")
    args = parser.parse_args()
    device = args.device

    cfg = json.load(open(OUT_ROOT / args.run / "config.json"))
    if args.target == "pile4l":
        from nano_param_decomp.pile_4L import (C_PER_MODULE_4L,
                                               load_paper_target_model, make_loader)
        MODULES[:] = list(C_PER_MODULE_4L.keys())
        target = load_paper_target_model().float().to(device)
    else:
        target = load_pythia14m_target().float().to(device)
    for p in target.parameters():
        p.requires_grad_(True)
    banks = build_banks(target, cfg["C"], cfg["m"]).to(device)
    banks.load_state_dict(torch.load(OUT_ROOT / args.run / "banks.pt",
                                     weights_only=True, map_location=device))
    runner = Runner(target, banks, cfg.get("tau", 1.0))
    C = cfg["C"]

    if args.target == "pile4l":
        loader = make_loader(args.n_seqs, args.seq_len, 0, 1, "train", args.seed)
        idx = next(loader).to(device)
    elif args.data == "pile":
        from nano_param_decomp.pile_4L import make_loader as make_pile_loader
        loader = make_pile_loader(args.n_seqs, args.seq_len, 0, 1, "train", args.seed)
        idx = next(loader).to(device)
    else:
        idx = generate_pool(target, args.n_seqs, args.seq_len, torch.device(device),
                            seed=args.seed).to(device)

    T = idx.shape[1]
    K = args.top_per_comp
    acc_v = torch.zeros(0, C, device=device)
    acc_p = torch.zeros(0, C, dtype=torch.long, device=device)
    top1_counts = torch.zeros(C, device=device)
    n_tok = 0
    from nano_apd.polar_lm import attribution_inner
    K_ig = cfg.get("ig_steps", 1)
    for i in range(0, idx.shape[0], 32):
        b = idx[i:i + 32]
        inner = None
        for k_ in range(1, K_ig + 1):
            runner.wscale = k_ / K_ig
            logits_t, tcache = runner.target_pass(b)
            s = F.cross_entropy(logits_t[:, :-1].flatten(0, 1), b[:, 1:].flatten(),
                                reduction="sum")
            posts = [tcache[p]["post"] for p in MODULES]
            gposts = {p: g.detach() for p, g in zip(
                MODULES, torch.autograd.grad(s, posts), strict=True)}
            with torch.no_grad():
                contrib = attribution_inner(banks, tcache, gposts)
            inner = contrib if inner is None else inner + contrib
        runner.wscale = 1.0
        A = ((inner / K_ig) ** 2).detach()
        denom = (A.sum(-1, keepdim=True) if cfg.get("gate_norm") == "sum"
                 else A.amax(-1, keepdim=True))
        g = (A / (denom + 1e-12)) ** cfg.get("tau", 1.0)
        gf = g.flatten(0, 1)                                     # [n, C]
        top1_counts += torch.bincount(gf.argmax(-1), minlength=C).float()
        pos = torch.arange(i * T, i * T + gf.shape[0],
                           device=device).unsqueeze(1).expand(-1, C)
        cv = torch.cat([acc_v, gf]); cp = torch.cat([acc_p, pos])
        k = min(K, cv.shape[0])
        acc_v, ti = cv.topk(k, dim=0)
        acc_p = cp.gather(0, ti)
        n_tok += gf.shape[0]
        if (i // 32) % 50 == 0:
            print(f"scored {i + b.shape[0]}/{idx.shape[0]}", flush=True)

    torch.save({
        "idx": idx.cpu(), "vals": acc_v.cpu(), "pos": acc_p.cpu(),
        "top1_counts": top1_counts.cpu(), "n_tok": n_tok, "T": T,
    }, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
