"""Evaluation for a polar_lm decomposition of Pythia-14M.

Metrics (no ground-truth features exist at LM scale — these are the functional referees):
  keep_topj   per-token KL(target || gated) keeping only the top-j attribution
              components per token, j in {1, 2, 4, 8, 16, all} — the LM analog of
              keep-only; low KL at small j = few components suffice per token.
  gates_l0    mean number of gates above 0.01 per token.
  induction   repeated-random-token sequences: copy accuracy (argmax matches the
              repeated token) for target vs gated model on the second half — a known
              behavior the decomposition must preserve.

    python -m nano_apd.eval_lm --run pythia14m_polar_C1024_v3_phased
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/param-decomp")
from nano_param_decomp.pythia14m import generate_pool, load_pythia14m_target  # noqa: E402
from nano_apd.polar_lm import (MODULES, Runner, attribution_inner, build_banks,  # noqa: E402
                               compute_attributions)

OUT_ROOT = Path(__file__).parent / "out"


def attributions_for(runner, banks, idx, K):
    """Attributions with the SAME gate recipe the run was trained with: K=1 is the
    single-point sensor; K>1 averages the inner sums over the IG-over-mask path
    (gate recipes are not interchangeable across decompositions — a mismatch
    invalidates the gated pass). Returns (A detached, final-pass logits detached)."""
    inner = None
    for k in range(1, K + 1):
        runner.wscale = k / K
        logits_t, tcache = runner.target_pass(idx)
        s = F.cross_entropy(logits_t[:, :-1].flatten(0, 1), idx[:, 1:].flatten(),
                            reduction="sum")
        posts = [tcache[p]["post"] for p in MODULES]
        gposts = {p: g.detach() for p, g in zip(
            MODULES, torch.autograd.grad(s, posts), strict=True)}
        with torch.no_grad():
            contrib = attribution_inner(banks, tcache, gposts)
        inner = contrib if inner is None else inner + contrib
    runner.wscale = 1.0
    return ((inner / K) ** 2).detach(), logits_t.detach()


@torch.no_grad()
def kl_per_token(logits_g, logits_t):
    return F.kl_div(F.log_softmax(logits_g, -1), F.softmax(logits_t, -1),
                    reduction="none").sum(-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_seqs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--target", choices=["pythia14m", "pile4l"], default="pythia14m")
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

    # held-out sequences (fresh seed, never in the training pool)
    if args.target == "pile4l":
        try:
            loader = make_loader(args.n_seqs, args.seq_len, 0, 1, "validation", 777)
            idx = next(loader).to(device)
        except Exception:
            loader = make_loader(args.n_seqs, args.seq_len, 0, 1, "train", 777_777)
            idx = next(loader).to(device)
    else:
        idx = generate_pool(target, args.n_seqs, args.seq_len, torch.device(device),
                            seed=777).to(device)

    K = cfg.get("ig_steps", 1)
    A, logits_t = attributions_for(runner, banks, idx, K)

    report = {"run": args.run}
    C = cfg["C"]
    order = A.argsort(dim=-1, descending=True)                    # [B, T, C]
    for j in [1, 2, 4, 8, 16, C]:
        mask = torch.zeros_like(A)
        mask.scatter_(-1, order[..., :j], 1.0)
        g = (A / (A.amax(-1, keepdim=True) + 1e-12)) ** cfg.get("tau", 1.0)
        with torch.no_grad():
            logits_g, _ = runner.gated_pass(idx, g * mask)
        report[f"keep_top{j if j < C else 'ALL'}_kl"] = round(
            kl_per_token(logits_g[:, :-1], logits_t[:, :-1]).item(), 4)
    g_full = (A / (A.amax(-1, keepdim=True) + 1e-12)) ** cfg.get("tau", 1.0)
    report["gates_l0_0.01"] = round((g_full > 0.01).float().sum(-1).mean().item(), 1)

    # induction: [random half | same half repeated]
    half = args.seq_len // 2
    vocab = target.config.vocab_size
    gen = torch.Generator(device=device).manual_seed(123)
    first = torch.randint(0, vocab, (args.n_seqs, half), device=device, generator=gen)
    seq = torch.cat([first, first], dim=1)
    with torch.no_grad():
        lt = target(seq)
    lt_acc = (lt[:, half:-1].argmax(-1) == seq[:, half + 1:]).float().mean().item()
    A2, _ = attributions_for(runner, banks, seq, K)
    g2 = (A2 / (A2.amax(-1, keepdim=True) + 1e-12)) ** cfg.get("tau", 1.0)
    with torch.no_grad():
        lg, _ = runner.gated_pass(seq, g2)
    lg_acc = (lg[:, half:-1].argmax(-1) == seq[:, half + 1:]).float().mean().item()
    report["induction_copy_acc_target"] = round(lt_acc, 3)
    report["induction_copy_acc_gated"] = round(lg_acc, 3)

    print(json.dumps(report, indent=2))
    with open(OUT_ROOT / args.run / "eval.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
