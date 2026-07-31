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
from nano_apd.lm_target import one_loader_batch  # noqa: E402
from nano_apd.polar_lm import (  # noqa: E402
    MODULES,
    Runner,
    attribution_inner,
    build_banks,
)
from nano_param_decomp.pythia14m import generate_pool, load_pythia14m_target  # noqa: E402

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


def normalize_gates(A, cfg):
    """Apply the exact gate normalization saved by the training run."""
    norm = cfg.get("gate_norm", "max")
    if norm == "sum":
        denom = A.sum(-1, keepdim=True)
    elif norm == "max":
        denom = A.amax(-1, keepdim=True)
    else:
        raise ValueError(f"unknown gate_norm={norm!r}")
    gates = (A / (denom + 1e-12)) ** cfg.get("tau", 1.0)
    threshold = cfg.get("gate_thresh", 0.0)
    if not 0 <= threshold < 1:
        raise ValueError(f"gate_thresh must be in [0, 1), got {threshold}")
    if threshold > 0:
        gates = F.relu(gates - threshold) / (1 - threshold)
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_seqs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--target", choices=["pythia14m", "pile4l"], default=None,
                        help="override the target saved in config.json")
    parser.add_argument("--allow_train_fallback", action="store_true",
                        help="fall back to a fresh train stream if validation is unavailable")
    args = parser.parse_args()
    device = args.device

    with open(OUT_ROOT / args.run / "config.json") as config_file:
        cfg = json.load(config_file)
    target_name = args.target or cfg.get("target", "pythia14m")
    if args.target is not None and cfg.get("target") not in (None, args.target):
        raise ValueError(f"--target={args.target} disagrees with saved target={cfg['target']}")
    if target_name == "pile4l":
        from nano_param_decomp.pile_4L import C_PER_MODULE_4L, load_paper_target_model, make_loader
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

    # Held-out sequences. Falling back to train data must be explicit and is reported.
    eval_split = "generated"
    if target_name == "pile4l":
        try:
            loader = make_loader(args.n_seqs, args.seq_len, 0, 1, "validation", 777)
            idx = one_loader_batch(loader).to(device)
            eval_split = "validation"
        except Exception:
            if not args.allow_train_fallback:
                raise
            loader = make_loader(args.n_seqs, args.seq_len, 0, 1, "train", 777_777)
            idx = one_loader_batch(loader).to(device)
            eval_split = "train-fallback"
    else:
        idx = generate_pool(target, args.n_seqs, args.seq_len, torch.device(device),
                            seed=777).to(device)

    K = cfg.get("ig_steps", 1)
    A, logits_t = attributions_for(runner, banks, idx, K)

    report = {"run": args.run, "target": target_name, "eval_split": eval_split,
              "gate_norm": cfg.get("gate_norm", "max")}
    C = cfg["C"]
    order = A.argsort(dim=-1, descending=True)                    # [B, T, C]
    for j in [1, 2, 4, 8, 16, C]:
        mask = torch.zeros_like(A)
        mask.scatter_(-1, order[..., :j], 1.0)
        g = normalize_gates(A, cfg)
        with torch.no_grad():
            logits_g, _ = runner.gated_pass(idx, g * mask)
        report[f"keep_top{j if j < C else 'ALL'}_kl"] = round(
            kl_per_token(logits_g[:, :-1], logits_t[:, :-1]).item(), 4)
    g_full = normalize_gates(A, cfg)
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
    g2 = normalize_gates(A2, cfg)
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
