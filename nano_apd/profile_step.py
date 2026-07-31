"""Profile one polar_lm training step: per-phase forward CUDA timing + total
backward + optimizer, averaged over N steps after warmup. Single GPU.

    python -m nano_apd.profile_step --C 1024 --m 4 --steps 40
"""
import argparse, time, json
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/workspace/param-decomp")
from nano_param_decomp.pythia14m import generate_pool, load_pythia14m_target, pool_loader
from nano_apd.polar_lm import (MODULES, Runner, build_banks, compute_attributions,
                               pair_geometry)


class Timer:
    def __init__(self):
        self.acc = defaultdict(float)
        self.n = 0
    def __call__(self, name):
        return _Scope(self, name)
class _Scope:
    def __init__(self, t, name):
        self.t, self.name = t, name
    def __enter__(self):
        self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
        self.s.record(); return self
    def __exit__(self, *a):
        self.e.record(); torch.cuda.synchronize()
        self.t.acc[self.name] += self.s.elapsed_time(self.e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--batch_seqs", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--n_pairs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    target = load_pythia14m_target().float().to(device)
    for p in target.parameters():
        p.requires_grad_(True)
    pool_path = Path(f"/tmp/toy/pythia14m_pool_4096x{args.seq_len}.pt")
    pool = torch.load(pool_path, weights_only=True, map_location="cpu")
    loader = pool_loader(pool, args.batch_seqs, seed=0)
    banks = build_banks(target, args.C, args.m).to(device)
    runner = Runner(target, banks, 1.0)
    opt = torch.optim.AdamW(banks.parameters(), lr=1e-3)
    W2 = {p: (target.get_submodule(p).weight.detach().t() ** 2) for p in MODULES}
    n_params = sum(target.get_submodule(p).weight.numel() for p in MODULES)

    T = Timer()
    step_ms = 0.0
    for step in range(args.steps):
        rec = step >= args.warmup
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        opt.zero_grad(set_to_none=True)
        idx = next(loader).to(device)

        def timed(name):
            return T(name) if rec else _Null()
        with timed("1_target_pass"):
            logits_t, tcache = runner.target_pass(idx)
            s = F.cross_entropy(logits_t[:, :-1].flatten(0, 1), idx[:, 1:].flatten(), reduction="sum")
        with timed("2_attrib_backward"):
            posts = [tcache[p]["post"] for p in MODULES]
            gposts_l = torch.autograd.grad(s, posts, retain_graph=True)
            gposts = {p: g.detach() for p, g in zip(MODULES, gposts_l, strict=True)}
        with timed("3_compute_attrib"):
            A = compute_attributions(banks, tcache, gposts)
            g = (A / (A.amax(-1, keepdim=True) + 1e-12))
        with timed("4_gated_pass"):
            logits_g, gcache = runner.gated_pass(idx, g)
        with timed("5_loss_faith"):
            loss_faith = sum(((banks[p.replace('.', '/')].weight()
                               - target.get_submodule(p).weight.detach().t()) ** 2).sum()
                             for p in MODULES) / n_params
        with timed("6_loss_recon"):
            loss_recon = F.kl_div(F.log_softmax(logits_g[:, :-1], -1),
                                  F.softmax(logits_t[:, :-1].detach(), -1), reduction="batchmean")
        with timed("7_loss_act"):
            loss_act = sum(((gcache[p]["post"] - tcache[p]["post"].detach()) ** 2).mean()
                           for p in MODULES) / len(MODULES)
        with timed("8_pair_geometry"):
            BT = idx.shape[0] * idx.shape[1]
            pi = torch.randint(0, BT, (args.n_pairs,), device=device)
            pj = torch.randint(0, BT, (args.n_pairs,), device=device)
            w_sim = pair_geometry(tcache, gposts, W2, pi, pj).detach()
            Af = F.normalize(A.flatten(0, 1), dim=-1)
            u = (Af[pi] * Af[pj]).sum(-1)
            loss_geom = ((u - w_sim) ** 2).mean()
        loss = 100 * loss_faith + 1.0 * loss_recon + 30 * loss_geom + 0.1 * loss_act
        with timed("9_backward"):
            loss.backward()
        with timed("10_optimizer"):
            opt.step()
        t1.record(); torch.cuda.synchronize()
        if rec:
            step_ms += t0.elapsed_time(t1)
            T.n += 1

    print(f"\nprofiled {T.n} steps (C={args.C}, m={args.m}, batch {args.batch_seqs}x{args.seq_len})")
    print(f"{'phase':22s} {'ms/step':>9s}  {'% of step':>9s}")
    tot = sum(T.acc.values())
    for k in sorted(T.acc):
        v = T.acc[k] / T.n
        print(f"{k:22s} {v:9.2f}  {100*T.acc[k]/tot:8.1f}%")
    print(f"{'SUM phases':22s} {tot/T.n:9.2f}")
    print(f"{'full step (measured)':22s} {step_ms/T.n:9.2f}")


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): pass


if __name__ == "__main__":
    main()
