"""OUR method on the Christensen & Riggs induction toy: the decisive regime test.

Their paper (arXiv 2511.08854) recovers the 2-step induction circuit with SPD on a 2-layer,
1-head-per-layer, d=16 attention-only transformer trained on: sample tokens, insert a separator
s at a random position and again at the end; predict the token that FOLLOWED the first s.
The model must form an induction circuit (layer-0 positional/previous-token machinery writes
where the target token is; layer-1 attends from the final s back to it and copies).

Here we train that target, decompose it with OUR machinery (whole-network rank-1 components, one
shared gate, hidden recon, moderate L1), and score against the same answer key they use:
  - position-resolved CI: which components are active at s1 / m (=token after s1) / s2 / random?
  - per-matrix ownership: do dedicated components own Q1/K1 (the s2->m attention) and layer-0
    machinery, the way their Table 1 finds 1 subcomponent each for Q0,K0,V0,Q1,K1 (+11 for V1)?
  - causal: ablate the s2-active components -> task accuracy must die; keep-only them (+ backbone)
    -> must survive.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.apd_induction_toy
Env:  STEPS (decomp, default 8000), C (default 48), IMP, L1, HIDDEN, SEED, TRAIN_STEPS
      Variable-rank variants: R (pieces per matrix per component, default 1),
      FROB (V1: unweighted variational nuclear norm), NESTED=1 (V2: Matryoshka rank prefixes),
      RANK (V3: capacity-x-usage piece-count penalty)
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .apd_lm import SharedCI, decompose_lm, faithfulness_eval
from .apd_mask import ApdConfig, clear_masks, refresh_caches
from .apd_lm import masked_forward
from .run import Config as VpdConfig

VOCAB, SEQ, D, S_TOK = 128, 64, 16, 0  # token 0 reserved as the separator


class ToyInductionTransformer(nn.Module):
    """2-layer, 1-head, attention-only, learned positional embeddings, pre-LN."""

    def __init__(self, vocab: int = VOCAB, d: int = D, seq: int = SEQ) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList()
        for _ in range(2):
            blk = nn.ModuleDict({
                "ln": nn.LayerNorm(d),
                "q": nn.Linear(d, d, bias=False), "k": nn.Linear(d, d, bias=False),
                "v": nn.Linear(d, d, bias=False), "o": nn.Linear(d, d, bias=False),
            })
            self.blocks.append(blk)
        self.ln_f = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, vocab, bias=False)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        self.register_buffer("causal", mask, persistent=False)

    def forward(self, idx: Tensor) -> Tensor:  # [B, S] -> [B, S, V]
        B, S = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(S, device=idx.device))
        for blk in self.blocks:
            h = blk["ln"](x)
            q, k, v = blk["q"](h), blk["k"](h), blk["v"](h)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            att = att.masked_fill(~self.causal[:S, :S], float("-inf")).softmax(-1)
            x = x + blk["o"](att @ v)
        return self.unembed(self.ln_f(x))


def gen_batch(B: int, device, gen) -> tuple[Tensor, Tensor, Tensor]:
    """Sequences with s at a random position p (2..S-8) and at the end; returns (seq, p, m)."""
    seq = torch.randint(1, VOCAB, (B, SEQ), device=device, generator=gen)
    p = torch.randint(2, SEQ - 8, (B,), device=device, generator=gen)
    seq[torch.arange(B, device=device), p] = S_TOK
    seq[:, -1] = S_TOK
    m = seq[torch.arange(B, device=device), p + 1]  # answer: token after first s
    return seq, p, m


def train_target(device, steps: int = 4000, seed: int = 0) -> ToyInductionTransformer:
    torch.manual_seed(seed)
    model = ToyInductionTransformer().to(device)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(steps):
        seq, p, m = gen_batch(512, device, gen)
        logits = model(seq)[:, -1, :]  # predict at the final s position
        loss = F.cross_entropy(logits, m)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            acc = (logits.argmax(-1) == m).float().mean().item()
            print(f"  target step {step}: loss {loss.item():.3f} acc {acc:.3f}", flush=True)
    seq, p, m = gen_batch(4096, device, gen)
    acc = (model(seq)[:, -1, :].argmax(-1) == m).float().mean().item()
    print(f"target task accuracy: {acc:.3f}", flush=True)
    return model


@torch.no_grad()
def rubric(model, banks, ci, cfg, device) -> None:
    gen = torch.Generator(device=device).manual_seed(99)
    seq, p, m = gen_batch(256, device, gen)
    B = seq.shape[0]
    clear_masks(banks)
    logits = model(seq)
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g, _ = ci(acts)  # [B, S, C]
    C = g.shape[-1]
    ar = torch.arange(B, device=device)

    # position-resolved activity
    pos_sets = {"s1": g[ar, p], "m(=s1+1)": g[ar, p + 1], "s2(last)": g[:, -1],
                "random-mid": g[ar, (p + 10) % (SEQ - 2)]}
    rate = (g > 0.5).float().mean(dim=(0, 1))
    print("\n=== position-resolved component activity ===", flush=True)
    for name, gp in pos_sets.items():
        act = (gp > 0.5).float()
        print(f"  {name:<10} mean active comps: {act.sum(-1).mean():.2f}", flush=True)

    # components selectively active at s2 (the induction query position)
    p_s2 = (g[:, -1] > 0.5).float().mean(0)
    sel = p_s2 - rate
    order = sel.argsort(descending=True)
    W = {n: b.materialized_weights() for n, b in banks.items()}
    print("\n=== top s2-selective components (P(active@s2) - overall rate) + matrix ownership ===", flush=True)
    for c in order[:8].tolist():
        mass = {n.split('.')[-1] + n.split('.')[1]: W[n][c].pow(2).sum().item() for n in W}
        tot = sum(mass.values()) + 1e-12
        fp = " ".join(f"{k}:{v/tot:.2f}" for k, v in sorted(mass.items(), key=lambda kv: -kv[1])[:3])
        print(f"  comp {c:>3} P@s2={p_s2[c]:.2f} rate={rate[c]:.2f} sel={sel[c]:+.2f} | {fp}", flush=True)

    # per-matrix ownership: who owns each of the 8 matrices? (their Table-1 analog)
    print("\n=== per-matrix ownership (top comp's share of each matrix's weight energy) ===", flush=True)
    for n in sorted(W):
        w = W[n]
        tot = banks[n].W_target.pow(2).sum()
        share = w.pow(2).sum(dim=(1, 2)) / tot
        srt = share.sort(descending=True)
        n50 = int((srt.values.cumsum(0) < 0.5).sum()) + 1
        c0 = srt.indices[0].item()
        print(f"  {n:<14} top comp {c0:>3} owns {srt.values[0]:.1%} (P@s2={p_s2[c0]:.2f}, rate={rate[c0]:.2f}); "
              f"comps for 50%: {n50}", flush=True)

    # variable-rank runs: what effective rank did each component CHOOSE, per matrix?
    # (C&R answer key for this toy: ~1 piece each in Q0/K0/V0/Q1/K1, ~11 in V1.)
    if getattr(next(iter(banks.values())), "r", 1) > 1:
        from .apd_mask import rank_profile
        prof = rank_profile(banks)  # {module: [C, 3] (svd_rank, piece_count, participation)}
        live = rate > 0.05
        print("\n=== effective rank per matrix (SVD of materialized component; pieces in brackets) ===",
              flush=True)
        for n in sorted(W):
            svd_r, counts = prof[n][:, 0], prof[n][:, 1]
            share = W[n].pow(2).sum(dim=(1, 2)) / banks[n].W_target.pow(2).sum()
            c0 = int(share.argmax())
            lc = svd_r[live] if live.any() else svd_r
            print(f"  {n:<14} top-owner comp {c0:>3}: svd rank {int(svd_r[c0]):>2} "
                  f"[{int(counts[c0]):>2} pieces] | live comps: mean {lc.mean():.1f}, max {int(lc.max())}",
                  flush=True)
        svd_all = torch.stack([prof[n][:, 0] for n in sorted(prof)], dim=1)  # [C, n_mod]
        el = svd_all[live].mean(dim=1) if live.any() else svd_all.mean(dim=1)
        print(f"live components: {int(live.sum())}; svd rank (mean over matrices): "
              f"mean={el.mean():.2f} median={el.median():.1f} max={el.max():.1f}", flush=True)

    # causal: ablate the top-k s2-selective comps vs random; keep-only them + always-on backbone
    zeros = {n: torch.zeros(B, SEQ, device=device) for n in banks}
    base_acc = (logits[:, -1].argmax(-1) == m).float().mean().item()
    ci_acc = (masked_forward(model, banks, seq, g, zeros)[:, -1].argmax(-1) == m).float().mean().item()
    print(f"\ntask accuracy: target={base_acc:.3f}  ci-masked={ci_acc:.3f}", flush=True)
    backbone = torch.nonzero(rate > 0.9).squeeze(1)
    print(f"always-on backbone comps (rate>0.9): {len(backbone)}", flush=True)
    for K in (2, 4, 8):
        top = order[:K]
        gm = g.clone(); gm[..., top] = 0.0
        abl = (masked_forward(model, banks, seq, gm, zeros)[:, -1].argmax(-1) == m).float().mean().item()
        rnd = torch.randperm(C, generator=torch.Generator().manual_seed(K))[:K].to(device)
        gr = g.clone(); gr[..., rnd] = 0.0
        abl_r = (masked_forward(model, banks, seq, gr, zeros)[:, -1].argmax(-1) == m).float().mean().item()
        gk = torch.zeros_like(g); keep = torch.cat([top, backbone]); gk[..., keep] = g[..., keep]
        keep_acc = (masked_forward(model, banks, seq, gk, zeros)[:, -1].argmax(-1) == m).float().mean().item()
        print(f"  K={K}: ablate top-K s2-selective -> {abl:.3f} | ablate K random -> {abl_r:.3f} | "
              f"keep-only top-K + backbone -> {keep_acc:.3f}", flush=True)


def _run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = int(os.environ.get("STEPS", "8000"))
    C = int(os.environ.get("C", "48"))
    imp = float(os.environ.get("IMP", "1e-3"))
    l1 = float(os.environ.get("L1", "1e-3"))
    hidden = float(os.environ.get("HIDDEN", "1.0"))
    seed = int(os.environ.get("SEED", "0"))
    R = int(os.environ.get("R", "1"))                 # pieces per matrix per component (rank cap)
    frob = float(os.environ.get("FROB", "0.0"))       # V1
    nested = os.environ.get("NESTED", "0") == "1"     # V2
    rank_pen = float(os.environ.get("RANK", "0.0"))   # V3
    rank_floor = float(os.environ.get("RANKFLOOR", "0.05"))  # V3 usage-weight floor

    print("training target ...", flush=True)
    target = train_target(device, steps=int(os.environ.get("TRAIN_STEPS", "4000")), seed=seed)

    # pool of task sequences for decomposition
    gen = torch.Generator(device=device).manual_seed(7)
    pool = torch.cat([gen_batch(512, device, gen)[0] for _ in range(4)], 0).cpu()

    modules = [f"blocks.{i}.{mm}" for i in range(2) for mm in ("q", "k", "v", "o")]
    cfg = ApdConfig(modules=modules, n_components=C, simplicity_impl="factored", factor_rank=R,
                    lowrank_forward=True, coeff_faith=1e8, coeff_imp=imp, coeff_simplicity=0.0,
                    coeff_hidden=hidden, coeff_weight_l1=l1, p_start=2.0, p_end=0.4, seed=seed,
                    coeff_frob=frob, nested_rank=nested, coeff_rank=rank_pen,
                    rank_freq_floor=rank_floor)
    ci_cfg = VpdConfig(C_per_module={m: C for m in modules}, seq_len=SEQ,
                       ci_d_model=64, ci_n_blocks=2, ci_n_heads=4, ci_mlp_hidden=256,
                       coeff_stoch=0.5, coeff_ppgd=0.5, ppgd_lr=0.01, ppgd_inner_steps=2)
    print(f"config: C={C} R={R} steps={steps} imp={imp} l1={l1} hidden={hidden} "
          f"frob={frob} nested={nested} rank={rank_pen}", flush=True)
    tag = os.environ.get("TAG", "")
    out = decompose_lm(target, pool, cfg, ci_cfg, device, n_steps=steps, batch=64, seq_len=SEQ,
                       warmup_steps=300, save_path=f"/tmp/ind_toy_c{C}_r{R}{'_' + tag if tag else ''}.pt")
    ev = pool[:64].to(device)
    fe = faithfulness_eval(out["model"], out["banks"], out["ci"], ev, cfg, device)
    print(f"\nfaithfulness: kl_ci={fe['kl_ci_masked']:.4f} ce_rec={fe['ce_recovered_pct']:.1f}% "
          f"kl_unmasked={fe['kl_unmasked']:.2e} L0={fe['L0']:.1f}/{C} ratio={fe['l1_ratio']:.2f}", flush=True)
    rubric(out["model"], out["banks"], out["ci"], cfg, device)
    print("APD_INDUCTION_TOY DONE", flush=True)


if __name__ == "__main__":
    _run()
