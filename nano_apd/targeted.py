"""Targeted parameter decomposition: extract a SMALL bank (C~16) of rank-m components
that carry ONE supervised behavior (induction on pile-4L 67M), with the rest of the
network as an IMPLICIT residual. Unlike the full C=2048 decompositions, the residual is
never materialized or trained: the edited forward is

    out = x W^T - sum_c (1 - m_c) (x A_c) B_c        m_c in [0,1], 1 = keep, 0 = ablate

so faithfulness is exact by construction (all masks 1 -> target model bit-for-bit), and
training cost is LoRA-scale regardless of model size. The shipped artifact is the global
weight edit W - sum_c A_c B_c.

Losses (mixed batches: pile retain rows + repeated-random-token induction forget rows):
  retain   KL(edited || target) on pile rows under MIXED masks — half the rows fully
           ablated (the shipped edit), half with random per-component subset/partial
           masks (VPD-style: kills cross-component cancellation mass, the failure mode
           that made full-decomposition components unsubtractable), plus the random
           first half of induction rows under full ablation.
  forget   log-prob hinge on induction-copy predictions (2nd half of repeated rows)
           under FULL ablation: relu(logp_correct - logp0). Bounded — pushes the
           correct-token prob down to ~chance (logp0) and stops; no gradient-ascent
           blowup, no anti-induction overshoot.
  reg      optional sum ||A_c B_c||_F^2 / n_params minimality guard.

--retain_excl_ind excludes induction-predictable pile positions (bigram completed
earlier in context) from the retain KL: an honest induction edit MUST change those
predictions, so training them toward the target either fights the forget loss or forces
a shallow synthetic-only hack. Both arms are worth measuring.

Run:  python3.12 -m nano_apd.targeted --steps 3000 --tag v1 --wandb
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, "/workspace/param-decomp")
from nano_apd.lm_target import one_loader_batch  # noqa: E402
from nano_param_decomp.pile_4L import (  # noqa: E402
    C_PER_MODULE_4L,
    load_paper_target_model,
    make_loader,
)

OUT_ROOT = Path(__file__).parent / "out"
MODULES = list(C_PER_MODULE_4L.keys())
VOCAB_DRAW = 50000  # induction sequences drawn from the first 50k token ids


class ComponentBank(nn.Module):
    """Per decomposed matrix: piece_c = A_c B_c, A [C, d_in, m], B [C, m, d_out].
    B zero-init (LoRA-style): at step 0 every piece is exactly 0, so the full edit
    is a no-op and the retain loss starts at 0 — mass enters only where the forget
    loss pulls it and the retain loss tolerates it."""

    def __init__(self, d_in: int, d_out: int, C: int, m: int):
        super().__init__()
        self.C, self.m = C, m
        self.A = nn.Parameter(torch.empty(C, d_in, m))
        self.B = nn.Parameter(torch.zeros(C, m, d_out))
        nn.init.xavier_normal_(self.A)

    def piece_sq_mass(self) -> Tensor:  # [C], ||A_c B_c||_F^2 without materializing
        M = torch.einsum("cim,cin->cmn", self.A.float(), self.A.float())
        N = torch.einsum("cmo,cno->cmn", self.B.float(), self.B.float())
        return (M @ N).diagonal(dim1=-2, dim2=-1).sum(-1)


def build_banks(target: nn.Module, C: int, m: int) -> nn.ModuleDict:
    banks = nn.ModuleDict()
    for path in MODULES:
        lin = target.get_submodule(path)
        d_out, d_in = lin.weight.shape
        banks[path.replace(".", "/")] = ComponentBank(d_in, d_out, C, m)
    return banks


class Editor:
    """Monkeypatch decomposed linears: masks None -> exact target forward;
    masks [B, 1, C] (per-sequence, matching the global-weight-edit granularity)
    -> subtract each component's contribution scaled by (1 - mask)."""

    def __init__(self, target: nn.Module, banks: nn.ModuleDict):
        self.banks = banks
        self.masks: Tensor | None = None
        for path in MODULES:
            lin = target.get_submodule(path)
            lin._path = path
            lin.forward = self._make_forward(lin)

    def _make_forward(self, lin: nn.Linear):
        def fwd(x: Tensor) -> Tensor:
            out = F.linear(x, lin.weight, lin.bias)
            if self.masks is not None:
                bank: ComponentBank = self.banks[lin._path.replace(".", "/")]
                inner = torch.einsum("btd,cdm->btcm", x, bank.A)
                w = (1.0 - self.masks).unsqueeze(-1)             # [B, 1, C, 1]
                out = out - torch.einsum("btcm,cmo->bto", inner * w, bank.B)
            return out
        return fwd


def induction_batch(n_seq: int, half: int, device, seed: int) -> Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    first = torch.randint(0, VOCAB_DRAW, (n_seq, half), device=device, generator=gen)
    return torch.cat([first, first], 1)


def induction_predictable(idx: Tensor) -> Tensor:
    """[B, T] bool at position t (t >= 2): the bigram (x_{t-1}, x_t) already occurred
    at some s < t-1 — i.e. the token is completable by induction/copying from context.
    Used to SPLIT pile CE (truth metric) and optionally to exclude those positions
    from the retain KL."""
    B, T = idx.shape
    eq = idx.unsqueeze(1) == idx.unsqueeze(2)                    # [B, s, t]
    # match[b, s, t] = (x_s == x_{t-1}) & (x_{s+1} == x_t), valid for s <= t-2
    match = eq[:, : T - 1, : T - 1] & eq[:, 1:, 1:]              # [B, T-1, T-1] (s, t-1)
    s_idx = torch.arange(T - 1, device=idx.device)
    valid = (s_idx.view(1, -1, 1) <= s_idx.view(1, 1, -1) - 1)   # s <= (t-1) - 1
    out = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
    out[:, 1:] = (match & valid).any(1)
    return out


def kl_per_pos(logits_e: Tensor, logits_t: Tensor) -> Tensor:
    """[B, T-1] forward KL(target || edited) per predicting position."""
    return F.kl_div(
        F.log_softmax(logits_e[:, :-1].float(), -1),
        F.softmax(logits_t[:, :-1].float().detach(), -1),
        reduction="none",
    ).sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=16)
    p.add_argument("--rank_m", type=int, default=4, dest="m")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_pile", type=int, default=48)
    p.add_argument("--batch_ind", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--retain_w", type=float, default=1.0)
    p.add_argument("--forget_w", type=float, default=1.0)
    p.add_argument("--full_p", type=float, default=0.5,
                   help="prob a pile retain row uses the FULL edit (all comps ablated) "
                        "vs a random subset/partial mask")
    p.add_argument("--logp0", type=float, default=-10.0,
                   help="forget hinge floor: push logp(correct copy token) down to "
                        "this and stop (~2x uniform chance over 50k vocab)")
    p.add_argument("--reg_w", type=float, default=0.0)
    p.add_argument("--forget_nat_w", type=float, default=0.0,
                   help="ALSO forget natural-text induction: push down the edited "
                        "model's log-prob on pile positions that are induction-"
                        "predictable AND that the target gets right (top-1). Floor "
                        "--logp0_nat, milder than synthetic (natural tokens keep "
                        "other evidence sources). Use with --retain_excl_ind, else "
                        "retain and forget fight over the same positions.")
    p.add_argument("--logp0_nat", type=float, default=-5.0)
    p.add_argument("--retain_excl_ind", action="store_true",
                   help="exclude induction-predictable pile positions from retain KL")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--probe_split", default="validation")
    p.add_argument("--allow_train_fallback", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modules_filter", default="",
                   help="comma-separated substrings; decompose ONLY matching modules "
                        "(e.g. 'h.1.attn,h.2.attn,h.3.' = the known induction-circuit "
                        "sites). Tests whether mechanism-level localization improves "
                        "edit robustness, per the mechanistic-unlearning literature.")
    p.add_argument("--tag", default="v1")
    args = p.parse_args()
    if args.modules_filter:
        keep = args.modules_filter.split(",")
        MODULES[:] = [m_ for m_ in MODULES if any(k in m_ for k in keep)]
        print(f"decomposing {len(MODULES)} modules: {MODULES}", flush=True)

    device = "cuda"
    torch.manual_seed(args.seed)
    target = load_paper_target_model().float().to(device)
    for prm in target.parameters():
        prm.requires_grad_(False)          # only banks train; no target grads needed

    banks = build_banks(target, args.C, args.m).to(device)
    editor = Editor(target, banks)
    opt = torch.optim.AdamW(banks.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(target.get_submodule(p_).weight.numel() for p_ in MODULES)
    out_dir = OUT_ROOT / f"pile4l_targeted_{args.tag}"
    ckpt_path = out_dir / "ckpt.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        banks.load_state_dict(checkpoint["banks"])
        if "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        print(f"resumed {ckpt_path} at step {start_step}", flush=True)

    loader = make_loader(args.batch_pile, args.seq_len, 0, 1, "train", args.seed)
    half = args.seq_len // 2

    # Fixed probes. A train-stream fallback is never silent and is recorded in config.
    probe_ind = induction_batch(24, half, device, 999)
    try:
        probe_pile = one_loader_batch(make_loader(
            32, args.seq_len, 0, 1, args.probe_split, 999_999
        )).to(device)
        probe_split_actual = args.probe_split
    except Exception:
        if not args.allow_train_fallback:
            raise
        probe_pile = one_loader_batch(make_loader(
            32, args.seq_len, 0, 1, "train", 999_999
        )).to(device)
        probe_split_actual = "train-fallback"
    probe_mask_ind = induction_predictable(probe_pile)[:, 1:]   # aligns with CE positions

    @torch.no_grad()
    def probe(full_masks_zero: Tensor):
        editor.masks = None
        lt_i = target(probe_ind)
        lt_p = target(probe_pile)
        acc_t = (lt_i[:, half:-1].argmax(-1) == probe_ind[:, half + 1:]).float().mean()
        ce_t = F.cross_entropy(lt_p[:, :-1].flatten(0, 1), probe_pile[:, 1:].flatten(),
                               reduction="none").view(probe_pile.shape[0], -1)
        editor.masks = full_masks_zero[: probe_ind.shape[0]]
        le_i = target(probe_ind)
        editor.masks = full_masks_zero[: probe_pile.shape[0]]
        le_p = target(probe_pile)
        editor.masks = None
        acc_e = (le_i[:, half:-1].argmax(-1) == probe_ind[:, half + 1:]).float().mean()
        ce_e = F.cross_entropy(le_p[:, :-1].flatten(0, 1), probe_pile[:, 1:].flatten(),
                               reduction="none").view(probe_pile.shape[0], -1)
        d = ce_e - ce_t
        return {
            "probe_acc_target": round(acc_t.item(), 3),
            "probe_acc_edit": round(acc_e.item(), 3),
            "probe_dCE_pile": round(d.mean().item(), 4),
            "probe_dCE_indtok": round(d[probe_mask_ind].mean().item(), 4),
            "probe_dCE_other": round(d[~probe_mask_ind].mean().item(), 4),
            "indtok_frac": round(probe_mask_ind.float().mean().item(), 4),
        }

    zeros_masks = torch.zeros(max(64, args.batch_pile + args.batch_ind), 1, args.C,
                              device=device)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project="nano-apd", name=f"targeted_{args.tag}",
                        config=vars(args))

    for step in range(start_step, args.steps + 1):
        lr = args.lr * min(1.0, (step + 1) / 100) * \
            (0.5 * (1 + torch.cos(torch.tensor(step / args.steps * 3.14159)))).item()
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)

        pile = next(loader).to(device)
        ind = induction_batch(args.batch_ind, half, device,
                              7_000_000 + args.seed * 100_000 + step)
        idx = torch.cat([pile, ind], 0)
        Bp, Bi = pile.shape[0], ind.shape[0]

        with torch.no_grad():
            editor.masks = None
            logits_t = target(idx)

        # masks: pile rows full-zero w.p. full_p else random subset/partial;
        # induction rows always full-zero (the forget objective is about the edit)
        mode_full = torch.rand(Bp, 1, 1, device=device) < args.full_p
        u = torch.rand(Bp, 1, args.C, device=device)
        keep = (torch.rand(Bp, 1, args.C, device=device) < 0.5).float()
        rand_mask = keep + (1 - keep) * u
        masks = torch.cat([
            torch.where(mode_full, torch.zeros_like(rand_mask), rand_mask),
            torch.zeros(Bi, 1, args.C, device=device)], 0)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            editor.masks = masks
            logits_e = target(idx)
            editor.masks = None

        kl = kl_per_pos(logits_e, logits_t)                      # [B, T-1]
        retain_sel = torch.zeros_like(kl, dtype=torch.bool)
        retain_sel[:Bp] = True
        pile_itok = induction_predictable(pile)[:, 1:]
        if args.retain_excl_ind:
            retain_sel[:Bp] &= ~pile_itok
        retain_sel[Bp:, : half - 1] = True                       # random 1st half of ind
        loss_retain = kl[retain_sel].mean()

        logp = F.log_softmax(logits_e[Bp:, half:-1].float(), -1)
        logp_c = logp.gather(-1, ind[:, half + 1:].unsqueeze(-1)).squeeze(-1)
        loss_forget = F.relu(logp_c - args.logp0).mean()

        loss_forget_nat = torch.tensor(0.0, device=device)
        if args.forget_nat_w > 0:
            # forget positions: induction-predictable, target top-1 correct, AND the
            # pile row was fully edited this step (mask exactly 0 — the shipped edit)
            lp_p = F.log_softmax(logits_e[:Bp, :-1].float(), -1)
            lpc = lp_p.gather(-1, pile[:, 1:].unsqueeze(-1)).squeeze(-1)
            t_ok = logits_t[:Bp, :-1].argmax(-1) == pile[:, 1:]
            fully = (masks[:Bp, 0] == 0).all(-1).unsqueeze(-1)
            fsel = pile_itok & t_ok & fully
            if fsel.any():
                loss_forget_nat = F.relu(lpc[fsel] - args.logp0_nat).mean()

        loss_reg = torch.tensor(0.0, device=device)
        if args.reg_w > 0:
            loss_reg = sum(b.piece_sq_mass().sum() for b in banks.values()) / n_params

        loss = (args.retain_w * loss_retain + args.forget_w * loss_forget
                + args.forget_nat_w * loss_forget_nat + args.reg_w * loss_reg)

        if step % 50 == 0:
            rec = {"step": step, "retain_kl": round(loss_retain.item(), 5),
                   "forget": round(loss_forget.item(), 4),
                   "forget_nat": round(loss_forget_nat.item(), 4),
                   "logp_copy": round(logp_c.mean().item(), 3),
                   "mass": round(sum(b.piece_sq_mass().sum().item()
                                     for b in banks.values()), 2),
                   "lr": round(lr, 6)}
            if step % args.eval_every == 0:
                rec.update(probe(zeros_masks))
            print(json.dumps(rec), flush=True)
            if wb:
                wb.log(rec, step=step)

        if step != args.steps:
            loss.backward()
            opt.step()
            if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp = out_dir / "ckpt.tmp"
                torch.save({"banks": banks.state_dict(), "optimizer": opt.state_dict(),
                            "step": step, "probe_split": probe_split_actual}, tmp)
                tmp.replace(out_dir / "ckpt.pt")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(banks.state_dict(), out_dir / "banks.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args) | {"probe_split_actual": probe_split_actual}, f, indent=2)
    (out_dir / "ckpt.pt").unlink(missing_ok=True)
    close_loader = getattr(loader, "close", None)
    if close_loader is not None:
        close_loader()
    print(f"saved to {out_dir}", flush=True)
    print("TARGETED_DONE", flush=True)


if __name__ == "__main__":
    main()
