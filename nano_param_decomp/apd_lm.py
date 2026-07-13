"""APD-basis decomposition of a small LM with a KNOWN cross-layer circuit: Neel Nanda's `attn-only-2l`
induction circuit (prev-token head (0,3) -> induction head (1,6), K-composition across layers).

This is the LM analog of the toy resid experiment: whole-network parameter components (one gate per
component, shared across ALL decomposed q/k/v/o matrices), trained with a learned CI function under
stochastic reconstruction (KL), faithfulness, importance-minimality, and the factored (SVD-free,
low-rank-forward) backend for efficiency. We then score how well the decomposition RECOVERS the
induction circuit (rubric R1-R4 in `circuit_recovery`), for head-to-head vs VPD (+A.8 clustering).

Reuses run.py's CI transformer blocks (context-aware CI is required — induction is contextual) and
apd_mask's ComponentBankLinear (factored low-rank forward) + shared-gate masking.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.apd_lm
Env:  STEPS, C, R (factor_rank), IMP, LIFE, SEQ, B, SMOKE
"""

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _dist_info() -> tuple[int, int]:
    """(rank, world_size); (0, 1) when not launched under torchrun."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

from .apd_mask import (
    ApdConfig,
    clear_masks,
    faithfulness_loss,
    frob_loss,
    install_banks,
    masked_forward,
    rank_count_loss,
    refresh_caches,
    sample_rank_rung,
    set_masks,
    simplicity_loss,
)
from .run import (
    CIBlock,
    Config as VpdConfig,
    _ce_next_token,
    cosine_lr,
    importance_minimality_loss,
    induction_copy_acc,
    kl_logits,
    lower_leaky,
    precompute_rope,
    upper_leaky,
)


# --- Persistent adversarial PGD on the SHARED gate (VPD-style, adapted to one gate/component) -------


class SharedPGD:
    """Persistent adversarial sources for the shared component gate [B,S,C] + per-module delta [B,S].
    Mirrors run.PersistentPGD but with ONE gate shared across all modules. Sources persist across
    steps with Adam state; each step: warmup (inner PGD ascent on KL), recon_loss (enters total),
    external Adam step after total.backward()."""

    def __init__(self, banks, C, local_B, seq_len, device, cfg: VpdConfig) -> None:
        self.cfg = cfg
        self.names = list(banks)
        self.gate = torch.rand(local_B, seq_len, C, device=device).requires_grad_(True)
        self.gm = torch.zeros_like(self.gate)
        self.gv = torch.zeros_like(self.gate)
        self.delta = {n: torch.rand(local_B, seq_len, device=device).requires_grad_(True) for n in self.names}
        self.dm = {n: torch.zeros_like(v) for n, v in self.delta.items()}
        self.dv = {n: torch.zeros_like(v) for n, v in self.delta.items()}
        self.t = 0

    def _sources(self) -> list[Tensor]:
        return [self.gate] + [self.delta[n] for n in self.names]

    def recon_loss(self, model, banks, idx, target, ci_lower) -> Tensor:
        mask = ci_lower + (1 - ci_lower) * self.gate
        return kl_logits(masked_forward(model, banks, idx, mask, dict(self.delta)), target)

    def warmup(self, model, banks, idx, target, ci_lower, lr: float) -> None:
        for _ in range(self.cfg.ppgd_inner_steps):
            loss = self.recon_loss(model, banks, idx, target, ci_lower)
            grads = torch.autograd.grad(loss, self._sources())
            self._adam(grads, lr)

    def external_step(self, grads, lr: float) -> None:
        self._adam(grads, lr)

    def _adam(self, grads, lr: float) -> None:
        self.t += 1
        c = self.cfg
        bc1, bc2 = 1 - c.ppgd_beta1**self.t, 1 - c.ppgd_beta2**self.t
        ms = [self.gm] + [self.dm[n] for n in self.names]
        vs = [self.gv] + [self.dv[n] for n in self.names]
        with torch.no_grad():
            for src, g, m, v in zip(self._sources(), grads, ms, vs, strict=True):
                m.mul_(c.ppgd_beta1).add_(g, alpha=1 - c.ppgd_beta1)
                v.mul_(c.ppgd_beta2).addcmul_(g, g, value=1 - c.ppgd_beta2)
                src.add_(lr * (m / bc1) / ((v / bc2).sqrt() + c.ppgd_eps))
                src.clamp_(0.0, 1.0)


# --- Shared-gate CI transformer (one gate per whole-network component) ------------------------------


class SharedCI(nn.Module):
    """Context-aware CI: reads every decomposed module's pre-weight activations (RMS-normed,
    concatenated) and emits ONE gate per component, shared across all modules -> [B, S, C]."""

    def __init__(self, d_in_per_module: dict[str, int], C: int, cfg: VpdConfig) -> None:
        super().__init__()
        self.module_order = sorted(d_in_per_module)
        self.alpha = cfg.leaky_alpha
        total_in = sum(d_in_per_module.values())
        self.proj_in = nn.Linear(total_in, cfg.ci_d_model)
        self.blocks = nn.ModuleList([CIBlock(cfg) for _ in range(cfg.ci_n_blocks)])
        self.proj_out = nn.Linear(cfg.ci_d_model, C)
        head_dim = cfg.ci_d_model // cfg.ci_n_heads
        cos, sin = precompute_rope(cfg.seq_len, head_dim, cfg.ci_rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, acts: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        normed = [F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order]
        x = self.proj_in(torch.cat(normed, dim=-1))
        S = x.shape[1]
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.proj_out(x)  # [B, S, C]
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha)


# --- training ---------------------------------------------------------------------------------------


def decompose_lm(model: nn.Module, pool: Tensor, cfg: ApdConfig, ci_cfg: VpdConfig,
                 device: torch.device, n_steps: int, batch: int, seq_len: int,
                 lr: float = 5e-4, warmup_steps: int = 400, eval_every: int = 500,
                 log_every: int = 100, save_path: str | None = None) -> dict[str, object]:
    rank, world = _dist_info()  # DDP: identical init on all ranks (same seed), sharded data,
    rank0 = rank == 0           # grad all-reduce after backward, rank-0-only eval/save.
    torch.manual_seed(cfg.seed)
    banks = install_banks(model, cfg)
    model = model.to(device)
    order = sorted(cfg.modules)
    d_in = {n: int(b.W_target.shape[1]) for n, b in banks.items()}
    ci = SharedCI(d_in, cfg.n_components, ci_cfg).to(device)
    if rank0:
        print(f"installed {len(banks)} banks, CI {sum(p.numel() for p in ci.parameters()):,} params "
              f"(world_size={world}, local batch={batch})", flush=True)
    use_wandb = rank0 and cfg.use_wandb
    if use_wandb:
        import dataclasses

        import wandb  # type: ignore[import-untyped]

        # sweep agents init the run before calling us; don't double-init (matryoshka pattern)
        run_cfg = {**dataclasses.asdict(cfg), **{f"ci_{k}": v for k, v in dataclasses.asdict(ci_cfg).items()},
                   "n_steps": n_steps, "global_batch": batch * world, "seq_len": seq_len, "lr": lr}
        if wandb.run is None:
            wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, group=cfg.wandb_group,
                       job_type=cfg.wandb_job_type, name=cfg.wandb_run_name,
                       tags=list(cfg.wandb_tags), notes=cfg.wandb_notes, config=run_cfg)
        else:
            wandb.config.update(run_cfg, allow_val_change=True)
        print(f"wandb url: {wandb.run.url if wandb.run else '?'}", flush=True)

    comp_params = [p for b in banks.values() for p in b.params()]
    # faithfulness warmup
    wopt = torch.optim.AdamW(comp_params, lr=1e-3)
    for _ in range(warmup_steps):
        refresh_caches(banks)
        loss = faithfulness_loss(banks)
        wopt.zero_grad(); loss.backward(); wopt.step()
        # no all-reduce needed: identical init + identical (data-free) loss => identical trajectories
    if rank0:
        print(f"faithfulness after warmup: {loss.item():.3e}", flush=True)

    opt = torch.optim.AdamW(comp_params + list(ci.parameters()), lr=lr)
    trainable = comp_params + list(ci.parameters())
    g = torch.Generator().manual_seed(cfg.seed + 7919 * rank)  # per-rank data sharding
    ppgd = SharedPGD(banks, cfg.n_components, batch, seq_len, device, ci_cfg)
    best_score = float("inf")

    def sample_batch() -> Tensor:
        idx = torch.randint(0, pool.shape[0], (batch,), generator=g)
        return pool[idx].to(device)

    # bf16 autocast for the heavy forwards (same pattern as the VPD stack in run.py): ~2x matmul
    # throughput on tensor cores and halved activation memory (the [tokens, C, r] intermediates
    # dominate at large C). Faithfulness, minimality, and the loss summation stay in fp32 outside
    # the context; the KL/MSE reductions upcast via .float() inside kl_logits and the losses below.
    use_amp = os.environ.get("AMP", "0") == "1" and device.type == "cuda"
    import contextlib

    def amp_ctx():
        return torch.autocast("cuda", torch.bfloat16) if use_amp else contextlib.nullcontext()

    if rank0 and use_amp:
        print("bf16 autocast ON for forwards (losses fp32)", flush=True)

    for step in range(n_steps):
        p = cfg.p_start + (cfg.p_end - cfg.p_start) * (step / n_steps)
        ppgd_lr = cosine_lr(step, n_steps, ci_cfg.ppgd_lr, ci_cfg.ppgd_lr_final_frac, ci_cfg.ppgd_warmup_pct)
        idx = sample_batch()
        clear_masks(banks)
        with torch.no_grad(), amp_ctx():
            target_logits = model(idx)
        acts = {n: b.last_input for n, b in banks.items()}
        refresh_caches(banks)
        with amp_ctx():
            g_lower, g_upper = ci(acts)  # [B, S, C]
        g_lower, g_upper = g_lower.float(), g_upper.float()
        B, S = idx.shape

        with amp_ctx():
            ppgd.warmup(model, banks, idx, target_logits, g_lower, ppgd_lr)  # inner adversarial ascent

        loss_faith = faithfulness_loss(banks)
        # stochastic-mask KL reconstruction (shared gate across modules). With subset routing (VPD
        # uniform-k-subset), only a random k of the modules are masked; the rest run at target.
        u = torch.rand_like(g_lower)
        mask = g_lower + (1 - g_lower) * u
        deltas = {n: torch.rand(B, S, device=device) for n in order} if cfg.use_delta else None
        subset: set[str] | None = None
        if cfg.subset_routing:
            k = int(torch.randint(1, len(order) + 1, (1,), generator=g).item())
            subset = {order[i] for i in torch.randperm(len(order), generator=g)[:k].tolist()}
        if cfg.nested_rank:  # V2: stochastic recon (+hidden, same pass) under a random rank-prefix;
            # faithfulness, PPGD and eval stay full-rank. Sample the rung from the LARGEST cap; each
            # bank truncates to min(rung, its own r) -- the slice [:rung] clamps naturally -- so piece
            # index i means "i-th most important piece" consistently across heterogeneous-rank banks.
            rung = sample_rank_rung(max(b.r for b in banks.values()), g)
            for b in banks.values():
                b.rank_keep = rung
        with amp_ctx():
            stoch_logits = masked_forward(model, banks, idx, mask, deltas, subset)
        loss_stoch = kl_logits(stoch_logits.float(), target_logits.float())
        # APD hidden-activation recon: match each masked module's output to its target-weights
        # output (cached during the target forward). Only masked modules contribute (unmasked = 0).
        loss_hidden = torch.zeros((), device=device)
        if cfg.coeff_hidden > 0:
            hid_mods = sorted(subset) if subset is not None else order
            for n in hid_mods:
                tgt = banks[n].last_target_out.float()
                loss_hidden = loss_hidden + F.mse_loss(banks[n].last_masked_out.float(), tgt) / (tgt.var() + 1e-8)
            loss_hidden = loss_hidden / len(hid_mods)
        if cfg.nested_rank:
            for b in banks.values():
                b.rank_keep = None
        with amp_ctx():
            loss_ppgd = ppgd.recon_loss(model, banks, idx, target_logits, g_lower)  # adversarial recon
        loss_ppgd = loss_ppgd.float()
        loss_imp = importance_minimality_loss({"g": g_upper}, p, cfg.imp_eps, cfg.imp_beta, world)
        loss_simp = (simplicity_loss(banks, g_upper.mean(dim=(0, 1)), cfg)
                     if cfg.coeff_simplicity > 0 else torch.zeros((), device=device))
        loss_frob = frob_loss(banks) if cfg.coeff_frob > 0 else torch.zeros((), device=device)
        loss_rank = (rank_count_loss(banks, g_upper.mean(dim=(0, 1)), cfg)
                     if cfg.coeff_rank > 0 else torch.zeros((), device=device))
        # anti-redundancy interaction loss (ported from apd_alg): penalize super-additive pairwise
        # ablation damage (redundant role overlap); sub-additive (shared pathway) not penalized.
        loss_inter = torch.zeros((), device=device)
        if cfg.coeff_interaction > 0:
            probs = g_upper.mean(dim=(0, 1)).detach() + 1e-6
            n_s = min(2 * cfg.interaction_pairs, cfg.n_components)
            cidx = torch.multinomial(probs, n_s, replacement=False)
            pairs = cidx[: (n_s // 2) * 2].view(-1, 2)
            zero_d = {n: torch.zeros(B, S, device=device) for n in order}

            def _abl(ablate: list[int]) -> Tensor:
                gm = g_lower.clone()
                if ablate:
                    gm[..., ablate] = 0.0
                return kl_logits(masked_forward(model, banks, idx, gm, zero_d), target_logits)

            l_base = _abl([])
            for i, j in pairs.tolist():
                loss_inter = loss_inter + F.relu(_abl([i, j]) - _abl([i]) - _abl([j]) + l_base)
            loss_inter = loss_inter / max(1, len(pairs))

        # entrywise L1 on components. Rank-1 exact form ||a b^T||_1 = ||a||_1 ||b||_1 -- no [C,d,d]
        # materialization (which would be ~4GB/step at C=512). Falls back to materializing otherwise.
        loss_l1 = torch.zeros((), device=device)
        if cfg.coeff_weight_l1 > 0:
            for b in banks.values():
                if b.impl == "factored" and b.r == 1:
                    loss_l1 = loss_l1 + (b.A.abs().sum(dim=(1, 2)) * b.B.abs().sum(dim=(1, 2))).sum()
                else:
                    loss_l1 = loss_l1 + b.materialized_weights().abs().sum()

        loss_life = torch.zeros((), device=device)
        if cfg.coeff_lifetime > 0.0:
            life_c = cfg.coeff_lifetime
            if cfg.lifetime_ramp_frac > 0.0:
                life_c = life_c * min(1.0, step / max(1, int(cfg.lifetime_ramp_frac * n_steps)))
            freq = g_upper.mean(dim=(0, 1))
            taxed = (freq - cfg.lifetime_target).clamp(min=0.0) if cfg.lifetime_target > 0 else freq
            loss_life = life_c * taxed.pow(cfg.lifetime_pow).sum()

        loss = (cfg.coeff_faith * loss_faith + ci_cfg.coeff_stoch * loss_stoch
                + ci_cfg.coeff_ppgd * loss_ppgd + cfg.coeff_imp * loss_imp
                + cfg.coeff_simplicity * loss_simp + loss_life
                + cfg.coeff_hidden * loss_hidden + cfg.coeff_interaction * loss_inter
                + cfg.coeff_weight_l1 * loss_l1
                + cfg.coeff_frob * loss_frob + cfg.coeff_rank * loss_rank)
        ppgd_grads = torch.autograd.grad(loss_ppgd, ppgd._sources(), retain_graph=True)
        opt.zero_grad(); loss.backward()
        if world > 1:  # data-parallel: average trainable grads; PGD sources stay per-rank.
            # ONE flat-bucket all_reduce instead of ~100 per-parameter NCCL calls (launch latency
            # dominated the sync); COMM_BF16=1 additionally halves the wire traffic — bf16 rounding
            # of averaged grads is far below Adam's noise floor at these scales.
            grads = [prm.grad for prm in trainable if prm.grad is not None]
            flat = torch._utils._flatten_dense_tensors(grads)
            if os.environ.get("COMM_BF16", "0") == "1":
                flat16 = flat.to(torch.bfloat16)
                dist.all_reduce(flat16, op=dist.ReduceOp.AVG)
                flat = flat16.to(torch.float32)
            else:
                dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for g_, s_ in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                g_.copy_(s_)
        if cfg.grad_clip > 0:  # VPD clips the component-factor grads (not the CI net)
            torch.nn.utils.clip_grad_norm_(comp_params, cfg.grad_clip)
        opt.step()
        if cfg.unit_norm_A:  # APD gauge-fix: unit-norm A rank-slices, magnitude folded into B
            with torch.no_grad():
                for b in banks.values():
                    if b.impl == "factored":
                        nrm = b.A.norm(dim=1, keepdim=True).clamp_min(1e-8)  # [C,1,r]
                        b.A.div_(nrm); b.B.mul_(nrm.transpose(1, 2))
        ppgd.external_step(ppgd_grads, ppgd_lr)

        if (step % log_every == 0 or step == n_steps - 1) and rank0:
            l0 = (g_lower > 0.5).float().sum(-1).mean().item()
            print(f"  step {step:>5} faith={loss_faith.item():.2e} kl={loss_stoch.item():.4f} "
                  f"ppgd={loss_ppgd.item():.4f} imp={loss_imp.item():.2f} hid={loss_hidden.item():.4f} "
                  f"inter={loss_inter.item():.4f} l1={loss_l1.item():.0f} "
                  f"frob={loss_frob.item():.2f} rank={loss_rank.item():.2f} "
                  f"L0={l0:.1f}/{cfg.n_components}", flush=True)
            if use_wandb:
                wandb.log({"train/faithfulness": loss_faith.item(), "train/kl_stoch": loss_stoch.item(),
                           "train/kl_ppgd": loss_ppgd.item(), "train/importance_min": loss_imp.item(),
                           "train/hidden_recon": loss_hidden.item(), "train/interaction": loss_inter.item(),
                           "train/weight_l1": loss_l1.item(), "train/lifetime": loss_life.item(),
                           "train/frob": loss_frob.item(), "train/rank_count": loss_rank.item(),
                           "train/L0": l0, "train/p_anneal": p}, step=step)
        if (step % eval_every == 0 or step == n_steps - 1) and rank0:
            fe = faithfulness_eval(model, banks, ci, idx, cfg, device)
            print(f"    [eval] kl_ci_masked={fe['kl_ci_masked']:.4f} ce_recovered={fe['ce_recovered_pct']:.1f}% "
                  f"kl_unmasked={fe['kl_unmasked']:.2e} L0={fe['L0']:.1f}", flush=True)
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in fe.items()}, step=step)
            # best-checkpoint tracking (Matryoshka lesson: the final step is not necessarily the
            # best — L1 keeps reorganizing after recon peaks). Gate on the sanity check so we never
            # crown an unfaithful decomposition.
            score = fe["kl_ci_masked"]
            if save_path is not None and fe["kl_unmasked"] < 0.05 and score < best_score:
                best_score = score
                import dataclasses
                torch.save({"banks": {n: b.state_dict() for n, b in banks.items()},
                            "ci": ci.state_dict(), "cfg": dataclasses.asdict(cfg),
                            "ci_cfg": dataclasses.asdict(ci_cfg), "step": step,
                            "kl_ci_masked": score}, save_path + ".best.pt")
                print(f"    [best] saved at step {step} (kl_ci={score:.4f})", flush=True)

    if save_path is not None and rank0:
        import dataclasses
        torch.save({
            "banks": {n: b.state_dict() for n, b in banks.items()},
            "ci": ci.state_dict(),
            "cfg": dataclasses.asdict(cfg),
            "ci_cfg": dataclasses.asdict(ci_cfg),
        }, save_path)
        print(f"saved decomposition -> {save_path}", flush=True)
    return {"banks": banks, "ci": ci, "model": model}


def load_decomp(path: str, model: nn.Module, device: torch.device):
    """Reload a saved APD-LM decomposition onto a fresh target model. Returns (banks, ci, cfg)."""
    ck = torch.load(path, weights_only=False)
    cfg = ApdConfig(**ck["cfg"])
    ci_cfg = VpdConfig(**ck["ci_cfg"])
    banks = install_banks(model, cfg)
    model = model.to(device)
    for n, b in banks.items():
        b.load_state_dict(ck["banks"][n])
    d_in = {n: int(b.W_target.shape[1]) for n, b in banks.items()}
    ci = SharedCI(d_in, cfg.n_components, ci_cfg).to(device)
    ci.load_state_dict(ck["ci"])
    return banks, ci, cfg, model


# --- induction eval ---------------------------------------------------------------------------------


@torch.no_grad()
def faithfulness_eval(model, banks, ci, eval_batch: Tensor, cfg: ApdConfig,
                      device: torch.device) -> dict[str, float]:
    """The paper metric: KL to target + CE-recovered under CI-masking (components only, delta off).
    kl_unmasked (~0) is a sanity check; ce_recovered = 1 - (ce_ci - ce_target)/(ce_zero - ce_target)."""
    clear_masks(banks)
    target = model(eval_batch)
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g_lower, _ = ci(acts)
    B, S = eval_batch.shape
    C = cfg.n_components
    zeros = {n: torch.zeros(B, S, device=device) for n in banks}

    def kl_ce(gate: Tensor) -> tuple[float, float]:
        pred = masked_forward(model, banks, eval_batch, gate, zeros)
        return kl_logits(pred, target).item(), _ce_next_token(pred, eval_batch)

    kl_ci, ce_ci = kl_ce(g_lower)
    kl_un, _ = kl_ce(torch.ones(B, S, C, device=device))
    _, ce_zero = kl_ce(torch.zeros(B, S, C, device=device))
    ce_target = _ce_next_token(target, eval_batch)
    denom = ce_zero - ce_target
    ce_unrec = (ce_ci - ce_target) / denom if denom != 0 else float("nan")
    l0 = (g_lower > 0.0).float().sum(-1).mean().item()
    l1_sum = 0.0
    for b in banks.values():
        if b.impl == "factored" and b.r == 1:
            l1_sum += (b.A.abs().sum(dim=(1, 2)) * b.B.abs().sum(dim=(1, 2))).sum().item()
        else:
            l1_sum += b.materialized_weights().abs().sum().item()
    l1_w = sum(b.W_target.abs().sum().item() for b in banks.values())
    clear_masks(banks)
    return {"kl_ci_masked": kl_ci, "kl_unmasked": kl_un, "ce_unrecovered": ce_unrec,
            "ce_recovered_pct": 100 * (1 - ce_unrec), "L0": l0, "l1_ratio": l1_sum / l1_w}


def adversarial_kl(model, banks, ci, eval_batch: Tensor, cfg: ApdConfig, device: torch.device,
                   steps: int = 20, lr: float = 0.1) -> float:
    """Worst-case KL under adversarial gate ablation (sign-SGD PGD), delta off — the strict
    faithfulness check VPD argues is the meaningful one."""
    with torch.no_grad():
        clear_masks(banks)
        target = model(eval_batch)
        acts = {n: b.last_input for n, b in banks.items()}
        refresh_caches(banks)
        g_lower, _ = ci(acts)
    B, S = eval_batch.shape
    zeros = {n: torch.zeros(B, S, device=device) for n in banks}
    src = torch.rand(B, S, cfg.n_components, device=device, requires_grad=True)
    with torch.enable_grad():
        for _ in range(steps):
            mask = g_lower + (1 - g_lower) * src
            kl = kl_logits(masked_forward(model, banks, eval_batch, mask, zeros), target)
            gr = torch.autograd.grad(kl, src)[0]
            src = (src + lr * gr.sign()).clamp(0, 1).detach().requires_grad_(True)
        mask = g_lower + (1 - g_lower) * src
        final = kl_logits(masked_forward(model, banks, eval_batch, mask, zeros), target).item()
    clear_masks(banks)
    return final


@torch.no_grad()
def induction_ci_masked(model, banks, ci, eval_batch: Tensor, cfg: ApdConfig,
                        device: torch.device) -> dict[str, float]:
    """Induction copy accuracy on an in-distribution repeated probe: unmasked (ceiling) vs CI-masked
    (important components only, delta off)."""
    L = min(64, eval_batch.shape[1] // 2)
    first = eval_batch[:32, :L]
    seq = torch.cat([first, first], dim=1)
    clear_masks(banks)
    unmasked = model(seq)
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g_lower, _ = ci(acts)
    B, S = seq.shape
    zeros = {n: torch.zeros(B, S, device=device) for n in banks}
    ci_masked = masked_forward(model, banks, seq, g_lower, zeros)
    clear_masks(banks)
    return {"unmasked": induction_copy_acc(unmasked, first, L),
            "ci_masked": induction_copy_acc(ci_masked, first, L)}


# --- circuit recovery rubric (R1-R4) ---------------------------------------------------------------


def _head_of_row(row: int, d_head: int) -> int:
    return row // d_head


@torch.no_grad()
def circuit_recovery(model, banks, ci, pool: Tensor, cfg: ApdConfig, device: torch.device,
                     prev_head: tuple[int, int], ind_head: tuple[int, int], d_head: int,
                     n_probe: int = 32, seq_len: int = 128) -> dict[str, object]:
    """Score how well the decomposition recovers the induction circuit.
    R1 sufficiency, R2 sparsity (keep-only top-k curve), R3 head attribution, R4 cross-layer span."""
    # induction probe: repeat the first half of IN-DISTRIBUTION pool sequences (the CI does NOT
    # generalize to OOD repeated-random tokens -- those read ~0 spuriously; repo eval_induction_vpd
    # uses in-distribution repeats for the same reason).
    L = seq_len // 2
    first = pool[:n_probe, :L].to(device)  # [n_probe, L] in-distribution
    seq = torch.cat([first, first], dim=1)

    clear_masks(banks)
    unmasked = model(seq)
    acts = {n: b.last_input for n, b in banks.items()}
    refresh_caches(banks)
    g_lower, g_upper = ci(acts)  # [B, S, C]
    C = cfg.n_components

    # importance per component = mean CI over 2nd-copy positions (where induction fires)
    imp = g_upper[:, L:, :].mean(dim=(0, 1))  # [C]
    order_c = torch.argsort(imp, descending=True)
    B, S = seq.shape
    zeros = {n: torch.zeros(B, S, device=device) for n in banks}

    def keep_only(top_idx: Tensor) -> float:
        gate = torch.zeros(B, S, C, device=device)
        gate[..., top_idx] = 1.0
        pred = masked_forward(model, banks, seq, gate, zeros)
        return induction_copy_acc(pred, first.to(device), L)

    unmasked_copy = induction_copy_acc(unmasked, first.to(device), L)
    ci_gate_copy = induction_copy_acc(masked_forward(model, banks, seq, g_lower, zeros), first.to(device), L)

    # R2: keep-only top-k curve
    ks = [k for k in [1, 2, 3, 5, 8, 16, 32] if k <= C]
    curve = {k: keep_only(order_c[:k]) for k in ks}
    # R2 metric: min k to reach 90% of unmasked induction
    thresh = 0.9 * unmasked_copy
    min_k = next((k for k in ks if curve[k] >= thresh), None)

    # R3 + R4: for the top components carrying induction, weight-mass per (layer, head) and x-layer span
    # component c's mass on (layer l, head h) = sum over that layer's q/k/v/o of the head's rows/cols.
    top = order_c[: (min_k or 8)]
    prev_l, prev_h = prev_head
    ind_l, ind_h = ind_head
    on_known, on_prev, on_ind, xlayer, peak = 0.0, 0.0, 0.0, 0.0, 0.0
    W = {n: b.materialized_weights() for n, b in banks.items()}  # [C,d_out,d_in]
    for c in top.tolist():
        # per-(layer,head) mass for this component
        mass = {}  # (layer,head)->norm
        layer_mass = {0: 0.0, 1: 0.0}
        for n, w in W.items():
            layer = 0 if "blocks.0" in n else 1
            proj = n.split(".")[-1]
            wc = w[c]  # [d_out, d_in]
            if proj == "o_proj":  # heads index the INPUT (columns)
                per = wc.pow(2).sum(dim=0)  # [d_in = n_heads*d_head]
                nh = per.shape[0] // d_head
                for h in range(nh):
                    m = per[h * d_head:(h + 1) * d_head].sum().item()
                    mass[(layer, h)] = mass.get((layer, h), 0.0) + m
                    layer_mass[layer] += m
            else:  # q/k/v: heads index the OUTPUT (rows)
                per = wc.pow(2).sum(dim=1)  # [d_out = n_heads*d_head]
                nh = per.shape[0] // d_head
                for h in range(nh):
                    m = per[h * d_head:(h + 1) * d_head].sum().item()
                    mass[(layer, h)] = mass.get((layer, h), 0.0) + m
                    layer_mass[layer] += m
        total = sum(mass.values()) + 1e-12
        peak += max(mass.values()) / total  # head concentration: 1/n_heads_total = uniform blob
        on_prev += mass.get((prev_l, prev_h), 0.0) / total
        on_ind += mass.get((ind_l, ind_h), 0.0) / total
        on_known += (mass.get((prev_l, prev_h), 0.0) + mass.get((ind_l, ind_h), 0.0)) / total
        lm = layer_mass
        span = min(lm[0], lm[1]) / (max(lm[0], lm[1]) + 1e-12)
        xlayer += 1.0 if span > 0.1 else 0.0
    n_top = len(top)
    clear_masks(banks)
    return {
        "unmasked_copy": unmasked_copy,
        "ci_gate_copy": ci_gate_copy,          # R1: sufficiency of CI-gated components
        "keepk_curve": curve,                  # R2
        "min_k_90pct": min_k,                  # R2: sparsity
        "top_on_known_heads": on_known / n_top,   # R3: mass on {prev,induction} heads
        "top_on_prev_head": on_prev / n_top,
        "top_on_ind_head": on_ind / n_top,
        "top_cross_layer_frac": xlayer / n_top,   # R4
        "top_peak_head_mass": peak / n_top,       # localization: uniform blob = 1/16 = 0.0625
        "n_top": n_top,
    }


# --- entry -----------------------------------------------------------------------------------------


def _run() -> None:
    from .attn_only_2l import (
        C_PER_MODULE_ATTN_ONLY_2L, INDUCTION_HEAD, PREV_TOKEN_HEAD,
        generate_pool, load_attn_only_2l_target,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = os.environ.get("SMOKE", "0") == "1"
    steps = int(os.environ.get("STEPS", "300" if smoke else "5000"))
    C = int(os.environ.get("C", "256"))
    R = int(os.environ.get("R", "64"))
    imp = float(os.environ.get("IMP", "1e-3"))
    simp = float(os.environ.get("SIMP", "0.0"))  # OFF for apples-to-apples: VPD/SPD use no simplicity
    life = float(os.environ.get("LIFE", "0.0"))
    seq_len = int(os.environ.get("SEQ", "128"))
    batch = int(os.environ.get("B", "16"))
    hidden = float(os.environ.get("HIDDEN", "0.0"))   # APD hidden-activation recon coeff
    subset_rt = os.environ.get("SUBSET", "0") == "1"  # VPD uniform-k-subset routing
    clip = float(os.environ.get("CLIP", "0.0"))       # VPD grad clip on component factors
    unorm = os.environ.get("UNORM", "0") == "1"       # APD per-step A unit-norm gauge

    model = load_attn_only_2l_target().to(device)
    d_head = model.blocks[0].d_head
    pool_path = "/tmp/attn2l_compare/pool.pt"
    if os.path.exists(pool_path):
        pool = torch.load(pool_path, weights_only=True)
        print(f"loaded pool {tuple(pool.shape)}", flush=True)
    else:
        print("generating pool ...", flush=True)
        pool = generate_pool(model, n_seqs=(256 if smoke else 4096), seq_len=seq_len, device=device)
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        torch.save(pool, pool_path)
    pool = pool[:, :seq_len]

    modules = list(C_PER_MODULE_ATTN_ONLY_2L.keys())
    # p_start/p_end + coeff_imp matched to the VPD run.Config defaults for apples-to-apples.
    cfg = ApdConfig(modules=modules, n_components=C, simplicity_impl="factored", factor_rank=R,
                    lowrank_forward=True, coeff_faith=float(os.environ.get("FAITH", "1e7")),
                    coeff_imp=imp, coeff_simplicity=simp, coeff_lifetime=life,
                    lifetime_ramp_frac=(0.6 if life > 0 else 0.0), p_start=2.0, p_end=0.4,
                    coeff_hidden=hidden, subset_routing=subset_rt, grad_clip=clip, unit_norm_A=unorm,
                    coeff_interaction=float(os.environ.get("INTER", "0.0")),
                    interaction_pairs=int(os.environ.get("PAIRS", "4")),
                    coeff_weight_l1=float(os.environ.get("L1", "0.0")),
                    use_wandb=os.environ.get("WANDB", "0") == "1",
                    wandb_project=os.environ.get("WANDB_PROJECT", "apd-basis"),
                    wandb_group=os.environ.get("WANDB_GROUP", "attn-only-2l"),
                    wandb_job_type="attn_only_2l",
                    wandb_run_name=os.environ.get("WANDB_NAME"))
    # VpdConfig carries CI dims + PPGD params + recon coeffs; matched to the VPD baseline run.
    ci_cfg = VpdConfig(C_per_module=C_PER_MODULE_ATTN_ONLY_2L, seq_len=seq_len,
                       ci_d_model=256, ci_n_blocks=4, ci_n_heads=8, ci_mlp_hidden=1024,
                       coeff_stoch=0.5, coeff_ppgd=0.5, ppgd_lr=0.01, ppgd_inner_steps=2)
    print(f"config: C={C} R={R} steps={steps} imp={imp} simp={simp} life={life} seq={seq_len} B={batch} "
          f"hidden={hidden} subset={subset_rt} clip={clip} unorm={unorm} "
          f"[PGD on, simplicity {'on' if simp>0 else 'OFF'}]", flush=True)

    out = decompose_lm(model, pool, cfg, ci_cfg, device, n_steps=steps, batch=batch, seq_len=seq_len,
                       warmup_steps=(100 if smoke else 400),
                       save_path=os.environ.get("SAVE", "/tmp/attn2l_compare/apd_lm.pt"))
    ev = pool[:batch, :seq_len].to(device)
    fe = faithfulness_eval(out["model"], out["banks"], out["ci"], ev, cfg, device)
    adv = adversarial_kl(out["model"], out["banks"], out["ci"], ev, cfg, device)
    print("\n=== APD-basis FAITHFULNESS (paper metric) ===", flush=True)
    print(f"kl_ci_masked={fe['kl_ci_masked']:.4f}  ce_recovered={fe['ce_recovered_pct']:.1f}%  "
          f"kl_adversarial={adv:.4f}  kl_unmasked(sanity)={fe['kl_unmasked']:.2e}  L0={fe['L0']:.1f}/{C}  "
          f"l1_ratio={fe['l1_ratio']:.2f}", flush=True)
    # induction-circuit rubric retired as a default metric (Rohan, 2026-07-09) — revisit later.
    if os.environ.get("INDUCTION", "0") == "1":
        rec = circuit_recovery(out["model"], out["banks"], out["ci"], pool, cfg, device,
                               PREV_TOKEN_HEAD, INDUCTION_HEAD, d_head, seq_len=seq_len)
        print("=== cross-layer localization (induction rubric, opt-in) ===", flush=True)
        print(f"top comps mass on known heads={rec['top_on_known_heads']:.2f} "
              f"(prev {rec['top_on_prev_head']:.2f}, induction {rec['top_on_ind_head']:.2f}); "
              f"peak head mass={rec['top_peak_head_mass']:.3f} (uniform=0.0625); "
              f"cross-layer frac={rec['top_cross_layer_frac']:.2f} (n_top={rec['n_top']}); "
              f"induction copy ci={rec['ci_gate_copy']:.3f} vs unmasked {rec['unmasked_copy']:.3f}", flush=True)
    print("APD_LM DONE", flush=True)


if __name__ == "__main__":
    _run()
