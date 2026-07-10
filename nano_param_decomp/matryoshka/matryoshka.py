"""Matryoshka Parameter Decomposition — a variant of `run.py`.

Idea: instead of treating every rank-1 subcomponent as its own unit of causal importance,
group the rank-1 atoms into a fixed number `G` of cross-layer *components*. The causal-
importance (CI) function emits one mask per *component* (G of them, not one per atom). A
learned membership matrix `M[A, G]` (A = total atoms across all modules) maps each component
mask onto the atoms that belong to it:

    atom_mask[a] = (sum_g M[a, g] * component_mask[g]) / (sum_g M[a, g] + eps)   # weighted average

`M = sigmoid(M_logits / tau)`: each (atom, component) gate is INDEPENDENT (not row-normalized), so
an atom can belong to several components at once (shareable subcomponents) or to none. The
temperature `tau` anneals high -> low so the gates sharpen toward binary {0, 1}. The aggregation is
a convex weighted average (divided by the per-atom membership mass), which keeps every atom mask in
[0, 1] regardless of how many components it joins. Membership penalties (membership_l1 /
component_size_l2 / membership_dist_entropy) provide the pressure toward sparse, small components.

What is unchanged from `run.py` (and imported, not re-implemented):
  - the rank-1 atoms themselves (`ComponentLinear`, per-module `V`/`U`) -> faithfulness identical
  - leaky-hard sigmoids, RoPE, CI transformer blocks, LR schedule, KL, eval CE helpers

What changes here:
  - `MatryoshkaCI`: CI transformer whose output head is `d_model -> G`
  - `ComponentAssignment`: the learned membership `M[A, G]` + component->atom mask mapping
  - component-level mask sampling (stochastic + persistent PGD), per-module delta kept separate
  - importance-minimality summed over G components; new assignment-entropy loss + tau anneal

Single-GPU:  python -m nano_param_decomp.matryoshka_pile_4L
Multi-GPU:   torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.matryoshka_pile_4L
"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, override

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .run import (
    CIBlock,
    ComponentLinear,
    _all_reduce_mean,
    _ce_next_token,
    _require,
    anneal_p,
    clear_wrapper_masks,
    cosine_lr,
    faithfulness_loss,
    importance_minimality_loss,
    induction_copy_acc,
    init_dist,
    install_components,
    kl_logits,
    lower_leaky,
    precompute_rope,
    sample_uniform_k_subset_routing,
    set_wrapper_masks,
    upper_leaky,
)

# Chunk width (in components) for the `max` atom-mask aggregation: bounds the transient
# [B, S, C_m, chunk] product so the full [B, S, C_m, G] tensor is never materialized.
_MAX_AGG_GROUP_CHUNK = 64

# --- Section A: Config ---


@dataclass
class Config:
    """Matryoshka PD config. Defaults track the pile-4L run; matryoshka-specific fields are
    grouped at the bottom and are the ones we expect to sweep."""

    C_per_module: dict[str, int]

    n_steps: int = 400_000
    batch_size: int = 64
    seq_len: int = 512
    seed: int = 0

    main_lr: float = 5e-5
    main_lr_final_frac: float = 0.1

    faithfulness_warmup_steps: int = 400
    faithfulness_warmup_lr: float = 1e-3

    # Loss coefficients
    coeff_faith: float = 1e7
    coeff_imp: float = 2e-4  # importance-minimality on the G component gates
    coeff_imp_atoms: float = 0.0  # importance-minimality ALSO on per-atom effective importance
    coeff_stoch: float = 0.5
    coeff_ppgd: float = 0.5

    p_start: float = 2.0
    p_end: float = 0.4
    imp_eps: float = 1e-12
    imp_beta: float = 0.5

    leaky_alpha: float = 0.01

    # CI transformer (global_shared_transformer); output head is d_model -> n_components
    ci_d_model: int = 2048
    ci_n_blocks: int = 8
    ci_n_heads: int = 16
    ci_mlp_hidden: int = 8192
    ci_rope_base: float = 10000.0

    # Persistent PGD
    ppgd_lr: float = 0.01
    ppgd_lr_final_frac: float = 1.0
    ppgd_warmup_pct: float = 0.025
    ppgd_beta1: float = 0.5
    ppgd_beta2: float = 0.99
    ppgd_eps: float = 1e-8
    ppgd_inner_steps: int = 2

    grad_clip_components: float = 0.01

    eval_freq: int = 1000
    slow_eval_freq: int = 10000
    eval_batch_size: int = 128
    ci_alive_threshold: float = 0.0
    rounding_threshold: float = 0.0

    log_every: int = 200
    use_wandb: bool = False
    wandb_project: str = "matryoshka-pd"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "matryoshka"
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: str = ""
    save_path: str | None = None  # if set, rank0 saves the decomposition (V/U + CI net + M) at the end

    # --- Matryoshka-specific (the knobs to sweep) ---
    n_components: int = 1024  # G: total cross-layer components
    tau_start: float = 2.0  # membership sigmoid temperature at step 0 (soft gates)
    tau_end: float = 0.5  # temperature floor: anneal stops here and HOLDS. Going lower (e.g. 0.05)
                          # gives the membership penalty a high-gain (1/tau) regime that, over enough
                          # steps, collapses every component to 1 atom and wrecks reconstruction.
    tau_anneal_frac: float = 0.6  # fraction of n_steps to anneal tau over; hold at tau_end after
    coeff_assign_entropy: float = 0.01  # per-gate binarization pressure
    coeff_membership: float = 0.001  # L1 on membership mass (sparse / small components)
    coeff_membership_entropy: float = 0.0  # entropy of each atom's normalized membership row
                                           # (concentrate each atom onto few components; scale-free)
    coeff_comp_size: float = 0.0  # quadratic per-component size tax (hits biggest components hardest)
    m_logits_init_std: float = 1.0  # init scale of the membership logits
    m_logits_init_bias: float = 0.0  # 0 = warm start (~half gates on), let L1 prune downward
    membership_type: str = "sigmoid"  # "sigmoid" (independent shareable gates) | "softmax" (one row
                                      # per atom sums to 1 -> atom commits to ~one component)
    aggregation: str = "mean"  # how component gates -> atom gate. "mean": membership-weighted average
                               # (dilutes a shared atom by 1/degree). "max": atom on if ANY of its
                               # components is on (no dilution; correct for shareable subcomponents)


# --- Section B: Learned component<->atom assignment ---


def _layer_of(module_path: str) -> int:
    """Layer index = first integer token in the dotted module path. Handles both `h.0.attn.q_proj`
    (SS2L) and `gpt_neox.layers.0.attention.query_key_value` (Pythia/GPTNeoX)."""
    for tok in module_path.split("."):
        if tok.isdigit():
            return int(tok)
    return 0  # no integer token -> single-layer model (e.g. toy "W"); LM paths always have one


class ComponentAssignment(nn.Module):
    """Membership matrix `M[A, G]` mapping the A rank-1 atoms onto G components.

    Atoms are laid out in `module_order` (alphabetical, matching the CI transformer), each
    module contributing `C_per_module[name]` consecutive rows. `M = sigmoid(M_logits / tau)`
    elementwise, so each gate is independent: an atom can belong to several components, one, or
    none (shareable subcomponents). `tau -> 0` sharpens every gate toward binary {0, 1}.
    """

    def __init__(self, c_per_module: dict[str, int], module_order: list[str], cfg: Config) -> None:
        super().__init__()
        self.module_order = module_order
        self.c_splits: list[int] = [c_per_module[n] for n in module_order]
        offsets: list[int] = []
        acc = 0
        for c in self.c_splits:
            offsets.append(acc)
            acc += c
        self.offsets = offsets
        self.n_atoms = acc
        self.n_components = cfg.n_components
        self.membership_type = cfg.membership_type
        self.aggregation = cfg.aggregation
        self.M_logits = nn.Parameter(
            torch.randn(self.n_atoms, cfg.n_components) * cfg.m_logits_init_std
            + cfg.m_logits_init_bias
        )
        layers = [
            _layer_of(name) for name, c in zip(module_order, self.c_splits) for _ in range(c)
        ]
        self.register_buffer("atom_layer", torch.tensor(layers, dtype=torch.long), persistent=False)
        self.n_layers = max(layers) + 1

    def membership(self, tau: float) -> Tensor:
        # sigmoid: each (atom, component) gate is independent, so a row is NOT normalized -> an atom
        # can belong to several components at once (shareable subcomponents), or to none.
        # softmax: each atom's row sums to 1 -> the atom commits to ~one component (no sharing), which
        # forces atoms to distribute across components instead of piling into one mega-component.
        # tau sharpens either toward its committed form as it shrinks.
        if self.membership_type == "softmax":
            return torch.softmax(self.M_logits / tau, dim=1)  # [A, G], rows sum to 1
        return torch.sigmoid(self.M_logits / tau)  # [A, G], entries in [0, 1]

    def atom_masks(self, component_mask: Tensor, tau: float) -> dict[str, Tensor]:
        """Map a per-component mask [B, S, G] to per-module atom masks {name: [B, S, C_m]}.

        `mean` aggregation: atom_mask[..., c] = (sum_g M[c,g]*mask[...,g]) / (sum_g M[c,g] + eps), a
        membership-weighted AVERAGE. Bounded in [0,1] but DILUTES a shared atom by ~1/degree, so an
        atom in many components can't reach 1 when a single owning component fires (this throttled
        reconstruction badly on the toy).

        `max` aggregation: atom_mask[..., c] = max_g M[c,g]*mask[...,g] — the atom is on if ANY of its
        components is on, with no 1/degree dilution. Computed by chunking over components and keeping a
        running max so the full [B,S,C,G] product is never materialized (otherwise prohibitive on LMs).
        """
        M = self.membership(tau)
        out: dict[str, Tensor] = {}
        for name, off, c in zip(self.module_order, self.offsets, self.c_splits, strict=True):
            M_m = M[off : off + c]  # [C_m, G]
            if self.aggregation == "max":
                # Find the argmax component per atom in no_grad (chunked, intermediates discarded), then
                # gather the winning M*mask WITH grad. Backward touches only the argmax element, so memory
                # is O(B*S*C_m) not O(B*S*C_m*G) -- materializing the latter OOMs even chunked, because
                # autograd would save every chunk for the backward pass.
                B, S = component_mask.shape[:2]
                with torch.no_grad():
                    best_val: Tensor | None = None
                    best_idx: Tensor | None = None
                    for g0 in range(0, self.n_components, _MAX_AGG_GROUP_CHUNK):
                        g1 = g0 + _MAX_AGG_GROUP_CHUNK
                        v, i = (component_mask[..., g0:g1].unsqueeze(-2) * M_m[:, g0:g1]).max(dim=-1)
                        i = i + g0  # [B,S,C_m]
                        if best_val is None:
                            best_val, best_idx = v, i
                        else:
                            take = v > best_val
                            best_val = torch.where(take, v, best_val)
                            best_idx = torch.where(take, i, best_idx)
                assert best_idx is not None
                cm_sel = torch.gather(component_mask, -1, best_idx)  # [B,S,C_m] from [B,S,G]
                m_sel = torch.gather(M_m, 1, best_idx.permute(2, 0, 1).reshape(c, B * S))  # [C_m,B*S]
                out[name] = cm_sel * m_sel.reshape(c, B, S).permute(1, 2, 0)
            else:
                num = torch.einsum("bsg,cg->bsc", component_mask, M_m)
                den = M_m.sum(dim=1) + 1e-6  # [C_m]
                out[name] = num / den
        return out

    def entropy(self, tau: float) -> Tensor:
        """Per-gate binarization pressure: mean over all gates of -M*log(M). Each term is 0 at
        m in {0,1} and positive between, so minimizing it pushes every gate to commit to 0 or 1.
        Scale-invariant: a per-element mean, so independent of atom count A and component count G."""
        M = self.membership(tau)
        return -(M * (M + 1e-12).log()).mean()

    def membership_l1(self, tau: float) -> Tensor:
        """Mean membership gate value over all (atom, component) pairs, in [0,1]. Penalizing it
        keeps memberships sparse (fewer/weaker gates -> few components per atom, small components).
        Scale-invariant: a per-element mean, so independent of A and G (was previously the per-atom
        row sum, which grew with G; normalized so the coefficient transfers across model sizes)."""
        return self.membership(tau).mean()

    def component_size_l2(self, tau: float) -> Tensor:
        """Per-component (column) size tax: mean over components of the SQUARED atom-fraction
        (atoms_in_component_g / A)^2, in [0,1]. Quadratic -> taxes the BIGGEST components hardest,
        unlike the linear total-mass tax (membership_l1) which is size-blind across columns.
        Scale-invariant: uses the *fraction* of atoms (size / A), so independent of atom count A
        (was previously raw size^2, which grew with A^2; normalized so the coefficient transfers)."""
        frac = self.membership(tau).sum(dim=0) / self.n_atoms  # [G], fraction of atoms per comp
        return (frac**2).mean()

    def membership_dist_entropy(self, tau: float) -> Tensor:
        """Mean over atoms of the entropy of each atom's NORMALIZED membership distribution
        p[a,g] = M[a,g] / sum_g M[a,g], divided by log(G) so it lands in [0,1]. Penalizes the
        *shape* of the row -> low entropy concentrates each atom onto a few components.
        Scale-invariant in G via the log(G) normalization. Atoms with ~zero mass contribute ~0."""
        M = self.membership(tau)
        p = M / (M.sum(dim=1, keepdim=True) + 1e-12)
        ent = -(p * (p + 1e-12).log()).sum(dim=1).mean()
        return ent / math.log(self.n_components)

    def hardness(self, tau: float) -> Tensor:
        """Mean over gates of max(M, 1-M); 1.0 means every gate is a clean 0 or 1."""
        M = self.membership(tau)
        return torch.maximum(M, 1.0 - M).mean()

    def cross_layer_stats(self, tau: float) -> dict[str, float]:
        """Threshold the (shareable) membership at 0.5 and measure component structure: layer
        span, size (atoms/component), and how much atoms are shared across components."""
        member = (self.membership(tau) > 0.5).float()  # [A, G] binary, sharing allowed
        comp_size = member.sum(dim=0)  # [G] atoms per component
        used = comp_size > 0
        onehot_layer = F.one_hot(self.atom_layer, self.n_layers).float()  # [A, L]
        comp_layer = member.t() @ onehot_layer  # [G, L]
        layers_per = (comp_layer > 0).sum(dim=1).float()[used]
        atom_degree = member.sum(dim=1)  # [A] components per atom
        if used.sum() == 0:
            return {k: 0.0 for k in (
                "n_used", "mean_layers", "crosslayer_frac", "mean_comp_size",
                "max_comp_size", "mean_atom_degree", "shared_atom_frac",
            )}
        return {
            "n_used": float(used.sum().item()),
            "mean_layers": layers_per.mean().item(),
            "crosslayer_frac": (layers_per >= 2).float().mean().item(),
            "mean_comp_size": comp_size[used].mean().item(),
            "max_comp_size": comp_size.max().item(),
            "mean_atom_degree": atom_degree.mean().item(),
            "shared_atom_frac": (atom_degree > 1).float().mean().item(),
        }


def tau_at(step: int, cfg: Config) -> float:
    anneal_steps = max(1, int(cfg.tau_anneal_frac * cfg.n_steps))
    if step >= anneal_steps:
        return cfg.tau_end
    return cosine_lr(step, anneal_steps, cfg.tau_start, cfg.tau_end / cfg.tau_start)


def _all_reduce_grads(params: list[nn.Parameter], world_size: int) -> None:
    """Average each param's grad across ranks (manual DDP). Correct regardless of which forward
    produced the grad, so it covers M_logits / V / U whose grads come from the recon forwards."""
    if world_size <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# --- Section C: CI transformer with a G-wide output head ---


class MatryoshkaCI(nn.Module):
    """Causal-importance transformer that emits one importance per *component*.

    Identical to `run.CITransformer` except the output head maps `d_model -> G` (not -> total_C),
    and the returned CI tensors are component-level [B, S, G] rather than per-module dicts.
    """

    def __init__(self, d_in_per_module: dict[str, int], cfg: Config) -> None:
        super().__init__()
        self.module_order = sorted(d_in_per_module.keys())
        self.cfg = cfg
        total_in = sum(d_in_per_module.values())
        self.proj_in = nn.Linear(total_in, cfg.ci_d_model)
        self.blocks = nn.ModuleList([CIBlock(cfg) for _ in range(cfg.ci_n_blocks)])
        self.proj_out = nn.Linear(cfg.ci_d_model, cfg.n_components)
        head_dim = cfg.ci_d_model // cfg.ci_n_heads
        cos, sin = precompute_rope(cfg.seq_len, head_dim, cfg.ci_rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @override
    def forward(self, acts: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        normed = [F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order]
        x = self.proj_in(torch.cat(normed, dim=-1))
        S = x.shape[1]
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.proj_out(x)  # [B, S, G]
        alpha = self.cfg.leaky_alpha
        return lower_leaky(logits, alpha), upper_leaky(logits, alpha), logits


# --- Section D: mask sampling (component level) ---


def sample_component_masks(
    ci_lower: Tensor, module_names: list[str]
) -> tuple[Tensor, dict[str, Tensor]]:
    """component_mask = ci + (1 - ci) * U(0,1) over the G components; per-module delta ~ U(0,1)."""
    u = torch.rand_like(ci_lower)
    component_mask = ci_lower + (1 - ci_lower) * u
    B, S, _ = ci_lower.shape
    delta_masks = {
        n: torch.rand(B, S, device=ci_lower.device, dtype=ci_lower.dtype) for n in module_names
    }
    return component_mask, delta_masks


# --- Section E: Persistent PGD (component-level source + per-module delta) ---


class PersistentPGD:
    """Adversarial sources persisted across steps. Scope `per_batch_per_position`.

    Unlike the baseline (one source per module of width C_m + 1), the matryoshka sources are:
      - one shared component source `comp` of shape [local_B, S, G] (the mask multiplier per
        component), since masking is component-level;
      - one per-module delta source of shape [local_B, S] (the Δ-component scalar, kept per
        module exactly as in the baseline).
    Adam state (m, v) is kept alongside every source; no cross-rank sync.
    """

    def __init__(
        self,
        module_names: list[str],
        n_components: int,
        local_B: int,
        seq_len: int,
        device: torch.device,
        cfg: Config,
    ) -> None:
        self.cfg = cfg
        self.module_names = module_names
        comp = torch.rand(local_B, seq_len, n_components, device=device).requires_grad_(True)
        self.sources: dict[str, Tensor] = {"comp": comp}
        for n in module_names:
            self.sources[f"delta::{n}"] = torch.rand(
                local_B, seq_len, device=device
            ).requires_grad_(True)
        self.m = {k: torch.zeros_like(v) for k, v in self.sources.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.sources.items()}
        self.t = 0

    def _masks_from_sources(
        self, ci_lower: Tensor, assign: ComponentAssignment, tau: float
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        comp = self.sources["comp"]
        component_mask = ci_lower + (1 - ci_lower) * comp
        masks = assign.atom_masks(component_mask, tau)
        delta_masks = {n: self.sources[f"delta::{n}"] for n in self.module_names}
        return masks, delta_masks

    def recon_loss(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        assign: ComponentAssignment,
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: Tensor,
        tau: float,
    ) -> Tensor:
        masks, delta_masks = self._masks_from_sources(ci_lower, assign, tau)
        set_wrapper_masks(wrappers, masks, delta_masks, routing=None)
        try:
            pred = target_model(input_ids)
        finally:
            clear_wrapper_masks(wrappers)
        return kl_logits(pred, target_logits)

    def warmup(
        self,
        target_model: nn.Module,
        wrappers: dict[str, ComponentLinear],
        assign: ComponentAssignment,
        input_ids: Tensor,
        target_logits: Tensor,
        ci_lower: Tensor,
        tau: float,
        lr: float,
    ) -> None:
        for _ in range(self.cfg.ppgd_inner_steps):
            loss = self.recon_loss(
                target_model, wrappers, assign, input_ids, target_logits, ci_lower, tau
            )
            grads = torch.autograd.grad(loss, list(self.sources.values()), retain_graph=False)
            self._adam_step(dict(zip(self.sources, grads, strict=True)), lr)

    def external_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self._adam_step(grads, lr)

    def _adam_step(self, grads: dict[str, Tensor], lr: float) -> None:
        self.t += 1
        bc1 = 1 - self.cfg.ppgd_beta1**self.t
        bc2 = 1 - self.cfg.ppgd_beta2**self.t
        with torch.no_grad():
            for name, src in self.sources.items():
                g = grads[name]
                m, v = self.m[name], self.v[name]
                m.mul_(self.cfg.ppgd_beta1).add_(g, alpha=1 - self.cfg.ppgd_beta1)
                v.mul_(self.cfg.ppgd_beta2).addcmul_(g, g, value=1 - self.cfg.ppgd_beta2)
                src.add_(lr * (m / bc1) / ((v / bc2).sqrt() + self.cfg.ppgd_eps))
                src.clamp_(0.0, 1.0)


# --- Section F: stochastic recon + container ---


def stochastic_recon_loss(
    target_model: nn.Module,
    wrappers: dict[str, ComponentLinear],
    assign: ComponentAssignment,
    input_ids: Tensor,
    target_logits: Tensor,
    ci_lower: Tensor,
    tau: float,
) -> Tensor:
    B, S = input_ids.shape
    component_mask, delta_masks = sample_component_masks(ci_lower, list(wrappers))
    masks = assign.atom_masks(component_mask, tau)
    routing = sample_uniform_k_subset_routing(list(wrappers), (B, S), input_ids.device)
    set_wrapper_masks(wrappers, masks, delta_masks, routing)
    try:
        pred = target_model(input_ids)
    finally:
        clear_wrapper_masks(wrappers)
    return kl_logits(pred, target_logits)


class MatryoshkaModule(nn.Module):
    """Container so DDP tracks target component params, CI transformer params, and M_logits.

    Mirrors `run.SPDModule`: the wrapped forward runs the target (caching pre-weight acts) then
    the CI transformer; the masked recon forwards go through `self.target` directly."""

    def __init__(
        self,
        target: nn.Module,
        ci_fn: MatryoshkaCI,
        assign: ComponentAssignment,
        wrappers: dict[str, ComponentLinear],
    ) -> None:
        super().__init__()
        self.target = target
        self.ci_fn = ci_fn
        self.assign = assign
        self._wrappers = wrappers

    @override
    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        clear_wrapper_masks(self._wrappers)
        target_logits = self.target(input_ids)
        acts = {n: _require(w.last_input) for n, w in self._wrappers.items()}
        ci_lower, ci_upper, _pre = self.ci_fn(acts)
        return target_logits, ci_lower, ci_upper


# --- Section G: Training loop ---


def _best_path(save_path: str) -> str:
    root, ext = os.path.splitext(save_path)
    return f"{root}.best{ext}"


def _save_decomp(
    path: str,
    wrappers: dict[str, ComponentLinear],
    ci_fn: nn.Module,
    assign: "ComponentAssignment",
    cfg: Config,
    metrics: dict[str, float],
) -> None:
    import dataclasses

    torch.save(
        {
            "method": "matryoshka",
            "wrappers": {
                n: {"V": w.V.detach().cpu(), "U": w.U.detach().cpu()} for n, w in wrappers.items()
            },
            "ci_fn": {k: v.detach().cpu() for k, v in ci_fn.state_dict().items()},
            "M_logits": assign.M_logits.detach().cpu(),
            "C_per_module": dict(cfg.C_per_module),
            "n_components": cfg.n_components,
            "tau_end": cfg.tau_end,
            "cfg": dataclasses.asdict(cfg),
            "final_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        },
        path,
    )


def decompose(
    target_model: nn.Module,
    cfg: Config,
    train_loader: Iterator[Tensor],
    eval_loader: Iterator[Tensor],
) -> dict[str, Any]:
    rank, world_size, local_rank, device = init_dist()
    assert cfg.batch_size % world_size == 0
    local_B = cfg.batch_size // world_size

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    def _log(msg: str) -> None:
        if rank == 0:
            print(f"[rank0] {msg}", flush=True)

    target_model.eval()
    wrappers = install_components(target_model, cfg.C_per_module)
    module_order = sorted(wrappers.keys())
    _log(f"installed {len(wrappers)} components ({sum(cfg.C_per_module.values())} atoms)")

    d_in_per_module = {name: int(w.W_target.shape[1]) for name, w in wrappers.items()}
    ci_fn = MatryoshkaCI(d_in_per_module, cfg)
    assign = ComponentAssignment(cfg.C_per_module, module_order, cfg)
    _log(
        f"CI head -> {cfg.n_components} components | "
        f"M_logits {tuple(assign.M_logits.shape)} "
        f"({assign.M_logits.numel():,} params)"
    )
    target_model.to(device)
    ci_fn.to(device)
    assign.to(device)

    # Faithfulness warmup (component params only; identical across ranks -> no sync).
    component_params = [p for w in wrappers.values() for p in (w.V, w.U)]
    warmup_opt = torch.optim.AdamW(component_params, lr=cfg.faithfulness_warmup_lr, weight_decay=0.0)
    _log(f"faithfulness warmup ({cfg.faithfulness_warmup_steps} steps)")
    for _ in range(cfg.faithfulness_warmup_steps):
        warmup_opt.zero_grad()
        faithfulness_loss(wrappers).backward()
        warmup_opt.step()

    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)

    # No DDP wrapper: M_logits and V/U get gradients from the recon forwards that bypass the
    # module's forward, which DDP would not all-reduce. We average grads manually after backward.
    module = MatryoshkaModule(target_model, ci_fn, assign, wrappers).to(device)
    wrapped: nn.Module = module

    ppgd = PersistentPGD(module_order, cfg.n_components, local_B, cfg.seq_len, device, cfg)
    main_params = component_params + list(ci_fn.parameters()) + [assign.M_logits]
    opt = torch.optim.AdamW(main_params, lr=cfg.main_lr, weight_decay=0.0)
    _log("optimizer ready; starting main loop")

    if rank == 0 and cfg.use_wandb:
        import dataclasses

        import wandb  # type: ignore[import-untyped]

        # A wandb-sweep agent inits the run before calling decompose; don't double-init -- just
        # record the resolved config onto the existing run. Standalone runs init here.
        if wandb.run is None:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                job_type=cfg.wandb_job_type,
                name=cfg.wandb_run_name,
                tags=list(cfg.wandb_tags),
                notes=cfg.wandb_notes,
                config=dataclasses.asdict(cfg),
            )
        else:
            wandb.config.update(dataclasses.asdict(cfg), allow_val_change=True)

    best_kl = float("inf")
    for step in range(cfg.n_steps):
        main_lr = cosine_lr(step, cfg.n_steps, cfg.main_lr, cfg.main_lr_final_frac)
        ppgd_lr = cosine_lr(
            step, cfg.n_steps, cfg.ppgd_lr, cfg.ppgd_lr_final_frac, cfg.ppgd_warmup_pct
        )
        tau = tau_at(step, cfg)
        p = anneal_p(step, cfg.n_steps, cfg.p_start, cfg.p_end)
        for g in opt.param_groups:
            g["lr"] = main_lr

        input_ids = next(train_loader).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            target_logits, ci_lower, ci_upper = wrapped(input_ids)

            ppgd.warmup(
                target_model, wrappers, assign, input_ids, target_logits, ci_lower, tau, ppgd_lr
            )

            loss_faith = faithfulness_loss(wrappers)
            loss_imp = importance_minimality_loss(
                {"components": ci_upper}, p, cfg.imp_eps, cfg.imp_beta, world_size
            )
            # Per-atom effective importance = weighted-average of component upper-CI through M
            # (same [B,S,C_m] per module as baseline VPD), so atoms get direct usage pressure too.
            atom_upper = assign.atom_masks(ci_upper, tau)
            loss_imp_atoms = importance_minimality_loss(
                atom_upper, p, cfg.imp_eps, cfg.imp_beta, world_size
            )
            loss_stoch = stochastic_recon_loss(
                target_model, wrappers, assign, input_ids, target_logits, ci_lower, tau
            )
            loss_ppgd = ppgd.recon_loss(
                target_model, wrappers, assign, input_ids, target_logits, ci_lower, tau
            )
            loss_assign = assign.entropy(tau)
            loss_membership = assign.membership_l1(tau)
            loss_memb_entropy = assign.membership_dist_entropy(tau)
            loss_comp_size = assign.component_size_l2(tau)

        total = (
            cfg.coeff_faith * loss_faith
            + cfg.coeff_imp * loss_imp
            + cfg.coeff_imp_atoms * loss_imp_atoms
            + cfg.coeff_stoch * loss_stoch
            + cfg.coeff_ppgd * loss_ppgd
            + cfg.coeff_assign_entropy * loss_assign
            + cfg.coeff_membership * loss_membership
            + cfg.coeff_membership_entropy * loss_memb_entropy
            + cfg.coeff_comp_size * loss_comp_size
        )

        ppgd_grads = torch.autograd.grad(loss_ppgd, list(ppgd.sources.values()), retain_graph=True)
        ppgd_grads_dict = dict(zip(ppgd.sources, ppgd_grads, strict=True))

        opt.zero_grad()
        total.backward()
        _all_reduce_grads(main_params, world_size)  # data-parallel sync across ranks
        torch.nn.utils.clip_grad_norm_(component_params, cfg.grad_clip_components)
        opt.step()
        ppgd.external_step(ppgd_grads_dict, ppgd_lr)  # PPGD sources stay per-rank (adversarial)

        if step % cfg.eval_freq == 0:
            eval_batch = next(eval_loader).to(device)
            metrics = run_eval(target_model, ci_fn, assign, wrappers, cfg, world_size, eval_batch, tau, p)
            if rank == 0 and cfg.use_wandb:
                import wandb

                wandb.log(metrics, step=step)
            if rank == 0 and cfg.save_path is not None:
                kl = metrics.get("eval/kl_ci_masked", float("inf"))
                if kl < best_kl:
                    best_kl = kl
                    _save_decomp(_best_path(cfg.save_path), wrappers, ci_fn, assign, cfg, metrics)
                    _log(f"new best kl_ci={kl:.4g} @ step {step} -> {_best_path(cfg.save_path)}")

        if rank == 0 and step % cfg.log_every == 0:
            xl = assign.cross_layer_stats(tau)
            metrics = {
                "loss/faith": loss_faith.detach().item(),
                "loss/imp": loss_imp.detach().item(),
                "loss/imp_atoms": loss_imp_atoms.detach().item(),
                "loss/stoch": loss_stoch.detach().item(),
                "loss/ppgd": loss_ppgd.detach().item(),
                "loss/assign_entropy": loss_assign.detach().item(),
                "loss/membership": loss_membership.detach().item(),
                "loss/membership_entropy": loss_memb_entropy.detach().item(),
                "loss/comp_size": loss_comp_size.detach().item(),
                "assign/tau": tau,
                "assign/hardness": assign.hardness(tau).detach().item(),
                "comp/mean_size": xl["mean_comp_size"],
                "comp/max_size": xl["max_comp_size"],
                "comp/crosslayer_frac": xl["crosslayer_frac"],
                "comp/shared_atom_frac": xl["shared_atom_frac"],
                "comp/used": xl["n_used"],
                "lr/main": main_lr,
                "step": step,
            }
            if cfg.use_wandb:
                import wandb

                wandb.log(metrics, step=step)
            else:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)

    eval_batch = next(eval_loader).to(device)
    final_tau = tau_at(cfg.n_steps - 1, cfg)
    final_p = anneal_p(cfg.n_steps - 1, cfg.n_steps, cfg.p_start, cfg.p_end)
    final_metrics = run_eval(
        target_model, ci_fn, assign, wrappers, cfg, world_size, eval_batch, final_tau, final_p
    )
    if rank == 0 and cfg.save_path is not None:
        _save_decomp(cfg.save_path, wrappers, ci_fn, assign, cfg, final_metrics)
        _log(f"saved final decomposition -> {cfg.save_path} (best-by-kl at {_best_path(cfg.save_path)})")
    if world_size > 1:
        dist.destroy_process_group()
    return final_metrics


# --- Section H: Eval (slim) ---


def run_eval(
    target_model: nn.Module,
    ci_fn: MatryoshkaCI,
    assign: ComponentAssignment,
    wrappers: dict[str, ComponentLinear],
    cfg: Config,
    world_size: int,
    eval_batch: Tensor,
    tau: float,
    p: float,
) -> dict[str, Any]:
    with torch.no_grad():
        clear_wrapper_masks(wrappers)
        target_logits = target_model(eval_batch)
        acts = {n: _require(w.last_input) for n, w in wrappers.items()}
        ci_lower, _ci_upper, _pre = ci_fn(acts)

        # CI-masked forward: component_mask = ci, mapped to atoms.
        masks = assign.atom_masks(ci_lower, tau)
        B, S = eval_batch.shape
        zeros_delta = {n: torch.zeros(B, S, device=eval_batch.device) for n in wrappers}
        set_wrapper_masks(wrappers, masks, zeros_delta, routing=None)
        try:
            ci_logits = target_model(eval_batch)
        finally:
            clear_wrapper_masks(wrappers)

        # atoms/token active (comparable to baseline eval/l0/0.0_total)
        l0_atoms = sum((m > cfg.ci_alive_threshold).float().sum(-1).mean() for m in masks.values())
        l0_components = (ci_lower > cfg.ci_alive_threshold).float().sum(-1).mean()

        # stochastic-mask reconstruction (component-level sampling), comparable to baseline stoch
        component_mask, delta_masks = sample_component_masks(ci_lower, list(wrappers))
        stoch_masks = assign.atom_masks(component_mask, tau)
        set_wrapper_masks(wrappers, stoch_masks, delta_masks, routing=None)
        try:
            stoch_logits = target_model(eval_batch)
        finally:
            clear_wrapper_masks(wrappers)

        # Induction: does masking to the important components preserve the copy behavior? Uses an
        # IN-DISTRIBUTION repeat (first half of eval-batch sequences) -- the CI fn doesn't generalize
        # to OOD random tokens, which spuriously read as ~0.
        L = min(64, cfg.seq_len // 2)
        ind_first = eval_batch[:32, :L]
        ind_seq = torch.cat([ind_first, ind_first], dim=1)
        clear_wrapper_masks(wrappers)
        ind_unmasked = target_model(ind_seq)
        ind_acts = {n: _require(w.last_input) for n, w in wrappers.items()}
        ind_ci, _u, _pp = ci_fn(ind_acts)
        ind_masks = assign.atom_masks(ind_ci, tau)
        ind_B, ind_S = ind_seq.shape
        set_wrapper_masks(
            wrappers, ind_masks, {n: torch.zeros(ind_B, ind_S, device=eval_batch.device) for n in wrappers}, None
        )
        try:
            ind_ci_logits = target_model(ind_seq)
        finally:
            clear_wrapper_masks(wrappers)
        induction_unmasked = induction_copy_acc(ind_unmasked, ind_first, L)
        induction_ci_masked = induction_copy_acc(ind_ci_logits, ind_first, L)

        target_ce = _ce_next_token(target_logits, eval_batch)
        ci_ce = _ce_next_token(ci_logits, eval_batch)
        xl = assign.cross_layer_stats(tau)
        out: dict[str, Any] = {
            "eval/kl_ci_masked": _all_reduce_mean(
                kl_logits(ci_logits, target_logits).clone(), world_size
            ).item(),
            "eval/kl_stoch_masked": _all_reduce_mean(
                kl_logits(stoch_logits, target_logits).clone(), world_size
            ).item(),
            "eval/ce_difference_ci_masked": _all_reduce_mean(
                torch.tensor(ci_ce - target_ce, device=eval_batch.device), world_size
            ).item(),
            "eval/l0_atoms": _all_reduce_mean(l0_atoms.clone(), world_size).item(),
            "eval/l0_components": _all_reduce_mean(l0_components.clone(), world_size).item(),
            "eval/faithfulness": _all_reduce_mean(
                faithfulness_loss(wrappers).clone(), world_size
            ).item(),
            "eval/induction/copy_unmasked": induction_unmasked,
            "eval/induction/copy_ci_masked": induction_ci_masked,
            "eval/assign_hardness": assign.hardness(tau).item(),
            "eval/comp_used": xl["n_used"],
            "eval/comp_mean_layers": xl["mean_layers"],
            "eval/comp_crosslayer_frac": xl["crosslayer_frac"],
            "eval/comp_mean_size": xl["mean_comp_size"],
            "eval/comp_max_size": xl["max_comp_size"],
            "eval/shared_atom_frac": xl["shared_atom_frac"],
            "eval/mean_atom_degree": xl["mean_atom_degree"],
        }
    return out
