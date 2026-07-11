"""APD-basis decomposition with a trained (S/VPD-style) mask.

The research idea (Rohan): take APD's *basis* -- a set of `C` whole-network **parameter
components**, each a full-rank copy of every decomposed weight matrix, summing to the target
weights -- but replace APD's gradient *attribution* + batch-top-k with S/VPD's machinery: a learned
causal-importance (CI) net that emits one gate per component, trained under stochastic and
adversarial (PGD) ablation plus an L_p importance-minimality penalty. Plus APD's "simplicity"
penalty on active components, since with full-rank components nothing else pushes a component toward
low rank.

The one-line contrast with the existing files here:

  - VPD (`run.py`):        each weight matrix -> `C` **rank-1 atoms** (V@U); the CI net emits one
                           gate **per atom per matrix** (gates are independent across matrices).
  - MPD (`matryoshka.py`): rank-1 atoms grouped into components via a learned membership table.
  - THIS (`apd_mask.py`):  each component is a **full-rank matrix per module**, and the CI net emits
                           **one gate per component, shared across ALL modules** -- i.e. the mask is
                           trained on the whole network, not on individual subcomponents. This is
                           APD's parameterization (paper Sec 2.2) with attribution swapped for a
                           trained mask.

Simplicity backends (`simplicity_impl`), which is the thing we profile:
  - "svd":      components are free matrices `P [C, d_out, d_in]`; simplicity = Schatten-p of the
                singular values (`torch.linalg.svdvals` every step). Exact but needs an SVD.
  - "factored": each component is reparameterized `P = A @ B` (A [C,d_out,r], B [C,r,d_in], r =
                min(d_out,d_in) so still full-rank-capable); simplicity = 1/2(||A||_F^2 + ||B||_F^2),
                the variational form of the nuclear norm -- SVD-free. Only equals p=1 (nuclear).
  Both materialize the same `[C, d_out, d_in]` effective weights into a per-step cache, so the masked
  forward is identical between them; only the simplicity term (and a per-step A@B matmul) differ.

Deliberately toy-scoped (no sequence dim assumed in the masked forward's memory layout, MSE
reconstruction, MLP CI net). The masked forward materializes a per-datapoint effective weight
`W_eff = sum_c g_c P_c` per module, cheap for the toy targets but needing a smarter kernel for an LM.

Run:
    CUDA_VISIBLE_DEVICES="" python -m nano_param_decomp.apd_mask            # TMS (default)
    MODEL=resid python -m nano_param_decomp.apd_mask                        # cross-layer resid-MLP
    IMPL=factored MODEL=resid python -m nano_param_decomp.apd_mask          # SVD-free simplicity
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .run import importance_minimality_loss, lower_leaky, upper_leaky
from .toy_decompose import mmcs


# --- Config ----------------------------------------------------------------------------------------


@dataclass
class ApdConfig:
    # Which nn.Linear submodules to decompose. Every module shares the SAME C global components
    # (a component exists in every module); the gate is one scalar per component, network-wide.
    modules: list[str]
    n_components: int  # C -- total whole-network parameter components
    n_steps: int = 10000
    warmup_steps: int = 500
    batch: int = 2048
    lr: float = 3e-3
    warmup_lr: float = 1e-3
    ci_hidden: int = 256
    ci_layers: int = 2
    leaky_alpha: float = 0.01

    # loss coefficients
    coeff_faith: float = 1e4
    coeff_imp: float = 3e-2
    coeff_stoch: float = 1.0
    coeff_adv: float = 1.0
    coeff_simplicity: float = 1e-3  # APD "simplicity" on active components

    # importance-minimality L_p (linear anneal, as in run.py) -- PER-DATAPOINT sparsity
    p_start: float = 2.0
    p_end: float = 0.7
    imp_eps: float = 1e-12
    imp_beta: float = 0.5

    # lifetime (frequency) minimality -- ACROSS-DATAPOINT sparsity. Penalize each component by how
    # often it fires over the batch, with a convex power (>1) so a component that bundles k features
    # (~k x the firing frequency) pays ~k^pow, far more than k separate rarely-firing components pay
    # (k x 1). Pushes many rarely-active per-feature components instead of few shared bundles.
    # Per-datapoint minimality alone can't do this: a shared component that only fires for one of its
    # features at a time looks just as sparse per datapoint as a dedicated one. (Same term the
    # Matryoshka line uses; `coeff_lifetime`=0 disables it.)
    coeff_lifetime: float = 0.0
    lifetime_pow: float = 2.0
    lifetime_target: float = 0.0  # if >0, hinge: only tax firing frequency ABOVE this rate
    lifetime_ramp_frac: float = 0.0  # if >0, ramp coeff_lifetime 0->full over this fraction of
    # steps (explore-then-commit): let reconstruction FORM the components early, then split them late
    # -- so a strong final coeff splits bundles instead of suppressing components before they exist.

    # component parameterization + simplicity backend. "svd" -> free P[C,d_out,d_in], Schatten-p of
    # singular values. "factored" -> per-component A@B, variational nuclear norm 1/2(||A||^2+||B||^2).
    # "tucker" -> per-module Tucker of the bank tensor P[C,d_out,d_in]: shared factors F_C[C,r_C],
    #   F_out[d_out,r_mode], F_in[d_in,r_mode] + core[r_C,r_mode,r_mode]; masked forward runs entirely
    #   in the small core space (never materializes W_eff). Compresses BOTH the component mode (C->r_C)
    #   and the larger weight mode; r_mode >= rank(W)=min(d_out,d_in) keeps faithfulness exact.
    simplicity_impl: str = "svd"  # "svd" | "factored" | "tucker"
    simplicity_p: float = 1.0     # p=1 -> nuclear norm -> low rank (svd backend only)
    factor_rank: int | None = None  # inner rank r for "factored"; None -> min(d_out,d_in) (full)
    tucker_rc: int = 64           # component-mode rank r_C for "tucker" (<= C)
    tucker_r_mode: int | None = None  # weight-mode rank r_out=r_in for "tucker"; None -> min(d_out,d_in)

    # adversarial (fresh PGD per step, as in toy_decompose)
    pgd_steps: int = 4
    pgd_lr: float = 0.05

    # forward path for the "factored" backend: if True, do the masked forward in the r-dim factored
    # space (x@B_c -> gate -> @A_c), cost B*C*r*(d_in+d_out), NEVER materializing the per-datapoint
    # W_eff=[B,d_out,d_in]. If False, materialize W_eff (the same path the "svd" backend must use).
    # No effect for the "svd" backend (free P has no low-rank structure to exploit).
    lowrank_forward: bool = True

    # --- paper ingredients we initially skipped (each an independent toggle; apd_lm loop only) ---
    # APD hidden-activation recon: MSE(masked module output, target module output)/var, averaged
    # over the masked modules, attached to the stochastic pass. Forces components to route through
    # the right INTERMEDIATE activations, not just match the output distribution.
    coeff_hidden: float = 0.0
    # VPD uniform-k-subset routing: each step the stochastic mask hits a random k of the modules
    # (k ~ U{1..n_modules}); the rest run at target weights (SPD's layerwise-recon analog).
    subset_routing: bool = False
    # VPD clips U,V gradient norm to 0.01; applied to the component factors only (not the CI net).
    grad_clip: float = 0.0
    # APD per-step U normalization: renorm each rank-slice A[c,:,m] to unit norm, folding magnitude
    # into B (pure gauge change, P=A@B unchanged; factored backend only).
    unit_norm_A: bool = False

    # entrywise L1 on ALL component weights (unweighted): sum_c ||P_c||_1 under the faithfulness
    # constraint sum_c P_c = W is minimized by the disjoint-support (axis-aligned) carving -- the
    # rotation-symmetry breaker the nuclear-norm simplicity (rotation-invariant) cannot provide.
    # Diagnostic: sum_c ||P_c||_1 / ||W||_1 = 1.0 at perfect disjoint support.
    coeff_weight_l1: float = 0.0

    # --- variable-rank components (factored backend, factor_rank R > 1) ---------------------------
    # Three independent pressures that let a component's EFFECTIVE rank land anywhere in 0..R,
    # per matrix, while the cap R blocks mega-components. All three keep the one-gate-per-component
    # binding (the whole-network coupling) untouched.
    #
    # V1: plain UNWEIGHTED variational nuclear norm sum_c 1/2(||A_c||^2+||B_c||^2). Under the
    # faithfulness constraint sum_c P_c = W this is minimized by carving W's spectrum across
    # components without overlap (the rank-analog of the entrywise L1); unused rank shrinks to 0,
    # including in DORMANT components (unlike coeff_simplicity, which is importance-weighted).
    coeff_frob: float = 0.0
    # V2: Matryoshka-style nested ranks. Each step sample a rung k from {1,2,4,...,R} and run the
    # stochastic recon with every component truncated to its FIRST k rank-pieces (faithfulness,
    # PGD and eval stay full-rank). Prefixes must stand alone -> pieces become importance-ordered
    # and effective rank is where the tail dies. Structurally hostile to blobs.
    nested_rank: bool = False
    # V3: effective-rank (piece-count) penalty with capacity-x-usage coupling. Piece magnitude
    # s(c,r) = ||A[c,:,r]|| * ||B[c,r,:]|| (gauge-invariant); penalize sum_r s^rank_p (p<1 counts
    # pieces rather than magnitude), weighted per component by (rank_freq_floor + firing rate),
    # detached: frequent components must be low-rank, high-rank components must be rare, and the
    # floor makes dormant components shed junk rank too.
    coeff_rank: float = 0.0
    rank_p: float = 0.5
    rank_freq_floor: float = 0.05

    # in-training anti-redundancy (interaction) loss: sample component pairs each step, penalize
    # positive second-difference ablation damage I(i,j) = L(ij) - L(i) - L(j) + L(base) -- i.e.
    # super-additive damage = the pair backs each other up = redundant role overlap. The first
    # identity force that prices ROLE overlap rather than firing rate (rate-based forces are blind
    # on dense-activity targets). Negative I (shared pathway) is NOT penalized.
    coeff_interaction: float = 0.0
    interaction_pairs: int = 4  # sampled pairs per step (usage-weighted, distinct comps)

    # a VPD-style masked weight-delta spillover, active (with a U(0,1) mask) during stochastic/adv
    # recon to absorb whatever the components don't yet explain; forced to 0 at ci-masked eval so we
    # measure the components alone. APD has no delta; it's a training aid. Toggle off for purity.
    use_delta: bool = True

    seed: int = 0

    # wandb (mirrors the matryoshka setup; used by the apd_lm loop, rank 0 only)
    use_wandb: bool = False
    wandb_project: str = "apd-basis"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "decompose"
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_notes: str = ""


# --- Full-rank component bank (replaces one nn.Linear) ---------------------------------------------


class ComponentBankLinear(nn.Module):
    """Replaces one `nn.Linear` with a bank of `C` full-rank component matrices, summing to
    `W_target`. Two backends for the component params (see `ApdConfig.simplicity_impl`):

      "svd":      free `P [C, d_out, d_in]`.
      "factored": `A [C, d_out, r]`, `B [C, r, d_in]`, component = A@B (full-rank-capable at r=min).

    `materialized_weights()` returns `[C, d_out, d_in]` either way. The training loop calls
    `refresh_cache()` once per step so the (possibly A@B) weights are built once and reused across the
    several masked forwards; `W_cache` holds them (graph-connected, so grads flow).

    Modes:
      "target":    returns `F.linear(x, W_target, bias)` and caches `x` for the CI net.
      "component": returns `sum_c g_c (x @ P_c^T)` (+ bias) (+ optional masked delta), gate `g` set
                   externally and SHARED across all bank modules.
    """

    def __init__(self, linear: nn.Linear, C: int, impl: str, factor_rank: int | None,
                 tucker_rc: int = 64, tucker_r_mode: int | None = None) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = C
        self.impl = impl
        self.register_buffer("W_target", linear.weight.detach().clone())
        bias = linear.bias
        self.register_buffer("bias", bias.detach().clone() if bias is not None else None)
        std = 1.0 / math.sqrt(d_in * C)  # break symmetry; faithfulness warmup fits sum_c P_c to W
        if impl == "svd":
            self.P = nn.Parameter(torch.empty(C, d_out, d_in).normal_(0.0, std))
        elif impl == "factored":
            r = factor_rank if factor_rank is not None else min(d_out, d_in)
            self.r = r
            # init so (A@B) entries have variance std^2:  var(AB) = r * varA * varB
            sigma = (std * std / r) ** 0.25
            self.A = nn.Parameter(torch.empty(C, d_out, r).normal_(0.0, sigma))
            self.B = nn.Parameter(torch.empty(C, r, d_in).normal_(0.0, sigma))
        elif impl == "tucker":
            r_mode = tucker_r_mode if tucker_r_mode is not None else min(d_out, d_in)
            r_mode = min(r_mode, d_out, d_in)   # can't exceed either dim
            r_C = min(tucker_rc, C)
            self.r_out = self.r_in = r_mode
            self.r_C = r_C
            # P_c = F_out @ (sum_a F_C[c,a] core[a]) @ F_in^T. Init factors ~orthonormal-scale, small
            # core; faithfulness warmup fits sum_c P_c to W (core capacity r_mode >= rank(W)).
            self.F_out = nn.Parameter(torch.empty(d_out, r_mode).normal_(0.0, 1.0 / math.sqrt(r_mode)))
            self.F_in = nn.Parameter(torch.empty(d_in, r_mode).normal_(0.0, 1.0 / math.sqrt(r_mode)))
            self.F_C = nn.Parameter(torch.empty(C, r_C).normal_(0.0, 1.0 / math.sqrt(r_C)))
            self.core = nn.Parameter(torch.empty(r_C, r_mode, r_mode).normal_(0.0, std))
        else:
            raise ValueError(f"unknown simplicity_impl {impl!r}")
        # transient per-forward state
        self.mode: str = "target"
        self.mask: Tensor | None = None       # [*batch, C] gate, shared across modules
        self.rank_keep: int | None = None      # nested-rank truncation: use only pieces [:k] (factored)
        self.delta_mask: Tensor | None = None  # [*batch] scalar for the spillover
        self.last_input: Tensor | None = None
        self.W_cache: Tensor | None = None     # [C, d_out, d_in] (only when a materialize path needs it)
        self.W_sum: Tensor | None = None       # [d_out, d_in] = sum_c P_c, for faithfulness/delta (cheap)
        self.lowrank_fwd: bool = False         # set by install_banks; factored-space masked forward

    def params(self) -> list[Tensor]:
        if self.impl == "svd":
            return [self.P]
        if self.impl == "factored":
            return [self.A, self.B]
        return [self.F_C, self.F_out, self.F_in, self.core]  # tucker

    def _tucker_M(self) -> Tensor:
        """Per-component core-space matrices M_c = sum_a F_C[c,a] core[a]  -> [C, r_out, r_in]."""
        return torch.einsum("ca,aoi->coi", self.F_C, self.core)

    def materialized_weights(self) -> Tensor:
        if self.impl == "svd":
            return self.P
        if self.impl == "factored":
            return torch.einsum("cor,cri->coi", self.A, self.B)  # [C, d_out, d_in]
        # tucker: P_c[o,i] = sum_{r,s} F_out[o,r] M[c,r,s] F_in[i,s]
        M = self._tucker_M()  # [C, r_out, r_in]
        return torch.einsum("or,crs,is->coi", self.F_out, M, self.F_in)  # [C, d_out, d_in]

    def refresh_cache(self) -> None:
        """Build only what the step needs. W_sum = sum_c P_c ([d_out,d_in], cheap) is always set for
        faithfulness/delta. The full [C,d_out,d_in] W_cache is materialized ONLY when a forward/
        simplicity path actually needs it (svd, or factored-materialize) — avoids the LM-scale
        per-step [C,d,d] blow-up when using the low-rank forward."""
        if self.impl == "svd":
            self.W_cache = self.P
            self.W_sum = self.P.sum(dim=0)
        elif self.impl == "factored":
            # sum_c A_c B_c directly, without the [C,d,d] intermediate
            self.W_sum = torch.einsum("cor,cri->oi", self.A, self.B)
            self.W_cache = None if self.lowrank_fwd else self.materialized_weights()
        else:  # tucker: sum_c P_c = F_out @ (sum_c M_c) @ F_in^T
            m_sum = torch.einsum("a,ars->rs", self.F_C.sum(dim=0), self.core)  # [r_out, r_in]
            self.W_sum = torch.einsum("or,rs,is->oi", self.F_out, m_sum, self.F_in)
            self.W_cache = None

    def weight_delta(self) -> Tensor:
        assert self.W_sum is not None
        return self.W_target - self.W_sum  # [d_out, d_in]

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "target":
            self.last_input = x.detach()
            out = F.linear(x, self.W_target, self.bias)
            self.last_target_out = out.detach()  # for hidden-activation recon (APD L_hidden)
            return out
        assert self.mask is not None
        if self.rank_keep is not None:
            assert self.impl == "factored" and self.lowrank_fwd, \
                "nested_rank needs the factored backend with lowrank_forward=True"
        if self.impl == "tucker":
            # entirely in core space: gate contracts into the component mode, forward never leaves
            # the r_mode-dim space. gh = g @ F_C [.,r_C]; x~ = x @ F_in [.,r_in];
            # z = sum_a gh_a (x~ @ core[a]^T) [.,r_out]; out = z @ F_out^T. No [.,d_out,d_in] anywhere.
            gh = self.mask @ self.F_C                          # [*batch, r_C]
            xt = torch.einsum("...i,is->...s", x, self.F_in)   # [*batch, r_in]
            t = torch.einsum("...s,ars->...ar", xt, self.core)  # [*batch, r_C, r_out]
            z = torch.einsum("...a,...ar->...r", gh, t)        # [*batch, r_out]
            out = torch.einsum("...r,or->...o", z, self.F_out)  # [*batch, d_out]
        elif self.impl == "factored" and self.lowrank_fwd:
            # stay in r-dim factored space: x@B_c -> gate -> @A_c. Cost B*C*r*(d_in+d_out); never
            # materializes W_eff=[*batch,d_out,d_in]. Equivalent to sum_c g_c (A_c B_c) applied to x.
            # nested-rank truncation (rank_keep=k): only the first k pieces of every component run.
            rk = self.rank_keep
            Bf = self.B if rk is None else self.B[:, :rk, :]
            Af = self.A if rk is None else self.A[:, :, :rk]
            h = torch.einsum("...i,cri->...cr", x, Bf)  # [*batch, C, r]
            h = h * self.mask.unsqueeze(-1)                  # gate components
            out = torch.einsum("...cr,cor->...o", h, Af)  # [*batch, d_out]
        else:
            # materialize W_eff = sum_c g_c P_c, then out = x @ W_eff^T. [*batch, d_out, d_in].
            assert self.W_cache is not None
            w_eff = torch.einsum("...c,coi->...oi", self.mask, self.W_cache)
            out = torch.einsum("...oi,...i->...o", w_eff, x)
        if self.bias is not None:
            out = out + self.bias
        if self.delta_mask is not None:
            delta_out = F.linear(x, self.weight_delta())
            out = out + self.delta_mask.unsqueeze(-1) * delta_out
        self.last_masked_out = out  # graph-connected ref (no extra memory; used by L_hidden)
        return out


def install_banks(model: nn.Module, cfg: "ApdConfig") -> dict[str, ComponentBankLinear]:
    """Freeze all target params and replace each listed `nn.Linear` with a `ComponentBankLinear`."""
    for p in model.parameters():
        p.requires_grad_(False)
    banks: dict[str, ComponentBankLinear] = {}
    for path in cfg.modules:
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        linear = model.get_submodule(path)
        assert isinstance(linear, nn.Linear), f"{path} is not nn.Linear: {type(linear)}"
        bank = ComponentBankLinear(linear, cfg.n_components, cfg.simplicity_impl, cfg.factor_rank,
                                   tucker_rc=cfg.tucker_rc, tucker_r_mode=cfg.tucker_r_mode)
        bank.lowrank_fwd = cfg.lowrank_forward
        setattr(parent, attr, bank)
        banks[path] = bank
    return banks


def refresh_caches(banks: dict[str, ComponentBankLinear]) -> None:
    for b in banks.values():
        b.refresh_cache()


def set_masks(banks: dict[str, ComponentBankLinear], gate: Tensor, deltas: dict[str, Tensor] | None,
              subset: set[str] | None = None) -> None:
    """Apply the SAME network-wide gate to every bank (that's the whole point). If `subset` is given
    (VPD uniform-k-subset routing), only those banks are masked; the rest run at target weights."""
    for name, b in banks.items():
        if subset is not None and name not in subset:
            b.mode = "target"
            b.mask = None
            b.delta_mask = None
            continue
        b.mode = "component"
        b.mask = gate
        b.delta_mask = None if deltas is None else deltas[name]


def clear_masks(banks: dict[str, ComponentBankLinear]) -> None:
    for b in banks.values():
        b.mode = "target"
        b.mask = None
        b.delta_mask = None


# --- Losses ----------------------------------------------------------------------------------------


def faithfulness_loss(banks: dict[str, ComponentBankLinear]) -> Tensor:
    """MSE(W_target, sum_c P_c) summed over modules, per element. Needs `refresh_caches` first."""
    sq = torch.zeros((), device=next(iter(banks.values())).W_target.device)
    numel = 0
    for b in banks.values():
        d = b.weight_delta()
        sq = sq + d.pow(2).sum()
        numel += d.numel()
    return sq / numel


def simplicity_loss(banks: dict[str, ComponentBankLinear], importance_c: Tensor, cfg: ApdConfig) -> Tensor:
    """APD simplicity: sum_c s_c * sum_modules simplicity(P_{module,c}), `s_c` = detached importance.

    "svd":      Schatten-p of singular values (p from cfg). "factored": 1/2(||A||^2+||B||^2), the
    variational nuclear norm (== p=1). Both reshape only *active* components toward low rank."""
    per_c = torch.zeros_like(importance_c)  # [C]
    for b in banks.values():
        if b.impl == "svd":
            sv = torch.linalg.svdvals(b.W_cache)          # [C, min(d_out, d_in)]
            per_c = per_c + sv.pow(cfg.simplicity_p).sum(dim=1)
        elif b.impl == "factored":  # 1/2 (||A_c||_F^2 + ||B_c||_F^2) = variational nuclear norm
            per_c = per_c + 0.5 * (b.A.pow(2).sum(dim=(1, 2)) + b.B.pow(2).sum(dim=(1, 2)))
        else:  # tucker: Frobenius of the per-component core-space matrix M_c (complexity proxy;
            # r_out/r_in already bound rank structurally, this taxes magnitude of active components)
            per_c = per_c + b._tucker_M().pow(2).sum(dim=(1, 2))
    return (importance_c.detach() * per_c).sum()


def frob_loss(banks: dict[str, ComponentBankLinear]) -> Tensor:
    """V1: unweighted variational nuclear norm sum_c 1/2(||A_c||_F^2 + ||B_c||_F^2), all modules.
    Under faithfulness (sum_c P_c = W) minimized by splitting W's spectrum across components with
    no overlap; drives unused rank-pieces to zero in every component, dormant ones included."""
    total = torch.zeros((), device=next(iter(banks.values())).W_target.device)
    for b in banks.values():
        assert b.impl == "factored", "coeff_frob needs the factored backend"
        total = total + 0.5 * (b.A.pow(2).sum() + b.B.pow(2).sum())
    return total


def piece_norms(bank: ComponentBankLinear) -> Tensor:
    """Per-piece magnitudes s[c, r] = ||A[c,:,r]||_2 * ||B[c,r,:]||_2 (gauge-invariant)."""
    return bank.A.norm(dim=1) * bank.B.norm(dim=2)  # [C, r]


def rank_count_loss(banks: dict[str, ComponentBankLinear], freq_c: Tensor,
                    cfg: ApdConfig) -> Tensor:
    """V3: soft piece-count penalty with capacity-x-usage coupling. Per module,
    sum_c (floor + freq_c) * sum_r s(c,r)^p with p<1 -- sub-linear in piece magnitude, so it
    concentrates each component's mass on few pieces (rank selection) rather than shrinking all
    of them; the (floor + firing-rate) weight makes frequent components low-rank and lets
    high-rank specialists exist only if they are rare."""
    w = (cfg.rank_freq_floor + freq_c).detach()  # [C]
    total = torch.zeros((), device=freq_c.device)
    for b in banks.values():
        assert b.impl == "factored", "coeff_rank needs the factored backend"
        s = piece_norms(b)  # [C, r]
        # smoothed power: s^p has an infinite gradient at s -> 0, which explodes exactly when
        # another pressure (nested prefixes) drives tail pieces to zero; (s^2 + eps^2)^(p/2) is
        # the same count-like penalty with a bounded gradient near zero.
        total = total + (w * s.pow(2).add(1e-8).pow(cfg.rank_p / 2).sum(dim=1)).sum()
    return total


def sample_rank_rung(r: int, gen: torch.Generator) -> int:
    """V2 rung sampler: uniform over {1, 2, 4, ..., R} (powers of two, full rank included)."""
    rungs = [1 << i for i in range(r.bit_length()) if (1 << i) <= r]
    if rungs[-1] != r:
        rungs.append(r)
    return rungs[int(torch.randint(0, len(rungs), (1,), generator=gen).item())]


@torch.no_grad()
def rank_profile(banks: dict[str, ComponentBankLinear], thresh: float = 0.05) -> dict[str, Tensor]:
    """Effective rank per component per module, three ways. THE number is the SVD rank of the
    materialized component (singular values above `thresh` of the largest) — the outer-product
    pieces are not unique, so 16 mixed pieces can sum to a rank-3 matrix and the piece count only
    matches the true rank once a penalty has aligned pieces with singular directions. Piece count
    and participation ratio are kept as secondary diagnostics (V2/V3 act on pieces directly).
    Returns {module: [C, 3] (svd_rank, piece_count, participation)} for factored banks."""
    out: dict[str, Tensor] = {}
    for n, b in banks.items():
        if b.impl != "factored":
            continue
        sv = torch.linalg.svdvals(b.materialized_weights())  # [C, min(d_out, d_in)]
        svd_rank = (sv > thresh * sv.max(dim=1, keepdim=True).values.clamp_min(1e-12)).float().sum(dim=1)
        s = piece_norms(b)  # [C, r]
        top = s.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        count = (s > thresh * top).float().sum(dim=1)
        part = s.sum(dim=1).pow(2) / s.pow(2).sum(dim=1).clamp_min(1e-24)
        out[n] = torch.stack([svd_rank, count, part], dim=1)  # [C, 3]
    return out


# --- CI net (one gate per component, shared across modules) ----------------------------------------


class CIMLP(nn.Module):
    """Reads the decomposed modules' input activations (RMS-normed, concatenated) and emits `C`
    component gate logits -- ONE gate per component, shared across the whole network."""

    def __init__(self, module_order: list[str], d_in: dict[str, int], C: int,
                 hidden: int, n_layers: int, alpha: float) -> None:
        super().__init__()
        self.module_order = module_order
        self.alpha = alpha
        d_total = sum(d_in[n] for n in module_order)
        layers: list[nn.Module] = [nn.Linear(d_total, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, C)

    def forward(self, acts: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = torch.cat([F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order], dim=-1)
        logits = self.head(self.trunk(x))  # [*batch, C]
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha)


# --- Masked forward + reconstruction (module-level so the profiler hits the same paths) ------------


def masked_forward(model: nn.Module, banks: dict[str, ComponentBankLinear], x: Tensor,
                   gate: Tensor, deltas: dict[str, Tensor] | None,
                   subset: set[str] | None = None) -> Tensor:
    set_masks(banks, gate, deltas, subset)
    try:
        return model(x)
    finally:
        clear_masks(banks)


def recon_pair(model: nn.Module, banks: dict[str, ComponentBankLinear], cfg: ApdConfig,
               g_lower: Tensor, x: Tensor, target_out: Tensor, tvar: Tensor,
               deltas: dict[str, Tensor] | None) -> tuple[Tensor, Tensor]:
    """Stochastic + fresh-PGD adversarial recon, both under the SHARED gate `g_lower` [B, C]."""
    u = torch.rand_like(g_lower)
    stoch = F.mse_loss(masked_forward(model, banks, x, g_lower + (1 - g_lower) * u, deltas), target_out) / tvar
    s = torch.rand_like(g_lower).requires_grad_(True)
    for _ in range(cfg.pgd_steps):
        gm = g_lower.detach() + (1 - g_lower.detach()) * s
        adv = F.mse_loss(masked_forward(model, banks, x, gm, deltas), target_out)
        gr = torch.autograd.grad(adv, s)[0]
        s = (s + cfg.pgd_lr * gr.sign()).clamp(0, 1).detach().requires_grad_(True)
    gm = g_lower + (1 - g_lower) * s.detach()
    adv = F.mse_loss(masked_forward(model, banks, x, gm, deltas), target_out) / tvar
    return stoch, adv


# --- Decompose -------------------------------------------------------------------------------------


def decompose_apd(model: nn.Module, data_fn: Callable[[], Tensor], cfg: ApdConfig,
                  device: torch.device) -> dict[str, object]:
    order = sorted(cfg.modules)
    torch.manual_seed(cfg.seed)
    rgen = torch.Generator().manual_seed(cfg.seed + 31)  # nested-rank rung sampling
    banks = install_banks(model, cfg)
    model = model.to(device)
    d_in = {n: int(b.W_target.shape[1]) for n, b in banks.items()}
    ci = CIMLP(order, d_in, cfg.n_components, cfg.ci_hidden, cfg.ci_layers, cfg.leaky_alpha).to(device)

    comp_params = [p for b in banks.values() for p in b.params()]
    other = list(ci.parameters())

    # faithfulness warmup: fit sum_c P_c to the frozen weights before the CI net enters.
    wopt = torch.optim.AdamW(comp_params, lr=cfg.warmup_lr)
    for _ in range(cfg.warmup_steps):
        refresh_caches(banks)
        loss = faithfulness_loss(banks)
        wopt.zero_grad(); loss.backward(); wopt.step()

    opt = torch.optim.AdamW(comp_params + other, lr=cfg.lr)

    def target_forward(x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        clear_masks(banks)
        out = model(x)
        return out, {n: b.last_input for n, b in banks.items()}

    for step in range(cfg.n_steps):
        p = cfg.p_start + (cfg.p_end - cfg.p_start) * (step / cfg.n_steps)
        x = data_fn().to(device)
        target_out, acts = target_forward(x)
        target_out = target_out.detach()
        tvar = target_out.var() + 1e-8  # normalize recon to O(1) so minimality can't dominate it
        refresh_caches(banks)           # build (A@B or P) once; reused across the masked forwards
        g_lower, g_upper = ci(acts)  # [B, C]
        B = x.shape[0]
        deltas = {n: torch.rand(B, device=device) for n in order} if cfg.use_delta else None

        loss_faith = faithfulness_loss(banks)
        if cfg.nested_rank:  # V2: recon under a random rank-prefix; faithfulness/eval stay full
            rung = sample_rank_rung(next(iter(banks.values())).r, rgen)
            for b in banks.values():
                b.rank_keep = rung
        loss_stoch, loss_adv = recon_pair(model, banks, cfg, g_lower, x, target_out, tvar, deltas)
        if cfg.nested_rank:
            for b in banks.values():
                b.rank_keep = None
        loss_imp = importance_minimality_loss({"g": g_upper}, p, cfg.imp_eps, cfg.imp_beta, 1)
        loss_simp = simplicity_loss(banks, g_upper.mean(dim=0), cfg)
        loss_frob = frob_loss(banks) if cfg.coeff_frob > 0 else torch.zeros((), device=device)
        loss_rank = (rank_count_loss(banks, g_upper.mean(dim=0), cfg)
                     if cfg.coeff_rank > 0 else torch.zeros((), device=device))

        loss_life = torch.zeros((), device=device)
        life_coeff = cfg.coeff_lifetime
        if cfg.coeff_lifetime > 0.0:
            if cfg.lifetime_ramp_frac > 0.0:  # explore-then-commit: ramp 0 -> full
                life_coeff = cfg.coeff_lifetime * min(1.0, step / max(1, int(cfg.lifetime_ramp_frac * cfg.n_steps)))
            freq = g_upper.mean(dim=0)  # [C] mean firing over the batch (how consistently active)
            taxed = (freq - cfg.lifetime_target).clamp(min=0.0) if cfg.lifetime_target > 0.0 else freq
            loss_life = taxed.pow(cfg.lifetime_pow).sum()

        loss_l1 = torch.zeros((), device=device)
        if cfg.coeff_weight_l1 > 0:  # entrywise L1 over all components (materialize is fine on toys)
            loss_l1 = sum(b.materialized_weights().abs().sum() for b in banks.values())

        loss = (cfg.coeff_faith * loss_faith
                + cfg.coeff_stoch * loss_stoch
                + cfg.coeff_adv * loss_adv
                + cfg.coeff_imp * loss_imp
                + cfg.coeff_simplicity * loss_simp
                + life_coeff * loss_life
                + cfg.coeff_weight_l1 * loss_l1
                + cfg.coeff_frob * loss_frob
                + cfg.coeff_rank * loss_rank)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % max(1, cfg.n_steps // 10) == 0 or step == cfg.n_steps - 1:
            print(f"  step {step:>6} faith={loss_faith.item():.2e} stoch={loss_stoch.item():.4f} "
                  f"adv={loss_adv.item():.4f} imp={loss_imp.item():.3f} simp={loss_simp.item():.3f} "
                  f"life={loss_life.item():.4f} frob={loss_frob.item():.2f} rank={loss_rank.item():.2f}",
                  flush=True)

    # final eval
    out: dict[str, object] = {"banks": banks, "ci": ci}
    with torch.no_grad():
        x = data_fn().to(device)
        target_out, acts = target_forward(x)
        tvar = target_out.var() + 1e-8
        refresh_caches(banks)
        g_lower, g_upper = ci(acts)
        out["recon_ci"] = (F.mse_loss(masked_forward(model, banks, x, g_lower, None), target_out) / tvar).item()
        out["recon_off"] = (F.mse_loss(masked_forward(model, banks, x, torch.zeros_like(g_lower), None), target_out) / tvar).item()
        out["recon_on"] = (F.mse_loss(masked_forward(model, banks, x, torch.ones_like(g_lower), None), target_out) / tvar).item()
        out["l0"] = (g_lower > 0.5).float().sum(dim=-1).mean().item()
        out["mean_ci"] = g_upper.mean(dim=0).detach().cpu()  # [C]
        l1_sum = sum(b.materialized_weights().abs().sum().item() for b in banks.values())
        l1_w = sum(b.W_target.abs().sum().item() for b in banks.values())
        out["l1_ratio"] = l1_sum / l1_w  # 1.0 = disjoint support; superposition GT is >> 1
    return out


def component_matrices(bank: ComponentBankLinear) -> Tensor:
    """The learned components for one module as full weight matrices `[C, d_out, d_in]`."""
    return bank.materialized_weights().detach()


# --- Cross-layer mechanism recovery (ResidMLP) -----------------------------------------------------


@torch.no_grad()
def feature_recovery_resid(model: nn.Module, banks: dict[str, ComponentBankLinear],
                           ci: CIMLP, cfg: ApdConfig, device: torch.device,
                           n_probe: int = 128) -> dict[str, float]:
    """Does each input feature's cross-layer computation land on ONE dedicated component?

    For every input feature i we build single-feature probes (only i active, magnitude ~U(0,1) --
    in-distribution since the target was trained at ~1 active feature) and measure:

      active_frac    fraction of features whose top component fires confidently (mean gate > 0.5)
      injectivity    (# distinct assigned components) / (# features) over confidently-assigned
                     features -- 1.0 = each feature gets its own component (monosemantic, no sharing)
      purity         mean of top-component gate / total gate mass -- 1.0 = a single component carries it
      cross_layer    fraction of assigned components with substantial weight in BOTH layers (the
                     signal that a component captures a *cross-layer* mechanism, not a single layer)
      keep_only      mean normalized recon of feature i's OWN output dim using ONLY its assigned
                     component (vs passthrough baseline `keep_only_off`) -- low & << off = the
                     component causally supplies feature i's MLP correction (sufficiency)
    """
    order = sorted(cfg.modules)
    nf = int(model.n_features)  # type: ignore[attr-defined]
    C = cfg.n_components

    # pass 1: mean CI gate per feature -> assignment
    A = torch.zeros(nf, C, device=device)
    for i in range(nf):
        x = torch.zeros(n_probe, nf, device=device)
        x[:, i] = torch.rand(n_probe, device=device)
        clear_masks(banks)
        _ = model(x)
        acts = {n: b.last_input for n, b in banks.items()}
        g_lower, _ = ci(acts)
        A[i] = g_lower.mean(0)
    assigned = A.argmax(dim=1)                      # [nf]
    top1 = A.gather(1, assigned[:, None]).squeeze(1)  # [nf] top mean-gate
    active = top1 > 0.5
    purity = (top1 / (A.sum(dim=1) + 1e-8))[active].mean().item() if active.any() else 0.0
    injectivity = (assigned[active].unique().numel() / int(active.sum())) if active.any() else 0.0

    # cross-layer span of each component: weight norm in layer 0 vs layer 1
    refresh_caches(banks)
    comp_layer_norm = torch.zeros(C, 2, device=device)
    for n, b in banks.items():
        l = 0 if "blocks.0" in n else 1
        comp_layer_norm[:, l] += b.materialized_weights().pow(2).sum(dim=(1, 2))
    comp_layer_norm = comp_layer_norm.sqrt()
    a_norms = comp_layer_norm[assigned[active]]     # [n_active, 2]
    span = a_norms.min(dim=1).values / (a_norms.max(dim=1).values + 1e-8)
    cross_layer = (span > 0.1).float().mean().item() if active.any() else 0.0

    # causal sufficiency: keep ONLY the assigned component, recon feature i's own output dim
    keep_only, keep_off = [], []
    for i in range(nf):
        if not bool(active[i]):
            continue
        x = torch.zeros(n_probe, nf, device=device)
        x[:, i] = torch.rand(n_probe, device=device)
        clear_masks(banks)
        tgt = model(x).detach()
        var_i = tgt[:, i].var() + 1e-8
        gate = torch.zeros(n_probe, C, device=device)
        gate[:, assigned[i]] = 1.0
        pred = masked_forward(model, banks, x, gate, None)
        keep_only.append((F.mse_loss(pred[:, i], tgt[:, i]) / var_i).item())
        off = masked_forward(model, banks, x, torch.zeros(n_probe, C, device=device), None)
        keep_off.append((F.mse_loss(off[:, i], tgt[:, i]) / var_i).item())
    n_active = int(active.sum())
    return {
        "active_frac": n_active / nf,
        "injectivity": injectivity,
        "purity": purity,
        "cross_layer": cross_layer,
        "keep_only": (sum(keep_only) / len(keep_only)) if keep_only else float("nan"),
        "keep_only_off": (sum(keep_off) / len(keep_off)) if keep_off else float("nan"),
    }


# --- Entry points ----------------------------------------------------------------------------------


def _run_tms() -> None:
    import copy
    import os

    from .toy_models import TMS, feature_batch, train_tms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, nh = 5, 2
    C = int(os.environ.get("C", "20"))
    steps = int(os.environ.get("STEPS", "10000"))
    adv = float(os.environ.get("ADV", "1.0"))
    simp = float(os.environ.get("SIMP", "1e-3"))
    impl = os.environ.get("IMPL", "svd")
    fprob = float(os.environ.get("FPROB", "0.05"))
    ckpt = "/tmp/toy/tms_5-2.pt"

    os.makedirs("/tmp/toy", exist_ok=True)
    if os.path.exists(ckpt):
        base = TMS(nf, nh).to(device)
        base.load_state_dict(torch.load(ckpt, weights_only=True))
        print("loaded cached TMS target", flush=True)
    else:
        print("training TMS_5-2 target ...", flush=True)
        base = train_tms(nf, nh, steps=5000, batch=1024, feature_prob=0.05, lr=1e-2, device=device, seed=0)
        torch.save(base.state_dict(), ckpt)
    gt = base.ground_truth().to(device)  # [nf, nh, nf] -- rank-1 mechanism per feature
    gen = torch.Generator(device=device).manual_seed(123)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    print(f"config: C={C} steps={steps} adv={adv} simp={simp} impl={impl} fprob={fprob}", flush=True)
    model = copy.deepcopy(base)
    model.freeze_for_decomposition()
    cfg = ApdConfig(modules=["W"], n_components=C, n_steps=steps, warmup_steps=500,
                    coeff_imp=3e-2, coeff_adv=adv, coeff_simplicity=simp, simplicity_impl=impl)
    out = decompose_apd(model, data_fn, cfg, device)
    learned = component_matrices(out["banks"]["W"])  # [C, nh, nf]
    print(f"APD-basis MMCS to ground truth = {mmcs(learned, gt):.4f}  (1.0 = perfect)", flush=True)
    print(f"recon ci={out['recon_ci']:.4f} off={out['recon_off']:.4f} on={out['recon_on']:.4f} "
          f"L0={out['l0']:.2f} / C={C}", flush=True)
    print("APD_MASK TMS DONE", flush=True)


def _run_resid() -> None:
    import copy
    import os

    from .toy_models import ResidMLP, feature_batch, train_resid_mlp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, d_embed, d_mlp, n_layers = 100, 256, 40, 2
    C = int(os.environ.get("C", "130"))
    steps = int(os.environ.get("STEPS", "8000"))
    simp = float(os.environ.get("SIMP", "1e-3"))
    imp = float(os.environ.get("IMP", "3e-2"))
    life = float(os.environ.get("LIFE", "0.0"))
    life_target = float(os.environ.get("TARGET", "0.0"))
    life_ramp = float(os.environ.get("RAMP", "0.0"))
    impl = os.environ.get("IMPL", "svd")
    tucker_rc = int(os.environ.get("TUCKER_RC", "64"))
    fr = os.environ.get("R", "")  # factor_rank for the factored backend; "" -> min(d_out,d_in)
    factor_rank = int(fr) if fr else None
    weight_l1 = float(os.environ.get("L1", "0.0"))
    frob = float(os.environ.get("FROB", "0.0"))       # V1: unweighted variational nuclear norm
    rank_pen = float(os.environ.get("TRIM", os.environ.get("RANK", "0.0")))  # V3 piece-count
    # penalty (TRIM; RANK kept as a single-process fallback -- torchrun clobbers RANK under DDP)
    rank_floor = float(os.environ.get("RANKFLOOR", "0.05"))  # V3 usage-weight floor (set BELOW the
    # typical live-component firing rate or the usage coupling washes out)
    nested = os.environ.get("NESTED", "0") == "1"     # V2: Matryoshka nested rank prefixes
    seed = int(os.environ.get("SEED", "0"))
    fprob = 0.01
    ckpt = f"/tmp/toy/resid_2l_apd_dmlp{d_mlp}.pt"
    os.makedirs("/tmp/toy", exist_ok=True)

    if os.path.exists(ckpt):
        base = ResidMLP(nf, d_embed, d_mlp, n_layers, seed=0).to(device)
        base.load_state_dict(torch.load(ckpt, weights_only=True))
        print("loaded cached resid-MLP target", flush=True)
    else:
        print("training 2-layer cross-layer resid-MLP target ...", flush=True)
        base = train_resid_mlp(nf, d_embed, d_mlp, n_layers, steps=4000, batch=2048,
                               feature_prob=fprob, lr=3e-3, device=device, seed=0)
        torch.save(base.state_dict(), ckpt)
    gen = torch.Generator(device=device).manual_seed(123)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    modules = [f"blocks.{i}.{proj}" for i in range(n_layers) for proj in ("in_proj", "out_proj")]
    print(f"config: C={C} steps={steps} simp={simp} imp={imp} life={life} impl={impl} modules={modules}", flush=True)
    model = copy.deepcopy(base)
    cfg = ApdConfig(modules=modules, n_components=C, n_steps=steps, warmup_steps=500,
                    coeff_imp=imp, coeff_simplicity=simp, coeff_lifetime=life,
                    lifetime_target=life_target, lifetime_ramp_frac=life_ramp,
                    simplicity_impl=impl, tucker_rc=tucker_rc, factor_rank=factor_rank,
                    coeff_weight_l1=weight_l1, coeff_frob=frob, coeff_rank=rank_pen,
                    rank_freq_floor=rank_floor, nested_rank=nested, seed=seed)
    out = decompose_apd(model, data_fn, cfg, device)
    print(f"recon ci={out['recon_ci']:.4f} off={out['recon_off']:.4f} on={out['recon_on']:.4f} "
          f"L0={out['l0']:.2f} / C={C} l1_ratio={out['l1_ratio']:.2f}", flush=True)
    if impl == "factored" and factor_rank is not None and factor_rank > 1:
        prof = rank_profile(out["banks"])  # {module: [C, 3] (svd_rank, piece_count, participation)}
        live = out["mean_ci"] > 0.1  # only components that actually fire
        svd_r = torch.stack([prof[n][:, 0] for n in sorted(prof)], dim=1).cpu()  # [C, n_modules]
        pieces = torch.stack([prof[n][:, 1] for n in sorted(prof)], dim=1).cpu()
        el = (svd_r[live] if live.any() else svd_r).mean(dim=1)
        pl = (pieces[live] if live.any() else pieces).mean(dim=1)
        hist = torch.histc(el.float(), bins=factor_rank, min=0.5, max=factor_rank + 0.5)
        print(f"effective rank, SVD of materialized comps (live, mean over modules): mean={el.mean():.2f} "
              f"median={el.median():.1f} max={el.max():.1f}  [pieces alive: mean={pl.mean():.2f}]", flush=True)
        print("  svd-rank histogram (1..R): " + " ".join(str(int(v)) for v in hist), flush=True)
    if os.environ.get("RECOVERY", "0") == "1":
        rec = feature_recovery_resid(model, out["banks"], out["ci"], cfg, device)
        print(f"RECOVERY active_frac={rec['active_frac']:.2f} injectivity={rec['injectivity']:.2f} "
              f"purity={rec['purity']:.2f} cross_layer={rec['cross_layer']:.2f} "
              f"keep_only={rec['keep_only']:.3f} (vs off={rec['keep_only_off']:.3f})", flush=True)
    print("APD_MASK RESID DONE", flush=True)


if __name__ == "__main__":
    import os

    if os.environ.get("MODEL", "tms") == "resid":
        _run_resid()
    else:
        _run_tms()
