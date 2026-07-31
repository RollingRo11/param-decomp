"""Geometry+activation-matched decomposition (polar.py recipe) scaled to Pythia-14M.

Same four losses as the toy winner, adapted per the scaling notes:
  faithfulness      components sum to every decomposed matrix (24 nn.Linear: QKV,
                    attn.dense, mlp in/out x 6 layers)
  gated recon       KL(target next-token dist || gated forward), gates = per-token
                    attribution shares (A/Amax)^tau — no k, no mask net
  usage geometry    match component-usage cosine to the target's own per-parameter
                    usage cosine for sampled token pairs. At LM scale the fingerprints
                    are ~14M-dim; we never materialize them — for v = vec(outer(pre,
                    gpost) * W), the EXACT pair dot is (pre_i*pre_j) @ W^2 @
                    (gpost_i*gpost_j), summed over matrices (P sampled pairs/step).
  act matching      post-activations of all 24 decomposed matrices under the gated
                    forward must match the target's (read+write anti-shortcut).
Attribution: s = summed next-token CE -> ONE backward for all per-token gpost
(the toy's per-output-dim convention collapses at LM scale, as planned).

Run (DDP, both GPUs):
  torchrun --standalone --nproc_per_node=2 -m nano_apd.polar_lm --steps 10000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, "/workspace/param-decomp")
from nano_param_decomp.pythia14m import (  # noqa: E402
    C_PER_MODULE_PYTHIA_14M,
    generate_pool,
    load_pythia14m_target,
    pool_loader,
)

OUT_ROOT = Path(__file__).parent / "out"
MODULES = list(C_PER_MODULE_PYTHIA_14M.keys())


class ComponentBank(nn.Module):
    """Per decomposed matrix: W ~ sum_c A_c B_c, A [C, d_in, m], B [C, m, d_out]."""

    def __init__(self, d_in: int, d_out: int, C: int, m: int):
        super().__init__()
        self.C, self.m = C, m
        self.A = nn.Parameter(torch.empty(C, d_in, m))
        self.B = nn.Parameter(torch.empty(C, m, d_out))
        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)

    def weight(self) -> Tensor:  # [d_in, d_out]
        return torch.einsum("cim,cmo->io", self.A, self.B)


def build_banks(target: nn.Module, C: int, m: int) -> nn.ModuleDict:
    banks = nn.ModuleDict()
    for path in MODULES:
        lin = target.get_submodule(path)
        d_out, d_in = lin.weight.shape  # nn.Linear stores [d_out, d_in]
        banks[path.replace(".", "/")] = ComponentBank(d_in, d_out, C, m)
    return banks


class Runner:
    """Target forward with activation capture + gated forward via linear monkeypatch."""

    def __init__(self, target: nn.Module, banks: nn.ModuleDict, tau: float):
        self.target = target
        self.banks = banks
        self.tau = tau
        self.mode = "target"          # target | gated
        self.wscale = 1.0             # target-pass weight scale (IG-over-mask path)
        self.gates: Tensor | None = None  # [B, T, C]
        self.cache: dict[str, dict] = {}
        for path in MODULES:
            lin = target.get_submodule(path)
            lin._orig_forward = lin.forward
            lin._path = path
            lin.forward = self._make_forward(lin)

    def _make_forward(self, lin: nn.Linear):
        def fwd(x: Tensor) -> Tensor:
            path = lin._path
            if self.mode == "target":
                w = lin.weight if self.wscale == 1.0 else lin.weight * self.wscale
                out = F.linear(x, w, lin.bias)
                self.cache[path] = {"pre": x, "post": out}
                return out
            bank: ComponentBank = self.banks[path.replace(".", "/")]
            lead = x.shape[:-1]
            xf = x.reshape(-1, x.shape[-1])                      # [N, d_in], row-major
            gf = self.gates.reshape(-1, self.gates.shape[-1])    # [N, C], same order
            inner = torch.einsum("nd,cdm->ncm", xf, bank.A) * gf.unsqueeze(-1)
            out = inner.reshape(xf.shape[0], -1) @ bank.B.reshape(-1, bank.B.shape[-1])
            if lin.bias is not None:
                out = out + lin.bias
            out = out.reshape(*lead, -1)
            self.cache[path] = {"post": out}
            return out
        return fwd

    def target_pass(self, idx: Tensor):
        self.mode, self.cache = "target", {}
        logits = self.target(idx)
        return logits, self.cache

    def gated_pass(self, idx: Tensor, gates: Tensor):
        self.mode, self.cache, self.gates = "gated", {}, gates
        logits = self.target(idx)
        self.mode = "target"
        return logits, self.cache


def grow_banks(banks, new_m, device):
    """Expand every ComponentBank's rank cap to new_m: copy learned A,B into the
    first slices, xavier-init the new A slices, ZERO-init the new B slices (so the
    added rank contributes exactly 0 to the weight at growth — faithfulness is
    preserved, and gradients still flow into B_new via the nonzero A_new)."""
    for bank in banks.values():
        C, d_in, m = bank.A.shape
        d_out = bank.B.shape[2]
        nA = torch.empty(C, d_in, new_m, device=device)
        nn.init.xavier_normal_(nA)
        nB = torch.zeros(C, new_m, d_out, device=device)
        with torch.no_grad():
            nA[..., :m] = bank.A.data
            nB[:, :m, :] = bank.B.data
        bank.A = nn.Parameter(nA)
        bank.B = nn.Parameter(nB)
        bank.m = new_m


def rank_budget_loss(banks, g, target_R: float):
    """batchAverageRank: usage-weighted batch-average effective rank vs a target,
    TWO-SIDED (a budget, not a minimality pressure — deviation in either direction
    costs, so there is no degenerate escape to zero; some components are pushed up
    in capacity while others shrink). Per-matrix effective rank via the
    participation ratio of sigma^2: (tr MN)^2 / tr(MN MN) with M = A^T A,
    N = B B^T (m x m — no eigendecomposition, fully differentiable). Component
    rank = mass-weighted mean over matrices (mass = ||piece||_F^2 = tr MN, so
    near-empty pieces don't pollute). Usage weights are DETACHED mean gates
    (else the budget is satisfied by reshuffling gating instead of rank)."""
    rs, ws = [], []
    for path in MODULES:
        bank = banks[path.replace(".", "/")]
        A, B = bank.A.float(), bank.B.float()
        M = torch.einsum("cim,cin->cmn", A, A)
        N = torch.einsum("cmo,cno->cmn", B, B)
        MN = M @ N                                              # [C, m, m]
        tr1 = MN.diagonal(dim1=-2, dim2=-1).sum(-1)             # ||A_cB_c||_F^2
        tr2 = (MN * MN.transpose(-1, -2)).sum((-1, -2)).clamp_min(1e-20)
        rs.append(tr1.pow(2) / tr2)
        ws.append(tr1)
    R, W = torch.stack(rs), torch.stack(ws)                     # [M, C]
    r_c = (R * (W / W.sum(0, keepdim=True).clamp_min(1e-20))).sum(0)
    u = g.detach().float().mean((0, 1)).clamp_min(1e-12)        # [C]
    avg = (u * r_c).sum() / u.sum()
    return (avg - target_R) ** 2, avg


def attribution_sparsity(A: Tensor):
    """Sparsity of the PRE-GATING attribution A [B, T, C], measured per token over
    components and averaged. All quantities are scale-invariant, so they are directly
    comparable across gate_norm=max/sum (which only changes the downstream gating, not
    A). Returns:
      n_eff : smooth effective-L0 = participation ratio (sum A)^2 / sum A^2 — the
              differentiable count of active components (this doubles as the L0
              pressure loss when --l0_w > 0; minimizing it concentrates attribution).
      h10   : hard count of components with A within 10% of the per-token max.
      h01   : hard count of components with A within 1% of the per-token max."""
    Af = A.flatten(0, 1).float()                            # [N, C]
    s1 = Af.sum(-1)
    s2 = (Af * Af).sum(-1).clamp_min(1e-20)
    n_eff = (s1 * s1) / s2                                   # [N]
    rel = Af / (Af.amax(-1, keepdim=True) + 1e-20)
    h10 = (rel > 0.1).float().sum(-1)
    h01 = (rel > 0.01).float().sum(-1)
    return n_eff.mean(), h10.mean(), h01.mean()


def attribution_inner(banks, tcache, gposts) -> Tensor:
    """sum_modules gpost . (pre A_c B_c) — the pre-square inner sum [B, T, C]."""
    total = None
    for path in MODULES:
        bank = banks[path.replace(".", "/")]
        pre = tcache[path]["pre"].detach()
        gp = gposts[path]
        Bg = torch.einsum("cmo,bto->btcm", bank.B, gp)
        inner = torch.einsum("btd,cdm->btcm", pre, bank.A)
        contrib = (inner * Bg).sum(-1)                           # [B, T, C]
        total = contrib if total is None else total + contrib
    return total


def compute_attributions(banks, tcache, gposts) -> Tensor:
    """A[b, t, c] = ( sum_modules gpost . (pre A_c B_c) )^2 — one backward already done."""
    return attribution_inner(banks, tcache, gposts) ** 2


def row_geometry(tcache, gposts, targets_W2, anchors, candidates) -> Tensor:
    """Exact fingerprint-cosine rows for anchors vs candidates (both index the
    flattened token axis), via the same bilinear identity as pair_geometry."""
    def flat(path):
        return (tcache[path]["pre"].detach().flatten(0, 1),
                gposts[path].flatten(0, 1))
    D = None
    dii = None
    djj = None
    for path, W2 in targets_W2.items():
        pre, gp = flat(path)
        PI, PJ, GI, GJ = pre[anchors], pre[candidates], gp[anchors], gp[candidates]
        X = PI.unsqueeze(1) * PJ.unsqueeze(0)                 # [Na, Nc, d_in]
        Y = GI.unsqueeze(1) * GJ.unsqueeze(0)                 # [Na, Nc, d_out]
        t = ((X @ W2) * Y).sum(-1)                            # [Na, Nc]
        D = t if D is None else D + t
        si = (((PI ** 2) @ W2) * (GI ** 2)).sum(-1)           # [Na]
        sj = (((PJ ** 2) @ W2) * (GJ ** 2)).sum(-1)           # [Nc]
        dii = si if dii is None else dii + si
        djj = sj if djj is None else djj + sj
    return D / (dii.clamp_min(1e-20).sqrt().unsqueeze(1)
                * djj.clamp_min(1e-20).sqrt().unsqueeze(0))


def pair_geometry(tcache, gposts, targets_W2, pairs_i, pairs_j) -> Tensor:
    """Exact cosine of per-parameter fingerprints for sampled token pairs.
    v_i . v_j = sum_modules (pre_i*pre_j) @ W^2 @ (gpost_i*gpost_j)."""
    def dots(ii, jj):
        d = None
        for path in MODULES:
            pre = tcache[path]["pre"].detach().flatten(0, 1)
            gp = gposts[path].flatten(0, 1)
            t = torch.einsum("pd,do,po->p", pre[ii] * pre[jj], targets_W2[path],
                             gp[ii] * gp[jj])
            d = t if d is None else d + t
        return d
    dij = dots(pairs_i, pairs_j)
    dii = dots(pairs_i, pairs_i).clamp_min(1e-20)
    djj = dots(pairs_j, pairs_j).clamp_min(1e-20)
    return dij / (dii.sqrt() * djj.sqrt())


def pair_geometry_modular(tcache, gposts, targets_W2, pi, pj) -> Tensor:
    """Per-MODULE fingerprint cosines for sampled token pairs: [M, P]. Unlike
    pair_geometry, does NOT sum across modules before normalizing — the aggregate
    rewards merging tokens that differ in one module but agree elsewhere. Mechanism
    identity is a conjunction: differ anywhere = different mechanism.
    Also returns per-module TARGET-side fingerprint mass sqrt(dii*djj) [M, P] for
    weighting: where function lives is the target model's fact — weighting by
    attribution mass instead lets the decomposition dump mass into function-agreeing
    modules and hide merges from the loss (observed: use_eff collapse, gviol ~1)."""
    rows, wrows = [], []
    for path in MODULES:
        pre = tcache[path]["pre"].detach().flatten(0, 1)
        gp = gposts[path].flatten(0, 1)
        W2 = targets_W2[path]

        def d(ii, jj, pre=pre, gp=gp, weight_sq=W2):
            return torch.einsum("pd,do,po->p", pre[ii] * pre[jj], weight_sq,
                                gp[ii] * gp[jj])
        dij = d(pi, pj)
        dii = d(pi, pi).clamp_min(1e-20)
        djj = d(pj, pj).clamp_min(1e-20)
        wrows.append((dii * djj).sqrt())
        rows.append(dij / (dii.sqrt() * djj.sqrt()))
    return torch.stack(rows), torch.stack(wrows)                 # [M,P],[M,P]


def attribution_module_rows(banks, tcache, gposts, idx):
    """Per-module squared-attribution rows a_m[n, c] = (gpost . pre A_c B_c)^2 at
    flattened token indices idx, stacked [M, n, C], plus per-row mass norms [M, n].
    Differentiable w.r.t. banks (pre/gposts detached, as in attribution_inner)."""
    rows, mass = [], []
    for path in MODULES:
        bank = banks[path.replace(".", "/")]
        pre = tcache[path]["pre"].detach().flatten(0, 1)[idx]
        gp = gposts[path].flatten(0, 1)[idx]
        Bg = torch.einsum("cmo,no->ncm", bank.B, gp)
        inner = torch.einsum("nd,cdm->ncm", pre, bank.A)
        a = ((inner * Bg).sum(-1)) ** 2                          # [n, C]
        rows.append(a)
        mass.append(a.float().norm(dim=-1))
    return torch.stack(rows), torch.stack(mass)                  # [M,n,C],[M,n]


def block_prefix(path: str) -> str:
    """'h.0.attn.q_proj' -> 'h.0'; 'gpt_neox.layers.3.mlp.dense_4h_to_h' ->
    'gpt_neox.layers.3' (module path up to and including the block index)."""
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return ".".join(parts[: i + 1])
    raise ValueError(path)


def install_perturb_hooks(target: nn.Module) -> dict:
    """Forward-pre-hooks on every block that (a) record the block input's RMS during
    capture, (b) add PERT['vec'] to the block input when PERT['path'] matches.
    Handles positional (pile4l Block(x)) and kwargs (HF hidden_states=) calls."""
    PERT = {"path": None, "vec": None, "capture": False, "rms": {}, "d": {}}
    for bp in sorted({block_prefix(p) for p in MODULES}):
        def mk(bp):
            def pre(mod, hargs, hkwargs):
                h = hargs[0] if hargs else hkwargs["hidden_states"]
                if PERT["capture"]:
                    PERT["rms"][bp] = h.detach().float().pow(2).mean().sqrt().item()
                    PERT["d"][bp] = h.shape[-1]
                if PERT["path"] == bp:
                    h = h + PERT["vec"]
                    if hargs:
                        return (h,) + hargs[1:], hkwargs
                    hkwargs["hidden_states"] = h
                return hargs, hkwargs
            return pre
        target.get_submodule(bp).register_forward_pre_hook(mk(bp), with_kwargs=True)
    return PERT


def jacobian_match_loss(runner, PERT, idx_sel, gates_sel, logits_t_sel, args, device):
    """L_tan = 1 - cos(d_tgt, d_gated) + beta * log^2(|d_gated|/|d_tgt|), where d_* are
    finite-difference responses at the logits to the same eps*u resid perturbation at a
    randomly sampled block. Gates detached (circuit fidelity, not selection dynamics)."""
    bps = sorted(PERT["rms"].keys())
    bp = bps[torch.randint(len(bps), (1,)).item()]
    eps = args.jac_eps * max(PERT["rms"][bp], 1e-8)
    u = torch.randn(*idx_sel.shape, PERT["d"][bp], device=device)
    u = F.normalize(u, dim=-1) * eps
    g_det = gates_sel.detach()
    PERT["path"], PERT["vec"] = None, None
    logits_gb, _ = runner.gated_pass(idx_sel, g_det)             # base, detached gates
    PERT["path"], PERT["vec"] = bp, u
    with torch.no_grad():
        logits_tp, _ = runner.target_pass(idx_sel)               # perturbed target
    logits_gp, _ = runner.gated_pass(idx_sel, g_det)             # perturbed gated
    PERT["path"], PERT["vec"] = None, None
    d_tgt = (logits_tp.float() - logits_t_sel.detach().float()) / eps
    d_gat = (logits_gp.float() - logits_gb.float()) / eps
    cos = F.cosine_similarity(d_gat, d_tgt, dim=-1)
    nrat = ((d_gat.norm(dim=-1) + 1e-6) / (d_tgt.norm(dim=-1) + 1e-6)).log()
    return (1 - cos).mean() + args.jac_beta * (nrat ** 2).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=int, default=1024)
    # --rank_m alias: torchrun's argparser greedily prefix-matches a bare --m
    # against its own --master*/--module/--monitor* options and refuses to pass
    # it through, in both `-m module` and script-path invocation forms
    parser.add_argument("--m", "--rank_m", dest="m", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_seqs", type=int, default=24)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--faith", type=float, default=100.0)
    parser.add_argument("--recon", type=float, default=10.0)
    parser.add_argument("--geom", type=float, default=30.0)
    parser.add_argument("--act", type=float, default=3.0)
    parser.add_argument("--jac", type=float, default=0.0,
                        help="Jacobian (tangent-space) matching: perturb the residual "
                             "stream at a sampled block by eps*u and require the gated "
                             "model's finite-difference response to match the target's "
                             "in direction (cosine) and magnitude (log-norm-ratio^2, "
                             "weight jac_beta). Gates are DETACHED here: the question "
                             "is whether the selected circuit implements the target's "
                             "local computation, not gate-selection dynamics. Testing "
                             "only — leave 0 for production runs.")
    parser.add_argument("--jac_beta", type=float, default=1.0)
    parser.add_argument("--jac_eps", type=float, default=0.1,
                        help="perturbation scale relative to per-token resid RMS")
    parser.add_argument("--jac_every", type=int, default=1,
                        help="apply the jac loss every N steps (lazy regularization: "
                             "the weight is scaled by N on firing steps, so average "
                             "pressure matches every-step application at ~1/N cost)")
    parser.add_argument("--lr_steps", type=int, default=0,
                        help="cosine LR horizon (0 = use --steps). Set to the full "
                             "run length when doing a short validation, so the LR "
                             "matches that run's early slice instead of decaying fully.")
    parser.add_argument("--m_grow", type=int, default=0,
                        help="grow the per-matrix rank cap up to this value (0 = "
                             "fixed at --m). Starts at --m and doubles when the "
                             "usage-weighted avg rank saturates the current cap — "
                             "so memory is only spent where the budget needs it.")
    parser.add_argument("--grow_every", type=int, default=250)
    parser.add_argument("--grow_thresh", type=float, default=0.85,
                        help="grow when avg effective rank >= this fraction of cap")
    parser.add_argument("--rank_target", type=float, default=0.0,
                        help="batchAverageRank: two-sided budget on the usage-"
                             "weighted batch-average effective rank (0 = off). "
                             "Use with a raised cap, e.g. --m 8 --rank_target 6.")
    parser.add_argument("--rank_w", type=float, default=1.0,
                        help="weight of the rank-budget loss")
    parser.add_argument("--l0_w", type=float, default=0.0,
                        help="weight of the L0 sparsity pressure on the number of "
                             "components with PRE-GATING attribution (smooth effective-"
                             "L0 = participation ratio of A). Default 0 = metric only, "
                             "tracked in wandb but inert in the loss.")
    parser.add_argument("--l0_target", type=float, default=0.0,
                        help="two-sided per-token L0 budget: loss becomes "
                             "(l0_eff - target)^2 instead of minimizing l0_eff. "
                             "Deviation in EITHER direction costs, so per-token "
                             "sparsity cannot race below the target (the 30-owner "
                             "collapse mode of pure minimization). 0 = minimize.")
    parser.add_argument("--gate_norm", choices=["max", "sum"], default="max",
                        help="gate normalization: divide attribution by max_c (winner"
                             "-take-most, gates in [0,1]) or by sum_c (partition of "
                             "unity, gates are shares summing to 1).")
    parser.add_argument("--gate_thresh", type=float, default=0.0,
                        help="hard-zero the gate tail after max-normalization: "
                             "g = relu(g - t)/(1 - t). Losers get exactly 0, "
                             "winners keep full amplitude.")
    parser.add_argument("--comp_dropout", type=float, default=0.0,
                        help="ablation-robustness: per sequence in the gated/recon "
                             "pass, drop each component's gate with this prob. "
                             "Keep small (~0.1): uniform dropout of ACTIVE "
                             "components doubles as a redundancy pressure.")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pool_seqs", type=int, default=4096)
    parser.add_argument("--ramp_frac", type=float, default=0.3,
                        help="fraction of steps over which functional losses ramp in")
    parser.add_argument("--sub_frac", type=float, default=1.0,
                        help="fraction of batch sequences run through the gated pass "
                             "(recon+act computed on that subset; <1 = cheaper backward)")
    parser.add_argument("--tf32", action="store_true",
                        help="allow TF32 matmuls (cheaper precision on H100)")
    parser.add_argument("--bf16", action="store_true",
                        help="bf16 autocast on target/gated/attribution passes; "
                             "faith loss stays exact fp32")
    parser.add_argument("--bf16_frac", type=float, default=1.0,
                        help="fraction of steps with bf16 enabled; the rest run fp32 "
                             "(precision annealing — the bf16 floor only binds near "
                             "convergence)")
    parser.add_argument("--geom_margin", type=float, default=0.1,
                        help="modular geometry: slack before a pair counts as a "
                             "violation in either direction")
    parser.add_argument("--geom_frag", type=float, default=0.3,
                        help="modular geometry: weight of fragmentation violations "
                             "(function-similar, attribution-different) relative to "
                             "merging violations (attribution-similar, function-"
                             "different), which get weight 1")
    parser.add_argument("--n_cand_pairs", type=int, default=16384,
                        help="modular geometry: candidate pool mined (no_grad) for "
                             "violation pairs; n_pairs of them get the full "
                             "differentiable per-module loss")
    parser.add_argument("--geom_mode", choices=["gram", "cliprow", "modular"],
                        default="gram",
                        help="gram = MSE on sampled pair cosines; cliprow = CLIP-style "
                             "soft-target contrast on anchor-vs-candidate similarity rows")
    parser.add_argument("--tau_t", type=float, default=0.1,
                        help="target-side softmax temperature for geom_mode=cliprow")
    parser.add_argument("--n_anchors", type=int, default=128)
    parser.add_argument("--n_cands", type=int, default=256)
    parser.add_argument("--ig_steps", type=int, default=1,
                        help=">1 = IG-over-mask attribution: K target passes with all "
                             "decomposed weights scaled alpha=k/K; inner products "
                             "averaged over the path before squaring (own-gate-at-zero "
                             "counterfactual). The k=K pass is the ordinary clean pass, "
                             "reused for recon/act/geom.")
    parser.add_argument("--target", choices=["pythia14m", "pile4l"], default="pythia14m",
                        help="pile4l = VPD paper's 4-layer LlamaSimpleMLP (67M, Pile)")
    parser.add_argument("--wandb", action="store_true",
                        help="log metrics to WandB (project nano-apd, rank 0 only)")
    parser.add_argument("--ckpt_every", type=int, default=10000,
                        help="atomic banks+optimizer checkpoint every N steps "
                             "(rank 0; 0 disables). A crash costs at most N steps.")
    parser.add_argument("--resume", action="store_true",
                        help="resume from <out_dir>/ckpt.pt if present (banks, "
                             "optimizer, step; data stream reshuffled from the "
                             "resume step, not replayed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{rank}" if ddp else "cuda"
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    torch.manual_seed(args.seed + rank)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.target == "pile4l":
        from nano_param_decomp.pile_4L import C_PER_MODULE_4L, load_paper_target_model, make_loader
        MODULES[:] = list(C_PER_MODULE_4L.keys())
        target = load_paper_target_model().float().to(device)
        for p in target.parameters():
            p.requires_grad_(True)
        loader = make_loader(args.batch_seqs * world, args.seq_len, rank, world,
                             "train", args.seed)
    else:
        target = load_pythia14m_target().float().to(device)
        for p in target.parameters():
            p.requires_grad_(True)  # needed for activation grads; never optimized

        pool_path = Path(f"/tmp/toy/pythia14m_pool_{args.pool_seqs}x{args.seq_len}.pt")
        if rank == 0 and not pool_path.exists():
            pool_path.parent.mkdir(exist_ok=True)
            pool = generate_pool(target, args.pool_seqs, args.seq_len,
                                 torch.device(device))
            torch.save(pool, pool_path)
        if ddp:
            dist.barrier()
        pool = torch.load(pool_path, weights_only=True, map_location="cpu")
        loader = pool_loader(pool, args.batch_seqs, seed=args.seed + 100 * rank)

    banks = build_banks(target, args.C, args.m).to(device)
    runner = Runner(target, banks, args.tau)
    PERT = install_perturb_hooks(target) if args.jac > 0 else None
    logit_scale = None
    train_params = list(banks.parameters())
    if args.geom_mode == "cliprow":
        import math
        logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(1 / 0.07), device=device))
        train_params.append(logit_scale)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)

    W2 = {p: (target.get_submodule(p).weight.detach().t() ** 2) for p in MODULES}
    n_params = sum(target.get_submodule(p).weight.numel() for p in MODULES)

    out_dir = OUT_ROOT / f"{args.target}_polar_C{args.C}{('_' + args.tag) if args.tag else ''}"
    ckpt_path = out_dir / "ckpt.pt"
    start_step = 0
    cur_m = args.m
    lr_horizon = args.lr_steps if args.lr_steps > 0 else args.steps
    if args.resume and ckpt_path.exists():
        # load to CPU: a device-mapped ck dict stays referenced for the whole run
        # (~3.6GB banks+opt held on GPU for nothing); load_state_dict moves to device
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        if ck.get("cur_m", args.m) > cur_m:      # grow to saved cap before loading
            grow_banks(banks, ck["cur_m"], device)
            cur_m = ck["cur_m"]
            train_params = list(banks.parameters()) + (
                [logit_scale] if logit_scale is not None else [])
            opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
        banks.load_state_dict(ck["banks"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        if rank == 0:
            print(f"resumed from {ckpt_path} at step {start_step}, m={cur_m}", flush=True)
        del ck
        torch.cuda.empty_cache()

    for step in range(start_step, args.steps + 1):
        lr = args.lr * (0.5 * (1 + torch.cos(torch.tensor(
            min(step, lr_horizon) / lr_horizon * 3.14159)))).item()
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        # Target parameters require gradients so the captured activations have an
        # autograd graph. They are not optimized, so clear accumulating gradients.
        target.zero_grad(set_to_none=True)
        idx = next(loader).to(device)

        # heavy passes optionally under bf16 autocast; the faith loss is computed
        # OUTSIDE autocast so the weight-matching residual stays exact fp32
        # NOTE <= with frac>=1.0: the final (log-only) iteration must stay bf16 —
        # an fp32 final forward spikes ~+8GB and OOM'd two 30k runs at step 30000
        amp_on = args.bf16 and (args.bf16_frac >= 1.0
                                or step < int(args.bf16_frac * args.steps))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            if PERT is not None:
                PERT["capture"] = True
            inner_tot = None
            for k in range(1, args.ig_steps + 1):
                runner.wscale = k / args.ig_steps                # == 1.0 at k = K
                last = k == args.ig_steps
                logits_t, tcache = runner.target_pass(idx)
                s = F.cross_entropy(logits_t[:, :-1].flatten(0, 1),
                                    idx[:, 1:].flatten(), reduction="sum")
                posts = [tcache[p]["post"] for p in MODULES]
                gposts_l = torch.autograd.grad(s, posts, retain_graph=last)
                gposts = {p: g.detach() for p, g in zip(MODULES, gposts_l, strict=True)}
                contrib = attribution_inner(banks, tcache, gposts)
                inner_tot = contrib if inner_tot is None else inner_tot + contrib
            A = (inner_tot / args.ig_steps) ** 2                 # [B, T, C]
            denom = (A.sum(-1, keepdim=True) if args.gate_norm == "sum"
                     else A.amax(-1, keepdim=True))
            g = (A / (denom + 1e-12)) ** args.tau
            if args.gate_thresh > 0:
                g = F.relu(g - args.gate_thresh) / (1 - args.gate_thresh)

            # optionally run the gated pass (the cost bottleneck) on a subset of
            # the batch sequences; recon+act are computed on that subset only
            if args.sub_frac < 1.0:
                k = max(1, int(round(args.sub_frac * idx.shape[0])))
                sel = torch.randperm(idx.shape[0], device=device)[:k]
            else:
                sel = torch.arange(idx.shape[0], device=device)
            if PERT is not None:
                PERT["capture"] = False
            g_run = g[sel]
            if args.comp_dropout > 0:
                keep = (torch.rand(len(sel), 1, args.C, device=device)
                        > args.comp_dropout).to(g.dtype)
                g_run = g_run * keep
            logits_g, gcache = runner.gated_pass(idx[sel], g_run)

            loss_jac = torch.tensor(0.0, device=device)
            if args.jac > 0 and step % args.jac_every == 0:
                loss_jac = jacobian_match_loss(runner, PERT, idx[sel], g[sel],
                                               logits_t[sel], args, device)

            # Mean nats per predicting token. `batchmean` divided only by batch size,
            # making the old training metric scale linearly with context length.
            loss_recon = F.kl_div(
                F.log_softmax(logits_g[:, :-1], -1),
                F.softmax(logits_t[sel][:, :-1].detach(), -1),
                reduction="none",
            ).sum(-1).mean()
            loss_act = sum(((gcache[p]["post"] - tcache[p]["post"][sel].detach()) ** 2)
                           .mean() for p in MODULES) / len(MODULES)

            BT = idx.shape[0] * idx.shape[1]
            geom_viol = 0.0
            if args.geom_mode == "cliprow":
                # CLIP-style: anchor rows of the exact fingerprint-cosine matrix
                # (softmaxed) are soft targets for the usage-cosine rows
                perm = torch.randperm(BT, device=device)
                J = perm[:args.n_cands]
                anchors = J[:args.n_anchors]
                with torch.no_grad():
                    w_rows = row_geometry(tcache, gposts, W2, anchors, J).float()
                    tgt = F.softmax(w_rows / args.tau_t, dim=-1)
                An = F.normalize(A.flatten(0, 1).float(), dim=-1)
                scale = logit_scale.exp().clamp(max=100.0)
                logits = (An[anchors] @ An[J].t()) * scale
                loss_geom = -(tgt * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            elif args.geom_mode == "modular":
                # ---- 1) mine violation pairs cheaply on a large candidate pool ----
                with torch.no_grad():
                    ci = torch.randint(0, BT, (args.n_cand_pairs,), device=device)
                    cj = torch.randint(0, BT, (args.n_cand_pairs,), device=device)
                    w_mod_c, fmass_c = pair_geometry_modular(
                        tcache, gposts, W2, ci, cj)
                    w_mod_c = w_mod_c.float().clamp(0, 1)               # [M, Pc]
                    # conjunction target: fingerprint-mass-weighted SOFT-min. A raw
                    # min over 24 modules always finds a near-dead module with a
                    # noisy ~0 cosine, so every pair scores "function-different"
                    # (gviol saturates ~1, mining unselective). Weighting by target-
                    # side mass + softmin keeps conjunction semantics where function
                    # actually lives.
                    fw_c = fmass_c.float() / fmass_c.float().sum(0, keepdim=True) \
                        .clamp_min(1e-20)
                    T_sm = 0.15
                    w_min = -T_sm * torch.log(
                        (fw_c * torch.exp(-w_mod_c / T_sm)).sum(0).clamp_min(1e-20))
                    Af = F.normalize(A.flatten(0, 1).float(), dim=-1)
                    u_c = (Af[ci] * Af[cj]).sum(-1)  # global-A proxy for mining
                    n_m = args.n_pairs // 2
                    n_f = args.n_pairs // 4
                    n_r = args.n_pairs - n_m - n_f
                    sel = torch.cat([
                        (u_c - w_min).topk(n_m).indices,          # merging viols
                        (w_min - u_c).topk(n_f).indices,          # fragmentation
                        torch.randint(0, args.n_cand_pairs, (n_r,), device=device)])
                    pi, pj = ci[sel], cj[sel]
                    w_tgt = w_min[sel]                            # [P]
                    geom_viol = (u_c - w_min).topk(n_m).values.mean().item()
                # ---- 2) differentiable GLOBAL hinge on the selected pairs ----
                # Penalize the mined quantity itself: global attribution cosine may
                # not exceed functional agreement (weighted-softmin conjunction).
                # v1-v3 penalized per-module cosines while mining global ones — the
                # optimizer merged in the aggregate (hub dominating global A) while
                # keeping per-module profiles dissimilar, so gviol saturated at ~1
                # with zero hinge gradient. Training the mined quantity closes the
                # metric/gradient gap: gviol IS the pre-margin trained value.
                Ag = F.normalize(A.flatten(0, 1).float(), dim=-1)
                u_sel = (Ag[pi] * Ag[pj]).sum(-1)                 # [P], with grad
                merge = F.relu(u_sel - w_tgt - args.geom_margin)
                if args.geom_frag > 0:
                    merge = merge + args.geom_frag * F.relu(
                        w_tgt - u_sel - args.geom_margin)
                loss_geom = merge.mean()
            else:
                pi = torch.randint(0, BT, (args.n_pairs,), device=device)
                pj = torch.randint(0, BT, (args.n_pairs,), device=device)
                w_sim = pair_geometry(tcache, gposts, W2, pi, pj).detach()
                Af = F.normalize(A.flatten(0, 1).float(), dim=-1)
                u = (Af[pi] * Af[pj]).sum(-1)
                loss_geom = ((u - w_sim.float()) ** 2).mean()

        loss_faith = sum(
            ((banks[p.replace('.', '/')].weight()
              - target.get_submodule(p).weight.detach().t()) ** 2).sum()
            for p in MODULES) / n_params

        # phased schedule: faithfulness full-strength from step 0; functional pressures
        # ramp in over the first ramp_frac of training (they compete with faithfulness
        # when applied full-strength to random components — v1/v2 finding)
        ramp = min(1.0, step / max(1, int(args.ramp_frac * args.steps)))
        loss_rank = torch.tensor(0.0, device=device)
        rank_avg = 0.0
        if args.rank_target > 0:
            loss_rank, ravg = rank_budget_loss(banks, g, args.rank_target)
            rank_avg = ravg.item()

        # L0 sparsity pressure on pre-gating attribution (participation ratio of A).
        # Only built into the graph when weighted; otherwise logged as a metric below.
        # With --l0_target: two-sided budget (see arg help) instead of minimization.
        loss_l0 = torch.tensor(0.0, device=device)
        if args.l0_w > 0:
            n_eff, _, _ = attribution_sparsity(A)
            loss_l0 = ((n_eff - args.l0_target) ** 2 if args.l0_target > 0 else n_eff)

        loss = (args.faith * loss_faith + ramp * (args.recon * loss_recon
                + args.geom * loss_geom + args.act * loss_act
                + args.jac * args.jac_every * loss_jac
                + args.rank_w * loss_rank + args.l0_w * loss_l0))

        if step % 100 == 0 and rank == 0:
            with torch.no_grad():
                eff = g.sum(-1).mean().item()
                l0_eff, l0_h10, l0_h01 = attribution_sparsity(A)
                shm = A.flatten(0, 1).float()
                shm = shm / (shm.sum(-1, keepdim=True) + 1e-20)
                um = shm.mean(0)
                use_eff = (1.0 / (um * um).sum()).item()   # effective #components in use
            rec = {"step": step, "faith": round(loss_faith.item(), 6),
                   "recon_kl": round(loss_recon.item(), 4),
                   "act": round(loss_act.item(), 5),
                   "geom": round(loss_geom.item(), 5),
                   "jac": round(loss_jac.item(), 4),
                   "rank": round(rank_avg, 2),
                   "l0_eff": round(l0_eff.item(), 2),
                   "l0_frac": round(l0_eff.item() / args.C, 5),
                   "l0_h10": round(l0_h10.item(), 2),
                   "l0_h01": round(l0_h01.item(), 2),
                   "use_eff": round(use_eff, 1),
                   "gviol": round(geom_viol, 4),
                   "eff_gates": round(eff, 1), "lr": round(lr, 6)}
            print(json.dumps(rec), flush=True)
            if args.wandb:
                if not hasattr(main, "_wb"):
                    import wandb
                    main._wb = wandb.init(
                        project="nano-apd",
                        name=f"{args.target}_C{args.C}_{args.tag or 'run'}",
                        config=vars(args))
                main._wb.log(rec, step=step)

        if step != args.steps:
            # DDP grad sync via a dummy forward-hooked module is avoided: sum losses on
            # bank params only; manual allreduce keeps the Runner monkeypatch simple
            loss.backward()
            if ddp:
                for p_ in train_params:
                    if p_.grad is not None:
                        dist.all_reduce(p_.grad, op=dist.ReduceOp.AVG)
            opt.step()
            if step > 0 and step % 2000 == 0:
                torch.cuda.empty_cache()   # cap allocator fragmentation growth
            # grow the rank cap when the budget saturates the current cap
            if (args.m_grow > cur_m and step > start_step
                    and step % args.grow_every == 0
                    and rank_avg >= args.grow_thresh * cur_m):
                new_m = min(cur_m * 2, args.m_grow)
                grow_banks(banks, new_m, device)
                if ddp:
                    for bnk in banks.values():
                        dist.broadcast(bnk.A.data, src=0)
                        dist.broadcast(bnk.B.data, src=0)
                train_params = list(banks.parameters()) + (
                    [logit_scale] if logit_scale is not None else [])
                opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.0)
                cur_m = new_m
                torch.cuda.empty_cache()
                if rank == 0:
                    print(json.dumps({"step": step, "GREW_rank_cap_to": new_m,
                                      "at_avg_rank": round(rank_avg, 2)}), flush=True)
            if (rank == 0 and args.ckpt_every > 0 and step > start_step
                    and (step % args.ckpt_every == 0 or step == args.steps - 1)):
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp = ckpt_path.with_suffix(".tmp")
                torch.save({"banks": banks.state_dict(), "cur_m": cur_m,
                            "opt": opt.state_dict(), "step": step}, tmp)
                tmp.replace(ckpt_path)     # atomic: never a torn checkpoint

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(banks.state_dict(), out_dir / "banks.pt")
        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        ckpt_path.unlink(missing_ok=True)
        print(f"saved to {out_dir}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
