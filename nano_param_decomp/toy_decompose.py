"""VPD vs MPD decomposition on the toy models, in one framework so the comparison is fair.

`mode="vpd"`: the CI MLP emits one gate per atom (per decomposed matrix), masks applied directly.
`mode="mpd"`: the CI MLP emits one gate per component (G); per-atom masks are the membership-weighted
average (the matryoshka machinery). Both use MSE reconstruction to the frozen target model's output,
under stochastic AND fresh-PGD adversarial masks, plus faithfulness + importance-minimality (+ the
membership penalties for MPD).

Ground-truth recovery is scored with MMCS (mean max cosine similarity) between learned rank-1 atoms
and the known mechanisms. Run: CUDA_VISIBLE_DEVICES="" python -m nano_param_decomp.toy_decompose"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# matryoshka (MPD) archived under matryoshka/ -- imported lazily by the mpd mode only
from .run import (
    ComponentLinear,
    clear_wrapper_masks,
    faithfulness_loss,
    importance_minimality_loss,
    install_components,
    lower_leaky,
    set_wrapper_masks,
    upper_leaky,
)


@dataclass
class ToyConfig:
    C_per_module: dict[str, int]
    mode: str  # "vpd" | "mpd"
    n_components: int = 200  # G (mpd only)
    n_steps: int = 5000
    warmup_steps: int = 200
    batch: int = 2048
    lr: float = 3e-3
    warmup_lr: float = 1e-3
    ci_hidden: int = 256
    ci_layers: int = 2
    leaky_alpha: float = 0.01
    # losses
    coeff_imp: float = 1e-2
    coeff_membership: float = 0.0  # CM (mpd)
    coeff_comp_size: float = 0.0  # CCS (mpd)
    coeff_assign_entropy: float = 0.0  # binary-entropy commitment: push each gate to {0,1}, tau-independent
    ae_ramp_frac: float = 0.6  # ramp the commitment coeff 0->full over this fraction of steps (explore early, commit late)
    # lifetime (frequency) minimality: penalize components that are active across MANY datapoints, hard.
    # per-datapoint minimality alone is happy with a few always-on blobs; this taxes consistent activation
    # so the optimum becomes many rarely-active (feature-specific) components instead of few fat ones.
    coeff_lifetime: float = 0.0  # CL
    lifetime_pow: float = 2.0  # convex power on per-component activation frequency (>1 taxes big/frequent hardest)
    lifetime_target: float = 0.0  # if >0, only tax activation ABOVE this rate (hinge): below it costs nothing,
    # so feature-rate components are free and only always-on blobs are penalized -> no collapse-to-silence
    # dual VPD: also run the original per-atom VPD losses on the SHARED atoms alongside the component path,
    # so atom-level structure and component-level grouping influence each other through the shared V/U.
    dual_vpd: bool = False
    dual_mode: str = "simultaneous"  # "simultaneous" (both losses every step) | "alternate" (VPD/MPD on alternating steps)
    dual_vpd_weight: float = 1.0  # scale on the VPD-path loss; <1 lets the component path do more of the reconstructing
    # binding: group atoms by causal INTERACTION (off-diagonal Hessian of recon loss w.r.t. atom masks),
    # not co-activation. Rewards co-membership of high-interaction atoms -> components capture bound
    # circuits (atoms that must be on together), which reconstruction+size alone leave underdetermined.
    coeff_bind: float = 0.0
    bind_every: int = 5  # recompute the interaction matrix every N steps (it needs A backward passes)
    p_start: float = 2.0
    p_end: float = 0.7
    imp_eps: float = 1e-12
    imp_beta: float = 0.5
    # adversarial (fresh PGD per step)
    pgd_steps: int = 4
    pgd_lr: float = 0.05
    coeff_stoch: float = 1.0
    coeff_adv: float = 1.0
    # stochastic-recon-layerwise (SPD eq.): mask ONE layer's atoms at a time (others left at full
    # weight), match the OUTPUT. Ties each layer's components to that layer's contribution instead of
    # allowing output-equivalent-but-internally-wrong solutions. 0 = off.
    coeff_layerwise: float = 0.0
    # tau (mpd): anneal tau_start->tau_end over tau_anneal_frac of steps, then hold
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_frac: float = 0.6
    m_logits_init_std: float = 1.0
    m_logits_init_bias: float = 0.0
    membership_type: str = "sigmoid"  # "sigmoid" | "softmax" (per-atom row sums to 1)
    aggregation: str = "mean"  # "mean" (membership-weighted average) | "max" (atom on if any component on)


def _tau(step: int, cfg: ToyConfig) -> float:
    a = max(1, int(cfg.tau_anneal_frac * cfg.n_steps))
    if step >= a:
        return cfg.tau_end
    prog = step / (a - 1) if a > 1 else 1.0
    return cfg.tau_end + 0.5 * (cfg.tau_start - cfg.tau_end) * (1 + math.cos(math.pi * prog))


class CIMLP(nn.Module):
    """Reads the decomposed matrices' input activations, emits `out_dim` gate logits (atoms for VPD,
    components for MPD). No sequence dim, so a plain MLP replaces the LM CI transformer."""

    def __init__(self, module_order: list[str], d_in: dict[str, int], out_dim: int,
                 hidden: int, n_layers: int, alpha: float) -> None:
        super().__init__()
        self.module_order = module_order
        self.alpha = alpha
        d_total = sum(d_in[n] for n in module_order)
        layers: list[nn.Module] = [nn.Linear(d_total, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, acts: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x = torch.cat([F.rms_norm(acts[n], (acts[n].shape[-1],)) for n in self.module_order], dim=-1)
        logits = self.head(self.trunk(x))
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha), logits


def _split(t: Tensor, cfg: ToyConfig, order: list[str]) -> dict[str, Tensor]:
    return dict(zip(order, t.split([cfg.C_per_module[n] for n in order], dim=-1), strict=True))


def _atom_masks_from_comp(comp_mask: Tensor, assign: "ComponentAssignment", tau: float,
                          mode: str, cfg: ToyConfig, order: list[str]) -> dict[str, Tensor]:
    if mode == "mpd":
        # atom_masks expects [B, S, G]; toy has no sequence dim, so add and drop a dummy S=1
        masks = assign.atom_masks(comp_mask.unsqueeze(1), tau)
        return {n: m.squeeze(1) for n, m in masks.items()}
    return _split(comp_mask, cfg, order)  # vpd: gates already per-atom


def decompose_toy(
    model: nn.Module,
    data_fn: Callable[[], Tensor],
    cfg: ToyConfig,
    device: torch.device,
) -> dict[str, object]:
    order = sorted(cfg.C_per_module)
    wrappers = install_components(model, cfg.C_per_module)
    model = model.to(device)  # the new V/U params are created on CPU; move the whole wrapped model
    d_in = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}
    n_atoms = sum(cfg.C_per_module.values())
    out_dim = cfg.n_components if cfg.mode == "mpd" else n_atoms
    ci = CIMLP(order, d_in, out_dim, cfg.ci_hidden, cfg.ci_layers, cfg.leaky_alpha).to(device)

    assign = None
    if cfg.mode == "mpd":
        from .matryoshka import matryoshka  # archived; only the mpd mode needs it
        from .matryoshka.matryoshka import ComponentAssignment
        mcfg = matryoshka.Config(
            C_per_module=cfg.C_per_module, n_components=cfg.n_components,
            m_logits_init_std=cfg.m_logits_init_std, m_logits_init_bias=cfg.m_logits_init_bias,
            membership_type=cfg.membership_type, aggregation=cfg.aggregation,
        )
        assign = ComponentAssignment(cfg.C_per_module, order, mcfg).to(device)

    # dual VPD path: a SECOND CI net emitting per-atom gates, applying the original VPD losses to the
    # shared atoms. Only meaningful for mpd (vpd already is the atom path).
    ci_atom = None
    if cfg.dual_vpd:
        assert cfg.mode == "mpd", "dual_vpd adds an atom path to the component (mpd) path"
        ci_atom = CIMLP(order, d_in, n_atoms, cfg.ci_hidden, cfg.ci_layers, cfg.leaky_alpha).to(device)

    comp_params = [p for w in wrappers.values() for p in (w.V, w.U)]
    other = list(ci.parameters()) + ([assign.M_logits] if assign is not None else [])
    if ci_atom is not None:
        other += list(ci_atom.parameters())

    # faithfulness warmup: fit V/U to the frozen weights before the main loop
    wopt = torch.optim.AdamW(comp_params, lr=cfg.warmup_lr)
    for _ in range(cfg.warmup_steps):
        loss = faithfulness_loss(wrappers)
        wopt.zero_grad(); loss.backward(); wopt.step()

    opt = torch.optim.AdamW(comp_params + other, lr=cfg.lr)

    def target_forward(x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        clear_wrapper_masks(wrappers)
        out = model(x)
        return out, {n: w.last_input for n, w in wrappers.items()}

    def masked_forward(x: Tensor, masks: dict[str, Tensor], deltas: dict[str, Tensor]) -> Tensor:
        set_wrapper_masks(wrappers, masks, deltas, routing=None)
        try:
            return model(x)
        finally:
            clear_wrapper_masks(wrappers)

    def recon_pair(ci_lower: Tensor, path_mode: str, x: Tensor, target_out: Tensor, tvar: Tensor,
                   tau: float, deltas: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Stochastic + fresh-PGD adversarial reconstruction for one gating path (component or atom)."""
        u = torch.rand_like(ci_lower)
        masks = _atom_masks_from_comp(ci_lower + (1 - ci_lower) * u, assign, tau, path_mode, cfg, order)
        loss_stoch = F.mse_loss(masked_forward(x, masks, deltas), target_out) / tvar
        s = torch.rand_like(ci_lower).requires_grad_(True)
        for _ in range(cfg.pgd_steps):
            cm = ci_lower.detach() + (1 - ci_lower.detach()) * s
            m = _atom_masks_from_comp(cm, assign, tau, path_mode, cfg, order)
            adv = F.mse_loss(masked_forward(x, m, deltas), target_out)
            g = torch.autograd.grad(adv, s)[0]
            s = (s + cfg.pgd_lr * g.sign()).clamp(0, 1).detach().requires_grad_(True)
        cm = ci_lower + (1 - ci_lower) * s.detach()
        masks_adv = _atom_masks_from_comp(cm, assign, tau, path_mode, cfg, order)
        loss_adv = F.mse_loss(masked_forward(x, masks_adv, deltas), target_out) / tvar
        return loss_stoch, loss_adv

    def _layer_idx(name: str) -> int:
        return next((int(t) for t in name.split(".") if t.isdigit()), 0)

    layers_present = sorted({_layer_idx(n) for n in order})

    def layerwise_recon(ci_lower: Tensor, path_mode: str, x: Tensor, target_out: Tensor,
                        tvar: Tensor, tau: float) -> Tensor:
        """Mask one layer's atoms at a time (other layers held at exact target weight), match output.
        SPD's L_stochastic-recon-layerwise (eq. 115): ties each layer's components to its own contribution."""
        u = torch.rand_like(ci_lower)
        masks = _atom_masks_from_comp(ci_lower + (1 - ci_lower) * u, assign, tau, path_mode, cfg, order)
        B = x.shape[0]
        total = torch.zeros((), device=device)
        for lyr in layers_present:
            m_lyr, d_lyr = {}, {}
            for n in order:
                if _layer_idx(n) == lyr:
                    m_lyr[n] = masks[n]
                    d_lyr[n] = torch.rand(B, device=device)
                else:  # other layers exact: mask=1 (full V@U) + delta=1 (full remainder) = W_target
                    m_lyr[n] = torch.ones(B, cfg.C_per_module[n], device=device)
                    d_lyr[n] = torch.ones(B, device=device)
            total = total + F.mse_loss(masked_forward(x, m_lyr, d_lyr), target_out) / tvar
        return total / len(layers_present)

    n_atoms_total = sum(cfg.C_per_module[n] for n in order)

    def interaction_H(x: Tensor, target_out: Tensor, tvar: Tensor) -> Tensor:
        """|d^2 recon / dm_a dm_b| at all-atoms-on, over a batch -> [A, A]. Large off-diagonal = the
        two atoms' causal effects interact (bound circuit); ~0 = separable. Diagonal zeroed."""
        m = torch.ones(n_atoms_total, device=device, requires_grad=True)
        masks, off = {}, 0
        z = {n: torch.zeros(x.shape[0], device=device) for n in order}
        for n in order:
            c = cfg.C_per_module[n]
            masks[n] = m[off:off + c].unsqueeze(0).expand(x.shape[0], c)
            off += c
        loss = F.mse_loss(masked_forward(x, masks, z), target_out) / tvar
        grad = torch.autograd.grad(loss, m, create_graph=True)[0]
        H = torch.stack([torch.autograd.grad(grad[i], m, retain_graph=True)[0] for i in range(n_atoms_total)])
        H = H.abs().detach()
        H.fill_diagonal_(0.0)
        return H

    H_cache: Tensor | None = None

    for step in range(cfg.n_steps):
        tau = _tau(step, cfg)
        p = cfg.p_start + (cfg.p_end - cfg.p_start) * (step / cfg.n_steps)
        x = data_fn().to(device)
        target_out, acts = target_forward(x)
        target_out = target_out.detach()
        tvar = target_out.var() + 1e-8  # normalize recon so it is O(1) and dominates minimality
        ci_lower, ci_upper, _raw = ci(acts)  # [B, out_dim] (components for mpd, atoms for vpd)
        B = x.shape[0]
        deltas = {n: torch.rand(B, device=device) for n in order}

        loss_faith = faithfulness_loss(wrappers)

        # --- main path (component for mpd, atom for vpd) -------------------------------------------
        loss_stoch, loss_adv = recon_pair(ci_lower, cfg.mode, x, target_out, tvar, tau, deltas)
        if cfg.mode == "vpd":
            loss_imp = importance_minimality_loss(_split(ci_upper, cfg, order), p, cfg.imp_eps, cfg.imp_beta, 1)
        else:
            loss_imp = importance_minimality_loss({"comp": ci_upper}, p, cfg.imp_eps, cfg.imp_beta, 1)
        loss_main = cfg.coeff_stoch * loss_stoch + cfg.coeff_adv * loss_adv + cfg.coeff_imp * loss_imp
        if cfg.coeff_layerwise > 0.0:
            loss_main = loss_main + cfg.coeff_layerwise * layerwise_recon(
                ci_lower, cfg.mode, x, target_out, tvar, tau)

        # lifetime (frequency) minimality: tax components active across MANY datapoints, hardest on
        # the always-on blobs. Per-datapoint minimality alone can't tell 4 fat blobs from 100 thin ones.
        loss_lifetime = torch.zeros((), device=device)
        if cfg.coeff_lifetime > 0.0:
            freq = ci_upper.mean(dim=0)  # [out_dim] mean activation per component over the batch
            taxed = (freq - cfg.lifetime_target).clamp(min=0.0) if cfg.lifetime_target > 0.0 else freq
            loss_lifetime = (taxed ** cfg.lifetime_pow).sum()
            loss_main = loss_main + cfg.coeff_lifetime * loss_lifetime

        if assign is not None:
            M = assign.membership(tau)
            if cfg.membership_type == "softmax":  # commit = low ROW entropy -> each atom one-hot
                commit = -(M * (M + 1e-12).log()).sum(dim=1).mean()
            else:  # commit = low BINARY entropy -> each independent gate to {0,1}
                commit = -(M * (M + 1e-12).log() + (1 - M) * (1 - M + 1e-12).log()).mean()
            ae_ramp = min(1.0, step / max(1, int(cfg.ae_ramp_frac * cfg.n_steps)))  # explore early, commit late
            loss_main = (loss_main + cfg.coeff_membership * assign.membership_l1(tau)
                         + cfg.coeff_comp_size * assign.component_size_l2(tau)
                         + cfg.coeff_assign_entropy * ae_ramp * commit)

            # binding: reward co-membership of high-interaction atoms (centered H so low-interaction
            # pairs are pushed APART). This restructures membership toward the bound circuits, which
            # reconstruction is blind to. M Mᵀ[a,b] = how much atoms a,b share components.
            if cfg.coeff_bind > 0.0:
                if H_cache is None or step % cfg.bind_every == 0:
                    H_cache = interaction_H(x, target_out, tvar)
                Hn = H_cache / (H_cache.max() + 1e-8)
                Hc = Hn - Hn.mean()  # center over ALL pairs: high-interaction -> +, the rest -> small neg
                co_membership = M @ M.t()  # [A, A]; softmax rows -> bounded, so reward can't inflate M
                loss_bind = -(Hc * co_membership).mean()
                loss_main = loss_main + cfg.coeff_bind * loss_bind

        # --- dual VPD path: original VPD losses on the shared atoms ---------------------------------
        loss_vpd = torch.zeros((), device=device)
        if ci_atom is not None:
            a_lower, a_upper, _ = ci_atom(acts)  # [B, A]
            a_stoch, a_adv = recon_pair(a_lower, "vpd", x, target_out, tvar, tau, deltas)
            a_imp = importance_minimality_loss(_split(a_upper, cfg, order), p, cfg.imp_eps, cfg.imp_beta, 1)
            loss_vpd = cfg.coeff_stoch * a_stoch + cfg.coeff_adv * a_adv + cfg.coeff_imp * a_imp

        # simultaneous: both pressures every step. alternate: VPD on even steps, MPD on odd, sharing atoms.
        if ci_atom is not None and cfg.dual_mode == "alternate":
            loss = 1e4 * loss_faith + (cfg.dual_vpd_weight * loss_vpd if step % 2 == 0 else loss_main)
        else:
            loss = 1e4 * loss_faith + loss_main + cfg.dual_vpd_weight * loss_vpd

        opt.zero_grad(); loss.backward(); opt.step()

        if step % max(1, cfg.n_steps // 5) == 0 or step == cfg.n_steps - 1:
            print(f"  step {step:>5} tau={tau:.2f} faith={loss_faith.item():.2e} "
                  f"stoch={loss_stoch.item():.4f} adv={loss_adv.item():.4f} imp={loss_imp.item():.3f} "
                  f"life={loss_lifetime.item():.4f} vpd={loss_vpd.item():.4f}", flush=True)

    # final eval: ci-masked reconstruction (faithfulness) + L0 (active units per datapoint)
    final_tau = _tau(cfg.n_steps - 1, cfg)
    out: dict[str, object] = {"wrappers": wrappers, "ci": ci, "assign": assign, "tau": final_tau}
    with torch.no_grad():
        x = data_fn().to(device)
        target_out, acts = target_forward(x)
        tvar = target_out.var() + 1e-8
        ci_lower, _u, _r = ci(acts)
        B = x.shape[0]
        zeros = {n: torch.zeros(B, device=device) for n in order}
        masks = _atom_masks_from_comp(ci_lower, assign, final_tau, cfg.mode, cfg, order)
        out["recon_ci"] = (F.mse_loss(masked_forward(x, masks, zeros), target_out) / tvar).item()
        out["l0"] = (ci_lower > 0.5).float().sum(dim=-1).mean().item()

        # brackets for the (compressed) recon scale: all atoms off (passthrough) vs all atoms on (faithful)
        all_off = {n: torch.zeros_like(m) for n, m in masks.items()}
        all_on = {n: torch.ones_like(m) for n, m in masks.items()}
        out["recon_off"] = (F.mse_loss(masked_forward(x, all_off, zeros), target_out) / tvar).item()
        out["recon_on"] = (F.mse_loss(masked_forward(x, all_on, zeros), target_out) / tvar).item()

        if ci_atom is not None:  # the VPD side: gate atoms directly, no membership in the way
            a_lower, _au, _ar = ci_atom(acts)
            a_masks = _split(a_lower, cfg, order)
            out["recon_atom"] = (F.mse_loss(masked_forward(x, a_masks, zeros), target_out) / tvar).item()
            out["l0_atom"] = (a_lower > 0.5).float().sum(dim=-1).mean().item()
        if assign is not None:  # is the membership blurry? hardness=1 means every gate is a clean 0/1
            M = assign.membership(final_tau)
            out["m_hardness"] = torch.maximum(M, 1 - M).mean().item()
            out["m_mean"] = M.mean().item()  # average gate value; ~0.5 = maximally blurry
            out["atom_degree"] = (M > 0.5).float().sum(dim=1).mean().item()  # components an atom commits to
    return out


def atom_weight_matrices(w: ComponentLinear) -> Tensor:
    """Each atom's rank-1 weight matrix [C, d_out, d_in] = U[c] outer V[:,c]."""
    V, U = w.V.detach(), w.U.detach()  # [d_in, C], [C, d_out]
    return torch.einsum("co,ic->coi", U, V)  # [C, d_out, d_in]


def mmcs(learned: Tensor, ground_truth: Tensor) -> float:
    """Mean over ground-truth mechanisms of the max cosine similarity to any learned atom.
    Both [n, d_out, d_in] (flattened for cosine)."""
    L = F.normalize(learned.flatten(1), dim=1)  # [C, D]
    G = F.normalize(ground_truth.flatten(1), dim=1)  # [n, D]
    sims = G @ L.t()  # [n, C]
    return sims.max(dim=1).values.mean().item()


def _run_tms() -> None:
    import copy
    import os

    from .toy_models import TMS, feature_batch, train_tms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, nh = 5, 2
    steps = int(os.environ.get("STEPS", "10000"))
    adv = float(os.environ.get("ADV", "1.0"))  # coeff_adv (0 = stochastic-only, i.e. SPD-style)
    fprob = float(os.environ.get("FPROB", "0.05"))
    cm = float(os.environ.get("CM", "0.3"))
    modes = os.environ.get("MODES", "vpd,mpd").split(",")
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
    gt = base.ground_truth().to(device)  # [nf, nh, nf]
    gen = torch.Generator(device=device).manual_seed(123)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    print(f"config: steps={steps} adv={adv} fprob={fprob} cm={cm} modes={modes}", flush=True)
    for mode in modes:
        model = copy.deepcopy(base)
        model.freeze_for_decomposition()
        cfg = ToyConfig(
            C_per_module={"W": 20}, mode=mode, n_components=20, n_steps=steps, warmup_steps=500,
            batch=2048, lr=3e-3, coeff_imp=3e-2, coeff_adv=adv,
            coeff_membership=(cm if mode == "mpd" else 0.0),
        )
        print(f"\n=== {mode.upper()} ===", flush=True)
        out = decompose_toy(model, data_fn, cfg, device)
        learned = atom_weight_matrices(out["wrappers"]["W"])  # [20, nh, nf]
        print(f"{mode.upper()} MMCS to ground truth = {mmcs(learned, gt):.4f}  (1.0 = perfect)", flush=True)
    print("TMS_DECOMPOSE DONE", flush=True)


def _run_resid() -> None:
    """Cross-layer residual MLP: matrices used once (clean recovery), ground truth spans layers.
    Reports recon faithfulness + sparsity for both modes, and for MPD the cross-layer grouping that
    VPD cannot produce without a separate clustering pass."""
    import copy
    import os

    from .toy_models import ResidMLP, feature_batch, target_fn, train_resid_mlp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, d_embed, d_mlp, n_layers = 100, 256, 40, 2
    fprob = 0.01
    steps = int(os.environ.get("STEPS", "8000"))
    cm = float(os.environ.get("CM", "0.1"))
    ae = float(os.environ.get("AE", "0.1"))  # binary-entropy commitment coeff
    modes = os.environ.get("MODES", "vpd,mpd").split(",")
    ckpt = "/tmp/toy/resid_2l.pt"
    os.makedirs("/tmp/toy", exist_ok=True)

    if os.path.exists(ckpt):
        base = ResidMLP(nf, d_embed, d_mlp, n_layers, seed=0).to(device)
        base.load_state_dict(torch.load(ckpt, weights_only=True))
        print("loaded cached resid-MLP target", flush=True)
    else:
        print("training 2-layer cross-layer resid-MLP target ...", flush=True)
        base = train_resid_mlp(nf, d_embed, d_mlp, n_layers, steps=8000, batch=2048,
                               feature_prob=fprob, lr=3e-3, device=device, seed=0)
        torch.save(base.state_dict(), ckpt)

    gen = torch.Generator(device=device).manual_seed(123)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    # per-matrix atom budget: in_proj out-dim = d_mlp_layer, out_proj out-dim = d_embed; keep modest
    dml = d_mlp // n_layers
    cpm = {}
    for i in range(n_layers):
        cpm[f"blocks.{i}.in_proj"] = dml * 2
        cpm[f"blocks.{i}.out_proj"] = dml * 2
    print(f"config: steps={steps} cm={cm} modes={modes} | atoms/matrix shown below", flush=True)

    results = {}
    for mode in modes:
        model = copy.deepcopy(base)
        cfg = ToyConfig(
            C_per_module=cpm, mode=mode, n_components=200, n_steps=steps, warmup_steps=500,
            batch=2048, lr=3e-3, coeff_imp=3e-2,
            tau_start=1.0, tau_end=1.0,  # constant tau: commitment comes from the entropy loss, not annealing
            coeff_membership=(cm if mode == "mpd" else 0.0),
            coeff_assign_entropy=(ae if mode == "mpd" else 0.0),
        )
        print(f"\n=== {mode.upper()} ===", flush=True)
        out = decompose_toy(model, data_fn, cfg, device)
        results[mode] = out

    print(f"\n{'='*60}\nCROSS-LAYER RESID MLP  VPD vs MPD  (steps={steps})\n{'='*60}")
    print(f"{'metric':<28} {'VPD':>14} {'MPD':>14}")
    v, m = results.get("vpd"), results.get("mpd")

    def row(label, key, fmt="{:.4f}"):
        vs = fmt.format(v[key]) if v and key in v else "-"
        ms = fmt.format(m[key]) if m and key in m else "-"
        print(f"{label:<28} {vs:>14} {ms:>14}")

    row("recon (ci-masked, norm)", "recon_ci")
    row("L0 units/datapoint", "l0", "{:.1f}")
    if m and m["assign"] is not None:
        xl = m["assign"].cross_layer_stats(m["tau"])
        print(f"{'MPD used components':<28} {'-':>14} {xl['n_used']:>14.0f}")
        print(f"{'MPD atoms/component':<28} {'-':>14} {xl['mean_comp_size']:>14.2f}")
        print(f"{'MPD cross-layer frac':<28} {'-':>14} {xl['crosslayer_frac']:>14.2f}")
    print("="*60)
    print("RESID_DECOMPOSE DONE", flush=True)


def _resid_setup(device: torch.device):
    """Load cached cross-layer resid-MLP target + its decomposition layout + data sampler."""
    from .toy_models import ResidMLP, feature_batch

    nf, d_embed, d_mlp, n_layers, fprob = 100, 256, 60, 2, 0.01  # d_mlp=60 (30/layer): crisper retrained target
    ckpt = "/tmp/toy/resid_2l.pt"
    assert __import__("os").path.exists(ckpt), "run TARGET=resid once to train+cache the target first"
    base = ResidMLP(nf, d_embed, d_mlp, n_layers, seed=0).to(device)
    base.load_state_dict(torch.load(ckpt, weights_only=True))
    gen = torch.Generator(device=device).manual_seed(123)
    dml = d_mlp // n_layers
    cpm = {f"blocks.{i}.{p}": dml * 2 for i in range(n_layers) for p in ("in_proj", "out_proj")}
    batch = int(__import__("os").environ.get("BATCH", "2048"))
    return base, cpm, (lambda: feature_batch(nf, batch, fprob, device, gen)), dml


def _sweep_one() -> None:
    """One config -> one tagged RESULT line. Driven by env: MODE, AE, CM, CCS, IMP, CL (lifetime),
    LIFEPOW, DUAL (none|sim|alt), MEMTYPE, STEPS."""
    import copy
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, cpm, data_fn, _dml = _resid_setup(device)
    mode = os.environ.get("MODE", "mpd")
    ae, cm, imp = float(os.environ["AE"]), float(os.environ["CM"]), float(os.environ["IMP"])
    ccs = float(os.environ.get("CCS", "0"))  # component-size tax
    cl = float(os.environ.get("CL", "0"))  # lifetime (frequency) minimality
    lifepow = float(os.environ.get("LIFEPOW", "2"))
    life_target = float(os.environ.get("LIFE_TARGET", "0"))  # hinge: only tax activation above this rate
    dual = os.environ.get("DUAL", "none")  # none | sim | alt
    memtype = os.environ.get("MEMTYPE", "sigmoid")
    agg = os.environ.get("AGG", "mean")  # mean | max
    dualw = float(os.environ.get("DUALW", "1"))  # scale on the VPD-path loss in dual mode
    steps = int(os.environ.get("STEPS", "6000"))
    cfg = ToyConfig(
        C_per_module=cpm, mode=mode, n_components=200, n_steps=steps, warmup_steps=500,
        batch=2048, lr=3e-3, coeff_imp=imp, tau_start=1.0, tau_end=1.0, membership_type=memtype, aggregation=agg,
        coeff_membership=(cm if mode == "mpd" else 0.0),
        coeff_comp_size=(ccs if mode == "mpd" else 0.0),
        coeff_assign_entropy=(ae if mode == "mpd" else 0.0),
        coeff_lifetime=(cl if mode == "mpd" else 0.0), lifetime_pow=lifepow, lifetime_target=life_target,
        dual_vpd=(dual != "none"),
        dual_mode=("alternate" if dual == "alt" else "simultaneous"),
        dual_vpd_weight=dualw,
    )
    out = decompose_toy(copy.deepcopy(base), data_fn, cfg, device)
    if out["assign"] is not None:
        xl = out["assign"].cross_layer_stats(out["tau"])
        used, apc, xlf = xl["n_used"], xl["mean_comp_size"], xl["crosslayer_frac"]
    else:
        used, apc, xlf = -1, -1, -1
    print(f"RESULT agg={agg} mem={memtype} mode={mode} dual={dual} ae={ae} cm={cm} ccs={ccs} imp={imp} cl={cl} "
          f"recon={out['recon_ci']:.4f} l0={out['l0']:.2f} used={used:.0f} apc={apc:.2f} xl={xlf:.2f}", flush=True)


def _diag_dual() -> None:
    """Break down WHY dual-VPD reconstructs poorly: compare the component side (membership-gated, the
    metric we score) against the atom side (gated directly) for both simultaneous and alternate, and
    check whether the membership table is blurry. DUAL env picks sim|alt|both (default both)."""
    import copy
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, cpm, data_fn, _dml = _resid_setup(device)
    steps = int(os.environ.get("STEPS", "10000"))
    which = os.environ.get("DUAL", "both")
    modes = ["sim", "alt"] if which == "both" else [which]
    for dual in modes:
        cfg = ToyConfig(
            C_per_module=cpm, mode="mpd", n_components=200, n_steps=steps, warmup_steps=500,
            batch=2048, lr=3e-3, coeff_imp=0.01, tau_start=1.0, tau_end=1.0, membership_type="sigmoid",
            coeff_assign_entropy=0.05, coeff_comp_size=0.01,
            dual_vpd=True, dual_mode=("alternate" if dual == "alt" else "simultaneous"),
        )
        o = decompose_toy(copy.deepcopy(base), data_fn, cfg, device)
        xl = o["assign"].cross_layer_stats(o["tau"])
        print(f"\n=== DUAL={dual} (steps={steps}) ===", flush=True)
        print(f"  recon: passthrough(all-off)={o['recon_off']:.3f}  all-on(faithful)={o['recon_on']:.3f}", flush=True)
        print(f"  recon COMPONENT side (scored) = {o['recon_ci']:.3f}   active comps/input = {o['l0']:.2f}", flush=True)
        print(f"  recon ATOM side (VPD, direct) = {o['recon_atom']:.3f}   active atoms/input = {o['l0_atom']:.2f}", flush=True)
        print(f"  membership hardness = {o['m_hardness']:.3f} (1.0=clean 0/1, 0.5=fully blurry)", flush=True)
        print(f"  membership mean gate = {o['m_mean']:.3f}   components per atom (M>0.5) = {o['atom_degree']:.1f}", flush=True)
        print(f"  used comps = {xl['n_used']:.0f}  atoms/comp = {xl['mean_comp_size']:.1f}  cross-layer = {xl['crosslayer_frac']:.2f}", flush=True)
    print("DIAG_DUAL_DONE", flush=True)


def _distinct_check() -> None:
    """Low size penalty fills all 200 component slots, but are they ~100 DISTINCT cross-layer
    mechanisms wearing 200 hats, or 200 genuinely different things? Train the low-penalty config,
    then drive ONE feature at a time and read which components switch on. A component that fires for
    exactly one feature is that feature's mechanism; count distinct features covered + redundancy."""
    import copy
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, cpm, data_fn, _dml = _resid_setup(device)
    mode = os.environ.get("MODE", "mpd")  # mpd: probe component gates | vpd: probe per-atom gates
    ccs = float(os.environ.get("CCS", "1.0"))
    steps = int(os.environ.get("STEPS", "5000"))
    model = copy.deepcopy(base)
    cfg = ToyConfig(
        C_per_module=cpm, mode=mode, n_components=200, n_steps=steps, warmup_steps=500,
        batch=2048, lr=3e-3, coeff_imp=0.001, tau_start=1.0, tau_end=1.0, membership_type="sigmoid",
        aggregation="max", coeff_assign_entropy=(0.05 if mode == "mpd" else 0.0),
        coeff_comp_size=(ccs if mode == "mpd" else 0.0),
    )
    out = decompose_toy(model, data_fn, cfg, device)
    wrappers, ci, assign = out["wrappers"], out["ci"], out["assign"]
    nf = 100

    # probe on NATURAL inputs (in-distribution): correlate each component's activation with each
    # feature's activation across many sparse inputs. Robust to the one-hot off-distribution issue.
    from .toy_models import feature_batch
    gen = torch.Generator(device=device).manual_seed(7)
    with torch.no_grad():
        X = feature_batch(nf, 20000, 0.01, device, gen)  # [N, nf] sparse, value U[0,1] when active
        clear_wrapper_masks(wrappers)
        _ = model(X)
        acts = {n: w.last_input for n, w in wrappers.items()}
        G, _u, _r = ci(acts)  # [N, comps] component gate per input

    feat_on = (X > 0).float()  # [N, nf]
    freq = (G > 0.5).float().mean(dim=0)  # [comps] how often each component fires across inputs
    ever = freq > 0  # fires at least sometimes
    # correlation(component activation, feature presence) over inputs
    Gc = G - G.mean(0, keepdim=True)
    Fc = feat_on - feat_on.mean(0, keepdim=True)
    corr = (Gc.t() @ Fc) / (Gc.std(0).unsqueeze(1) * Fc.std(0).unsqueeze(0) * len(X) + 1e-8)  # [comps, nf]
    best_corr, best_feat = corr.max(dim=1)  # each unit's best-matching feature
    xlf = assign.cross_layer_stats(out["tau"])["crosslayer_frac"] if assign is not None else -1.0

    active = ever.nonzero().squeeze(-1)
    selective = active[best_corr[active] > 0.5]  # unit tracks one feature tightly
    feats_covered = int(best_feat[selective].unique().numel())
    always_on = int((freq > 0.5).sum())  # near-always-on units (carry bulk, not feature-specific)
    unit, total = ("components", G.shape[1]) if mode == "mpd" else ("atoms", G.shape[1])

    print(f"\n=== distinct-mechanism check (mode={mode}, recon={out['recon_ci']:.3f}, "
          f"cross-layer={xlf:.2f}) [NATURAL inputs] ===", flush=True)
    print(f"{unit} that ever fire (>0.5 sometimes): {int(ever.sum())} / {total}", flush=True)
    print(f"  ...of those, near-always-on (freq>0.5)  : {always_on}  (bulk carriers, not feature-specific)", flush=True)
    print(f"components tightly tracking ONE feature   : {int(selective.numel())}", flush=True)
    print(f"distinct features so covered              : {feats_covered} / {nf}", flush=True)
    print(f"mean best feature-correlation (active)    : {best_corr[active].mean():.2f}", flush=True)
    print(f"component fire-frequency  min/med/max     : "
          f"{freq[active].min():.3f}/{freq[active].median():.3f}/{freq[active].max():.3f}", flush=True)
    print("DISTINCT_CHECK_DONE", flush=True)


def _run_bound() -> None:
    """Bound-circuit toy: each mechanism = a multiplicatively-bound cross-layer atom pair that does NOT
    co-activate. Does the interaction-driven binding term recover the M pairs as components (where plain
    MPD can't)? Recovery = used~M, atoms/comp~2, cross-layer~1, each component tracks one mechanism."""
    import copy
    import os

    from .toy_models import BoundPairs, feature_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = int(os.environ.get("M", "8"))
    nf = 2 * M
    steps = int(os.environ.get("STEPS", "6000"))
    bind = float(os.environ.get("BIND", "0"))
    ccs = float(os.environ.get("CCS", "1.0"))
    memtype = os.environ.get("MEMTYPE", "softmax")  # softmax: row sums to 1 -> co-membership bounded, no inflation
    ae = float(os.environ.get("AE", "0.05"))
    imp = float(os.environ.get("IMP", "0.003"))
    mode = os.environ.get("MODE", "mpd")  # vpd: per-atom sparsity (does it sharpen H?) | mpd: + components
    gen = torch.Generator(device=device).manual_seed(123)
    data_fn = lambda: feature_batch(nf, 2048, 0.35, device, gen)  # dense enough that pairs co-fire
    cpm = {"blocks.0.proj": M, "blocks.1.proj": M}  # exact-rank atoms (one per mechanism per layer)

    cfg = ToyConfig(
        C_per_module=cpm, mode=mode, n_components=2 * M, n_steps=steps, warmup_steps=500,
        batch=2048, lr=3e-3, coeff_imp=imp, tau_start=1.0, tau_end=1.0, membership_type=memtype,
        aggregation="max", coeff_assign_entropy=(ae if mode == "mpd" else 0.0),
        coeff_comp_size=(ccs if mode == "mpd" else 0.0), coeff_bind=(bind if mode == "mpd" else 0.0),
    )
    model = BoundPairs(M).to(device)  # the instance that gets wrapped in-place by decompose_toy
    out = decompose_toy(model, data_fn, cfg, device)
    wrappers, ci, assign = out["wrappers"], out["ci"], out["assign"]
    order_b = sorted(cpm)

    # is the interaction matrix clean on the LEARNED atoms? each atom's strongest interaction partner
    # should be its cross-layer mechanism partner (the other layer). If muddy, no grouping loss can work.
    A = 2 * M
    Xh = data_fn().to(device)
    mvar = torch.ones(A, device=device, requires_grad=True)
    clear_wrapper_masks(wrappers)
    outf = model(Xh).detach()
    masks, off = {}, 0
    z = {n: torch.zeros(Xh.shape[0], device=device) for n in order_b}
    for n in order_b:
        masks[n] = mvar[off:off + cpm[n]].unsqueeze(0).expand(Xh.shape[0], cpm[n]); off += cpm[n]
    set_wrapper_masks(wrappers, masks, z, routing=None)
    Lh = F.mse_loss(model(Xh), outf)
    clear_wrapper_masks(wrappers)
    gh = torch.autograd.grad(Lh, mvar, create_graph=True)[0]
    Hl = torch.stack([torch.autograd.grad(gh[i], mvar, retain_graph=True)[0] for i in range(A)]).abs().detach()
    Hl.fill_diagonal_(0.0)
    ar = torch.arange(A, device=device)
    top = Hl.argmax(dim=1)
    is_l0 = ar < M
    cross_partner = ((top < M) != is_l0).float().mean().item()  # strongest partner in the OTHER layer?
    second = Hl.sort(dim=1, descending=True).values[:, 1]
    sharp = (Hl.max(dim=1).values / (second + 1e-9)).median().item()
    mutual_cross = (top[top] == ar) & ((top < M) != is_l0)  # a<->b mutual top, in different layers
    n_pairs = int(mutual_cross.sum().item() // 2)  # bound pairs recovered by greedy H-pairing
    print(f"  H-on-learned-atoms: cross-partner-frac={cross_partner:.2f} sharpness={sharp:.1f} "
          f"GREEDY-PAIRS-RECOVERED={n_pairs}/{M}", flush=True)

    # does each component track exactly one mechanism? probe component gates vs mechanism activity.
    with torch.no_grad():
        X = feature_batch(nf, 16384, 0.35, device, gen)
        clear_wrapper_masks(wrappers)
        _ = model(X)
        acts = {n: w.last_input for n, w in wrappers.items()}
        G, _u, _r = ci(acts)  # [N, comps]
        mech = model.mech_active(X)  # [N, M]
    Gc = G - G.mean(0, keepdim=True)
    Mc = mech - mech.mean(0, keepdim=True)
    corr = (Gc.t() @ Mc) / (Gc.std(0).unsqueeze(1) * Mc.std(0).unsqueeze(0) * len(X) + 1e-8)  # [comps, M]
    ever = (G > 0.5).float().mean(0) > 0
    best = corr.abs().max(dim=1).values
    selective = ever & (best > 0.5)
    mechs_tracked = int(corr.abs().argmax(dim=1)[selective].unique().numel())

    if assign is not None:
        xl = assign.cross_layer_stats(out["tau"])
        used, apc, xlf = xl["n_used"], xl["mean_comp_size"], xl["crosslayer_frac"]
    else:
        used, apc, xlf = -1.0, -1.0, -1.0  # vpd: per-atom, no components
    print(f"RESULT mode={mode} bind={bind} imp={imp} recon={out['recon_ci']:.4f} l0={out['l0']:.2f} "
          f"used={used:.0f} atoms/comp={apc:.2f} xlayer={xlf:.2f} "
          f"mechs_tracked={mechs_tracked}/{M} | H_sharpness={sharp:.1f} cross_partner={cross_partner:.2f}", flush=True)


if __name__ == "__main__":
    import os

    task = os.environ.get("TASK", "")
    if task == "sweep_one":
        _sweep_one()
    elif task == "diag_dual":
        _diag_dual()
    elif task == "distinct":
        _distinct_check()
    elif task == "bound":
        _run_bound()
    elif os.environ.get("TARGET", "tms") == "resid":
        _run_resid()
    else:
        _run_tms()
