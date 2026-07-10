"""APD-basis decomposition of AlgZoo RNNs (ARC's alg-zoo, tiny algorithmic models).

Targets: the HANDCRAFTED 2nd-argmax RNNs first -- ground truth by construction. The seq-10 model
(726 params, 22 hidden neurons) has documented per-neuron semantics in alg_zoo/handcrafted.py:
  neurons 0-9   delay line: max(0, x_{t-10}), ..., max(0, x_{t-1})
  neurons 10-18 leave-one-out running maxima (minus x_{t-1})
  neurons 19-21 running max machinery (increment / prefix max / prefix max - last)
  logit i = max(0, x_i) - 2 * max(0, leave-one-out max_i)

Why this target matters: activity is DENSE (every input runs the whole algorithm), so per-datapoint
minimality can't separate mechanisms by WHEN they fire -- the regime that produced blobs on the LM,
now at 726 params. Mechanism identity must come from ROLE (which timesteps, which neurons).

Decomposition: whole-network components spanning {ih, hh, out}, ONE shared gate per component,
varying PER TIMESTEP (the RNN analog of the LM's per-position gate; the hh bank is the same matrix
reused every step, so a "cross-layer" mechanism here is a cross-TIMESTEP one). Machinery mirrors
apd_mask/apd_lm: factored banks (rank-1 default, the current best operating point), faithfulness,
stochastic + fresh-PGD adversarial KL recon, importance minimality (+ optional lifetime).

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.apd_alg          # handcrafted seq-10
Env:  MODEL=hc10|hc3, C, R, STEPS, B, IMP, LIFE, RAMP, SIMP, SEED, SAVE
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .apd_mask import (
    ApdConfig,
    ComponentBankLinear,
    clear_masks,
    faithfulness_loss,
    install_banks,
    refresh_caches,
    simplicity_loss,
)
from .run import (
    CIBlock,
    Config as VpdConfig,
    importance_minimality_loss,
    kl_logits,
    lower_leaky,
    precompute_rope,
    upper_leaky,
)

sys.path.insert(0, "/workspace/alg-zoo")


# --- Unrolled RNN target (explicit nn.Linear so banks can hook in) ----------------------------------


class UnrolledRNN(nn.Module):
    """DistRNN rewritten with explicit Linears: h_t = ReLU(ih(x_t) + hh(h_{t-1})), out(h_T).

    Per-timestep gating: when `gates` [B,T,C] is set (banks in component mode), each timestep t
    assigns gates[:,t] to the ih/hh banks before stepping; `out` gets the final step's gate.
    In target mode the forward records `trace_h_prev` [B,T,h] for the CI net.
    """

    def __init__(self, hidden_size: int, seq_len: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.ih = nn.Linear(1, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, seq_len, bias=False)
        self.gates: Tensor | None = None            # [B, T, C]
        self.delta_gates: dict[str, Tensor] | None = None  # name -> [B, T]
        self.trace_h_prev: Tensor | None = None      # [B, T, h] (target mode)

    @classmethod
    def from_dist_rnn(cls, m) -> "UnrolledRNN":
        u = cls(m.hidden_size, m.output_size)
        with torch.no_grad():
            u.ih.weight.copy_(m.rnn.weight_ih_l0)
            u.hh.weight.copy_(m.rnn.weight_hh_l0)
            u.out.weight.copy_(m.linear.weight)
        return u

    def _set_step_masks(self, t: int, last: bool) -> None:
        assert self.gates is not None
        for name in ("ih", "hh") if not last else ("ih", "hh", "out"):
            bank = getattr(self, name)
            g = self.gates[:, -1] if (name == "out") else self.gates[:, t]
            bank.mask = g
            if self.delta_gates is not None:
                bank.delta_mask = self.delta_gates[name][:, -1 if name == "out" else t]

    def forward(self, x: Tensor) -> Tensor:  # x [B, T]
        B, T = x.shape
        h = x.new_zeros(B, self.hidden_size)
        gated = self.gates is not None
        h_prevs = [] if not gated else None
        h_states = []  # post-ReLU h_t; graph-connected when gated (for hidden-trajectory recon)
        for t in range(T):
            if gated:
                self._set_step_masks(t, last=(t == T - 1))
            else:
                if h_prevs is not None:
                    h_prevs.append(h.detach())
            h = F.relu(self.ih(x[:, t, None]) + self.hh(h))
            h_states.append(h if gated else h.detach())
        if h_prevs is not None:
            self.trace_h_prev = torch.stack(h_prevs, dim=1)  # [B, T, h]
        self.trace_h = torch.stack(h_states, dim=1)          # [B, T, h]
        return self.out(h)


def masked_forward_rnn(model: UnrolledRNN, banks: dict[str, ComponentBankLinear], x: Tensor,
                       gates: Tensor, deltas: dict[str, Tensor] | None) -> Tensor:
    """Component-mode forward with a per-timestep shared gate [B,T,C]."""
    for b in banks.values():
        b.mode = "component"
    model.gates = gates
    model.delta_gates = deltas
    try:
        return model(x)
    finally:
        model.gates = None
        model.delta_gates = None
        clear_masks(banks)


# --- CI: bidirectional transformer over timesteps, one gate per component per step ------------------


class AlgCI(nn.Module):
    """Reads per-timestep [x_t, rms-normed h_{t-1}] and emits gates [B, T, C]. Bidirectional
    (2nd-argmax is a global property of the sequence), tiny."""

    def __init__(self, hidden_size: int, seq_len: int, C: int, cfg: VpdConfig) -> None:
        super().__init__()
        self.alpha = cfg.leaky_alpha
        self.proj_in = nn.Linear(1 + hidden_size, cfg.ci_d_model)
        self.blocks = nn.ModuleList([CIBlock(cfg) for _ in range(cfg.ci_n_blocks)])
        self.proj_out = nn.Linear(cfg.ci_d_model, C)
        head_dim = cfg.ci_d_model // cfg.ci_n_heads
        cos, sin = precompute_rope(seq_len, head_dim, cfg.ci_rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor, h_prev: Tensor) -> tuple[Tensor, Tensor]:
        # x [B,T] raw (N(0,1) scale); h_prev [B,T,h] rms-normed so hc3's lam~100 magnitudes
        # don't swamp the input feature.
        hn = F.rms_norm(h_prev, (h_prev.shape[-1],))
        z = self.proj_in(torch.cat([x[..., None], hn], dim=-1))
        S = z.shape[1]
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        for blk in self.blocks:
            z = blk(z, cos, sin)
        logits = self.proj_out(z)  # [B, T, C]
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha)


# --- decomposition -----------------------------------------------------------------------------------


def _D(pred: Tensor, target: Tensor, kind: str) -> Tensor:
    """Reconstruction divergence. "kl" matches the LM setup; "mse" (normalized by target variance)
    matches APD/SPD's non-LM toys and preserves near-tie logit MARGINS, which KL is blind to --
    2nd-argmax flips at tiny margins, so KL~0.08 can coexist with broken task accuracy."""
    if kind == "kl":
        return kl_logits(pred, target)
    return F.mse_loss(pred, target) / (target.var() + 1e-8)


def decompose_alg(model: UnrolledRNN, cfg: ApdConfig, ci_cfg: VpdConfig, device: torch.device,
                  batch: int, recon: str = "kl", log_every: int = 500,
                  save_path: str | None = None) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    banks = install_banks(model, cfg)
    model = model.to(device)
    T = model.seq_len
    ci = AlgCI(model.hidden_size, T, cfg.n_components, ci_cfg).to(device)
    comp_params = [p for b in banks.values() for p in b.params()]

    wopt = torch.optim.AdamW(comp_params, lr=cfg.warmup_lr)
    for _ in range(cfg.warmup_steps):
        refresh_caches(banks)
        loss = faithfulness_loss(banks)
        wopt.zero_grad(); loss.backward(); wopt.step()
    print(f"faithfulness after warmup: {loss.item():.3e}", flush=True)

    opt = torch.optim.AdamW(comp_params + list(ci.parameters()), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 7)

    for step in range(cfg.n_steps):
        p = cfg.p_start + (cfg.p_end - cfg.p_start) * (step / cfg.n_steps)
        x = torch.randn(batch, T, device=device, generator=gen)
        clear_masks(banks)
        with torch.no_grad():
            target = model(x)
        h_prev = model.trace_h_prev
        target_h = model.trace_h  # [B, T, h] detached target hidden trajectory
        refresh_caches(banks)
        g_lower, g_upper = ci(x, h_prev)  # [B, T, C]
        deltas = ({n: torch.rand(batch, T, device=device) for n in banks}
                  if cfg.use_delta else None)

        loss_faith = faithfulness_loss(banks)
        u = torch.rand_like(g_lower)
        loss_stoch = _D(masked_forward_rnn(model, banks, x, g_lower + (1 - g_lower) * u, deltas), target, recon)
        # hidden-trajectory recon (APD L_hidden, RNN form): masked h_t must match the target's h_t at
        # EVERY timestep -- per-neuron routing pressure; also kills the late-timestep "ramp cheat".
        loss_hidden = torch.zeros((), device=device)
        if cfg.coeff_hidden > 0:
            loss_hidden = F.mse_loss(model.trace_h, target_h) / (target_h.var() + 1e-8)
        # fresh PGD on the per-timestep shared gate (toy-style; targets are tiny)
        s = torch.rand_like(g_lower).requires_grad_(True)
        for _ in range(cfg.pgd_steps):
            gm = g_lower.detach() + (1 - g_lower.detach()) * s
            adv = _D(masked_forward_rnn(model, banks, x, gm, deltas), target, recon)
            gr = torch.autograd.grad(adv, s)[0]
            s = (s + cfg.pgd_lr * gr.sign()).clamp(0, 1).detach().requires_grad_(True)
        loss_adv = _D(masked_forward_rnn(model, banks, x, g_lower + (1 - g_lower) * s.detach(), deltas), target, recon)
        loss_imp = importance_minimality_loss({"g": g_upper}, p, cfg.imp_eps, cfg.imp_beta, 1)
        loss_simp = (simplicity_loss(banks, g_upper.mean(dim=(0, 1)), cfg)
                     if cfg.coeff_simplicity > 0 else torch.zeros((), device=device))
        loss_inter = torch.zeros((), device=device)
        if cfg.coeff_interaction > 0:
            probs = g_upper.mean(dim=(0, 1)).detach() + 1e-6
            n = min(2 * cfg.interaction_pairs, cfg.n_components)
            idx = torch.multinomial(probs, n, replacement=False)
            pairs = idx[: (n // 2) * 2].view(-1, 2)

            def _abl(ablate: list[int]) -> Tensor:
                g = g_lower.clone()
                g[..., ablate] = 0.0
                return _D(masked_forward_rnn(model, banks, x, g, None), target, recon)

            l_base = _abl([])
            for i, j in pairs.tolist():
                loss_inter = loss_inter + F.relu(_abl([i, j]) - _abl([i]) - _abl([j]) + l_base)
            loss_inter = loss_inter / max(1, len(pairs))

        loss_l1 = torch.zeros((), device=device)
        if cfg.coeff_weight_l1 > 0:  # W_cache materialized by refresh_caches (lowrank_forward=False)
            loss_l1 = sum(b.W_cache.abs().sum() for b in banks.values())

        loss_life = torch.zeros((), device=device)
        if cfg.coeff_lifetime > 0:
            life_c = cfg.coeff_lifetime
            if cfg.lifetime_ramp_frac > 0:
                life_c *= min(1.0, step / max(1, int(cfg.lifetime_ramp_frac * cfg.n_steps)))
            loss_life = life_c * g_upper.mean(dim=(0, 1)).pow(cfg.lifetime_pow).sum()

        loss = (cfg.coeff_faith * loss_faith + cfg.coeff_stoch * loss_stoch + cfg.coeff_adv * loss_adv
                + cfg.coeff_imp * loss_imp + cfg.coeff_simplicity * loss_simp + loss_life
                + cfg.coeff_hidden * loss_hidden + cfg.coeff_interaction * loss_inter
                + cfg.coeff_weight_l1 * loss_l1)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0 or step == cfg.n_steps - 1:
            l0 = (g_lower > 0.5).float().sum(-1).mean().item()
            print(f"  step {step:>6} faith={loss_faith.item():.2e} kl={loss_stoch.item():.4f} "
                  f"adv={loss_adv.item():.4f} hid={loss_hidden.item():.4f} inter={loss_inter.item():.4f} "
                  f"l1={loss_l1.item():.1f} imp={loss_imp.item():.2f} L0/t={l0:.1f}/{cfg.n_components}",
                  flush=True)

    if save_path is not None:
        import dataclasses
        torch.save({"banks": {n: b.state_dict() for n, b in banks.items()}, "ci": ci.state_dict(),
                    "cfg": dataclasses.asdict(cfg), "ci_cfg": dataclasses.asdict(ci_cfg)}, save_path)
        print(f"saved -> {save_path}", flush=True)
    return {"banks": banks, "ci": ci, "model": model}


# --- eval: faithfulness metrics + descriptive fingerprints vs the documented neuron groups ----------


@torch.no_grad()
def evaluate(model: UnrolledRNN, banks, ci, task_fn, device: torch.device, batch: int = 8192,
             groups: dict[str, list[int]] | None = None, topk: int = 12) -> None:
    T = model.seq_len
    x = torch.randn(batch, T, device=device)
    clear_masks(banks)
    target = model(x)
    h_prev = model.trace_h_prev
    refresh_caches(banks)
    g_lower, g_upper = ci(x, h_prev)
    C = g_lower.shape[-1]
    y = task_fn(x.cpu()).to(device)
    acc_target = (target.argmax(-1) == y).float().mean().item()

    zeros = {n: torch.zeros(batch, T, device=device) for n in banks}
    pred_ci = masked_forward_rnn(model, banks, x, g_lower, zeros)
    pred_on = masked_forward_rnn(model, banks, x, torch.ones_like(g_lower), zeros)
    pred_off = masked_forward_rnn(model, banks, x, torch.zeros_like(g_lower), zeros)
    print(f"\n=== faithfulness ===", flush=True)
    print(f"acc: target={acc_target:.4f} ci-masked={(pred_ci.argmax(-1) == y).float().mean().item():.4f} "
          f"all-on={(pred_on.argmax(-1) == y).float().mean().item():.4f} "
          f"all-off={(pred_off.argmax(-1) == y).float().mean().item():.4f}", flush=True)
    print(f"kl: ci={kl_logits(pred_ci, target).item():.4f} on={kl_logits(pred_on, target).item():.2e} "
          f"L0/t={(g_lower > 0.5).float().sum(-1).mean().item():.2f}/{C}", flush=True)
    l1_sum = sum(b.materialized_weights().abs().sum().item() for b in banks.values())
    l1_w = sum(b.W_target.abs().sum().item() for b in banks.values())
    print(f"l1 ratio sum_c|P_c| / |W| = {l1_sum / l1_w:.2f}  (1.0 = disjoint axis-aligned support)", flush=True)

    # descriptive fingerprints of the most-used components
    mean_g = g_upper.mean(dim=(0, 1))  # [C]
    W = {n: b.materialized_weights() for n, b in banks.items()}  # [C, d_out, d_in]
    print(f"\n=== top components (write-mass over neurons; gate profile over timesteps) ===", flush=True)
    for c in mean_g.topk(min(topk, C)).indices.tolist():
        mass_mod = {n: W[n][c].pow(2).sum().item() for n in W}
        tot = sum(mass_mod.values()) + 1e-12
        # which neurons this component writes (rows of ih/hh) / reads at output (cols of out)
        neuron_mass = W["ih"][c].pow(2).sum(1) + W["hh"][c].pow(2).sum(1) + W["out"][c].pow(2).sum(0)
        top_neurons = neuron_mass.topk(3).indices.tolist()
        prof = g_upper[:, :, c].mean(0)  # [T]
        prof_s = " ".join(f"{v:.2f}" for v in prof.tolist())
        gm = ""
        if groups:
            nm = neuron_mass / (neuron_mass.sum() + 1e-12)
            gm = "  groups: " + " ".join(f"{k}:{nm[v].sum().item():.2f}" for k, v in groups.items())
        print(f"  comp {c:>3} use={mean_g[c]:.3f} split ih/hh/out="
              f"{mass_mod['ih']/tot:.2f}/{mass_mod['hh']/tot:.2f}/{mass_mod['out']/tot:.2f} "
              f"top neurons {top_neurons}{gm}\n"
              f"           gate-by-timestep: {prof_s}", flush=True)


# --- entry -------------------------------------------------------------------------------------------


def _run() -> None:
    from alg_zoo.handcrafted import handcrafted_2nd_argmax
    from alg_zoo.tasks import task_2nd_argmax

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    which = os.environ.get("MODEL", "hc10")
    seq_len = {"hc10": 10, "hc3": 3}[which]
    C = int(os.environ.get("C", "40" if which == "hc10" else "12"))
    fr = os.environ.get("R", "1")  # factor rank; "" -> full (min(d_out,d_in) per module)
    R = int(fr) if fr else None
    steps = int(os.environ.get("STEPS", "8000"))
    batch = int(os.environ.get("B", "1024"))
    imp = float(os.environ.get("IMP", "3e-2"))
    simp = float(os.environ.get("SIMP", "0.0"))
    life = float(os.environ.get("LIFE", "0.0"))
    ramp = float(os.environ.get("RAMP", "0.0"))
    seed = int(os.environ.get("SEED", "0"))

    base = handcrafted_2nd_argmax(seq_len)
    model = UnrolledRNN.from_dist_rnn(base).to(device)
    with torch.no_grad():  # equivalence check
        x = torch.randn(4096, seq_len, device=device)
        d = (model(x) - base.to(device)(x)).abs().max().item()
    ranks = {n: torch.linalg.matrix_rank(getattr(model, n).weight).item() for n in ("ih", "hh", "out")}
    print(f"target {which}: unroll-equivalence maxdiff={d:.2e}, weight ranks={ranks} "
          f"(C={C} must cover max rank for rank-1 faithfulness)", flush=True)

    cfg = ApdConfig(modules=["ih", "hh", "out"], n_components=C, n_steps=steps, warmup_steps=500,
                    coeff_faith=float(os.environ.get("FAITH", "1e7")),  # near-tie argmax is precision-
                    # sensitive: tiny weight error flips outputs, so LM-grade faithfulness, not 1e4
                    coeff_hidden=float(os.environ.get("HIDDEN", "0.0")),
                    coeff_interaction=float(os.environ.get("INTER", "0.0")),
                    interaction_pairs=int(os.environ.get("PAIRS", "4")),
                    coeff_weight_l1=float(os.environ.get("L1", "0.0")),
                    coeff_imp=imp, coeff_simplicity=simp, coeff_lifetime=life,
                    lifetime_ramp_frac=ramp, simplicity_impl="factored", factor_rank=R,
                    lowrank_forward=False, seed=seed)  # tiny dims: materialize is fine
    ci_cfg = VpdConfig(C_per_module={}, seq_len=seq_len, ci_d_model=64, ci_n_blocks=2,
                       ci_n_heads=4, ci_mlp_hidden=256)
    print(f"config: C={C} R={R} steps={steps} B={batch} imp={imp} life={life} ramp={ramp} simp={simp}", flush=True)
    recon = os.environ.get("RECON", "mse")  # see _D: near-tie argmax needs margin-preserving recon
    out = decompose_alg(model, cfg, ci_cfg, device, batch, recon=recon,
                        save_path=os.environ.get("SAVE", f"/tmp/algzoo/{which}_r{fr or 'full'}.pt"))

    groups = None
    if which == "hc10":  # documented neuron groups from alg_zoo/handcrafted.py
        groups = {"delay0-9": list(range(10)), "loo10-18": list(range(10, 19)), "runmax19-21": [19, 20, 21]}
    evaluate(out["model"], out["banks"], out["ci"], task_2nd_argmax, device, groups=groups)
    print("APD_ALG DONE", flush=True)


if __name__ == "__main__":
    import os as _os
    _os.makedirs("/tmp/algzoo", exist_ok=True)
    _run()
