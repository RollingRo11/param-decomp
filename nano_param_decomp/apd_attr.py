"""Attribution-routed APD, corrected version — random-order-path (Shapley-style) attribution,
sparse-sigmoid routing on RAW attribution, stochastic + adversarial reconstruction, anti-redundancy,
and causal-role simplicity. Tested on the cross-layer resid-MLP toy against the trained-mask stack.

What changed vs the first prototype (each item maps to a diagnosed failure):

1. RANDOM-SUBSET attribution instead of straight-line IG. The diagonal path scales all components
   together, so two redundant components get identical "both important" attributions — the path
   never visits a subset where one is present and the other absent, which is the only place
   redundancy is visible. We sample K random subsets (density rho ~ U(0,1), membership Bernoulli)
   and average the gate-gradient there: the Monte-Carlo random-order-path / Shapley estimator.
   Also fixes the off-distribution path concern: subsets of WHOLE components, never a uniformly
   scaled-down model.
2. STOCHASTIC + ADVERSARIAL reconstruction (reusing `recon_pair`): the trained-mask stack trains
   every component every step via sampled masks g + (1-g)u; the first prototype gave zero gradient
   to below-threshold components (dead components could never be recruited). Also restores PGD
   parity with the baseline it is compared against.
3. RAW-attribution routing (no z-score). The z-score destroyed IG completeness, hardwired a fixed
   fraction of active components per input, and made minimality zero-sum through the normalization.
   Attributions are O(1) by normalizing the objective by target variance; the sparse-sigmoid
   thresholds them directly, with a learned per-component scale+threshold (2C scalars).
4. INTERACTION (anti-redundancy) penalty: faithfulness + reconstruction are indifferent between
   "one component per mechanism" and "five sharing it"; this removes the spread solutions by
   penalizing super-additive joint-deletion damage (deleting both is worse than the parts predict
   = they back each other up).
5. Causal-role simplicity now scores the footprint over OUTPUTS AND HIDDEN SITES (a component that
   moves one output by touching everything internally is not simple).
6. Parity: faithfulness pin 1e7, all-on/off sanity in the eval, default budget matched to the
   trained-mask reference (8000 steps).

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.apd_attr
Env:  C, STEPS, K (subset samples/step), THRESH, WIDTH, SIMP, SIMP_NC, IMP, INTER, PAIRS, FAITH,
      SEED, LEARNED_ROUTER (default 1), SECOND_ORDER (default 0), COMPARE (time trained-mask too)
"""

import copy
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from .apd_mask import (
    ApdConfig,
    clear_masks,
    faithfulness_loss,
    install_banks,
    masked_forward,
    recon_pair,
    refresh_caches,
)
from .run import importance_minimality_loss, lower_leaky, upper_leaky
from .toy_models import ResidMLP, feature_batch, train_resid_mlp


def subset_attribution(model, banks, x: Tensor, target_out: Tensor, tvar: Tensor, C: int, K: int,
                       device, second_order: bool = False) -> Tensor:
    """Random-order-path (Shapley-style) attribution in gate space. For K samples: draw a subset
    density rho ~ U(0,1) per input, a Bernoulli(rho) component subset S, and take the gradient of
    the (variance-normalized) reconstruction objective w.r.t. the gates AT g=S. The gradient at a
    random subset is the marginal value of each component GIVEN that random coalition — averaged
    over subsets this is the Monte-Carlo random-order-path integral (Shapley-like), which separates
    redundant components (a component's credit vanishes on subsets where its backup is present),
    unlike the straight-line path (all components scale together; redundant pairs look identical)."""
    B = x.shape[0]
    rho = torch.rand(K * B, 1, device=device)
    S = (torch.rand(K * B, C, device=device) < rho).float()
    return _subset_attr_core(model, banks, x, target_out, tvar, S, K, second_order)


def _subset_attr_core(model, banks, x: Tensor, target_out: Tensor, tvar: Tensor, S: Tensor,
                      K: int, second_order: bool, tiled: bool = True) -> Tensor:
    """One TILED forward+backward for all K subset samples (the K-loop was kernel-launch-bound:
    ~50 tiny sequential passes/step left the GPU at 12%). Gate-gradient rows are per-input, so one
    backward on the [K*B] batch yields every sample's attribution. `tiled=False` keeps the loop
    path for the equivalence test."""
    B = x.shape[0]
    C = S.shape[-1]
    if tiled:
        g = S.requires_grad_(True)
        out = masked_forward(model, banks, x.repeat(K, *([1] * (x.dim() - 1))), g, None)
        tt = target_out.repeat(K, *([1] * (target_out.dim() - 1)))
        obj = -(((out - tt) ** 2).sum(-1) / tvar).sum()
        grad = torch.autograd.grad(obj, g, create_graph=second_order)[0]  # [K*B, C]
        return grad.view(K, B, C).mean(0)
    A = torch.zeros(B, C, device=x.device)
    for k in range(K):
        g = S[k * B:(k + 1) * B].requires_grad_(True)
        out = masked_forward(model, banks, x, g, None)
        obj = -(((out - target_out) ** 2).sum(-1) / tvar).sum()
        A = A + torch.autograd.grad(obj, g, create_graph=second_order)[0]
    return A / K


def route(A: Tensor, thresh: float, width: float,
          scale: Tensor | None = None, bias: Tensor | None = None,
          alpha: float = 0.01) -> tuple[Tensor, Tensor]:
    """Sparse-sigmoid gate on RAW attribution, using the trained-mask stack's EXACT gate machinery
    downstream of the score (attribution only replaces the CI net's logit). Two properties of that
    machinery proved load-bearing the hard way:
      - LEAKY boundaries (slope alpha outside [0,1]): a gate pinned at 0 still receives gradient,
        so gate death is not an absorbing state. (Hard clamp + strong minimality killed every gate:
        thresholds ratcheted past all attributions and nothing could revive them.)
      - TWO gates from one logit: reconstruction routes on the pessimistic q_lower, minimality
        pushes on the optimistic q_upper — minimality cannot directly zero the routing gate.
    Learned per-component log-scale and threshold (2C scalars) let a component fire only when it is
    the clear winner for an input. Returns (q_lower, q_upper)."""
    z = A if scale is None else torch.exp(scale) * A
    b = thresh if bias is None else bias
    logit = (z - b) / (2 * width) + 0.5
    return lower_leaky(logit, alpha), upper_leaky(logit, alpha)


def causal_role_penalty(model, banks, x: Tensor, q: Tensor, C: int, n_score: int,
                        device) -> Tensor:
    """Simple causal role: a component's ablation footprint — over final outputs AND every hidden
    site — should be concentrated. For the n_score most-active components, penalize the
    importance-weighted participation ratio of |f(q) - f(q with c off)| concatenated across outputs
    and per-site mean deltas (=1 when it moves one coordinate, =N when uniform)."""
    q = q.detach()
    imp = q.mean(0)
    top = [c for c in imp.topk(min(n_score, C)).indices.tolist() if imp[c] >= 1e-4]
    if not top:
        return torch.zeros((), device=device)
    B = x.shape[0]
    V = len(top) + 1  # variant 0 = full, then one ablation per scored component; ONE tiled forward
    gates = q.repeat(V, 1)
    for vi, c in enumerate(top):
        gates[(vi + 1) * B:(vi + 2) * B, c] = 0.0
    out = masked_forward(model, banks, x.repeat(V, *([1] * (x.dim() - 1))), gates, None)
    outs = out.view(V, B, -1)
    hids = {n: b.last_masked_out.view(V, B, -1) for n, b in banks.items()}
    pens = []
    for vi, c in enumerate(top):
        parts = [(outs[0] - outs[vi + 1]).abs().mean(0)]
        for n in banks:
            parts.append((hids[n][0] - hids[n][vi + 1]).abs().mean(0))
        v = torch.cat(parts)
        pr = v.sum().pow(2) / (v.pow(2).sum() + 1e-12)
        pens.append(imp[c] * pr)
    return torch.stack(pens).sum() / (imp[torch.tensor(top, device=device)].sum() + 1e-12)


def interaction_penalty(model, banks, x: Tensor, q: Tensor, target_out: Tensor, tvar: Tensor,
                        n_pairs: int, device) -> Tensor:
    """Anti-redundancy: sample co-active pairs (usage-weighted) and penalize super-additive joint
    deletion damage relu(D(ij) - D(i) - D(j) + D(base)) — positive exactly when the two components
    back each other up (each alone harmless BECAUSE the other covers). Removes the spread/redundant
    decompositions that faithfulness + reconstruction are indifferent to. Sub-additive (shared
    pipeline) is left alone."""
    q = q.detach()
    probs = q.mean(0) + 1e-6
    n_s = min(2 * n_pairs, probs.shape[0])
    cidx = torch.multinomial(probs, n_s, replacement=False)
    pairs = cidx[: (n_s // 2) * 2].view(-1, 2)

    # ONE tiled forward for base + the 3 deletion variants of every pair
    B = x.shape[0]
    plist = pairs.tolist()
    V = 1 + 3 * len(plist)
    gates = q.repeat(V, 1)
    for k, (i, j) in enumerate(plist):
        o = (1 + 3 * k) * B
        gates[o:o + B, i] = 0.0                                   # i off
        gates[o + B:o + 2 * B, j] = 0.0                            # j off
        gates[o + 2 * B:o + 3 * B, i] = 0.0
        gates[o + 2 * B:o + 3 * B, j] = 0.0                        # both off
    out = masked_forward(model, banks, x.repeat(V, *([1] * (x.dim() - 1))), gates, None)
    tt = target_out.repeat(V, *([1] * (target_out.dim() - 1)))
    D = (((out - tt) ** 2).view(V, B, -1).mean(dim=(1, 2))) / tvar  # [V]
    total = torch.zeros((), device=device)
    for k in range(len(plist)):
        total = total + F.relu(D[3 * k + 3] - D[3 * k + 1] - D[3 * k + 2] + D[0])
    return total / max(1, len(plist))


def decompose_attr(model, data_fn, cfg: ApdConfig, device, K: int, thresh: float, width: float,
                   simp: float, simp_nc: int, imp: float, inter: float, n_pairs: int,
                   second_order: bool, learned_router: bool) -> dict:
    banks = install_banks(model, cfg)
    model = model.to(device)
    C = cfg.n_components
    comp_params = [p for b in banks.values() for p in b.params()]

    wopt = torch.optim.AdamW(comp_params, lr=cfg.warmup_lr)
    for _ in range(cfg.warmup_steps):
        refresh_caches(banks)
        loss = faithfulness_loss(banks)
        wopt.zero_grad(); loss.backward(); wopt.step()

    r_scale = torch.zeros(C, device=device, requires_grad=learned_router)
    r_bias = torch.full((C,), thresh, device=device, requires_grad=learned_router)
    router_params = [r_scale, r_bias] if learned_router else []
    opt = torch.optim.AdamW(comp_params + router_params, lr=cfg.lr)

    t_attr = t_pen = t_rest = 0.0
    for step in range(cfg.n_steps):
        x = data_fn().to(device)
        clear_masks(banks)
        target_out = model(x).detach()
        tvar = target_out.var() + 1e-8
        refresh_caches(banks)

        t0 = time.time()
        A = subset_attribution(model, banks, x, target_out, tvar, C, K, device, second_order)
        sc = r_scale if learned_router else None
        bs = r_bias if learned_router else None
        q, q_upper = route(A if second_order else A.detach(), thresh, width, sc, bs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_attr += time.time() - t0

        # stochastic (q + (1-q)u: every component sampled every step -> recruitment gradient for
        # dormant components) + fresh-PGD adversarial recon, both under the attribution gate.
        loss_faith = faithfulness_loss(banks)
        loss_stoch, loss_adv = recon_pair(model, banks, cfg, q, x, target_out, tvar, None)
        # minimality MUST be the stochastic term's matched counterweight (the stack's annealed Lp,
        # not a weak flat L1): the stochastic term rewards raising gates (a raised gate shields a
        # component from deletion), so under-dosed minimality lets gates inflate until routing is
        # decorative and recovery collapses (observed: L0 59/130, injectivity 0.01).
        p = cfg.p_start + (cfg.p_end - cfg.p_start) * (step / cfg.n_steps)
        loss_imp = importance_minimality_loss({"g": q_upper}, p, cfg.imp_eps, cfg.imp_beta, 1)

        t1 = time.time()
        loss_simple = (causal_role_penalty(model, banks, x, q, C, simp_nc, device)
                       if simp > 0 else torch.zeros((), device=device))
        loss_inter = (interaction_penalty(model, banks, x, q, target_out, tvar, n_pairs, device)
                      if inter > 0 else torch.zeros((), device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pen += time.time() - t1

        t2 = time.time()
        loss = (cfg.coeff_faith * loss_faith + cfg.coeff_stoch * loss_stoch
                + cfg.coeff_adv * loss_adv + imp * loss_imp
                + simp * loss_simple + inter * loss_inter)
        opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rest += time.time() - t2

        if step % max(1, cfg.n_steps // 10) == 0 or step == cfg.n_steps - 1:
            l0 = (q > 0.5).float().sum(-1).mean().item()
            print(f"  step {step:>5} faith={loss_faith.item():.2e} stoch={loss_stoch.item():.4f} "
                  f"adv={loss_adv.item():.4f} simple={loss_simple.item():.3f} "
                  f"inter={loss_inter.item():.4f} L0={l0:.1f}/{C}", flush=True)
    return {"banks": banks, "model": model, "t_attr": t_attr, "t_pen": t_pen, "t_rest": t_rest,
            "r_scale": r_scale.detach(), "r_bias": r_bias.detach(),
            "learned_router": learned_router}


def recovery_attr(model, banks, cfg: ApdConfig, device, K: int, thresh: float, width: float,
                  r_scale: Tensor | None = None, r_bias: Tensor | None = None, n_probe=128) -> dict:
    """Single-feature probes -> subset attribution -> routed gate -> assignment; separation,
    purity, cross-layer, keep-only sufficiency, plus all-on/off reconstruction sanity."""
    nf = int(model.n_features)
    C = cfg.n_components
    Amat = torch.zeros(nf, C, device=device)
    for i in range(nf):
        x = torch.zeros(n_probe, nf, device=device)
        x[:, i] = torch.rand(n_probe, device=device)
        clear_masks(banks)
        target = model(x).detach()
        tvar = target.var() + 1e-8
        refresh_caches(banks)
        A = subset_attribution(model, banks, x, target, tvar, C, K, device, second_order=False)
        Amat[i] = route(A.detach(), thresh, width, r_scale, r_bias)[0].mean(0)
    assigned = Amat.argmax(dim=1)
    top1 = Amat.gather(1, assigned[:, None]).squeeze(1)
    active = top1 > 0.5
    purity = (top1 / (Amat.sum(dim=1) + 1e-8))[active].mean().item() if active.any() else 0.0
    inj = (assigned[active].unique().numel() / int(active.sum())) if active.any() else 0.0

    with torch.no_grad():
        refresh_caches(banks)
        comp_layer_norm = torch.zeros(C, 2, device=device)
        for n, b in banks.items():
            l = 0 if "blocks.0" in n else 1
            comp_layer_norm[:, l] += b.materialized_weights().pow(2).sum(dim=(1, 2))
        comp_layer_norm = comp_layer_norm.sqrt()
        a = comp_layer_norm[assigned[active]]
        span = a.min(dim=1).values / (a.max(dim=1).values + 1e-8)
        cross = (span > 0.1).float().mean().item() if active.any() else 0.0

        keep, off = [], []
        gen = torch.Generator(device=device).manual_seed(777)
        for i in range(nf):
            if not bool(active[i]):
                continue
            x = torch.zeros(n_probe, nf, device=device)
            x[:, i] = torch.rand(n_probe, device=device, generator=gen)
            clear_masks(banks)
            tgt = model(x).detach()
            var_i = tgt[:, i].var() + 1e-8
            refresh_caches(banks)
            gate = torch.zeros(n_probe, C, device=device)
            gate[:, assigned[i]] = 1.0
            pred = masked_forward(model, banks, x, gate, None)
            keep.append((F.mse_loss(pred[:, i], tgt[:, i]) / var_i).item())
            z = masked_forward(model, banks, x, torch.zeros(n_probe, C, device=device), None)
            off.append((F.mse_loss(z[:, i], tgt[:, i]) / var_i).item())

        # all-on / all-off reconstruction sanity on a generic batch
        xg = feature_batch(nf, 2048, 0.01, device, torch.Generator(device=device).manual_seed(99))
        clear_masks(banks)
        tg = model(xg).detach()
        tv = tg.var() + 1e-8
        refresh_caches(banks)
        on = (F.mse_loss(masked_forward(model, banks, xg, torch.ones(2048, C, device=device), None), tg) / tv).item()
        offr = (F.mse_loss(masked_forward(model, banks, xg, torch.zeros(2048, C, device=device), None), tg) / tv).item()
    return {"active_frac": int(active.sum()) / nf, "injectivity": inj, "purity": purity,
            "cross_layer": cross, "keep_only": (sum(keep) / len(keep)) if keep else float("nan"),
            "keep_only_off": (sum(off) / len(off)) if off else float("nan"),
            "recon_on": on, "recon_off": offr}


def _run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, d_embed, d_mlp, n_layers = 100, 256, 40, 2
    C = int(os.environ.get("C", "130"))
    steps = int(os.environ.get("STEPS", "8000"))  # matched to the trained-mask reference budget
    K = int(os.environ.get("K", "6"))
    thresh = float(os.environ.get("THRESH", "0.3"))
    width = float(os.environ.get("WIDTH", "0.15"))
    simp = float(os.environ.get("SIMP", "1e-2"))
    simp_nc = int(os.environ.get("SIMP_NC", "24"))
    imp = float(os.environ.get("IMP", "3e-2"))  # matched to the trained-mask stack's coeff_imp
    inter = float(os.environ.get("INTER", "0.1"))
    n_pairs = int(os.environ.get("PAIRS", "4"))
    faith = float(os.environ.get("FAITH", "1e7"))
    second_order = os.environ.get("SECOND_ORDER", "0") == "1"
    learned_router = os.environ.get("LEARNED_ROUTER", "1") == "1"
    seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(seed)
    fprob = 0.01
    ckpt = f"/tmp/toy/resid_2l_apd_dmlp{d_mlp}.pt"
    os.makedirs("/tmp/toy", exist_ok=True)
    if os.path.exists(ckpt):
        base = ResidMLP(nf, d_embed, d_mlp, n_layers, seed=0).to(device)
        base.load_state_dict(torch.load(ckpt, weights_only=True))
        print("loaded cached resid-MLP target", flush=True)
    else:
        base = train_resid_mlp(nf, d_embed, d_mlp, n_layers, steps=4000, batch=2048,
                               feature_prob=fprob, lr=3e-3, device=device, seed=0)
        torch.save(base.state_dict(), ckpt)
    gen = torch.Generator(device=device).manual_seed(123 + seed)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    modules = [f"blocks.{i}.{p}" for i in range(n_layers) for p in ("in_proj", "out_proj")]
    cfg = ApdConfig(modules=modules, n_components=C, n_steps=steps, warmup_steps=500,
                    simplicity_impl="factored", factor_rank=1, lowrank_forward=True,
                    coeff_faith=faith, coeff_stoch=1.0, coeff_adv=1.0,
                    coeff_simplicity=0.0, seed=seed)
    print(f"config: C={C} steps={steps} K={K} thresh={thresh} width={width} simp={simp} "
          f"simp_nc={simp_nc} imp={imp} inter={inter} pairs={n_pairs} faith={faith:g} "
          f"second_order={second_order} learned_router={learned_router} seed={seed}", flush=True)
    model = copy.deepcopy(base)
    t0 = time.time()
    out = decompose_attr(model, data_fn, cfg, device, K, thresh, width, simp, simp_nc, imp,
                         inter, n_pairs, second_order, learned_router)
    wall = time.time() - t0
    rs = out["r_scale"] if out["learned_router"] else None
    rb = out["r_bias"] if out["learned_router"] else None
    rec = recovery_attr(out["model"], out["banks"], cfg, device, K, thresh, width, rs, rb)
    print(f"\nRECOVERY (attribution-routed, corrected) active_frac={rec['active_frac']:.2f} "
          f"injectivity={rec['injectivity']:.2f} purity={rec['purity']:.2f} "
          f"cross_layer={rec['cross_layer']:.2f} keep_only={rec['keep_only']:.3f} "
          f"(vs off={rec['keep_only_off']:.3f})", flush=True)
    print(f"SANITY recon_on={rec['recon_on']:.4f} (must be ~0) recon_off={rec['recon_off']:.4f}", flush=True)
    per = wall / steps
    print(f"COST attr: {wall:.1f}s / {steps} steps = {per*1000:.0f}ms/step  "
          f"[attr {out['t_attr']/steps*1000:.0f} + penalties {out['t_pen']/steps*1000:.0f} + "
          f"rest {out['t_rest']/steps*1000:.0f} ms]", flush=True)

    if os.environ.get("COMPARE", "0") == "1":
        from .apd_mask import decompose_apd
        cfg2 = ApdConfig(modules=modules, n_components=C, n_steps=steps, warmup_steps=500,
                         coeff_imp=3e-2, coeff_simplicity=1e-3, simplicity_impl="factored",
                         factor_rank=1, lowrank_forward=True, coeff_adv=1.0, seed=seed)
        m2 = copy.deepcopy(base)
        t0 = time.time()
        _ = decompose_apd(m2, data_fn, cfg2, device)
        wall2 = time.time() - t0
        print(f"COST trained-mask: {wall2:.1f}s = {wall2/steps*1000:.0f}ms/step -> attribution is "
              f"{wall/wall2:.2f}x the trained-mask wall time (>1 = slower)", flush=True)
    print("APD_ATTR DONE", flush=True)


if __name__ == "__main__":
    _run()
