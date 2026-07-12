"""Attribution-routed APD (a "better APD" prototype), tested on the cross-layer resid-MLP toy.

The trained-mask stack (SPD/VPD/ours) learns a side network to output the per-input component
gates. This module replaces that network with **attribution**: for each input we compute how much
each whole-network component matters (integrated gradients in gate space), pass it through a
**sparse-sigmoid** (2-D sparsemax) so the gate is sparse but still hands corrective gradient to
components near the boundary (unlike top-k), and route with those gates. Simplicity is measured as a
**simple causal role** (a component's ablation footprint over outputs should be concentrated on few
outputs) rather than low matrix rank (which SPD/our own toys showed fights reconstruction).

We ask the exact question we asked of VPD on this toy: does it recover the cross-layer
one-mechanism-per-feature structure (separation / coverage / cross-layer / keep-only), and at what
relative cost (no trained mask net to learn, but attribution and the causal-role penalty cost
gradients).

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.apd_attr
Env:  C, STEPS, K (IG path steps), TAU (sparse-sigmoid temperature), SIMP (causal-role coeff),
      SIMP_NC (components scored for causal role per step), SECOND_ORDER (1=grad through attribution),
      COMPARE (1=also time the trained-mask baseline)
"""

import copy
import math
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
    refresh_caches,
)
from .toy_models import ResidMLP, feature_batch, train_resid_mlp


def route(A: Tensor, thresh: float, width: float) -> Tensor:
    """Sparse-sigmoid gate on RELATIVE attribution. Standardize A per input (all components have some
    marginal reconstruction value, so absolute A is diffusely positive; what matters is which are
    important *relative to the rest*), then a clamped-linear (2-D sparsemax) gate: components more
    than `thresh` std above the mean are fully on, below `thresh-2*width` fully off, linear between.
    Exact zeros (sparsity, like top-k) with a gradient band around the boundary (unlike top-k)."""
    z = (A - A.mean(dim=-1, keepdim=True)) / (A.std(dim=-1, keepdim=True) + 1e-8)
    return torch.clamp((z - thresh) / (2 * width) + 0.5, 0.0, 1.0)


def ig_attribution(model, banks, x: Tensor, target_out: Tensor, C: int, K: int,
                   device, second_order: bool = False) -> Tensor:
    """Integrated gradients in gate space from the empty model (g=0) to the full model (g=1) of the
    reconstruction objective. A_c(x) = mean over the path of d(-||f_g(x) - target||^2)/dg_c: the
    marginal contribution of component c to reconstructing the target, averaged over subsets along
    the path (a straight-line IG; random-order paths would push it toward Shapley)."""
    B = x.shape[0]
    A = torch.zeros(B, C, device=device)
    for k in range(1, K + 1):
        alpha = float(k) / K
        g = torch.full((B, C), alpha, device=device, requires_grad=True)
        out = masked_forward(model, banks, x, g, None)
        obj = -((out - target_out) ** 2).sum()
        grad = torch.autograd.grad(obj, g, create_graph=second_order)[0]  # [B, C]
        A = A + grad
    return A / K


def causal_role_penalty(model, banks, x: Tensor, q: Tensor, C: int,
                        n_score: int, device) -> Tensor:
    """Simple causal role: a component's ablation footprint over the outputs should be concentrated
    on few outputs. For the n_score most-active components, delta_c = f(q) - f(q with c off) [B, O];
    penalize importance-weighted participation ratio PR(|delta_c|) = (sum v)^2 / sum v^2 over outputs
    (=1 when it moves one output, =O when uniform). Both forwards depend on the components, so this
    is first-order in the component weights (no second-order needed)."""
    q = q.detach()
    imp = q.mean(0)                                   # [C] mean gate
    top = imp.topk(min(n_score, C)).indices
    out_full = masked_forward(model, banks, x, q, None)  # [B, O]
    pens = []
    for c in top.tolist():
        if imp[c] < 1e-4:
            continue
        qa = q.clone()
        qa[:, c] = 0.0
        out_c = masked_forward(model, banks, x, qa, None)
        v = (out_full - out_c).abs().mean(0)          # [O] footprint
        pr = v.sum().pow(2) / (v.pow(2).sum() + 1e-12)
        pens.append(imp[c] * pr)
    if not pens:
        return torch.zeros((), device=device)
    return torch.stack(pens).sum() / (imp[top].sum() + 1e-12)


def decompose_attr(model, data_fn, cfg: ApdConfig, device, K: int, thresh: float, width: float,
                   simp: float, simp_nc: int, imp: float, second_order: bool) -> dict:
    banks = install_banks(model, cfg)
    model = model.to(device)
    C = cfg.n_components
    comp_params = [p for b in banks.values() for p in b.params()]

    # faithfulness warmup: fit sum_c P_c = W before routing
    wopt = torch.optim.AdamW(comp_params, lr=cfg.warmup_lr)
    for _ in range(cfg.warmup_steps):
        refresh_caches(banks)
        loss = faithfulness_loss(banks)
        wopt.zero_grad(); loss.backward(); wopt.step()

    opt = torch.optim.AdamW(comp_params, lr=cfg.lr)  # NO mask-net params: attribution routes
    t_attr = t_simp = t_rest = 0.0
    for step in range(cfg.n_steps):
        x = data_fn().to(device)
        clear_masks(banks)
        target_out = model(x).detach()
        tvar = target_out.var() + 1e-8
        refresh_caches(banks)

        t0 = time.time()
        A = ig_attribution(model, banks, x, target_out, C, K, device, second_order)
        q = route(A, thresh, width)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_attr += time.time() - t0

        out_q = masked_forward(model, banks, x, q, None)
        loss_recon = F.mse_loss(out_q, target_out) / tvar
        loss_faith = faithfulness_loss(banks)
        loss_imp = q.sum(-1).mean()  # light minimality: few components per input

        t1 = time.time()
        loss_simple = (causal_role_penalty(model, banks, x, q, C, simp_nc, device)
                       if simp > 0 else torch.zeros((), device=device))
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_simp += time.time() - t1

        t2 = time.time()
        loss = cfg.coeff_faith * loss_faith + loss_recon + simp * loss_simple + imp * loss_imp
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_rest += time.time() - t2

        if step % max(1, cfg.n_steps // 10) == 0 or step == cfg.n_steps - 1:
            l0 = (q > 0.5).float().sum(-1).mean().item()
            print(f"  step {step:>5} faith={loss_faith.item():.2e} recon={loss_recon.item():.4f} "
                  f"simple={loss_simple.item():.3f} L0={l0:.1f}/{C}", flush=True)
    return {"banks": banks, "model": model, "t_attr": t_attr, "t_simp": t_simp, "t_rest": t_rest,
            "K": K}


@torch.no_grad()
def _cross_layer_and_keep(model, banks, cfg, device, assigned, active, n_probe=128):
    """cross-layer span + keep-only sufficiency for the assigned components (weight-based, same as
    the trained-mask recovery eval)."""
    C = cfg.n_components
    refresh_caches(banks)
    comp_layer_norm = torch.zeros(C, 2, device=device)
    for n, b in banks.items():
        l = 0 if "blocks.0" in n else 1
        comp_layer_norm[:, l] += b.materialized_weights().pow(2).sum(dim=(1, 2))
    comp_layer_norm = comp_layer_norm.sqrt()
    a = comp_layer_norm[assigned[active]]
    span = a.min(dim=1).values / (a.max(dim=1).values + 1e-8)
    cross = (span > 0.1).float().mean().item() if active.any() else 0.0
    return cross


def recovery_attr(model, banks, cfg: ApdConfig, device, K: int, thresh: float, width: float,
                  n_probe=128) -> dict:
    """Attribution-routed analog of feature_recovery_resid: single-feature probes -> attribution
    gate q -> assignment; separation (injectivity), purity, cross-layer, keep-only sufficiency."""
    nf = int(model.n_features)
    C = cfg.n_components
    Amat = torch.zeros(nf, C, device=device)
    for i in range(nf):
        x = torch.zeros(n_probe, nf, device=device)
        x[:, i] = torch.rand(n_probe, device=device)
        clear_masks(banks)
        target = model(x).detach()
        refresh_caches(banks)
        A = ig_attribution(model, banks, x, target, C, K, device, second_order=False)
        Amat[i] = route(A, thresh, width).mean(0)
    assigned = Amat.argmax(dim=1)
    top1 = Amat.gather(1, assigned[:, None]).squeeze(1)
    active = top1 > 0.5
    purity = (top1 / (Amat.sum(dim=1) + 1e-8))[active].mean().item() if active.any() else 0.0
    inj = (assigned[active].unique().numel() / int(active.sum())) if active.any() else 0.0
    cross = _cross_layer_and_keep(model, banks, cfg, device, assigned, active, n_probe)

    # keep-only sufficiency: reconstruct feature i's own output dim using ONLY its assigned component
    keep, off = [], []
    for i in range(nf):
        if not bool(active[i]):
            continue
        x = torch.zeros(n_probe, nf, device=device)
        x[:, i] = torch.rand(n_probe, device=device)
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
    return {"active_frac": int(active.sum()) / nf, "injectivity": inj, "purity": purity,
            "cross_layer": cross, "keep_only": (sum(keep) / len(keep)) if keep else float("nan"),
            "keep_only_off": (sum(off) / len(off)) if off else float("nan")}


def _run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf, d_embed, d_mlp, n_layers = 100, 256, 40, 2
    C = int(os.environ.get("C", "130"))
    steps = int(os.environ.get("STEPS", "4000"))
    K = int(os.environ.get("K", "6"))
    thresh = float(os.environ.get("THRESH", "1.0"))   # gate on: >thresh std above mean attribution
    width = float(os.environ.get("WIDTH", "0.5"))
    simp = float(os.environ.get("SIMP", "1e-2"))
    simp_nc = int(os.environ.get("SIMP_NC", "24"))
    imp = float(os.environ.get("IMP", "1e-3"))
    second_order = os.environ.get("SECOND_ORDER", "0") == "1"
    seed = int(os.environ.get("SEED", "0"))
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
    gen = torch.Generator(device=device).manual_seed(123)

    def data_fn() -> Tensor:
        return feature_batch(nf, 2048, fprob, device, gen)

    modules = [f"blocks.{i}.{p}" for i in range(n_layers) for p in ("in_proj", "out_proj")]
    cfg = ApdConfig(modules=modules, n_components=C, n_steps=steps, warmup_steps=500,
                    simplicity_impl="factored", factor_rank=1, lowrank_forward=True,
                    coeff_faith=1e4, coeff_simplicity=0.0, seed=seed)
    print(f"config: C={C} steps={steps} K={K} thresh={thresh} width={width} simp={simp} "
          f"simp_nc={simp_nc} imp={imp} second_order={second_order}", flush=True)
    model = copy.deepcopy(base)
    t0 = time.time()
    out = decompose_attr(model, data_fn, cfg, device, K, thresh, width, simp, simp_nc, imp, second_order)
    wall = time.time() - t0
    rec = recovery_attr(out["model"], out["banks"], cfg, device, K, thresh, width)
    print(f"\nRECOVERY (attribution-routed) active_frac={rec['active_frac']:.2f} "
          f"injectivity={rec['injectivity']:.2f} purity={rec['purity']:.2f} "
          f"cross_layer={rec['cross_layer']:.2f} keep_only={rec['keep_only']:.3f} "
          f"(vs off={rec['keep_only_off']:.3f})", flush=True)
    per = wall / steps
    print(f"COST attr: {wall:.1f}s / {steps} steps = {per*1000:.0f}ms/step  "
          f"[attr {out['t_attr']/steps*1000:.0f} + simp {out['t_simp']/steps*1000:.0f} + "
          f"rest {out['t_rest']/steps*1000:.0f} ms]", flush=True)

    if os.environ.get("COMPARE", "0") == "1":
        # trained-mask baseline (our whole-network method) on the same toy, timed for comparison
        from .apd_mask import decompose_apd
        cfg2 = ApdConfig(modules=modules, n_components=C, n_steps=steps, warmup_steps=500,
                         coeff_imp=3e-2, coeff_simplicity=1e-3, simplicity_impl="factored",
                         factor_rank=1, lowrank_forward=True, coeff_adv=1.0, seed=seed)
        m2 = copy.deepcopy(base)
        t0 = time.time()
        _ = decompose_apd(m2, data_fn, cfg2, device)
        wall2 = time.time() - t0
        print(f"COST trained-mask (whole-network, PGD+CI-net): {wall2:.1f}s / {steps} steps = "
              f"{wall2/steps*1000:.0f}ms/step  -> attribution is {wall/wall2:.2f}x the trained-mask "
              f"wall time (>1 = slower)", flush=True)
    print("APD_ATTR DONE", flush=True)


if __name__ == "__main__":
    _run()
