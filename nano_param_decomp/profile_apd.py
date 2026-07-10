"""Profile the two simplicity backends of `apd_mask` on the ResidMLP target, to see whether the
per-step SVD (svd backend) is actually the cost worth removing vs the extra A@B materialization the
factored (SVD-free nuclear-norm) backend pays.

Replays the real training step (same module-level `recon_pair` / `simplicity_loss` the trainer uses)
and times four regions per step with CUDA syncs:
  refresh   -- refresh_caches: builds P (svd: free) or A@B (factored: one batched matmul), reused
               across the masked forwards
  simp      -- simplicity_loss forward: SVD (svd) vs Frobenius (factored)
  recon     -- recon_pair: the ~6 masked forwards (stochastic + PGD inner + adversarial)
  rest      -- target forward + CI net + faithfulness + minimality + backward + opt step
And the end-to-end total (the number that actually decides it).

Run:  CUDA_VISIBLE_DEVICES=1 python -m nano_param_decomp.profile_apd
Env:  STEPS_WARM, STEPS_MEASURE, C, MODEL(=resid|tms)
"""

import copy
import os
import time

import torch
import torch.nn.functional as F

from .apd_mask import (
    ApdConfig,
    CIMLP,
    faithfulness_loss,
    importance_minimality_loss,
    install_banks,
    recon_pair,
    refresh_caches,
    simplicity_loss,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_target(model_name: str, device: torch.device):
    os.makedirs("/tmp/toy", exist_ok=True)
    if model_name == "tms":
        from .toy_models import TMS, feature_batch, train_tms
        nf, nh = 5, 2
        ckpt = "/tmp/toy/tms_5-2.pt"
        if os.path.exists(ckpt):
            base = TMS(nf, nh).to(device); base.load_state_dict(torch.load(ckpt, weights_only=True))
        else:
            base = train_tms(nf, nh, 5000, 1024, 0.05, 1e-2, device, seed=0)
            torch.save(base.state_dict(), ckpt)
        base.freeze_for_decomposition()
        modules = ["W"]
        gen = torch.Generator(device=device).manual_seed(123)
        data_fn = lambda: feature_batch(nf, 2048, 0.05, device, gen)
        return base, modules, data_fn
    # resid
    from .toy_models import ResidMLP, feature_batch, train_resid_mlp
    nf, d_embed, d_mlp, n_layers = 100, 256, 40, 2
    ckpt = f"/tmp/toy/resid_2l_apd_dmlp{d_mlp}.pt"
    if os.path.exists(ckpt):
        base = ResidMLP(nf, d_embed, d_mlp, n_layers, seed=0).to(device)
        base.load_state_dict(torch.load(ckpt, weights_only=True))
    else:
        base = train_resid_mlp(nf, d_embed, d_mlp, n_layers, 4000, 2048, 0.01, 3e-3, device, seed=0)
        torch.save(base.state_dict(), ckpt)
    modules = [f"blocks.{i}.{proj}" for i in range(n_layers) for proj in ("in_proj", "out_proj")]
    gen = torch.Generator(device=device).manual_seed(123)
    data_fn = lambda: feature_batch(nf, 2048, 0.01, device, gen)
    return base, modules, data_fn


def profile_impl(impl: str, base, modules, data_fn, C: int, device: torch.device,
                 warm: int, measure: int, lowrank: bool = False) -> dict[str, float]:
    torch.manual_seed(0)
    model = copy.deepcopy(base)
    fr = os.environ.get("FR")
    cfg = ApdConfig(modules=modules, n_components=C, n_steps=warm + measure, warmup_steps=50,
                    coeff_simplicity=1e-3, simplicity_impl=impl, lowrank_forward=lowrank,
                    factor_rank=(int(fr) if fr and impl == "factored" else None))
    banks = install_banks(model, cfg)
    model = model.to(device)
    order = sorted(modules)
    d_in = {n: int(b.W_target.shape[1]) for n, b in banks.items()}
    ci = CIMLP(order, d_in, C, cfg.ci_hidden, cfg.ci_layers, cfg.leaky_alpha).to(device)
    comp_params = [p for b in banks.values() for p in b.params()]
    opt = torch.optim.AdamW(comp_params + list(ci.parameters()), lr=cfg.lr)

    acc = {k: 0.0 for k in ("refresh", "simp", "recon", "rest", "total")}
    for step in range(warm + measure):
        timing = step >= warm
        _sync(); t_step0 = time.perf_counter()

        x = data_fn().to(device)
        for b in banks.values():
            b.mode = "target"
        target_out = model(x).detach()
        acts = {n: b.last_input for n, b in banks.items()}
        tvar = target_out.var() + 1e-8

        _sync(); t0 = time.perf_counter()
        refresh_caches(banks)
        _sync(); t_refresh = time.perf_counter() - t0

        g_lower, g_upper = ci(acts)
        B = x.shape[0]
        deltas = {n: torch.rand(B, device=device) for n in order}

        loss_faith = faithfulness_loss(banks)

        _sync(); t0 = time.perf_counter()
        loss_stoch, loss_adv = recon_pair(model, banks, cfg, g_lower, x, target_out, tvar, deltas)
        _sync(); t_recon = time.perf_counter() - t0

        loss_imp = importance_minimality_loss({"g": g_upper}, 1.0, cfg.imp_eps, cfg.imp_beta, 1)

        _sync(); t0 = time.perf_counter()
        loss_simp = simplicity_loss(banks, g_upper.mean(dim=0), cfg)
        _sync(); t_simp = time.perf_counter() - t0

        loss = (cfg.coeff_faith * loss_faith + cfg.coeff_stoch * loss_stoch
                + cfg.coeff_adv * loss_adv + cfg.coeff_imp * loss_imp
                + cfg.coeff_simplicity * loss_simp)
        opt.zero_grad(); loss.backward(); opt.step()
        _sync(); t_total = time.perf_counter() - t_step0

        if timing:
            acc["refresh"] += t_refresh
            acc["simp"] += t_simp
            acc["recon"] += t_recon
            acc["rest"] += t_total - t_refresh - t_simp - t_recon
            acc["total"] += t_total
    return {k: 1000.0 * v / measure for k, v in acc.items()}  # ms/step


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.environ.get("MODEL", "resid")
    C = int(os.environ.get("C", "130"))
    warm = int(os.environ.get("STEPS_WARM", "20"))
    measure = int(os.environ.get("STEPS_MEASURE", "100"))
    base, modules, data_fn = _build_target(model_name, device)
    print(f"profiling MODEL={model_name} C={C} on {device}  (warm={warm}, measure={measure})", flush=True)
    print(f"modules={modules}\n", flush=True)

    configs = [("svd", False, "svd (materialize)"),
               ("factored", False, "factored (materialize)"),
               ("factored", True, "factored (low-rank fwd)"),
               ("tucker", False, "tucker (core-space)")]
    rows = {}
    for impl, lowrank, label in configs:
        r = profile_impl(impl, base, modules, data_fn, C, device, warm, measure, lowrank=lowrank)
        rows[label] = r
        print(f"[{label:24}] total={r['total']:7.2f}  refresh={r['refresh']:6.2f}  simp={r['simp']:6.2f}  "
              f"recon={r['recon']:6.2f}  rest={r['rest']:6.2f}   (ms/step)", flush=True)

    base_total = rows["svd (materialize)"]["total"]
    print()
    for label, r in rows.items():
        print(f"{label:24} total={r['total']:7.2f} ms/step  speedup_vs_svd={base_total / r['total']:.2f}x  "
              f"8k-step wall={r['total']*8000/1000/60:.1f} min", flush=True)
    print("PROFILE DONE", flush=True)


if __name__ == "__main__":
    main()
