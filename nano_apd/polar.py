"""Polarization-trained decomposition (from scratch — no warm start, no top-k, no
trained mask). The loss is about the attributions themselves:

  L = faith * ||W - sum_c P_c||^2 / n_params                     (faithfulness)
    + recon * MSE(forward with attribution-share gates, target)  (gated reconstruction)
    + polar * mean over batch pairs of  u^2 (1-u)^2,             (polarization)
        u = cosine between the two samples' attribution vectors — two inputs either
        use the same mechanisms (u -> 1) or different ones (u -> 0); the middle is
        what "polysemantic component" means and is penalized directly. This is also
        the anti-shattering force: shards of one mechanism serve overlapping inputs
        and get pushed to agree (merge), which per-input sparsity never provides.
    + dl * schatten_p(components, gates)                          (description length,
        the anti-merge guard: complexity x usage; without it the polarization loss
        has a degenerate one-giant-component solution)

Gates are g = (A / A.max)^tau per sample — a deterministic, differentiable function of
the gradient attributions. No k, no gate network, label-free (pairs need no annotation).

    python -m nano_apd.polar --polar 10 --dl 1.0
"""

import argparse
import json
from pathlib import Path

import einops
import torch
import torch.nn.functional as F

from nano_apd.apd import calc_grad_attributions, calc_recon_mse, calc_schatten_loss, \
    get_lr_schedule_fn, get_lr_with_warmup
from nano_apd.models import ResidMLPAPDModel, SparseFeatureDataset
from nano_apd.run_tmdr import OUT_ROOT, eval_tmdr, load_tmdr_target


def calc_grad_attributions_ig(target, batch, component_weights, K):
    """IG-over-mask attribution. Same inner product as calc_grad_attributions, but
    gradients and pre-acts are taken along the path alpha*W (all decomposed weights
    scaled together, alpha = k/K) and averaged BEFORE squaring — Aumann-Shapley
    credit between "every gate at 0" and the full model, so a component whose
    marginal slope at the intact model vanishes (saturation/backup) still earns
    its integrated share. Restores target weights exactly on exit."""
    names = list(component_weights.keys())
    tw = target.weights()
    W0 = {n: tw[n].data.clone() for n in names}
    inner = None
    for k in range(1, K + 1):
        for n in names:
            tw[n].data.copy_(W0[n] * (k / K))
        out_k, cache_k = target(batch)
        out_dim = out_k.shape[-1]
        eye = torch.eye(out_dim, device=out_k.device, dtype=out_k.dtype)
        post_list = [cache_k[n]["post"] for n in names]
        grads = torch.autograd.grad(
            einops.einsum(out_k, "batch i d_out -> d_out"), post_list,
            grad_outputs=eye, is_grads_batched=True)
        fa = None
        for grad, n in zip(grads, names, strict=True):
            ca = einops.einsum(
                cache_k[n]["pre"].detach().clone(), component_weights[n],
                "batch i d_in, i C d_in d_out -> batch i C d_out")
            t = einops.einsum(grad, ca,
                              "o batch i d_out, batch i C d_out -> o batch i C")
            fa = t if fa is None else fa + t
        inner = fa if inner is None else inner + fa
    for n in names:
        tw[n].data.copy_(W0[n])
    return einops.einsum((inner / K) ** 2, "o batch i C -> batch i C")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--C", type=int, default=130)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--faith", type=float, default=1.0)
    parser.add_argument("--recon", type=float, default=1.0)
    parser.add_argument("--polar", type=float, default=10.0)
    parser.add_argument("--dl", type=float, default=1.0)
    parser.add_argument("--dl_pnorm", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gate_norm", choices=["max", "sum"], default="max",
                        help="sum = mentor's competitive gating (shares sum to 1; "
                             "recon becomes an implicit concentration pressure). "
                             "Testing only — predicted failure mode is absorption.")
    parser.add_argument("--gate_thresh", type=float, default=0.0,
                        help="hard-zero the gate tail: g = relu(g - t)/(1 - t) after "
                             "normalization. Losers get exactly 0, winners keep full "
                             "amplitude — tail sparsity without starving co-winners.")
    parser.add_argument("--prior", type=float, default=1.0)
    parser.add_argument("--pair_prior", type=float, default=0.05)
    parser.add_argument("--geom", type=float, default=0.0,
                        help="usage-geometry matching loss weight")
    parser.add_argument("--geom_mode", choices=["gram", "cliprow", "clip"], default="gram",
                        help="gram = MSE on similarity Grams; cliprow = CLIP-style "
                             "soft-target contrast on similarity rows; clip = two-tower "
                             "InfoNCE (sketched fingerprint vs projected usage)")
    parser.add_argument("--clip_k", type=int, default=128,
                        help="sketch/embedding dim for geom_mode=clip")
    parser.add_argument("--tau_t", type=float, default=0.1,
                        help="target-side softmax temperature for geom_mode=cliprow")
    parser.add_argument("--count", type=float, default=0.0,
                        help="per-sample gate-count (co-firing merge) pressure")
    parser.add_argument("--act", type=float, default=0.0,
                        help="internal-activation reconstruction under gated forward")
    parser.add_argument("--jac", type=float, default=0.0,
                        help="Jacobian (tangent-space) matching: perturb the residual "
                             "before a sampled layer by eps*u; match the gated model's "
                             "finite-difference output response to the target's in "
                             "direction (cosine) and magnitude (log-norm-ratio^2). "
                             "Gates detached. Testing only.")
    parser.add_argument("--jac_beta", type=float, default=1.0)
    parser.add_argument("--jac_eps", type=float, default=0.1,
                        help="perturbation scale relative to resid RMS")
    parser.add_argument("--ig_steps", type=int, default=1,
                        help=">1 = IG-over-mask attribution: average grads/pre-acts "
                             "over K points on the all-components-scaled path before "
                             "squaring (own-gate-at-zero counterfactual)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=["benchmark", "paper"], default="benchmark",
                        help="benchmark = repo TMDR target; paper = APD-paper-dims 2L target")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    if args.target == "paper":
        from nano_apd.models import ResidMLPConfig
        from nano_apd.run_resid import get_target
        rc = ResidMLPConfig(n_instances=1, n_features=100, d_embed=1000, d_mlp=25,
                            n_layers=2, feature_probability=0.01, batch_size=2048,
                            steps=10_000, seed=0)
        target = get_target(rc, device)
        value_range = (-1.0, 1.0)
    else:
        target = load_tmdr_target(device)
        value_range = (0.0, 1.0)
    cfg = target.config
    apd = ResidMLPAPDModel(cfg, C=args.C, m=args.m, init_scale=1.0).to(device)
    apd.W_E.data[:] = target.W_E.data.clone()
    apd.W_U.data[:] = target.W_U.data.clone()
    if apd.bias1 is not None:
        for l in range(cfg.n_layers):
            apd.bias1[l].data[:] = target.bias1[l].data.clone()

    param_names = target.param_names()
    target_weights = {n: target.weights()[n].detach() for n in param_names}
    n_params = sum(w.numel() for w in target_weights.values())
    dataset = SparseFeatureDataset(1, cfg.n_features, cfg.feature_probability, device,
                                   value_range)
    geom_head, logit_scale, geom_P = None, None, None
    if args.geom > 0 and args.geom_mode != "gram":
        import math
        logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(1 / 0.07), device=device))
        extra = [logit_scale]
        if args.geom_mode == "clip":
            geom_head = torch.nn.Linear(args.C, args.clip_k, bias=False).to(device)
            gen = torch.Generator(device=device).manual_seed(0)
            geom_P = torch.randn(n_params, args.clip_k, device=device,
                                 generator=gen) / (n_params ** 0.5)
            extra += list(geom_head.parameters())
        opt = torch.optim.AdamW([{"params": apd.parameters()},
                                 {"params": extra}], lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(apd.parameters(), lr=args.lr, weight_decay=0.0)
    lr_fn = get_lr_schedule_fn("cosine")

    for step in range(args.steps + 1):
        step_lr = get_lr_with_warmup(step, args.steps, args.lr, lr_fn, 0.01)
        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)

        batch = dataset.generate_batch(args.batch_size)
        batch = batch[(batch.abs().sum(dim=(1, 2)) > 0)]         # drop empty samples
        target_out, tcache = target(batch)

        if args.ig_steps > 1:
            A = calc_grad_attributions_ig(
                target, batch,
                {n: apd.component_weights()[n] for n in param_names},
                K=args.ig_steps,
            )                                                    # [B, 1, C], >= 0
        else:
            A = calc_grad_attributions(
                target_out=target_out,
                pre_weight_acts={n: tcache[n]["pre"] for n in param_names},
                post_weight_acts={n: tcache[n]["post"] for n in param_names},
                component_weights={n: apd.component_weights()[n] for n in param_names},
                C=args.C,
            )                                                    # [B, 1, C], >= 0

        # gates: normalized attribution shares, differentiable in the components
        denom = (A.sum(dim=-1, keepdim=True) if args.gate_norm == "sum"
                 else A.max(dim=-1, keepdim=True).values)
        g = (A / (denom + 1e-12)) ** args.tau
        if args.gate_thresh > 0:
            g = F.relu(g - args.gate_thresh) / (1 - args.gate_thresh)
        out_g, gcache = apd(batch, topk_mask=g)
        # internal-activation reconstruction under the gated forward: forbids
        # re-implementing a mechanism's function in the wrong layer (a shortcut matches
        # the output but produces the wrong hidden states)
        loss_act = torch.tensor(0.0, device=device)
        if args.act > 0:
            for l in range(cfg.n_layers):
                n_in, n_out = f"layers.{l}.mlp_in", f"layers.{l}.mlp_out"
                b_l = apd.bias1[l] if apd.bias1 is not None else 0.0
                h_t = F.relu(tcache[n_in]["post"].detach() + b_l)
                h_g = F.relu(gcache[n_in]["post"] + b_l)
                loss_act = loss_act + ((h_g - h_t) ** 2).sum(dim=-1).mean()
                # constrain the WRITE side too: each layer's contribution to the
                # residual stream (a layer-0 write of ~0 is a shortcut, not a mechanism)
                w_t = tcache[n_out]["post"].detach()
                w_g = gcache[n_out]["post"]
                loss_act = loss_act + ((w_g - w_t) ** 2).sum(dim=-1).mean()
            loss_act = loss_act / (2 * cfg.n_layers)

        loss_faith = sum(
            ((apd.weights()[n] - target_weights[n]) ** 2).sum() for n in param_names
        ) / n_params
        loss_recon = calc_recon_mse(out_g, target_out).mean()

        Ac = A[:, 0]                                             # [B, C]
        off = ~torch.eye(Ac.shape[0], dtype=torch.bool, device=device)
        loss_geom = torch.tensor(0.0, device=device)
        if args.geom > 0:
            # usage-geometry matching: the components' per-sample usage similarity
            # matrix should mirror the target model's own per-PARAMETER usage
            # similarity (which separates mechanisms at 0.99 — diag_split.py).
            # Merged components fake u=1 for different-mechanism pairs; w exposes them.
            gps = torch.autograd.grad(
                (0.5 * (target_out ** 2)).sum(),
                [tcache[n]["post"] for n in param_names], retain_graph=True)
            with torch.no_grad():
                vs = []
                for n, gp in zip(param_names, gps, strict=True):
                    pre = tcache[n]["pre"].detach()[:, 0]
                    W = target_weights[n][0]
                    v = einops.einsum(pre, gp[:, 0], "b a, b c -> b a c") * W
                    vs.append(v.reshape(v.shape[0], -1))
                Vg = F.normalize(torch.cat(vs, dim=1), dim=1)
            if args.geom_mode == "gram":
                with torch.no_grad():
                    w_sim = (Vg @ Vg.t())[off]
                u_full = (F.normalize(Ac, dim=-1) @ F.normalize(Ac, dim=-1).t())[off]
                loss_geom = ((u_full - w_sim) ** 2).mean()
            elif args.geom_mode == "cliprow":
                # CLIP-style soft-target contrast: each sample's fingerprint-similarity
                # row (softmaxed) is the target distribution for its usage-cosine row
                with torch.no_grad():
                    tgt = F.softmax((Vg @ Vg.t()) / args.tau_t, dim=-1)
                scale = logit_scale.exp().clamp(max=100.0)
                An = F.normalize(Ac, dim=-1)
                loss_geom = -(tgt * F.log_softmax((An @ An.t()) * scale,
                                                  dim=-1)).sum(-1).mean()
            else:  # clip: two-tower InfoNCE, sketched fingerprint vs projected usage
                with torch.no_grad():
                    ef = F.normalize(Vg @ geom_P, dim=-1)
                eu = F.normalize(geom_head(Ac), dim=-1)
                scale = logit_scale.exp().clamp(max=100.0)
                logits = (eu @ ef.t()) * scale
                labels = torch.arange(logits.shape[0], device=device)
                loss_geom = 0.5 * (F.cross_entropy(logits, labels)
                                   + F.cross_entropy(logits.t(), labels))
        # polarization on batch-CENTERED attributions (correlation): in the all-merged
        # state raw cosines sit at the flat u=1 pole with zero gradient; centered
        # vectors lose that fake agreement, so the degenerate basin is not a minimum
        Ahat = F.normalize(Ac - Ac.mean(dim=0, keepdim=True), dim=-1)
        u = (Ahat @ Ahat.t())[off]
        loss_polar = (u ** 2 * (1 - u) ** 2).mean()
        # basin-breaker: mean RAW pair-cosine should be low (two random sparse inputs
        # rarely share a feature) — constant gradient out of the global-merge state
        u_raw = (F.normalize(Ac, dim=-1) @ F.normalize(Ac, dim=-1).t())[off]
        loss_prior = (u_raw.mean() - args.pair_prior) ** 2

        loss_jac = torch.tensor(0.0, device=device)
        if args.jac > 0:
            l_s = int(torch.randint(cfg.n_layers, (1,)).item())
            rms = tcache[f"layers.{l_s}.mlp_in"]["pre"].detach().float().pow(2).mean().sqrt()
            eps = args.jac_eps * max(rms.item(), 1e-8)
            u = torch.randn(batch.shape[0], 1, cfg.d_embed, device=device)
            u = F.normalize(u, dim=-1) * eps
            g_det = g.detach()
            out_gb, _ = apd(batch, topk_mask=g_det)
            with torch.no_grad():
                out_tp, _ = target(batch, perturb=(l_s, u))
            out_gp, _ = apd(batch, topk_mask=g_det, perturb=(l_s, u))
            d_tgt = (out_tp.float() - target_out.detach().float()) / eps
            d_gat = (out_gp.float() - out_gb.float()) / eps
            cos = F.cosine_similarity(d_gat, d_tgt, dim=-1)
            nrat = ((d_gat.norm(dim=-1) + 1e-6) / (d_tgt.norm(dim=-1) + 1e-6)).log()
            loss_jac = (1 - cos).mean() + args.jac_beta * (nrat ** 2).mean()

        loss_dl = calc_schatten_loss(
            As={n: apd.As()[n] for n in param_names},
            Bs={n: apd.Bs()[n] for n in param_names},
            mask=g, p=args.dl_pnorm, n_params=n_params,
        ).mean()

        # gentle per-sample gate-count pressure: one cross-layer component and two
        # co-firing layer-local components have identical usage geometry; this is the
        # only term that prefers the former (pays 1 gate instead of 2)
        loss_count = g.sum(dim=-1).mean()

        loss = (args.faith * loss_faith + args.recon * loss_recon
                + args.polar * loss_polar + args.dl * loss_dl
                + args.prior * loss_prior + args.geom * loss_geom
                + args.count * loss_count + args.act * loss_act
                + args.jac * loss_jac)

        if step % 250 == 0:
            with torch.no_grad():
                frac_high = (u > 0.8).float().mean().item()
                frac_low = (u < 0.2).float().mean().item()
                eff_gates = g.sum(dim=-1).mean().item()
            print(json.dumps({
                "step": step, "total": round(loss.item(), 5),
                "faith": round(loss_faith.item(), 6), "recon": round(loss_recon.item(), 6),
                "jac": round(loss_jac.item(), 4),
                "polar": round(loss_polar.item(), 5), "dl": round(loss_dl.item(), 6),
                "prior": round(loss_prior.item(), 5), "mean_u_raw": round(u_raw.mean().item(), 3),
                "pair_frac_hi": round(frac_high, 3), "pair_frac_lo": round(frac_low, 3),
                "eff_gates": round(eff_gates, 1), "lr": round(step_lr, 6),
            }), flush=True)

        if step != args.steps:
            loss.backward()
            opt.step()

    out_dir = OUT_ROOT / (f"{args.target}_polar_C{args.C}_p{args.polar:g}_dl{args.dl:g}"
                          f"_tau{args.tau:g}{('_' + args.tag) if args.tag else ''}"
                          if args.target != "benchmark" else
                          f"tmdr_polar_C{args.C}_p{args.polar:g}_dl{args.dl:g}"
                          f"_tau{args.tau:g}{('_' + args.tag) if args.tag else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(apd.state_dict(), out_dir / "apd_model.pth")
    with open(out_dir / "apd_config.json", "w") as f:
        json.dump({"C": args.C, "m": apd.m, **vars(args)}, f, indent=2)

    summary = eval_tmdr(target, apd, device, value_range=value_range)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "best_component_per_feature"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
