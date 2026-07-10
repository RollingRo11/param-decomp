"""Timed micro-benchmark of the matryoshka training step on pythia-14m at a given batch, DDP on N
GPUs. Reports s/step and peak GPU memory so we can size the overnight run. Mirrors the real
decompose() inner loop (target+CI forward, faith/imp/imp_atoms/stoch/ppgd losses, penalties,
backward, grad all-reduce, opt step) but skips warmup/eval/wandb.

    B=128 torchrun --standalone --nproc_per_node=2 -m nano_param_decomp.bench_pythia
"""

import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import torch.distributed as dist

from .matryoshka import (
    Config,
    MatryoshkaCI,
    ComponentAssignment,
    MatryoshkaModule,
    PersistentPGD,
    _all_reduce_grads,
    anneal_p,
    cosine_lr,
    sample_component_masks,
    stochastic_recon_loss,
    tau_at,
)
from .run import (
    faithfulness_loss,
    importance_minimality_loss,
    init_dist,
    install_components,
)
from .pythia14m import C_PER_MODULE_PYTHIA_14M, generate_pool, load_pythia14m_target, pool_loader

B = int(os.environ.get("B", "128"))
SEQ = int(os.environ.get("SEQ", "256"))
WARM = int(os.environ.get("WARM", "5"))
MEASURE = int(os.environ.get("MEASURE", "30"))


def main() -> None:
    rank, world_size, local_rank, device = init_dist()
    local_B = B // world_size
    cfg = Config(
        C_per_module=C_PER_MODULE_PYTHIA_14M, n_components=1024, seq_len=SEQ,
        batch_size=B, eval_batch_size=B, n_steps=100000, faithfulness_warmup_steps=0,
        ci_d_model=512, ci_n_blocks=4, ci_n_heads=8, ci_mlp_hidden=2048,
        coeff_imp=0.001, coeff_imp_atoms=0.0, coeff_membership=300.0, coeff_comp_size=5000.0,
        tau_end=0.05, main_lr=6e-4, use_wandb=False,
    )
    target = load_pythia14m_target().to(device)
    pool = generate_pool(target, 64, SEQ, device, seed=0)
    loader = pool_loader(pool, local_B, seed=rank)

    wrappers = install_components(target, cfg.C_per_module)
    module_order = sorted(wrappers.keys())
    d_in = {n: int(w.W_target.shape[1]) for n, w in wrappers.items()}
    ci_fn = MatryoshkaCI(d_in, cfg).to(device)
    assign = ComponentAssignment(cfg.C_per_module, module_order, cfg).to(device)
    module = MatryoshkaModule(target, ci_fn, assign, wrappers).to(device)
    ppgd = PersistentPGD(module_order, cfg.n_components, local_B, cfg.seq_len, device, cfg)
    comp_params = [p for w in wrappers.values() for p in (w.V, w.U)]
    main_params = comp_params + list(ci_fn.parameters()) + [assign.M_logits]
    opt = torch.optim.AdamW(main_params, lr=cfg.main_lr)

    def step(i: int) -> None:
        tau = tau_at(i, cfg)
        p = anneal_p(i, cfg.n_steps, cfg.p_start, cfg.p_end)
        ppgd_lr = cosine_lr(i, cfg.n_steps, cfg.ppgd_lr, cfg.ppgd_lr_final_frac, cfg.ppgd_warmup_pct)
        ids = next(loader).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tlogits, ci_lower, ci_upper = module(ids)
            ppgd.warmup(target, wrappers, assign, ids, tlogits, ci_lower, tau, ppgd_lr)
            lf = faithfulness_loss(wrappers)
            li = importance_minimality_loss({"components": ci_upper}, p, cfg.imp_eps, cfg.imp_beta, world_size)
            ls = stochastic_recon_loss(target, wrappers, assign, ids, tlogits, ci_lower, tau)
            lp = ppgd.recon_loss(target, wrappers, assign, ids, tlogits, ci_lower, tau)
            la = assign.entropy(tau)
            lm = assign.membership_l1(tau)
            lcs = assign.component_size_l2(tau)
        total = (cfg.coeff_faith * lf + cfg.coeff_imp * li + cfg.coeff_stoch * ls
                 + cfg.coeff_ppgd * lp + cfg.coeff_assign_entropy * la
                 + cfg.coeff_membership * lm + cfg.coeff_comp_size * lcs)
        pg = torch.autograd.grad(lp, list(ppgd.sources.values()), retain_graph=True)
        opt.zero_grad()
        total.backward()
        _all_reduce_grads(main_params, world_size)
        torch.nn.utils.clip_grad_norm_(comp_params, cfg.grad_clip_components)
        opt.step()
        ppgd.external_step(dict(zip(ppgd.sources, pg, strict=True)), ppgd_lr)

    for i in range(WARM):
        step(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(WARM, WARM + MEASURE):
        step(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / MEASURE
    if rank == 0:
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        print(f"BENCH B={B} world={world_size} | {dt:.3f} s/step | {B/dt:.0f} samples/s | "
              f"peak {peak:.1f} GB/GPU | 50k steps -> {dt*50000/3600:.1f} h", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
