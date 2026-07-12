"""APD-basis decomposition of Pythia-14M (EleutherAI, 6 layers, Pile-trained, MLPs included).

The first target past attn-only-2l. NO circuit answer key is assumed (per Rohan: we don't need to
"find induction"); evaluation is the intrinsic suite — faithfulness (CE-recovered / KL, all-on
sanity, adversarial KL, L0) plus two diagnostics this model makes newly interesting:

  - PER-MODULE l1 ratio: attention vs MLP matrices may live in different regimes (the Arora et al.
    neuron-basis-sparsity argument applies to MLPs, not attention subspaces) — the ratio measures,
    per matrix, how far the healthy carving is from disjoint support, i.e. where an L1 would be
    safe/destructive BEFORE we dose it.
  - Component module-fingerprints: each component's weight-mass split over (layer, matrix-type),
    the whole-network analog of the head fingerprints.

Decomposed: all 24 per-layer Linears (fused qkv [384,128], attn out [128,128], mlp up [512,128],
mlp down [128,512]). Biases stay frozen (papers do the same). Reuses apd_lm.decompose_lm verbatim.

Run:  CUDA_VISIBLE_DEVICES=0 python -m nano_param_decomp.apd_pythia
Env:  STEPS, C, R, IMP, HIDDEN, L1, INTER, FAITH, SEQ, B, SMOKE, SAVE
"""

import os

import torch
from torch import Tensor

from .apd_lm import adversarial_kl, decompose_lm, faithfulness_eval
from .apd_mask import ApdConfig
from .pythia14m import C_PER_MODULE_PYTHIA_14M, generate_pool, load_pythia14m_target
from .run import Config as VpdConfig, init_dist

POOL_PATH = "/tmp/pythia_compare/pool.pt"


def per_module_l1_ratio(banks) -> dict[str, float]:
    out = {}
    for n, b in banks.items():
        if b.impl == "factored" and b.r == 1:
            s = (b.A.abs().sum(dim=(1, 2)) * b.B.abs().sum(dim=(1, 2))).sum().item()
        else:
            s = b.materialized_weights().abs().sum().item()
        out[n] = s / (b.W_target.abs().sum().item() + 1e-12)
    return out


@torch.no_grad()
def module_fingerprints(banks, topk_comps: Tensor) -> None:
    """Weight-mass split of each top component over (layer, matrix-type)."""
    kinds = ("query_key_value", "attention.dense", "dense_h_to_4h", "dense_4h_to_h")
    short = {"query_key_value": "qkv", "attention.dense": "attnO", "dense_h_to_4h": "mlpUp",
             "dense_4h_to_h": "mlpDn"}
    for c in topk_comps.tolist():
        mass: dict[str, float] = {}
        layer_mass: dict[int, float] = {}
        for n, b in banks.items():
            w = b.materialized_weights()[c]
            m = w.pow(2).sum().item()
            layer = int(n.split(".")[2])
            kind = next(k for k in kinds if k in n)
            mass[short[kind]] = mass.get(short[kind], 0.0) + m
            layer_mass[layer] = layer_mass.get(layer, 0.0) + m
        tot = sum(mass.values()) + 1e-12
        kinds_s = " ".join(f"{k}:{v / tot:.2f}" for k, v in sorted(mass.items()))
        top_layers = sorted(layer_mass.items(), key=lambda kv: -kv[1])[:3]
        layers_s = " ".join(f"L{l}:{v / tot:.2f}" for l, v in top_layers)
        print(f"  comp {c:>3}: {kinds_s} | top layers {layers_s}", flush=True)


def _run() -> None:
    # DDP: launch with `torchrun --standalone --nproc_per_node=2 -m nano_param_decomp.apd_pythia`.
    # B env is the GLOBAL batch; each rank runs B/world locally, grads all-reduced in decompose_lm.
    if os.environ.get("TF32", "1") == "1":
        # fp32 matmul on H100 CUDA cores is ~1/7 of TF32 tensor-core throughput; TF32's ~1e-3
        # relative precision is far below this loop's SGD noise floor (train faith ~1e-6). The VPD
        # baseline stack (run.py) already trains under bf16 autocast.
        torch.set_float32_matmul_precision("high")
    rank, world, _local_rank, device = init_dist()
    rank0 = rank == 0
    smoke = os.environ.get("SMOKE", "0") == "1"
    steps = int(os.environ.get("STEPS", "300" if smoke else "3000"))
    C = int(os.environ.get("C", "512"))
    fr = os.environ.get("R", "1")
    R = int(fr) if fr else None
    # per-matrix-type rank caps (anatomy: MLP pieces saturate the cap, attention differentiates
    # below it). R_MLP raises the cap on the two MLP matrices only; empty -> uniform R everywhere.
    r_mlp = os.environ.get("R_MLP", "")
    rank_map = ({"dense_h_to_4h": int(r_mlp), "dense_4h_to_h": int(r_mlp)} if r_mlp else None)
    imp = float(os.environ.get("IMP", "1e-3"))
    hidden = float(os.environ.get("HIDDEN", "1.0"))
    l1 = float(os.environ.get("L1", "0.0"))
    inter = float(os.environ.get("INTER", "0.0"))
    nested = os.environ.get("NESTED", "0") == "1"     # V2: Matryoshka nested rank prefixes
    # env is TRIM, not RANK: torchrun exports RANK (the process rank) to every DDP worker, which
    # silently clobbers a RANK knob with 0/1/... per rank -- divergent configs across ranks.
    rank_pen = float(os.environ.get("TRIM", "0.0"))   # V3: capacity-x-usage piece-count trim
    rank_floor = float(os.environ.get("RANKFLOOR", "0.005"))
    frob = float(os.environ.get("FROB", "0.0"))       # V1 (rejected on toys; kept for ablations)
    # L1 fights the faithfulness pin; when L1 is on, default the pin 10x higher (attn2l lesson)
    faith = float(os.environ.get("FAITH", "1e8" if l1 > 0 else "1e7"))
    seq_len = int(os.environ.get("SEQ", "128"))
    batch_global = int(os.environ.get("B", "64"))
    lr = float(os.environ.get("LR", "5e-4"))          # decompose_lm default; lower for big C / small B
    grad_clip = float(os.environ.get("GRADCLIP", "0.0"))  # clip component-factor grad norm (spike guard)
    warmup = int(os.environ.get("WARMUP", "100" if smoke else "400"))  # faithfulness warmup steps
    assert batch_global % world == 0, "global batch must divide by world size"
    batch = batch_global // world
    seed = int(os.environ.get("SEED", "0"))

    model = load_pythia14m_target().float()  # HF ships fp16; our banks/CI are fp32
    assert os.path.exists(POOL_PATH) or world == 1, "generate the pool single-process first"
    if os.path.exists(POOL_PATH):
        pool = torch.load(POOL_PATH, weights_only=True)
        if rank0:
            print(f"loaded pool {tuple(pool.shape)}", flush=True)
    else:
        print("generating pool (autoregressive samples from the frozen target) ...", flush=True)
        pool = generate_pool(model, n_seqs=(256 if smoke else 2048), seq_len=max(seq_len, 256),
                             device=device, seed=0)
        os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)
        torch.save(pool, POOL_PATH)
    pool = pool[:, :seq_len]

    modules = list(C_PER_MODULE_PYTHIA_14M.keys())
    cfg = ApdConfig(modules=modules, n_components=C, simplicity_impl="factored", factor_rank=R,
                    factor_rank_map=rank_map,
                    lowrank_forward=True, coeff_faith=faith, coeff_imp=imp, coeff_simplicity=0.0,
                    coeff_hidden=hidden, coeff_weight_l1=l1, coeff_interaction=inter,
                    nested_rank=nested, coeff_rank=rank_pen, rank_freq_floor=rank_floor,
                    coeff_frob=frob, grad_clip=grad_clip,
                    p_start=2.0, p_end=0.4, seed=seed,
                    use_wandb=os.environ.get("WANDB", "0") == "1",
                    wandb_project=os.environ.get("WANDB_PROJECT", "apd-basis"),
                    wandb_group=os.environ.get("WANDB_GROUP", "pythia14m"),
                    wandb_job_type="pythia14m",
                    wandb_run_name=os.environ.get("WANDB_NAME"))
    ci_cfg = VpdConfig(C_per_module=C_PER_MODULE_PYTHIA_14M, seq_len=seq_len,
                       ci_d_model=256, ci_n_blocks=4, ci_n_heads=8, ci_mlp_hidden=1024,
                       coeff_stoch=0.5, coeff_ppgd=0.5, ppgd_lr=0.01, ppgd_inner_steps=2)
    if rank0:
        print(f"config: C={C} R={fr or 'full'} R_mlp={r_mlp or R} steps={steps} imp={imp} "
              f"hidden={hidden} l1={l1} inter={inter} nested={nested} rank={rank_pen} "
              f"rank_floor={rank_floor} frob={frob} faith={faith:g} lr={lr:g} "
              f"grad_clip={grad_clip:g} warmup={warmup} seq={seq_len} "
              f"B={batch_global} (x{world} ranks)", flush=True)

    out = decompose_lm(model, pool, cfg, ci_cfg, device, n_steps=steps, batch=batch, lr=lr,
                       seq_len=seq_len, warmup_steps=warmup,
                       save_path=os.environ.get("SAVE", "/tmp/pythia_compare/apd_pythia.pt"))
    if not rank0:  # final evaluation is rank-0 work
        import torch.distributed as dist
        dist.destroy_process_group()
        return
    ev = pool[:batch, :seq_len].to(device)
    fe = faithfulness_eval(out["model"], out["banks"], out["ci"], ev, cfg, device)
    adv = adversarial_kl(out["model"], out["banks"], out["ci"], ev, cfg, device)
    print("\n=== faithfulness (paper metric) ===", flush=True)
    print(f"kl_ci_masked={fe['kl_ci_masked']:.4f}  ce_recovered={fe['ce_recovered_pct']:.1f}%  "
          f"kl_adversarial={adv:.4f}  kl_unmasked(sanity)={fe['kl_unmasked']:.2e}  "
          f"L0={fe['L0']:.1f}/{C}  l1_ratio={fe['l1_ratio']:.2f}", flush=True)
    print("\n=== per-module l1 ratio (regime diagnostic: ~1 neuron-aligned, >>1 superposed) ===", flush=True)
    for n, r in sorted(per_module_l1_ratio(out["banks"]).items()):
        print(f"  {r:>7.2f}  {n}", flush=True)
    # fingerprints of the most-used components
    with torch.no_grad():
        from .apd_mask import clear_masks, refresh_caches
        clear_masks(out["banks"])
        _ = out["model"](ev)
        acts = {n: b.last_input for n, b in out["banks"].items()}
        refresh_caches(out["banks"])
        _, g_upper = out["ci"](acts)
        top = g_upper.mean(dim=(0, 1)).topk(10).indices
    print("\n=== top-component module fingerprints ===", flush=True)
    module_fingerprints(out["banks"], top)
    if cfg.use_wandb:
        import wandb  # type: ignore[import-untyped]
        if wandb.run is not None:
            wandb.log({**{f"final/{k}": v for k, v in fe.items()}, "final/kl_adversarial": adv,
                       **{f"final/l1_ratio_module/{n}": r
                          for n, r in per_module_l1_ratio(out["banks"]).items()}})
            wandb.finish()
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    print("APD_PYTHIA DONE", flush=True)


if __name__ == "__main__":
    _run()
