"""W&B sweep over coeff_comp_size -- the QUADRATIC per-component size tax (mean over components of
(atoms-in-component)^2), which hits the biggest components hardest. Unlike CM (linear total-mass,
size-blind across columns) this is a genuinely different signal. CM is turned OFF here to isolate
it; tau_end=0.05, assign/membership-entropy off.

Usage (one agent per GPU):

    python -m nano_param_decomp.sweep_compsize create
    CUDA_VISIBLE_DEVICES=0 SWEEP_ID=<id> python -m nano_param_decomp.sweep_compsize agent
    CUDA_VISIBLE_DEVICES=1 SWEEP_ID=<id> python -m nano_param_decomp.sweep_compsize agent
"""

import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

from . import matryoshka
from .ss2l import C_PER_MODULE_SS_2L, generate_pool, load_ss2l_target, pool_loader

STEPS = int(os.environ.get("STEPS", "4000"))
SEQ = int(os.environ.get("SEQ", "256"))
B = int(os.environ.get("B", "32"))
G = int(os.environ.get("G", "1024"))
N_POOL = int(os.environ.get("N_POOL", "1024"))

ENTITY = "rohan-kathuria-neu"
PROJECT = "matryoshka-pd"
GROUP = "ss2l-compsize"

CMP_DIR = os.environ.get("CMP_DIR", "/tmp/matry_compare")
POOL_PATH = os.path.join(CMP_DIR, "pool.pt")

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "final/eval/comp_mean_size", "goal": "minimize"},
    "parameters": {"ccs": {"values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3]}},
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_pools() -> tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(POOL_PATH):
        os.makedirs(CMP_DIR, exist_ok=True)
        pool = generate_pool(load_ss2l_target(), N_POOL, SEQ, _device(), seed=0)
        torch.save(pool, POOL_PATH)
    pool = torch.load(POOL_PATH, weights_only=True)
    n_tr = int(0.9 * pool.shape[0])
    return pool[:n_tr], pool[n_tr:]


def train() -> None:
    import wandb

    wandb.init(entity=ENTITY, project=PROJECT, group=GROUP, job_type="compsize")
    ccs = float(wandb.config.ccs)
    wandb.run.name = f"ccs{ccs}-s{STEPS}"

    train_pool, eval_pool = _load_pools()
    matryoshka.decompose(
        load_ss2l_target(),
        matryoshka.Config(
            C_per_module=C_PER_MODULE_SS_2L,
            n_components=G,
            coeff_membership=0.0,
            coeff_comp_size=ccs,
            coeff_membership_entropy=0.0,
            coeff_assign_entropy=0.0,
            tau_end=0.05,
            seq_len=SEQ,
            batch_size=B,
            eval_batch_size=B,
            n_steps=STEPS,
            faithfulness_warmup_steps=200,
            ci_d_model=512,
            ci_n_blocks=4,
            ci_n_heads=8,
            ci_mlp_hidden=2048,
            coeff_imp=0.001,
            main_lr=3e-4,
            log_every=200,
            eval_freq=max(STEPS // 6, 500),
            use_wandb=True,
            wandb_project=PROJECT,
            wandb_entity=ENTITY,
            wandb_group=GROUP,
            wandb_job_type="compsize",
            wandb_tags=("ss2l", "2L", "norm-agg", "compsize", f"G{G}"),
            wandb_notes="quadratic per-component size tax sweep; CM off",
        ),
        pool_loader(train_pool, B, seed=0),
        pool_loader(eval_pool, B, seed=1000),
    )
    wandb.finish()


def main() -> None:
    import wandb

    cmd = sys.argv[1] if len(sys.argv) > 1 else "create"
    if cmd == "create":
        sweep_id = wandb.sweep(SWEEP_CONFIG, entity=ENTITY, project=PROJECT)
        print(f"SWEEP_ID={sweep_id}", flush=True)
    elif cmd == "agent":
        wandb.agent(os.environ["SWEEP_ID"], function=train, entity=ENTITY, project=PROJECT)
    else:
        raise ValueError(f"unknown command {cmd!r} (use 'create' or 'agent')")


if __name__ == "__main__":
    main()
