"""W&B sweep over the membership-minimality coefficient (coeff_membership / "CM") for matryoshka
PD on the frozen 2-layer SimpleStories target. No baseline -- VPD reference is already known; this
sweep only pushes CM harder to drive components small/sparse and find where recon starts to suffer.

Two-step usage (one agent per GPU drains the grid in parallel):

    # 1. create the sweep, prints a SWEEP_ID
    python -m nano_param_decomp.sweep_cm create

    # 2. launch an agent on each GPU (run both, backgrounded)
    CUDA_VISIBLE_DEVICES=0 SWEEP_ID=<id> python -m nano_param_decomp.sweep_cm agent
    CUDA_VISIBLE_DEVICES=1 SWEEP_ID=<id> python -m nano_param_decomp.sweep_cm agent
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
GROUP = "ss2l-cmsweep"

CMP_DIR = os.environ.get("CMP_DIR", "/tmp/matry_compare")
POOL_PATH = os.path.join(CMP_DIR, "pool.pt")

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "final/eval/kl_ci_masked", "goal": "minimize"},
    "parameters": {"cm": {"values": [0.1, 0.3, 1.0, 3.0]}},
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

    # The sweep agent provides `cm` via wandb.config; this init creates the run and merges it.
    wandb.init(entity=ENTITY, project=PROJECT, group=GROUP, job_type="cmsweep")
    cm = float(wandb.config.cm)
    wandb.run.name = f"cm{cm}-s{STEPS}"

    train_pool, eval_pool = _load_pools()
    matryoshka.decompose(
        load_ss2l_target(),
        matryoshka.Config(
            C_per_module=C_PER_MODULE_SS_2L,
            n_components=G,
            coeff_membership=cm,
            coeff_assign_entropy=float(os.environ.get("CE", "0.01")),
            tau_end=float(os.environ.get("TAU_END", "0.05")),
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
            wandb_job_type="cmsweep",
            wandb_tags=("ss2l", "2L", "norm-agg", "cmsweep", f"G{G}"),
            wandb_notes="weighted-avg aggregation; CM grid sweep",
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
        sweep_id = os.environ["SWEEP_ID"]
        wandb.agent(sweep_id, function=train, entity=ENTITY, project=PROJECT)
    else:
        raise ValueError(f"unknown command {cmd!r} (use 'create' or 'agent')")


if __name__ == "__main__":
    main()
