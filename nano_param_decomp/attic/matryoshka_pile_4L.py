"""Matryoshka PD on the VPD paper's 4-layer LlamaSimpleMLP / Pile target.

Reuses the target loader + per-module atom counts from `pile_4L.py`; swaps the decomposition
for the matryoshka variant (G cross-layer components with a learned soft->hard assignment).

    # 8-GPU single-node
    torchrun --standalone --nproc_per_node=8 -m nano_param_decomp.matryoshka_pile_4L
    # single-GPU
    python -m nano_param_decomp.matryoshka_pile_4L
"""

import os

from .matryoshka import Config, decompose
from .pile_4L import C_PER_MODULE_4L, load_paper_target_model, make_loader

if __name__ == "__main__":
    cfg = Config(
        C_per_module=C_PER_MODULE_4L,
        n_components=1024,
        use_wandb=True,
        wandb_run_name="matryoshka_pile_4L_G1024",
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    decompose(
        load_paper_target_model(),
        cfg,
        make_loader(cfg.batch_size, cfg.seq_len, rank, world_size, "train", cfg.seed),
        make_loader(cfg.eval_batch_size, cfg.seq_len, rank, world_size, "val", cfg.seed + 1),
    )
