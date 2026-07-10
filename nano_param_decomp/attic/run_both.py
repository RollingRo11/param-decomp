"""Single run with BOTH membership penalties on: coeff_membership (atom participation, rows of M)
and coeff_comp_size (quadratic component size, columns of M). Tunable via env CM / CCS."""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

from . import matryoshka
from .ss2l import C_PER_MODULE_SS_2L, generate_pool, load_ss2l_target, pool_loader

STEPS = int(os.environ.get("STEPS", "10000"))
SEQ = int(os.environ.get("SEQ", "256"))
B = int(os.environ.get("B", "32"))
G = int(os.environ.get("G", "1024"))
# NORMALIZED-penalty coefficients (membership_l1=mean gate, component_size_l2=mean (size/A)^2,
# both [0,1]). CM~300 (=old 0.3 x G), CCS~5000 (=old 1e-4 x A^2) reproduce the validated regime.
CM = float(os.environ.get("CM", "300"))
CCS = float(os.environ.get("CCS", "5000"))
IMP_ATOMS = float(os.environ.get("IMP_ATOMS", "0.001"))

ENTITY = "rohan-kathuria-neu"
PROJECT = "matryoshka-pd"
GROUP = "ss2l-both"

CMP_DIR = os.environ.get("CMP_DIR", "/tmp/matry_compare")
POOL_PATH = os.path.join(CMP_DIR, "pool.pt")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_B = B // world
    if rank == 0 and not os.path.exists(POOL_PATH):
        os.makedirs(CMP_DIR, exist_ok=True)
        torch.save(generate_pool(load_ss2l_target(), 1024, SEQ, device, seed=0), POOL_PATH)
    pool = torch.load(POOL_PATH, weights_only=True)
    n_tr = int(0.9 * pool.shape[0])
    train_pool, eval_pool = pool[:n_tr], pool[n_tr:]

    mat = matryoshka.decompose(
        load_ss2l_target(),
        matryoshka.Config(
            C_per_module=C_PER_MODULE_SS_2L,
            n_components=G,
            coeff_membership=CM,
            coeff_comp_size=CCS,
            coeff_imp_atoms=IMP_ATOMS,
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
            wandb_job_type="both",
            wandb_run_name=f"both-CM{CM}-CCS{CCS}-impA{IMP_ATOMS}-s{STEPS}",
            wandb_tags=("ss2l", "2L", "norm-agg", "both", "imp-atoms", f"G{G}"),
            wandb_notes="CM + CCS membership penalties + atom-level importance minimality",
        ),
        pool_loader(train_pool, local_B, seed=rank),
        pool_loader(eval_pool, local_B, seed=1000 + rank),
    )
    if rank == 0:
        import wandb

        wandb.finish()
        print("FINAL:", {k: v for k, v in mat.items() if isinstance(v, (int, float))}, flush=True)


if __name__ == "__main__":
    main()
