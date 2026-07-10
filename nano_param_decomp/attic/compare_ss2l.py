"""Head-to-head: normal VPD (per-atom) vs matryoshka (G shareable cross-layer components) on the
same frozen 2-layer SimpleStories target, same model-sampled data / seed / budget. Logs both to
W&B (minimal metrics) under one group.

Runs each method on its own GPU in parallel via MODE (see orchestration at the bottom):
    MODE=genpool      -> sample the shared data pool once, save to disk
    MODE=baseline     -> load pool, run VPD, dump metrics json   (pin to one GPU)
    MODE=matryoshka   -> load pool, run matryoshka, dump json    (pin to the other GPU)
    MODE=table        -> load both jsons, print comparison
    MODE=both (default)-> sequential single-GPU (old behaviour)
"""

import json
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

from . import matryoshka, run
from .ss2l import C_PER_MODULE_SS_2L, generate_pool, load_ss2l_target, pool_loader

STEPS = int(os.environ.get("STEPS", "8000"))
SEQ = int(os.environ.get("SEQ", "256"))
B = int(os.environ.get("B", "32"))
G = int(os.environ.get("G", "1024"))
N_POOL = int(os.environ.get("N_POOL", "1024"))
MODE = os.environ.get("MODE", "both")

CMP_DIR = os.environ.get("CMP_DIR", "/tmp/matry_compare")
POOL_PATH = os.path.join(CMP_DIR, "pool.pt")
BASE_JSON = os.path.join(CMP_DIR, "base.json")
MAT_JSON = os.path.join(CMP_DIR, "mat.json")

ENTITY = "rohan-kathuria-neu"
PROJECT = "matryoshka-pd"
GROUP = os.environ.get("WGROUP", "ss2l-2L-sharedM")
TAGS = ["ss2l", "2L", "sharedM", f"G{G}"]

SHARED = dict(
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
    use_wandb=True,
)


def _floats(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def _finish_wandb(final: dict) -> None:
    import wandb

    wandb.summary.update({f"final/{k}": v for k, v in _floats(final).items()})
    wandb.finish()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_pool() -> None:
    os.makedirs(CMP_DIR, exist_ok=True)
    print(f"sampling {N_POOL} seqs (len {SEQ}) from target ...", flush=True)
    pool = generate_pool(load_ss2l_target(), N_POOL, SEQ, _device(), seed=0)
    torch.save(pool, POOL_PATH)
    print(f"saved pool {tuple(pool.shape)} -> {POOL_PATH}", flush=True)


def _load_pools() -> tuple[torch.Tensor, torch.Tensor]:
    pool = torch.load(POOL_PATH, weights_only=True)
    n_tr = int(0.9 * pool.shape[0])
    return pool[:n_tr], pool[n_tr:]


def _ddp_loaders(train_pool, eval_pool):
    """Per-rank data shards: each rank draws local_B-sized batches with a rank-specific seed, so
    the effective batch is world_size x local_B (true data parallelism)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_B = B // world
    return (
        pool_loader(train_pool, local_B, seed=rank),
        pool_loader(eval_pool, local_B, seed=1000 + rank),
        rank,
    )


def run_baseline() -> dict:
    train_pool, eval_pool = _load_pools()
    os.environ["WANDB_ENTITY"] = ENTITY
    os.environ["WANDB_RUN_GROUP"] = GROUP
    os.environ["WANDB_JOB_TYPE"] = "vpd-baseline"
    os.environ["WANDB_TAGS"] = ",".join(TAGS)
    train_loader, eval_loader, rank = _ddp_loaders(train_pool, eval_pool)
    base = run.decompose(
        load_ss2l_target(),
        run.Config(
            C_per_module=C_PER_MODULE_SS_2L,
            slow_eval_on_first_step=False,
            slow_eval_freq=STEPS + 1,
            eval_freq=STEPS + 1,
            wandb_project=PROJECT,
            wandb_run_name=f"vpd-baseline-s{STEPS}",
            **SHARED,
        ),
        train_loader,
        eval_loader,
    )
    if rank == 0:
        _finish_wandb(base)
        json.dump(_floats(base), open(BASE_JSON, "w"), indent=2)
    return base


def run_matry() -> dict:
    train_pool, eval_pool = _load_pools()
    train_loader, eval_loader, rank = _ddp_loaders(train_pool, eval_pool)
    mat = matryoshka.decompose(
        load_ss2l_target(),
        matryoshka.Config(
            C_per_module=C_PER_MODULE_SS_2L,
            n_components=G,
            coeff_membership=float(os.environ.get("CM", "0.003")),
            coeff_assign_entropy=float(os.environ.get("CE", "0.01")),
            tau_end=float(os.environ.get("TAU_END", "0.05")),
            eval_freq=max(STEPS // 6, 500),
            wandb_entity=ENTITY,
            wandb_project=PROJECT,
            wandb_group=GROUP,
            wandb_job_type="matryoshka",
            wandb_run_name=f"matry-G{G}-s{STEPS}-CM{os.environ.get('CM', '0.003')}",
            wandb_tags=tuple(TAGS),
            wandb_notes="shareable sigmoid membership + L1 + entropy; minimality over components",
            **SHARED,
        ),
        train_loader,
        eval_loader,
    )
    if rank == 0:
        _finish_wandb(mat)
        json.dump(_floats(mat), open(MAT_JSON, "w"), indent=2)
    return mat


def table(base: dict, mat: dict) -> None:
    def row(label: str, b, m) -> str:
        bs = f"{b:.4g}" if isinstance(b, (int, float)) else "-"
        ms = f"{m:.4g}" if isinstance(m, (int, float)) else "-"
        return f"{label:<36} {bs:>13} {ms:>13}"

    print(f"\n{'=' * 64}\nCOMPARISON  (steps={STEPS}, G={G})\n{'=' * 64}")
    print(f"{'metric':<36} {'VPD(per-atom)':>13} {'matryoshka':>13}")
    print("-" * 64)
    print(row("faithfulness loss", base.get("eval/loss/FaithfulnessLoss"), mat.get("eval/faithfulness")))
    print(row("KL ci-masked  (lower=better)", base.get("eval/ce_kl/kl_ci_masked"), mat.get("eval/kl_ci_masked")))
    print(row("KL stoch-masked", base.get("eval/ce_kl/kl_stoch_masked"), mat.get("eval/kl_stoch_masked")))
    print(row("CE diff ci-masked", base.get("eval/ce_kl/ce_difference_ci_masked"), mat.get("eval/ce_difference_ci_masked")))
    print(row("L0 atoms / token", base.get("eval/l0/0.0_total"), mat.get("eval/l0_atoms")))
    print("-" * 64)
    print("matryoshka-only:")
    print(row("  L0 components / token", None, mat.get("eval/l0_components")))
    print(row("  mean atoms / component", None, mat.get("eval/comp_mean_size")))
    print(row("  max atoms / component", None, mat.get("eval/comp_max_size")))
    print(row("  shared-atom fraction (>1 comp)", None, mat.get("eval/shared_atom_frac")))
    print(row("  mean layers / component", None, mat.get("eval/comp_mean_layers")))
    print(row("  cross-layer fraction (>=2)", None, mat.get("eval/comp_crosslayer_frac")))
    print(row("  assignment hardness (1=binary)", None, mat.get("eval/assign_hardness")))
    print("=" * 64)


def main() -> None:
    if MODE == "genpool":
        gen_pool()
    elif MODE == "baseline":
        run_baseline()
    elif MODE == "matryoshka":
        run_matry()
    elif MODE == "table":
        table(json.load(open(BASE_JSON)), json.load(open(MAT_JSON)))
    elif MODE == "both":
        gen_pool()
        table(run_baseline(), run_matry())
    else:
        raise ValueError(f"unknown MODE={MODE}")


if __name__ == "__main__":
    main()
