"""VPD baseline runner for pythia-14m (MPD arm archived under matryoshka/; APD comparisons live in apd_pythia.py). Originally head-to-head on the
frozen pythia-14m target (induction-capable, ~SS2L decomposition scale, 6 layers). Same
model-sampled data / seed / budget for both. DDP-ready (one run across both GPUs).

    MODE=genpool    python -m nano_param_decomp.compare_pythia14m            # sample shared pool once
    MODE=baseline   torchrun --standalone --nproc_per_node=2 -m nano_param_decomp.compare_pythia14m
    MODE=table      python -m nano_param_decomp.compare_pythia14m
"""

import json
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch

from . import run
from .pythia14m import C_PER_MODULE_PYTHIA_14M, generate_pool, load_pythia14m_target, pool_loader

STEPS = int(os.environ.get("STEPS", "20000"))
SEQ = int(os.environ.get("SEQ", "256"))
B = int(os.environ.get("B", "32"))
G = int(os.environ.get("G", "1024"))
N_POOL = int(os.environ.get("N_POOL", "1024"))
MODE = os.environ.get("MODE", "table")
# Coefficients for the NORMALIZED (scale-invariant) penalties: membership_l1 = mean gate in [0,1],
# component_size_l2 = mean (size/A)^2 in [0,1]. CM~300, CCS~5000 reproduce the SS2L-validated regime
# (old un-normalized CM=0.3 x G=1024; CCS=1e-4 x A^2~7104^2) and should now transfer across models.
CM = float(os.environ.get("CM", "300"))
CCS = float(os.environ.get("CCS", "5000"))

CMP_DIR = os.environ.get("CMP_DIR", "/tmp/pythia_compare")
POOL_PATH = os.path.join(CMP_DIR, "pool.pt")
BASE_JSON = os.path.join(CMP_DIR, "base.json")
MAT_JSON = os.path.join(CMP_DIR, "mat.json")

ENTITY = "rohan-kathuria-neu"
PROJECT = "matryoshka-pd"
GROUP = os.environ.get("WGROUP", "pythia14m-headtohead")
TAGS = ["pythia14m", "6L", f"G{G}"]

LR = float(os.environ.get("LR", "3e-4"))

SHARED = dict(
    seq_len=SEQ, batch_size=B, eval_batch_size=B, n_steps=STEPS,
    faithfulness_warmup_steps=200, ci_d_model=512, ci_n_blocks=4, ci_n_heads=8,
    ci_mlp_hidden=2048, coeff_imp=0.001, main_lr=LR, log_every=200, use_wandb=True,
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
    print(f"sampling {N_POOL} seqs (len {SEQ}) from pythia-14m ...", flush=True)
    pool = generate_pool(load_pythia14m_target().float(), N_POOL, SEQ, _device(), seed=0)
    torch.save(pool, POOL_PATH)
    print(f"saved pool {tuple(pool.shape)} -> {POOL_PATH}", flush=True)


def _load_pools() -> tuple[torch.Tensor, torch.Tensor]:
    pool = torch.load(POOL_PATH, weights_only=True)
    n_tr = int(0.9 * pool.shape[0])
    return pool[:n_tr], pool[n_tr:]


def _ddp_loaders(train_pool, eval_pool):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_B = B // world
    return pool_loader(train_pool, local_B, seed=rank), pool_loader(eval_pool, local_B, seed=1000 + rank), rank


def run_baseline() -> dict:
    train_pool, eval_pool = _load_pools()
    os.environ["WANDB_ENTITY"] = ENTITY
    os.environ["WANDB_RUN_GROUP"] = GROUP
    os.environ["WANDB_JOB_TYPE"] = "vpd-baseline"
    os.environ["WANDB_TAGS"] = ",".join(TAGS)
    train_loader, eval_loader, rank = _ddp_loaders(train_pool, eval_pool)
    base = run.decompose(
        load_pythia14m_target().float(),
        run.Config(
            C_per_module=C_PER_MODULE_PYTHIA_14M, slow_eval_on_first_step=False,
            slow_eval_freq=STEPS + 1, eval_freq=STEPS + 1, wandb_project=PROJECT,
            wandb_run_name=f"pythia14m-vpd-s{STEPS}",
            save_path=os.path.join(CMP_DIR, f"vpd_s{STEPS}.pt"), **SHARED,
        ),
        train_loader, eval_loader,
    )
    if rank == 0:
        _finish_wandb(base)
        json.dump(_floats(base), open(BASE_JSON, "w"), indent=2)
    return base




def table() -> None:
    base = json.load(open(BASE_JSON)) if os.path.exists(BASE_JSON) else {}
    mat = json.load(open(MAT_JSON)) if os.path.exists(MAT_JSON) else {}

    def row(label, b, m):
        bs = f"{b:.4g}" if isinstance(b, (int, float)) else "-"
        ms = f"{m:.4g}" if isinstance(m, (int, float)) else "-"
        return f"{label:<34} {bs:>13} {ms:>13}"

    print(f"\n{'='*62}\nPYTHIA-14M  VPD vs MPD  (steps={STEPS}, G={G})\n{'='*62}")
    print(f"{'metric':<34} {'VPD':>13} {'MPD':>13}")
    print(row("faithfulness", base.get("eval/loss/FaithfulnessLoss"), mat.get("eval/faithfulness")))
    print(row("KL ci-masked", base.get("eval/ce_kl/kl_ci_masked"), mat.get("eval/kl_ci_masked")))
    print(row("KL stoch-masked", base.get("eval/ce_kl/kl_stoch_masked"), mat.get("eval/kl_stoch_masked")))
    print(row("induction copy (unmasked)", base.get("eval/induction/copy_unmasked"), mat.get("eval/induction/copy_unmasked")))
    print(row("induction copy (ci-masked)", base.get("eval/induction/copy_ci_masked"), mat.get("eval/induction/copy_ci_masked")))
    print(row("L0 atoms/tok", base.get("eval/l0/0.0_total"), mat.get("eval/l0_atoms")))
    print(row("MPD: comps/tok", None, mat.get("eval/l0_components")))
    print(row("MPD: atoms/comp", None, mat.get("eval/comp_mean_size")))
    print(row("MPD: crosslayer frac", None, mat.get("eval/comp_crosslayer_frac")))
    print(row("MPD: mean layers/comp", None, mat.get("eval/comp_mean_layers")))
    print("="*62)


def main() -> None:
    if MODE == "genpool":
        gen_pool()
    elif MODE == "baseline":
        run_baseline()
    elif MODE == "table":
        table()
    else:
        raise ValueError(f"unknown MODE={MODE}")


if __name__ == "__main__":
    main()
