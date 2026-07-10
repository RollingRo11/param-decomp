"""Head-to-head VPD (per-atom) vs MPD (matryoshka) on Neel Nanda's `attn-only-2l` — the model whose
induction circuit is published: previous-token head 0.3 feeds induction head 1.6. After decomposing
we can ask whether the causally-important components localize to those two heads.

    # one-time: export processed weights (system python, needs transformer_lens)
    CUDA_VISIBLE_DEVICES="" python3.12 -m nano_param_decomp.attn_only_2l

    MODE=genpool    python -m nano_param_decomp.compare_attn_only
    MODE=baseline   torchrun --standalone --nproc_per_node=2 -m nano_param_decomp.compare_attn_only
    MODE=matryoshka torchrun --standalone --nproc_per_node=2 -m nano_param_decomp.compare_attn_only
    MODE=table      python -m nano_param_decomp.compare_attn_only
"""

import json
import os

import torch

from . import matryoshka, run
from .attn_only_2l import (
    C_PER_MODULE_ATTN_ONLY_2L,
    generate_pool,
    load_attn_only_2l_target,
    pool_loader,
)

STEPS = int(os.environ.get("STEPS", "20000"))
SEQ = int(os.environ.get("SEQ", "256"))
B = int(os.environ.get("B", "32"))
G = int(os.environ.get("G", "512"))  # cross-layer components (4096 atoms over 2 layers)
N_POOL = int(os.environ.get("N_POOL", "1024"))
MODE = os.environ.get("MODE", "table")
# Scale-invariant normalized penalties (mean gate / mean (size/A)^2), same regime as pythia run.
# Toy-validated committed regime (softmax membership + constant tau + AE commitment), ported from
# toy_decompose. Supersedes the old sigmoid/annealed-tau/CM300-CCS5000 dense regime that gave the
# degenerate mat.json (atom_degree 243, shared_atom_frac 1.0).
CM = float(os.environ.get("CM", "0.0"))
CCS = float(os.environ.get("CCS", "1.0"))
AGG = os.environ.get("AGG", "max")  # "max" = on-if-any-component-on (the dilution fix); "mean" = old
AE = float(os.environ.get("AE", "0.05"))  # commitment (binary entropy) pressure on the membership
MEMTYPE = os.environ.get("MEMTYPE", "softmax")  # softmax = per-atom row sums to 1 (each atom commits to ~1 comp)
TAU = float(os.environ.get("TAU", "1.0"))  # constant temperature (no anneal); tau_start=tau_end=TAU

CMP_DIR = os.environ.get("CMP_DIR", "/tmp/attn2l_compare")
POOL_PATH = os.path.join(CMP_DIR, "pool.pt")
BASE_JSON = os.path.join(CMP_DIR, "base.json")
MAT_JSON = os.path.join(CMP_DIR, "mat.json")

ENTITY = "rohan-kathuria-neu"
PROJECT = "matryoshka-pd"
GROUP = os.environ.get("WGROUP", "attn2l-headtohead")
TAGS = ["attn-only-2l", "2L", f"G{G}", "induction-circuit"]

LR = float(os.environ.get("LR", "3e-4"))
IMP = float(os.environ.get("IMP", "0.001"))  # importance-minimality coeff (toy used 0.003; sweepable)

SHARED = dict(
    seq_len=SEQ, batch_size=B, eval_batch_size=B, n_steps=STEPS,
    faithfulness_warmup_steps=200, ci_d_model=512, ci_n_blocks=4, ci_n_heads=8,
    ci_mlp_hidden=2048, coeff_imp=IMP, main_lr=LR, log_every=200, use_wandb=True,
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
    print(f"sampling {N_POOL} BOS-seeded seqs (len {SEQ}) from attn-only-2l ...", flush=True)
    pool = generate_pool(load_attn_only_2l_target(), N_POOL, SEQ, _device(), seed=0)
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
        load_attn_only_2l_target(),
        run.Config(
            C_per_module=C_PER_MODULE_ATTN_ONLY_2L, slow_eval_on_first_step=False,
            slow_eval_freq=STEPS + 1, eval_freq=STEPS + 1, wandb_project=PROJECT,
            wandb_run_name=f"attn2l-vpd-s{STEPS}",
            save_path=os.path.join(CMP_DIR, f"vpd_s{STEPS}.pt"), **SHARED,
        ),
        train_loader, eval_loader,
    )
    if rank == 0:
        _finish_wandb(base)
        json.dump(_floats(base), open(BASE_JSON, "w"), indent=2)
    return base


def run_matry() -> dict:
    train_pool, eval_pool = _load_pools()
    train_loader, eval_loader, rank = _ddp_loaders(train_pool, eval_pool)
    mat = matryoshka.decompose(
        load_attn_only_2l_target(),
        matryoshka.Config(
            C_per_module=C_PER_MODULE_ATTN_ONLY_2L, n_components=G, aggregation=AGG,
            membership_type=MEMTYPE, tau_start=TAU, tau_end=TAU,
            coeff_membership=CM, coeff_comp_size=CCS, coeff_imp_atoms=0.0,
            coeff_membership_entropy=0.0, coeff_assign_entropy=AE,
            eval_freq=max(STEPS // 20, 500), wandb_entity=ENTITY, wandb_project=PROJECT,
            wandb_group=GROUP, wandb_job_type="matryoshka",
            wandb_run_name=f"attn2l-mpd-{MEMTYPE}-{AGG}-G{G}-CCS{CCS}-AE{AE}-tau{TAU}-s{STEPS}",
            wandb_tags=tuple(TAGS),
            wandb_notes=f"attn-only-2l MPD: {AGG} agg, CM+CCS+AE, tau over n_steps",
            save_path=os.path.join(CMP_DIR, f"mpd_s{STEPS}.pt"),
            **SHARED,
        ),
        train_loader, eval_loader,
    )
    if rank == 0:
        _finish_wandb(mat)
        json.dump(_floats(mat), open(MAT_JSON, "w"), indent=2)
    return mat


def table() -> None:
    base = json.load(open(BASE_JSON)) if os.path.exists(BASE_JSON) else {}
    mat = json.load(open(MAT_JSON)) if os.path.exists(MAT_JSON) else {}

    def row(label, b, m):
        bs = f"{b:.4g}" if isinstance(b, (int, float)) else "-"
        ms = f"{m:.4g}" if isinstance(m, (int, float)) else "-"
        return f"{label:<34} {bs:>13} {ms:>13}"

    print(f"\n{'='*62}\nATTN-ONLY-2L  VPD vs MPD  (steps={STEPS}, G={G})\n{'='*62}")
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
    elif MODE == "matryoshka":
        run_matry()
    elif MODE == "table":
        table()
    else:
        raise ValueError(f"unknown MODE={MODE}")


if __name__ == "__main__":
    main()
