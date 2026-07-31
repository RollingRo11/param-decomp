"""Reproducible evaluation for legacy ``targeted.py`` artifacts.

The original rich targeted JSON files were produced by an evaluator that was not checked
into the repository. This replacement covers the central claims with strict split
reporting: legacy repeated-half induction, matched induction contrasts, natural-text CE,
component ablations, random subsets, and edit dose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nano_apd.carving import (
    CarvingEditor,
    build_banks,
    kl_per_position,
    make_matched_induction_batch,
    selected_log_probs,
)
from nano_apd.lm_target import one_loader_batch
from nano_apd.targeted import induction_predictable
from nano_param_decomp.pile_4L import (
    C_PER_MODULE_4L,
    load_paper_target_model,
    make_loader,
)

OUT_ROOT = Path(__file__).parent / "out"


def _modules(config: dict) -> list[str]:
    paths = list(C_PER_MODULE_4L)
    patterns = [x for x in config.get("modules_filter", "").split(",") if x]
    return [p for p in paths if not patterns or any(x in p for x in patterns)]


def _retain(args):
    try:
        return one_loader_batch(make_loader(
            args.n_retain, args.seq_len, 0, 1, "validation", args.seed
        )).to(args.device), "validation"
    except Exception:
        if not args.allow_train_fallback:
            raise
        return one_loader_batch(make_loader(
            args.n_retain, args.seq_len, 0, 1, "train", args.seed + 8_000_000
        )).to(args.device), "train-fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_pairs", type=int, default=32)
    parser.add_argument("--n_retain", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--n_subsets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=880_001)
    parser.add_argument("--allow_train_fallback", action="store_true")
    args = parser.parse_args()

    run_dir = OUT_ROOT / args.run
    config = json.loads((run_dir / "config.json").read_text())
    args.seq_len = args.seq_len or int(config.get("seq_len", 256))
    components, rank = int(config["C"]), int(config["m"])
    target = load_paper_target_model().float().to(args.device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    paths = _modules(config)
    banks = build_banks(target, paths, components, rank, "free").to(args.device)
    banks.load_state_dict(torch.load(run_dir / "banks.pt", weights_only=True,
                                     map_location=args.device))
    for parameter in banks.parameters():
        parameter.requires_grad_(False)
    editor = CarvingEditor(target, banks, paths)

    retain, retain_split = _retain(args)
    pairs = make_matched_induction_batch(
        args.n_pairs, args.seq_len, int(target.config.vocab_size), args.device, args.seed
    )
    pair_tokens = torch.cat([pairs.positive, pairs.negative])
    half = args.seq_len // 2
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    first = torch.randint(
        0, min(50_000, int(target.config.vocab_size)),
        (args.n_pairs, half), device=args.device, generator=generator,
    )
    legacy = torch.cat([first, first], -1)
    tokens = torch.cat([pair_tokens, legacy, retain])
    with torch.no_grad():
        editor.masks = None
        reference = target(tokens)
    editor.masks = torch.zeros(tokens.shape[0], 1, components, device=args.device)
    with torch.no_grad():
        edited = target(tokens)
    editor.masks = None

    batch = args.n_pairs
    pair_slice = slice(0, 2 * batch)
    legacy_slice = slice(2 * batch, 3 * batch)
    retain_slice = slice(3 * batch, None)
    target_gap = (
        selected_log_probs(reference[:batch], pairs.positions, pairs.labels)
        - selected_log_probs(reference[batch:2 * batch], pairs.positions, pairs.labels)
    )
    edited_gap = (
        selected_log_probs(edited[:batch], pairs.positions, pairs.labels)
        - selected_log_probs(edited[batch:2 * batch], pairs.positions, pairs.labels)
    )
    legacy_target_acc = (
        reference[legacy_slice][:, half:-1].argmax(-1) == legacy[:, half + 1:]
    ).float().mean()
    legacy_edit_acc = (
        edited[legacy_slice][:, half:-1].argmax(-1) == legacy[:, half + 1:]
    ).float().mean()
    retain_target_ce = F.cross_entropy(
        reference[retain_slice][:, :-1].flatten(0, 1), retain[:, 1:].flatten(),
        reduction="none",
    ).view(retain.shape[0], -1)
    retain_edit_ce = F.cross_entropy(
        edited[retain_slice][:, :-1].flatten(0, 1), retain[:, 1:].flatten(),
        reduction="none",
    ).view(retain.shape[0], -1)
    dce = retain_edit_ce - retain_target_ce
    predictable = induction_predictable(retain)[:, 1:]

    singles = []
    for component in range(components):
        masks = torch.ones(2 * batch, 1, components, device=args.device)
        masks[..., component] = 0
        editor.masks = masks
        with torch.no_grad():
            logits = target(pair_tokens)
        editor.masks = None
        gap = (
            selected_log_probs(logits[:batch], pairs.positions, pairs.labels)
            - selected_log_probs(logits[batch:], pairs.positions, pairs.labels)
        )
        singles.append((target_gap - gap).mean())

    random_subsets = []
    subset_generator = torch.Generator(device=args.device).manual_seed(args.seed + 2)
    for _ in range(args.n_subsets):
        remove = torch.rand(components, device=args.device,
                            generator=subset_generator) < 0.5
        masks = (~remove).float().view(1, 1, -1).expand(2 * batch, 1, -1)
        editor.masks = masks
        with torch.no_grad():
            logits = target(pair_tokens)
        editor.masks = None
        gap = (
            selected_log_probs(logits[:batch], pairs.positions, pairs.labels)
            - selected_log_probs(logits[batch:], pairs.positions, pairs.labels)
        )
        random_subsets.append({
            "n_removed": int(remove.sum()),
            "gap_removed": round((target_gap - gap).mean().item(), 6),
        })

    dose = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        editor.masks = torch.full(
            (2 * batch, 1, components), 1 - alpha, device=args.device
        )
        with torch.no_grad():
            logits = target(pair_tokens)
        editor.masks = None
        gap = (
            selected_log_probs(logits[:batch], pairs.positions, pairs.labels)
            - selected_log_probs(logits[batch:], pairs.positions, pairs.labels)
        )
        dose[str(alpha)] = round(gap.mean().item(), 6)

    singles_t = torch.stack(singles)
    report = {
        "run": args.run,
        "seed": args.seed,
        "retain_split": retain_split,
        "C": components,
        "rank": rank,
        "rank_matched_baseline": components * rank,
        "legacy_induction": {
            "target_acc": round(legacy_target_acc.item(), 4),
            "edited_acc": round(legacy_edit_acc.item(), 4),
        },
        "matched_induction": {
            "target_gap": round(target_gap.mean().item(), 6),
            "edited_gap": round(edited_gap.mean().item(), 6),
            "gap_removed": round((target_gap - edited_gap).mean().item(), 6),
            "negative_selected_kl": round(
                kl_per_position(edited[pair_slice][batch:], reference[pair_slice][batch:])[
                    torch.arange(batch, device=args.device), pairs.positions
                ].mean().item(), 6
            ),
        },
        "natural_retain": {
            "dce": round(dce.mean().item(), 6),
            "dce_induction_predictable": round(
                dce[predictable].mean().item() if predictable.any() else 0.0, 6
            ),
            "dce_other": round(dce[~predictable].mean().item(), 6),
        },
        "single_component_gap_removed": [round(x.item(), 6) for x in singles_t],
        "single_mean": round(singles_t.mean().item(), 6),
        "single_max": round(singles_t.max().item(), 6),
        "random_subsets": random_subsets,
        "dose_gap": dose,
    }
    path = run_dir / "eval_repro.json"
    path.write_text(json.dumps(report, indent=2))
    editor.restore()
    print(json.dumps(report, indent=2))
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
