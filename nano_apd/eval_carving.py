"""Evaluate a contrastive parameter carving without relying on training logs.

Reports full-edit efficacy, matched-negative and natural-text retention, individual atom
effects, random-subset additivity, dose response, an optional representation contrast, and
an optional held-out low-rank relearning attack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nano_apd.carving import (
    CarvingEditor,
    MatchedBatch,
    build_banks,
    kl_per_position,
    load_matched_batch,
    make_matched_induction_batch,
    pearson_correlation,
    selected_accuracy,
    selected_logits,
    selected_scores,
)
from nano_apd.lm_target import (
    DEFAULT_HF_MODEL,
    load_carving_target,
    one_loader_batch,
    vocab_size,
)
from nano_apd.targeted import induction_predictable
from nano_param_decomp.pile_4L import make_loader

OUT_ROOT = Path(__file__).parent / "out"


def _pairs(config: dict, args, target, seed: int) -> MatchedBatch:
    path = args.pairs_file or config.get("pairs_file", "")
    if path:
        saved = load_matched_batch(path, "cpu")
        generator = torch.Generator().manual_seed(seed)
        index = torch.randint(saved.positive.shape[0], (args.n_pairs,), generator=generator)
        return MatchedBatch(
            saved.positive[index], saved.negative[index], saved.positions[index],
            saved.labels[index]
        ).to(args.device)
    return make_matched_induction_batch(
        args.n_pairs, args.seq_len, vocab_size(target), args.device, seed
    )


def _retain(args, config: dict):
    path = args.retain_file or config.get("retain_file", "")
    if path:
        obj = torch.load(path, weights_only=True, map_location="cpu")
        tokens = obj["input_ids"] if isinstance(obj, dict) else obj
        if tokens.ndim != 2 or tokens.shape[0] < args.n_retain:
            raise ValueError("retain file must contain at least n_retain rows of token ids")
        if tokens.shape[1] < args.seq_len:
            raise ValueError("retain file sequences are shorter than seq_len")
        generator = torch.Generator().manual_seed(args.seed)
        index = torch.randperm(tokens.shape[0], generator=generator)[:args.n_retain]
        return tokens[index, :args.seq_len].long().to(args.device), "file"
    try:
        loader = make_loader(args.n_retain, args.seq_len, 0, 1, "validation", args.seed)
        return one_loader_batch(loader).to(args.device), "validation"
    except Exception:
        if not args.allow_train_fallback:
            raise
        loader = make_loader(
            args.n_retain, args.seq_len, 0, 1, "train", args.seed + 9_000_000
        )
        return one_loader_batch(loader).to(args.device), "train-fallback"


def _gap(logits, pairs: MatchedBatch, score_type: str) -> torch.Tensor:
    batch = pairs.positive.shape[0]
    return (
        selected_scores(logits[:batch], pairs.positions, pairs.labels, score_type)
        - selected_scores(logits[batch:], pairs.positions, pairs.labels, score_type)
    )


def _autocast(tokens, enabled: bool):
    return torch.autocast(
        tokens.device.type,
        dtype=torch.bfloat16,
        enabled=enabled and tokens.is_cuda,
    )


def _forward(editor, target, tokens, masks, use_bf16: bool):
    editor.masks = masks
    with torch.no_grad(), _autocast(tokens, use_bf16):
        logits = target(tokens)
    editor.masks = None
    return logits


def _representation_metrics(
    editor, target, pairs, paths, components, use_bf16: bool
):
    tokens = torch.cat([pairs.positive, pairs.negative])
    batch = pairs.positive.shape[0]
    rows = torch.arange(batch, device=tokens.device)

    def contrasts(masks):
        editor.masks = masks
        editor.start_capture()
        with torch.no_grad(), _autocast(tokens, use_bf16):
            target(tokens)
        result = {
            path: (
                editor.cache[path]["post"][:batch][rows, pairs.positions]
                - editor.cache[path]["post"][batch:][rows, pairs.positions]
            ).detach()
            for path in paths
        }
        editor.stop_capture()
        editor.cache = {}
        editor.masks = None
        return result

    target_contrasts = contrasts(
        torch.ones(2 * batch, 1, components, device=tokens.device)
    )
    edited_contrasts = contrasts(
        torch.zeros(2 * batch, 1, components, device=tokens.device)
    )
    metrics = {}
    for path in paths:
        target_contrast = target_contrasts[path].float()
        edited_contrast = edited_contrasts[path].float()
        target_norm = target_contrast.norm(dim=-1).mean()
        edited_norm = edited_contrast.norm(dim=-1).mean()
        cosine = F.cosine_similarity(target_contrast, edited_contrast, dim=-1)
        metrics[path] = {
            "target_contrast_norm": round(target_norm.item(), 6),
            "edited_contrast_norm": round(edited_norm.item(), 6),
            "norm_ratio": round(
                (edited_norm / target_norm.clamp_min(1e-8)).item(), 4
            ),
            "target_edit_cosine": round(cosine.mean().item(), 4),
        }
    return metrics


def _relearning_attack(args, config, target, editor, pairs_eval, retain_eval):
    if args.relearn_steps <= 0:
        return []
    paths = config["module_paths"]
    recovery = build_banks(target, paths, 1, args.relearn_rank, "free").to(args.device)
    optimizer = torch.optim.AdamW(recovery.parameters(), lr=args.relearn_lr, weight_decay=0)
    components = config["C"]
    eval_tokens = torch.cat([pairs_eval.positive, pairs_eval.negative])
    eval_masks = torch.zeros(eval_tokens.shape[0], 1, components, device=args.device)
    eval_keep = torch.ones_like(eval_masks)
    editor.set_recovery(None)
    eval_reference = _forward(
        editor, target, eval_tokens, eval_keep, args.use_bf16
    )
    score_type = config.get("score_type", "logit_margin")
    eval_target_gap = _gap(eval_reference, pairs_eval, score_type)
    eval_eligible = eval_target_gap > float(config.get("min_target_gap", 0.05))
    if not config.get("include_target_incorrect", False):
        eval_eligible &= (
            selected_logits(eval_reference[:pairs_eval.positive.shape[0]],
                            pairs_eval.positions).argmax(-1)
            == pairs_eval.labels
        )
    if not eval_eligible.any():
        raise RuntimeError("relearning evaluation has no eligible held-out pairs")
    editor.set_recovery(recovery)
    history = []

    def evaluate(step):
        logits = _forward(
            editor, target, eval_tokens, eval_masks, args.use_bf16
        )
        gap = _gap(logits, pairs_eval, score_type)
        history.append({
            "step": step,
            "eligible_count": int(eval_eligible.sum().item()),
            "target_gap": round(eval_target_gap[eval_eligible].mean().item(), 5),
            "recovered_gap": round(gap[eval_eligible].mean().item(), 5),
            "positive_acc": round(
                selected_accuracy(
                    logits[:pairs_eval.positive.shape[0]], pairs_eval.positions,
                    pairs_eval.labels,
                ).item(), 4
            ),
        })

    evaluate(0)
    for step in range(1, args.relearn_steps + 1):
        train = _pairs(config, args, target, args.seed + 20_000 + step)
        tokens = torch.cat([train.positive, train.negative, retain_eval])
        masks = torch.zeros(tokens.shape[0], 1, components, device=args.device)
        editor.set_recovery(None)
        editor.masks = torch.ones(
            tokens.shape[0], 1, components, device=args.device
        )
        with torch.no_grad(), _autocast(tokens, args.use_bf16):
            reference = target(tokens)
        editor.set_recovery(recovery)
        editor.masks = masks
        with _autocast(tokens, args.use_bf16):
            logits = target(tokens)
        editor.masks = None
        batch = train.positive.shape[0]
        gap = (
            selected_scores(logits[:batch], train.positions, train.labels, score_type)
            - selected_scores(
                logits[batch:2 * batch], train.positions, train.labels, score_type
            )
        )
        reference_gap = (
            selected_scores(
                reference[:batch], train.positions, train.labels, score_type
            )
            - selected_scores(
                reference[batch:2 * batch],
                train.positions,
                train.labels,
                score_type,
            )
        )
        train_eligible = reference_gap > float(config.get("min_target_gap", 0.05))
        if not config.get("include_target_incorrect", False):
            train_eligible &= (
                selected_logits(reference[:batch], train.positions).argmax(-1)
                == train.labels
            )
        preserve_kl = kl_per_position(logits, reference)
        preserve_mask = torch.ones_like(preserve_kl, dtype=torch.bool)
        preserve_mask[
            torch.arange(batch, device=args.device), train.positions
        ] = False
        preserve = preserve_kl[preserve_mask].mean()
        restore = (
            F.smooth_l1_loss(
                gap[train_eligible], reference_gap[train_eligible].detach()
            )
            if train_eligible.any()
            else gap.sum() * 0
        )
        loss = restore + args.relearn_retain_w * preserve
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.relearn_every == 0 or step == args.relearn_steps:
            evaluate(step)
    editor.set_recovery(None)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_pairs", type=int, default=32)
    parser.add_argument("--n_retain", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--pairs_file", default="")
    parser.add_argument("--retain_file", default="",
                        help="offline token tensor; overrides the training config")
    parser.add_argument("--n_subsets", type=int, default=24)
    parser.add_argument("--probe_module", default="")
    parser.add_argument("--precision", choices=["saved", "fp32", "bf16"],
                        default="saved",
                        help="evaluation forward precision; saved follows training config")
    parser.add_argument("--relearn_steps", type=int, default=0)
    parser.add_argument("--relearn_every", type=int, default=10)
    parser.add_argument("--relearn_rank", type=int, default=8)
    parser.add_argument("--relearn_lr", type=float, default=1e-3)
    parser.add_argument("--relearn_retain_w", type=float, default=1.0)
    parser.add_argument("--allow_train_fallback", action="store_true")
    parser.add_argument("--output", default="",
                        help="output JSON filename inside the run directory")
    parser.add_argument("--seed", type=int, default=990_001)
    args = parser.parse_args()

    run_dir = OUT_ROOT / args.run
    config = json.loads((run_dir / "config.json").read_text())
    args.seq_len = args.seq_len or int(config["seq_len"])
    args.use_bf16 = (
        bool(config.get("bf16", False))
        if args.precision == "saved"
        else args.precision == "bf16"
    )
    torch.manual_seed(args.seed)
    target_kind = config.get("target", "pile4l")
    model_name = config.get("model_name", DEFAULT_HF_MODEL)
    target = load_carving_target(target_kind, model_name).float().to(args.device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    paths = config["module_paths"]
    banks = build_banks(
        target, paths, int(config["C"]), int(config["rank"]), config["bank_type"]
    ).to(args.device)
    banks.load_state_dict(torch.load(run_dir / "banks.pt", weights_only=True,
                                     map_location=args.device))
    for parameter in banks.parameters():
        parameter.requires_grad_(False)
    editor = CarvingEditor(target, banks, paths)
    pairs = _pairs(config, args, target, args.seed)
    retain, retain_split = _retain(args, config)
    pair_tokens = torch.cat([pairs.positive, pairs.negative])
    all_tokens = torch.cat([pair_tokens, retain])
    components = int(config["C"])

    with torch.no_grad(), _autocast(all_tokens, args.use_bf16):
        editor.masks = torch.ones(
            all_tokens.shape[0], 1, components, device=args.device
        )
        reference = target(all_tokens)
    editor.masks = None
    full_masks = torch.zeros(all_tokens.shape[0], 1, components, device=args.device)
    edited = _forward(
        editor, target, all_tokens, full_masks, args.use_bf16
    )
    batch = pairs.positive.shape[0]
    score_type = config.get("score_type", "logit_margin")
    target_gap = _gap(reference[:2 * batch], pairs, score_type)
    edited_gap = _gap(edited[:2 * batch], pairs, score_type)
    target_correct = (
        selected_logits(reference[:batch], pairs.positions).argmax(-1)
        == pairs.labels
    )
    eligible = target_gap > float(config.get("min_target_gap", 0.05))
    if not config.get("include_target_incorrect", False):
        eligible &= target_correct
    if not eligible.any():
        raise RuntimeError(
            "held-out evaluation has no eligible pairs; increase --n_pairs or inspect "
            "the pair construction"
        )
    full_kl = kl_per_position(edited, reference)
    retain_kl = full_kl[2 * batch:]
    predictable = induction_predictable(retain)[:, 1:]
    ce_target = F.cross_entropy(
        reference[2 * batch:, :-1].flatten(0, 1), retain[:, 1:].flatten(),
        reduction="none",
    ).view(retain.shape[0], -1)
    ce_edit = F.cross_entropy(
        edited[2 * batch:, :-1].flatten(0, 1), retain[:, 1:].flatten(),
        reduction="none",
    ).view(retain.shape[0], -1)
    dce = ce_edit - ce_target

    single_effects = []
    for component in range(components):
        masks = torch.ones(2 * batch, 1, components, device=args.device)
        masks[..., component] = 0
        logits = _forward(
            editor, target, pair_tokens, masks, args.use_bf16
        )
        effect = target_gap - _gap(logits, pairs, score_type)
        single_effects.append(effect[eligible].mean())
    single_effects_t = torch.stack(single_effects)
    full_effect_eligible = (target_gap - edited_gap)[eligible].mean()
    single_sum = single_effects_t.sum()
    full_to_sum_ratio = (
        None
        if single_sum.abs().item() < 1e-8
        else round((full_effect_eligible / single_sum).item(), 4)
    )

    actual, predicted, subset_sizes = [], [], []
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    for _ in range(args.n_subsets):
        remove = torch.rand(components, device=args.device, generator=generator) < 0.5
        if not remove.any():
            remove[0] = True
        masks = (~remove).float().view(1, 1, -1).expand(2 * batch, 1, -1)
        logits = _forward(
            editor, target, pair_tokens, masks, args.use_bf16
        )
        effect = (target_gap - _gap(logits, pairs, score_type))[eligible].mean()
        actual.append(effect)
        predicted.append(single_effects_t[remove].sum())
        subset_sizes.append(int(remove.sum().item()))
    actual_t, predicted_t = torch.stack(actual), torch.stack(predicted)

    dose_curve = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        masks = torch.full(
            (all_tokens.shape[0], 1, components), 1 - alpha, device=args.device
        )
        dose_logits = _forward(
            editor, target, all_tokens, masks, args.use_bf16
        )
        gap = _gap(dose_logits[:2 * batch], pairs, score_type)
        dose_kl = kl_per_position(dose_logits, reference)
        negative_selected = dose_kl[batch:2 * batch][
            torch.arange(batch, device=args.device), pairs.positions
        ]
        dose_curve[str(alpha)] = {
            "gap_all": round(gap.mean().item(), 5),
            "gap_eligible": round(gap[eligible].mean().item(), 5),
            "gap_removed_eligible": round(
                (target_gap - gap)[eligible].mean().item(), 5
            ),
            "negative_selected_kl_all": round(
                negative_selected.mean().item(), 6
            ),
            "negative_selected_kl_eligible": round(
                negative_selected[eligible].mean().item(), 6
            ),
            "retain_kl": round(dose_kl[2 * batch:].mean().item(), 6),
        }

    probe_paths = [args.probe_module] if args.probe_module else paths
    if any(path not in paths for path in probe_paths):
        raise ValueError("--probe_module must be one of the carved module_paths")
    eligible_pairs = MatchedBatch(
        pairs.positive[eligible],
        pairs.negative[eligible],
        pairs.positions[eligible],
        pairs.labels[eligible],
    )
    representation = _representation_metrics(
        editor, target, eligible_pairs, probe_paths, components, args.use_bf16
    )
    relearning = _relearning_attack(args, config, target, editor, pairs, retain)

    report = {
        "run": args.run,
        "seed": args.seed,
        "retain_split": retain_split,
        "bank_type": config["bank_type"],
        "score_type": score_type,
        "precision": "bf16" if args.use_bf16 else "fp32",
        "C": components,
        "rank": int(config["rank"]),
        "rank_matched_baseline": components * int(config["rank"]),
        "eligibility": {
            "count": int(eligible.sum().item()),
            "fraction": round(eligible.float().mean().item(), 4),
            "requires_target_correct": not config.get(
                "include_target_incorrect", False
            ),
            "min_target_gap": float(config.get("min_target_gap", 0.05)),
        },
        "full_edit": {
            "target_gap_all": round(target_gap.mean().item(), 5),
            "edited_gap_all": round(edited_gap.mean().item(), 5),
            "gap_removed_all": round((target_gap - edited_gap).mean().item(), 5),
            "target_gap_eligible": round(target_gap[eligible].mean().item(), 5),
            "edited_gap_eligible": round(edited_gap[eligible].mean().item(), 5),
            "gap_removed_eligible": round(
                (target_gap - edited_gap)[eligible].mean().item(), 5
            ),
            "positive_acc_target": round(
                selected_accuracy(reference[:batch], pairs.positions, pairs.labels).item(), 4
            ),
            "positive_acc_edited": round(
                selected_accuracy(edited[:batch], pairs.positions, pairs.labels).item(), 4
            ),
            "negative_selected_kl_all": round(
                full_kl[batch:2 * batch][
                    torch.arange(batch, device=args.device), pairs.positions
                ].mean().item(), 6
            ),
            "negative_selected_kl_eligible": round(
                full_kl[batch:2 * batch][
                    torch.arange(batch, device=args.device), pairs.positions
                ][eligible].mean().item(), 6
            ),
            "retain_kl": round(retain_kl.mean().item(), 6),
            "retain_dce": round(dce.mean().item(), 6),
            "retain_dce_induction_predictable": round(
                dce[predictable].mean().item() if predictable.any() else 0.0, 6
            ),
            "retain_dce_other": round(dce[~predictable].mean().item(), 6),
        },
        "individual_components": {
            "effects": [round(x.item(), 6) for x in single_effects_t],
            "mean": round(single_effects_t.mean().item(), 6),
            "max": round(single_effects_t.max().item(), 6),
            "sum": round(single_effects_t.sum().item(), 6),
            "full_to_sum_ratio": full_to_sum_ratio,
        },
        "subset_composition": {
            "sizes": subset_sizes,
            "actual_effects": [round(x.item(), 6) for x in actual_t],
            "predicted_from_singles": [round(x.item(), 6) for x in predicted_t],
            "pearson": round(pearson_correlation(actual_t, predicted_t).item(), 4),
            "mae": round((actual_t - predicted_t).abs().mean().item(), 6),
        },
        "dose_curve": dose_curve,
        "representation": representation,
        "relearning": relearning,
    }
    output_name = args.output or (
        "eval_carving_bf16.json" if args.use_bf16 else "eval_carving_fp32.json"
    )
    if Path(output_name).name != output_name:
        raise ValueError("--output must be a filename, not a path")
    output_path = run_dir / output_name
    output_path.write_text(json.dumps(report, indent=2))
    editor.restore()
    print(json.dumps(report, indent=2))
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
