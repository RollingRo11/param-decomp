"""Audit what each learned component does across controlled induction variants.

This evaluator deliberately treats a component as one cross-layer object.  It never
scores or penalizes how many layers or attention heads the component touches.  The
question is functional instead: does removing the component hurt induction when the
cue match is present, and does it leave matched no-induction controls alone?

Examples are selected only by the frozen model's behavior on the standard induction
pair.  The exact same rows are then transformed into every variant, avoiding the
selection bias that would result from filtering each intervention separately.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from nano_apd.carving import (
    MatchedBatch,
    build_banks,
    capture_selected_usage,
    component_credits,
    selected_logits,
)
from nano_apd.induction_components import (
    effective_components,
    selected_distribution_kl,
    sum_normalized_gates,
)
from nano_apd.induction_data import target_correct_induction_batch
from nano_apd.induction_editor import InductionEditor
from nano_apd.induction_variants import (
    ABSENT_VARIANTS,
    PRESENT_VARIANTS,
    STRESS_VARIANTS,
    make_functional_variants,
)
from nano_apd.lm_target import load_carving_target, vocab_size


def _score(logits: Tensor, labels: Tensor, kind: str) -> Tensor:
    correct = logits.float().gather(-1, labels[:, None]).squeeze(-1)
    if kind == "target_logit":
        return correct
    if kind == "logprob":
        return F.log_softmax(logits.float(), -1).gather(
            -1, labels[:, None]
        ).squeeze(-1)
    if kind != "logit_margin":
        raise ValueError(kind)
    competitors = logits.float().scatter(-1, labels[:, None], float("-inf"))
    return correct - competitors.amax(-1)


def _gap(positive: Tensor, negative: Tensor, labels: Tensor, kind: str) -> Tensor:
    return _score(positive, labels, kind) - _score(negative, labels, kind)


def _selected_forward(target, tokens: Tensor, positions: Tensor) -> Tensor:
    return selected_logits(target(tokens), positions)


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def _mean(values: Tensor) -> float:
    if not values.numel():
        return float("nan")
    return round(values.detach().float().mean().item(), 6)


def _bootstrap_interval(values: Tensor, seed: int, samples: int = 1000) -> list[float]:
    values = values.detach().float().cpu()
    if not values.numel():
        return [float("nan"), float("nan")]
    if values.numel() == 1:
        value = round(values.item(), 6)
        return [value, value]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[indices].mean(-1)
    return [
        round(torch.quantile(means, 0.025).item(), 6),
        round(torch.quantile(means, 0.975).item(), 6),
    ]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _component_ablations(
    target,
    editor: InductionEditor,
    pairs: MatchedBatch,
    components: int,
    score_type: str,
    reference_positive: Tensor,
    reference_negative: Tensor,
    component_chunk: int,
    use_bf16: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove each component from the intact model, with no routing assumption."""
    batch = pairs.positive.shape[0]
    reference_gap = _gap(
        reference_positive, reference_negative, pairs.labels, score_type
    )
    damage = torch.empty(batch, components, device=pairs.positive.device)
    positive_kl = torch.empty_like(damage)
    negative_kl = torch.empty_like(damage)
    for start in range(0, components, component_chunk):
        ids = list(range(start, min(start + component_chunk, components)))
        count = len(ids)
        masks = torch.ones(
            count * batch,
            components,
            device=pairs.positive.device,
            dtype=reference_positive.dtype,
        )
        for variant, component in enumerate(ids):
            masks[variant * batch:(variant + 1) * batch, component] = 0
        tokens = torch.cat([
            pairs.positive.repeat(count, 1), pairs.negative.repeat(count, 1)
        ])
        positions = pairs.positions.repeat(2 * count)
        editor.masks = torch.cat([masks, masks]).unsqueeze(1)
        with torch.no_grad(), _autocast(use_bf16):
            selected = _selected_forward(target, tokens, positions)
        positive, negative = selected.split(count * batch)
        positive = positive.view(count, batch, -1)
        negative = negative.view(count, batch, -1)
        for variant, component in enumerate(ids):
            ablated_gap = _gap(
                positive[variant], negative[variant], pairs.labels, score_type
            )
            damage[:, component] = reference_gap - ablated_gap
            positive_kl[:, component] = selected_distribution_kl(
                positive[variant], reference_positive
            )
            negative_kl[:, component] = selected_distribution_kl(
                negative[variant], reference_negative
            )
    editor.masks = None
    return damage, positive_kl, negative_kl


def _routed_ablations(
    target,
    editor: InductionEditor,
    pairs: MatchedBatch,
    gates: Tensor,
    routed_gap: Tensor,
    components: int,
    score_type: str,
    component_chunk: int,
    use_bf16: bool,
) -> Tensor:
    """Remove each component from its attribution-routed model."""
    batch = pairs.positive.shape[0]
    damage = torch.empty(batch, components, device=pairs.positive.device)
    for start in range(0, components, component_chunk):
        ids = list(range(start, min(start + component_chunk, components)))
        count = len(ids)
        masks = gates.repeat(count, 1)
        for variant, component in enumerate(ids):
            masks[variant * batch:(variant + 1) * batch, component] = 0
        tokens = torch.cat([
            pairs.positive.repeat(count, 1), pairs.negative.repeat(count, 1)
        ])
        positions = pairs.positions.repeat(2 * count)
        editor.masks = torch.cat([masks, masks]).unsqueeze(1)
        with torch.no_grad(), _autocast(use_bf16):
            selected = _selected_forward(target, tokens, positions)
        positive, negative = selected.split(count * batch)
        positive = positive.view(count, batch, -1)
        negative = negative.view(count, batch, -1)
        for variant, component in enumerate(ids):
            ablated_gap = _gap(
                positive[variant], negative[variant], pairs.labels, score_type
            )
            damage[:, component] = routed_gap - ablated_gap
    editor.masks = None
    return damage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--n_pairs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=191_003)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--component_chunk", type=int, default=4)
    args = parser.parse_args()
    if min(args.n_pairs, args.batch_size, args.component_chunk) < 1:
        raise ValueError("n_pairs, batch_size, and component_chunk must be positive")

    config = json.loads((args.artifact / "config.json").read_text())
    device = torch.device("cuda")
    target = load_carving_target("hf", config["model_name"]).float().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    module_paths = config["module_paths"]
    components = int(config["C"])
    banks = build_banks(
        target, module_paths, components, int(config["rank"]), config["bank_type"]
    ).to(device)
    banks.load_state_dict(torch.load(
        args.artifact / "banks.pt", weights_only=True, map_location=device
    ))
    banks.eval()
    editor = InductionEditor(target, banks, module_paths)
    score_type = config["score_type"]
    variant_names = list(PRESENT_VARIANTS + STRESS_VARIANTS + ABSENT_VARIANTS)
    storage: dict[str, dict[str, list[Tensor]]] = {
        name: defaultdict(list) for name in variant_names
    }

    try:
        with torch.enable_grad():
            for start in range(0, args.n_pairs, args.batch_size):
                size = min(args.batch_size, args.n_pairs - start)
                standard = target_correct_induction_batch(
                    target,
                    batch_size=size,
                    seq_len=int(config["seq_len"]),
                    vocab_size=vocab_size(target),
                    device=device,
                    seed=args.seed + start * 10_007,
                    score_type=score_type,
                    min_target_gap=float(config["min_target_gap"]),
                    include_target_incorrect=False,
                    use_bf16=args.use_bf16,
                )
                variants = make_functional_variants(
                    standard, vocab_size(target), args.seed + 1_000_003 + start
                )
                for name in variant_names:
                    pairs = variants[name]
                    with _autocast(args.use_bf16):
                        positive_usage = capture_selected_usage(
                            target, editor, pairs.positive, pairs.positions,
                            pairs.labels, score_type,
                        )
                        negative_usage = capture_selected_usage(
                            target, editor, pairs.negative, pairs.positions,
                            pairs.labels, score_type,
                        )
                        credits = (
                            component_credits(
                                target, banks, module_paths, positive_usage
                            )
                            - component_credits(
                                target, banks, module_paths, negative_usage
                            )
                        )
                        gates, attribution = sum_normalized_gates(credits)

                    tokens = torch.cat([pairs.positive, pairs.negative])
                    positions = pairs.positions.repeat(2)
                    editor.masks = None
                    with torch.no_grad(), _autocast(args.use_bf16):
                        reference = _selected_forward(target, tokens, positions)
                    editor.masks = torch.cat([gates, gates]).unsqueeze(1)
                    with torch.no_grad(), _autocast(args.use_bf16):
                        routed = _selected_forward(target, tokens, positions)
                    editor.masks = torch.zeros(
                        2 * size, 1, components, device=device, dtype=gates.dtype
                    )
                    with torch.no_grad(), _autocast(args.use_bf16):
                        residual = _selected_forward(target, tokens, positions)
                    editor.masks = None
                    reference_positive, reference_negative = reference.split(size)
                    routed_positive, routed_negative = routed.split(size)
                    residual_positive, residual_negative = residual.split(size)
                    reference_gap = _gap(
                        reference_positive, reference_negative, pairs.labels, score_type
                    )
                    routed_gap = _gap(
                        routed_positive, routed_negative, pairs.labels, score_type
                    )
                    residual_gap = _gap(
                        residual_positive, residual_negative, pairs.labels, score_type
                    )
                    full_damage, full_positive_kl, full_negative_kl = (
                        _component_ablations(
                            target, editor, pairs, components, score_type,
                            reference_positive, reference_negative,
                            args.component_chunk, args.use_bf16,
                        )
                    )
                    routed_damage = _routed_ablations(
                        target, editor, pairs, gates, routed_gap, components,
                        score_type, args.component_chunk, args.use_bf16,
                    )

                    values = {
                        "reference_gap": reference_gap,
                        "reference_correct": (
                            reference_positive.argmax(-1) == pairs.labels
                        ),
                        "routed_gap": routed_gap,
                        "residual_gap": residual_gap,
                        "gates": gates,
                        "attribution": attribution,
                        "credits": credits,
                        "full_damage": full_damage,
                        "full_positive_kl": full_positive_kl,
                        "full_negative_kl": full_negative_kl,
                        "routed_damage": routed_damage,
                        "routed_kl": selected_distribution_kl(routed, reference),
                        "residual_kl": selected_distribution_kl(residual, reference),
                    }
                    for key, value in values.items():
                        storage[name][key].append(value.detach().cpu())
                    print(
                        f"rows {start + size}/{args.n_pairs} variant={name}",
                        flush=True,
                    )
    finally:
        editor.restore()

    merged = {
        name: {key: torch.cat(values) for key, values in rows.items()}
        for name, rows in storage.items()
    }
    variant_report = {}
    for name, values in merged.items():
        gates = values["gates"]
        component_rows = []
        for component in range(components):
            full_damage = values["full_damage"][:, component]
            routed_damage = values["routed_damage"][:, component]
            component_rows.append({
                "component": component,
                "mean_gate_share": _mean(gates[:, component]),
                "owner_fraction": _mean(
                    (gates.argmax(-1) == component).float()
                ),
                "mean_signed_credit": _mean(values["credits"][:, component]),
                "mean_abs_credit": _mean(values["credits"][:, component].abs()),
                "full_model_mean_gap_damage": _mean(full_damage),
                "full_model_gap_damage_ci95": _bootstrap_interval(
                    full_damage, args.seed + component + 10_000 * variant_names.index(name)
                ),
                "full_model_positive_damage_rate": _mean(
                    (full_damage > 0).float()
                ),
                "full_model_selected_kl": _mean(
                    0.5 * (
                        values["full_positive_kl"][:, component]
                        + values["full_negative_kl"][:, component]
                    )
                ),
                "routed_mean_gap_damage": _mean(routed_damage),
                "routed_positive_damage_rate": _mean(
                    (routed_damage > 0).float()
                ),
            })
        variant_report[name] = {
            "category": (
                "induction_present" if name in PRESENT_VARIANTS
                else "stress" if name in STRESS_VARIANTS
                else "induction_absent_control"
            ),
            "reference_gap": _mean(values["reference_gap"]),
            "reference_positive_accuracy": _mean(
                values["reference_correct"].float()
            ),
            "routed_gap": _mean(values["routed_gap"]),
            "residual_gap": _mean(values["residual_gap"]),
            "mean_effective_components": _mean(
                effective_components(values["attribution"])
            ),
            "mean_winner_share": _mean(gates.amax(-1)),
            "components_owning_examples": int(
                torch.unique(gates.argmax(-1)).numel()
            ),
            "routed_selected_kl": _mean(values["routed_kl"]),
            "residual_selected_kl": _mean(values["residual_kl"]),
            "components": component_rows,
        }

    component_report = []
    for component in range(components):
        present_damage = torch.stack([
            merged[name]["full_damage"][:, component].mean()
            for name in PRESENT_VARIANTS
        ])
        absent_damage = torch.stack([
            merged[name]["full_damage"][:, component].mean()
            for name in ABSENT_VARIANTS
        ])
        present_kl = torch.stack([
            0.5 * (
                merged[name]["full_positive_kl"][:, component].mean()
                + merged[name]["full_negative_kl"][:, component].mean()
            )
            for name in PRESENT_VARIANTS
        ])
        absent_kl = torch.stack([
            0.5 * (
                merged[name]["full_positive_kl"][:, component].mean()
                + merged[name]["full_negative_kl"][:, component].mean()
            )
            for name in ABSENT_VARIANTS
        ])
        component_report.append({
            "component": component,
            "present_variant_positive_sign_fraction": _mean(
                (present_damage > 0).float()
            ),
            "present_variant_mean_gap_damage": _mean(present_damage),
            "absent_control_mean_abs_gap_damage": _mean(absent_damage.abs()),
            "present_minus_absent_gap_damage": _mean(
                present_damage
            ) - _mean(absent_damage.abs()),
            "present_variant_mean_selected_kl": _mean(present_kl),
            "absent_control_mean_selected_kl": _mean(absent_kl),
            "variant_gap_damage": {
                name: _mean(merged[name]["full_damage"][:, component])
                for name in variant_names
            },
            "interpretation_guardrail": (
                "Consistency across these interventions is evidence of induction "
                "selectivity, not proof of monosemanticity on arbitrary natural text."
            ),
        })

    report = _json_safe({
        "artifact": str(args.artifact),
        "mechanism": (
            "Random-token induction: an earlier cue->label demonstration makes a "
            "later repetition of the cue predict the label."
        ),
        "selection": (
            "Rows are selected once for target-correct positive-gap standard "
            "induction, then reused without filtering for every intervention."
        ),
        "cross_layer_policy": (
            "A component is one object across all editable matrices; layer/head "
            "breadth is neither penalized nor treated as polysemanticity."
        ),
        "n_pairs": args.n_pairs,
        "components": components,
        "precision": "bf16" if args.use_bf16 else "fp32",
        "variant_definitions": {
            "present": list(PRESENT_VARIANTS),
            "stress": list(STRESS_VARIANTS),
            "absent_controls": list(ABSENT_VARIANTS),
        },
        "component_functional_signatures": component_report,
        "variants": variant_report,
    })
    suffix = "bf16" if args.use_bf16 else "fp32"
    output = args.artifact / f"induction_roles_{suffix}.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
