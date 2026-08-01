"""Run dictionary-free token-conditioned weight carving on an induction contrast.

The run has no optimizer.  It discovers fixed low-rank pieces from clean-minus-corrupt
selected-token gradients, then evaluates joint integrated attribution, independent
leave-one-out integration, prompt-family controls, and stochastic coalition marginals.
"""

import argparse
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.carving import (
    CarvingEditor,
    MatchedBatch,
    make_matched_induction_batch,
    selected_logits,
)
from nano_apd.induction_components import attention_head_credits, effective_components
from nano_apd.induction_variants import (
    ABSENT_VARIANTS,
    PRESENT_VARIANTS,
    STRESS_VARIANTS,
    make_functional_variants,
    source_positions,
)
from nano_apd.lm_target import (
    DEFAULT_HF_MODEL,
    candidate_linear_paths,
    load_carving_target,
    vocab_size,
)
from nano_apd.token_weight_carving import (
    FixedPieceCollection,
    TokenCarvingEditor,
    attribution_shares,
    capture_fixed_margin_usage,
    contrast_candidate_credits,
    extract_fixed_pieces,
    fixed_margin_scores,
    integrate_gate_path,
    paired_random_coalitions,
)

OUT_ROOT = Path(__file__).parent / "out"


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def _pair_distractors(batch: MatchedBatch, token_vocabulary: int) -> tuple[Tensor, int]:
    """Use the corrupt continuation, with a declared fallback for identical controls."""
    different = batch.positive != batch.negative
    result = []
    fallback_count = 0
    for row in range(batch.positive.shape[0]):
        positions = different[row].nonzero(as_tuple=False).flatten()
        candidates = batch.negative[row, positions]
        candidates = candidates[candidates != batch.labels[row]]
        if candidates.numel():
            result.append(candidates[-1])
        else:
            fallback_count += 1
            result.append((batch.labels[row] + 1) % token_vocabulary)
    return torch.stack(result), fallback_count


def _selected_fixed_gap(
    target: nn.Module,
    batch: MatchedBatch,
    distractors: Tensor,
    use_bf16: bool,
) -> Tensor:
    tokens = torch.cat([batch.positive, batch.negative])
    positions = batch.positions.repeat(2)
    labels = batch.labels.repeat(2)
    comparisons = distractors.repeat(2)
    with torch.no_grad(), _autocast(batch.positive.device, use_bf16):
        logits = target(tokens)
        scores = fixed_margin_scores(logits, positions, labels, comparisons)
    positive, negative = scores.split(batch.positive.shape[0])
    return positive - negative


def _select_batch_rows(batch: MatchedBatch, rows: Tensor) -> MatchedBatch:
    return MatchedBatch(
        batch.positive.index_select(0, rows),
        batch.negative.index_select(0, rows),
        batch.positions.index_select(0, rows),
        batch.labels.index_select(0, rows),
    )


def _fixed_target_correct_batch(
    target: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    token_vocabulary: int,
    device: torch.device,
    seed: int,
    min_target_gap: float,
    use_bf16: bool,
    variant_name: str,
    oversample: int = 4,
    max_rounds: int = 16,
) -> tuple[MatchedBatch, Tensor, dict[str, float | int]]:
    """Rejection-sample using the same fixed score used for carving and evaluation."""
    if variant_name not in PRESENT_VARIANTS:
        raise ValueError(f"discovery variant {variant_name!r} is not induction-present")
    selected_batches = []
    selected_distractors = []
    selected_count = 0
    generated_count = 0
    eligible_seen = 0
    for round_index in range(max_rounds):
        round_seed = seed + round_index * 1_000_003
        candidates = make_matched_induction_batch(
            batch_size * oversample,
            seq_len,
            token_vocabulary,
            device,
            round_seed,
        )
        if variant_name != "standard":
            candidates = make_functional_variants(
                candidates, token_vocabulary, round_seed + 500_009
            )[variant_name]
        distractors, _ = _pair_distractors(candidates, token_vocabulary)
        tokens = torch.cat([candidates.positive, candidates.negative])
        positions = candidates.positions.repeat(2)
        with torch.no_grad(), _autocast(device, use_bf16):
            chosen = selected_logits(target(tokens), positions).float()
        positive, negative = chosen.split(candidates.positive.shape[0])
        positive_margin = positive.gather(-1, candidates.labels[:, None]).squeeze(
            -1
        ) - positive.gather(-1, distractors[:, None]).squeeze(-1)
        negative_margin = negative.gather(-1, candidates.labels[:, None]).squeeze(
            -1
        ) - negative.gather(-1, distractors[:, None]).squeeze(-1)
        keep = (positive_margin - negative_margin) > min_target_gap
        keep &= positive.argmax(-1) == candidates.labels
        generated_count += candidates.positive.shape[0]
        eligible_seen += int(keep.sum())
        rows = keep.nonzero(as_tuple=False).flatten()
        if rows.numel():
            need = batch_size - selected_count
            rows = rows[:need]
            selected_batches.append(_select_batch_rows(candidates, rows))
            selected_distractors.append(distractors.index_select(0, rows))
            selected_count += rows.numel()
        if selected_count >= batch_size:
            break
    if selected_count < batch_size:
        raise RuntimeError(
            f"found only {selected_count}/{batch_size} fixed-score induction pairs "
            f"for {variant_name!r} after {generated_count} candidates"
        )
    batch = MatchedBatch(
        torch.cat([item.positive for item in selected_batches]),
        torch.cat([item.negative for item in selected_batches]),
        torch.cat([item.positions for item in selected_batches]),
        torch.cat([item.labels for item in selected_batches]),
    )
    batch.validate()
    return (
        batch,
        torch.cat(selected_distractors),
        {
            "generated_count": generated_count,
            "eligible_seen": eligible_seen,
            "observed_eligible_fraction": eligible_seen / generated_count,
        },
    )


def _repeat_rows(value: Tensor, repeats: int) -> Tensor:
    return (
        value.unsqueeze(0)
        .expand(repeats, *value.shape)
        .reshape(repeats * value.shape[0], *value.shape[1:])
    )


class GateScore:
    """Map physical gate rows to fixed-margin clean-minus-corrupt scores."""

    def __init__(
        self,
        target: nn.Module,
        editor: TokenCarvingEditor,
        batch: MatchedBatch,
        distractors: Tensor,
        use_bf16: bool,
    ):
        self.target = target
        self.editor = editor
        self.batch = batch
        self.distractors = distractors
        self.use_bf16 = use_bf16

    def selected(self, gates: Tensor) -> tuple[Tensor, Tensor]:
        """Return selected-position logits for clean and corrupt prompts."""
        batch_size = self.batch.positive.shape[0]
        if gates.shape[0] % batch_size:
            raise ValueError("gate rows must be a whole number of prompt batches")
        repeats = gates.shape[0] // batch_size
        positive = _repeat_rows(self.batch.positive, repeats)
        negative = _repeat_rows(self.batch.negative, repeats)
        positions = _repeat_rows(self.batch.positions, repeats)
        self.editor.masks = torch.cat([gates, gates]).unsqueeze(1)
        with _autocast(gates.device, self.use_bf16):
            logits = self.target(torch.cat([positive, negative]))
            positive_logits, negative_logits = logits.split(gates.shape[0])
        return (
            selected_logits(positive_logits, positions).float(),
            selected_logits(negative_logits, positions).float(),
        )

    def margins(self, gates: Tensor) -> tuple[Tensor, Tensor]:
        positive, negative = self.selected(gates)
        repeats = gates.shape[0] // self.batch.positive.shape[0]
        labels = _repeat_rows(self.batch.labels, repeats)
        distractors = _repeat_rows(self.distractors, repeats)

        def margin(logits):
            target = logits.gather(-1, labels[:, None]).squeeze(-1)
            comparison = logits.gather(-1, distractors[:, None]).squeeze(-1)
            return target - comparison

        return margin(positive), margin(negative)

    def __call__(self, gates: Tensor) -> Tensor:
        positive, negative = self.margins(gates)
        return positive - negative


class _HeadAblator:
    """Per-example masks over GPT-NeoX head outputs entering attention.dense."""

    def __init__(self, target: nn.Module, dense_paths: list[str], n_heads: int):
        self.masks: dict[str, Tensor] = {}
        self.n_heads = n_heads
        self.handles = []
        for path in dense_paths:
            module = target.get_submodule(path)

            def hook(_module, args, path=path):
                if path not in self.masks:
                    return args
                value = args[0]
                head_width = value.shape[-1] // self.n_heads
                mask = self.masks[path].to(value.dtype).view(
                    value.shape[0], 1, self.n_heads, 1
                )
                value = (
                    value.view(value.shape[0], value.shape[1], self.n_heads, head_width)
                    * mask
                ).reshape_as(value)
                return (value,) + args[1:]

            self.handles.append(module.register_forward_pre_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _discovery_usages(
    target: nn.Module,
    module_paths: list[str],
    args,
    device: torch.device,
) -> tuple[
    dict[str, tuple[object, object]],
    dict[str, MatchedBatch],
    dict[str, dict[str, float | int]],
]:
    variants = tuple(item.strip() for item in args.discovery_variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(PRESENT_VARIANTS))
    if not variants or unknown:
        raise ValueError(f"unknown discovery variants: {unknown}")
    if "standard" not in variants:
        raise ValueError("discovery_variants must include 'standard'")
    capture = CarvingEditor(target, nn.ModuleDict(), module_paths)
    usages = {}
    batches = {}
    selection_reports = {}
    try:
        for index, variant in enumerate(variants):
            batch, distractors, selection_report = _fixed_target_correct_batch(
                target,
                batch_size=args.discovery_pairs,
                seq_len=args.seq_len,
                token_vocabulary=vocab_size(target),
                device=device,
                seed=args.seed + 100_003 * index,
                min_target_gap=args.min_target_gap,
                use_bf16=args.bf16,
                variant_name=variant,
            )
            with _autocast(device, args.bf16):
                positive = capture_fixed_margin_usage(
                    target,
                    capture,
                    batch.positive,
                    batch.positions,
                    batch.labels,
                    distractors,
                )
                negative = capture_fixed_margin_usage(
                    target,
                    capture,
                    batch.negative,
                    batch.positions,
                    batch.labels,
                    distractors,
                )
            usages[variant] = (positive, negative)
            batches[variant] = batch
            selection_reports[variant] = selection_report
    finally:
        capture.restore()
    return usages, batches, selection_reports


def _position_regions(
    pieces: FixedPieceCollection,
    batch: MatchedBatch,
    position_credit: Tensor,
) -> list[dict[str, object]]:
    """Partition attribution mass and report enrichment relative to region size."""
    source = source_positions(batch)
    rows = torch.arange(batch.positive.shape[0], device=batch.positive.device)
    destination = batch.positions
    labels = ("source_cue", "source_label", "destination_cue", "other")
    token_count = position_credit.shape[-1]
    baseline = position_credit.new_tensor(
        [1 / token_count, 1 / token_count, 1 / token_count, (token_count - 3) / token_count]
    )
    regions = []
    for candidate in range(pieces.candidates):
        values = position_credit[:, candidate].abs()
        total = values.sum(-1).clamp_min(1e-20)
        named_per_row = torch.stack(
            [
                values[rows, source],
                values[rows, source + 1],
                values[rows, destination],
            ],
            -1,
        )
        other_per_row = (total - named_per_row.sum(-1)).clamp_min(0)
        fractions = torch.cat([named_per_row, other_per_row[:, None]], -1) / total[:, None]
        mean_fraction = fractions.mean(0)
        enrichment = mean_fraction / baseline
        regions.append(
            {
                "candidate_id": candidate,
                "largest_attribution_region": labels[int(enrichment.argmax())],
                "region_fraction": {
                    name: float(value)
                    for name, value in zip(labels, mean_fraction.tolist(), strict=True)
                },
                "region_enrichment": {
                    name: float(value)
                    for name, value in zip(labels, enrichment.tolist(), strict=True)
                },
            }
        )
    return regions


def _attribution_shortlist(
    pieces: FixedPieceCollection,
    candidate_score: Tensor,
    total_count: int,
    per_layer: int,
) -> Tensor:
    if candidate_score.shape != (pieces.candidates,):
        raise ValueError("candidate_score must have one value per fixed piece")
    score = candidate_score
    budget = min(total_count, score.numel())
    layers = sorted({item.layer for item in pieces.metadata})
    required = sum(
        min(per_layer, sum(item.layer == layer for item in pieces.metadata))
        for layer in layers
    )
    if required > budget:
        raise ValueError(
            f"screen_top={budget} cannot reserve {per_layer} candidate(s) "
            f"for each of {len(layers)} layers; need at least {required}"
        )
    chosen: set[int] = set()
    for layer in layers:
        ids = torch.tensor(
            [item.candidate_id for item in pieces.metadata if item.layer == layer],
            device=score.device,
        )
        count = min(per_layer, ids.numel())
        local = ids.index_select(0, torch.topk(score.index_select(0, ids), count).indices)
        chosen.update(local.tolist())
    for candidate in torch.argsort(score, descending=True).tolist():
        if len(chosen) >= budget:
            break
        chosen.add(candidate)
    ordered = sorted(chosen, key=lambda index: float(score[index]), reverse=True)
    return torch.tensor(ordered, device=score.device)


def _independent_integrals(
    score: GateScore,
    pieces: FixedPieceCollection,
    candidate_ids: Tensor,
    prompt_batch: int,
    steps: int,
    chunk: int,
) -> list[dict[str, object]]:
    rows = []
    for start_index in range(0, candidate_ids.numel(), chunk):
        ids = candidate_ids[start_index : start_index + chunk]
        repeated = ids.numel() * prompt_batch
        start = torch.ones(repeated, pieces.candidates, device=ids.device)
        end = torch.ones_like(start)
        for local, candidate in enumerate(ids.tolist()):
            start[local * prompt_batch : (local + 1) * prompt_batch, candidate] = 0
        result = integrate_gate_path(score, start, end, steps)
        endpoint = result.end_scores - result.start_scores
        for local, candidate in enumerate(ids.tolist()):
            interval = slice(local * prompt_batch, (local + 1) * prompt_batch)
            integrated = result.contributions[interval, candidate]
            exact = endpoint[interval]
            rows.append(
                {
                    "candidate_id": candidate,
                    "integrated_mean": float(integrated.mean()),
                    "integrated_abs_mean": float(integrated.abs().mean()),
                    "endpoint_mean": float(exact.mean()),
                    "endpoint_abs_mean": float(exact.abs().mean()),
                    "positive_fraction": float((exact > 0).float().mean()),
                    "quadrature_mae": float((integrated - exact).abs().mean()),
                    "per_example_endpoint": exact.tolist(),
                }
            )
    return rows


def _endpoint_leave_one_out(
    score: GateScore,
    pieces: FixedPieceCollection,
    candidate_ids: Tensor,
    prompt_batch: int,
    chunk: int,
) -> dict[int, Tensor]:
    full = score(torch.ones(prompt_batch, pieces.candidates, device=candidate_ids.device))
    result = {}
    for start_index in range(0, candidate_ids.numel(), chunk):
        ids = candidate_ids[start_index : start_index + chunk]
        gates = torch.ones(ids.numel() * prompt_batch, pieces.candidates, device=ids.device)
        for local, candidate in enumerate(ids.tolist()):
            gates[local * prompt_batch : (local + 1) * prompt_batch, candidate] = 0
        with torch.no_grad():
            ablated = score(gates).reshape(ids.numel(), prompt_batch)
        for local, candidate in enumerate(ids.tolist()):
            result[candidate] = full - ablated[local]
    return result


def _bootstrap_mean_ci(values: Tensor, seed: int, samples: int = 1000) -> list[float]:
    values = values.detach().float().cpu()
    if not values.numel():
        return [float("nan"), float("nan")]
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, values.numel(), (samples, values.numel()), generator=generator)
    means = values.index_select(0, rows.flatten()).reshape(samples, values.numel()).mean(-1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


def _effect_summary(values: Tensor, seed: int) -> dict[str, object]:
    return {
        "mean": float(values.mean()),
        "abs_mean": float(values.abs().mean()),
        "positive_fraction": float((values > 0).float().mean()),
        "mean_ci95_unadjusted": _bootstrap_mean_ci(values, seed),
    }


def _margin_from_selected(logits: Tensor, labels: Tensor, distractors: Tensor) -> Tensor:
    leading = (1,) * (logits.ndim - 2)
    target_index = labels.view(*leading, -1, 1).expand(*logits.shape[:-1], 1)
    distractor_index = distractors.view(*leading, -1, 1).expand(*logits.shape[:-1], 1)
    return logits.gather(-1, target_index).squeeze(-1) - logits.gather(
        -1, distractor_index
    ).squeeze(-1)


def _direct_head_ablation_reference(
    target: nn.Module,
    batch: MatchedBatch,
    distractors: Tensor,
    module_paths: list[str],
    discovery_head_contrast: Tensor,
    n_heads: int,
    chunk: int,
    seed: int,
) -> tuple[dict[str, object], Tensor]:
    """Build an objective- and population-matched empirical head reference."""
    dense_paths = [path for path in module_paths if path.endswith("attention.dense")]
    if len(dense_paths) != discovery_head_contrast.shape[1]:
        raise ValueError("head-attribution layer count does not match attention.dense paths")
    prompt_batch = batch.positive.shape[0]
    tokens = torch.cat([batch.positive, batch.negative])
    positions = batch.positions.repeat(2)
    with torch.no_grad():
        baseline_selected = selected_logits(target(tokens), positions).float()
    baseline_positive, baseline_negative = baseline_selected.split(prompt_batch)
    baseline_gap = _margin_from_selected(
        baseline_positive, batch.labels, distractors
    ) - _margin_from_selected(baseline_negative, batch.labels, distractors)

    damage = torch.empty(
        prompt_batch, len(dense_paths), n_heads, device=batch.positive.device
    )
    heads = [
        (layer, head)
        for layer in range(len(dense_paths))
        for head in range(n_heads)
    ]
    ablator = _HeadAblator(target, dense_paths, n_heads)
    try:
        for start in range(0, len(heads), chunk):
            selected_heads = heads[start : start + chunk]
            variants = len(selected_heads)
            ablator.masks = {
                path: torch.ones(
                    variants * 2 * prompt_batch,
                    n_heads,
                    device=batch.positive.device,
                )
                for path in dense_paths
            }
            for variant, (layer, head) in enumerate(selected_heads):
                positive_rows = slice(
                    variant * prompt_batch, (variant + 1) * prompt_batch
                )
                negative_rows = slice(
                    variants * prompt_batch + variant * prompt_batch,
                    variants * prompt_batch + (variant + 1) * prompt_batch,
                )
                ablator.masks[dense_paths[layer]][positive_rows, head] = 0
                ablator.masks[dense_paths[layer]][negative_rows, head] = 0
            repeated_tokens = torch.cat(
                [batch.positive.repeat(variants, 1), batch.negative.repeat(variants, 1)]
            )
            repeated_positions = batch.positions.repeat(2 * variants)
            with torch.no_grad():
                selected = selected_logits(
                    target(repeated_tokens), repeated_positions
                ).float()
            positive, negative = selected.split(variants * prompt_batch)
            positive = positive.reshape(variants, prompt_batch, -1)
            negative = negative.reshape(variants, prompt_batch, -1)
            ablated_gap = _margin_from_selected(
                positive, batch.labels, distractors
            ) - _margin_from_selected(negative, batch.labels, distractors)
            for variant, (layer, head) in enumerate(selected_heads):
                damage[:, layer, head] = baseline_gap - ablated_gap[variant]
    finally:
        ablator.close()

    attribution = discovery_head_contrast.square().mean(0)
    attribution_order = attribution.flatten().argsort(descending=True)
    attribution_top = set(attribution_order[: min(32, len(heads))].tolist())
    rows = []
    for layer, head in heads:
        values = damage[:, layer, head]
        interval = _bootstrap_mean_ci(values, seed + layer * n_heads + head)
        rows.append(
            {
                "layer": layer,
                "head": head,
                "mean_contrast_damage": float(values.mean()),
                "contrast_damage_ci95_unadjusted": interval,
                "discovery_squared_attribution_mean": float(attribution[layer, head]),
                "positive_causal_ci_unadjusted": interval[0] > 0,
            }
        )
    top_by_ablation = sorted(
        rows, key=lambda row: row["mean_contrast_damage"], reverse=True
    )[:32]
    top_by_attribution = sorted(
        rows, key=lambda row: row["discovery_squared_attribution_mean"], reverse=True
    )[:32]
    convergent = [
        row
        for row in rows
        if row["positive_causal_ci_unadjusted"]
        and row["layer"] * n_heads + row["head"] in attribution_top
    ]
    layer_mass = damage.new_zeros(len(dense_paths))
    for row in convergent:
        layer_mass[row["layer"]] += max(0.0, row["mean_contrast_damage"])
    layer_mass /= layer_mass.sum().clamp_min(1e-12)
    report = {
        "warning": (
            "Empirical reference on the same confirmation prompts and fixed readout; "
            "it is not literal ground truth. Confidence intervals are unadjusted."
        ),
        "tested_heads": len(heads),
        "full_gap_mean": float(baseline_gap.mean()),
        "top_by_direct_ablation": top_by_ablation,
        "top_by_discovery_attribution": top_by_attribution,
        "top32_attribution_and_positive_ablation": convergent,
        "reference_layer_mass": layer_mass.tolist(),
    }
    return report, layer_mass


def _endpoint_diagnostics(
    score: GateScore,
    pieces: FixedPieceCollection,
    candidate_ids: Tensor,
    prompt_batch: int,
    chunk: int,
    seed: int,
) -> dict[int, dict[str, object]]:
    """Exact LOO effects with clean/corrupt damage and selected-distribution side effects."""
    full_gates = torch.ones(prompt_batch, pieces.candidates, device=candidate_ids.device)
    with torch.no_grad():
        full_positive, full_negative = score.selected(full_gates)
    full_positive_margin = _margin_from_selected(
        full_positive, score.batch.labels, score.distractors
    )
    full_negative_margin = _margin_from_selected(
        full_negative, score.batch.labels, score.distractors
    )
    result = {}
    for start_index in range(0, candidate_ids.numel(), chunk):
        ids = candidate_ids[start_index : start_index + chunk]
        gates = torch.ones(ids.numel() * prompt_batch, pieces.candidates, device=ids.device)
        for local, candidate in enumerate(ids.tolist()):
            gates[local * prompt_batch : (local + 1) * prompt_batch, candidate] = 0
        with torch.no_grad():
            ablated_positive, ablated_negative = score.selected(gates)
        ablated_positive = ablated_positive.reshape(ids.numel(), prompt_batch, -1)
        ablated_negative = ablated_negative.reshape(ids.numel(), prompt_batch, -1)
        positive_margin = _margin_from_selected(
            ablated_positive, score.batch.labels, score.distractors
        )
        negative_margin = _margin_from_selected(
            ablated_negative, score.batch.labels, score.distractors
        )
        clean_damage = full_positive_margin[None] - positive_margin
        corrupt_damage = full_negative_margin[None] - negative_margin
        contrast_damage = clean_damage - corrupt_damage
        clean_kl = F.kl_div(
            F.log_softmax(ablated_positive, -1),
            F.softmax(full_positive, -1)[None].expand_as(ablated_positive),
            reduction="none",
        ).sum(-1)
        corrupt_kl = F.kl_div(
            F.log_softmax(ablated_negative, -1),
            F.softmax(full_negative, -1)[None].expand_as(ablated_negative),
            reduction="none",
        ).sum(-1)
        clean_flips = ablated_positive.argmax(-1) != full_positive.argmax(-1)[None]
        corrupt_flips = ablated_negative.argmax(-1) != full_negative.argmax(-1)[None]
        for local, candidate in enumerate(ids.tolist()):
            local_seed = seed + candidate * 1009
            result[candidate] = {
                "candidate_id": candidate,
                "contrast_damage": _effect_summary(contrast_damage[local], local_seed),
                "clean_margin_damage": _effect_summary(clean_damage[local], local_seed + 1),
                "corrupt_margin_damage": _effect_summary(corrupt_damage[local], local_seed + 2),
                "clean_selected_kl_mean": float(clean_kl[local].mean()),
                "corrupt_selected_kl_mean": float(corrupt_kl[local].mean()),
                "clean_prediction_flip_fraction": float(clean_flips[local].float().mean()),
                "corrupt_prediction_flip_fraction": float(corrupt_flips[local].float().mean()),
                "per_example_contrast_damage": contrast_damage[local].tolist(),
            }
    score.editor.masks = None
    return result


def _population_summary(
    score: GateScore,
    pieces: FixedPieceCollection,
    min_target_gap: float,
) -> dict[str, float]:
    gates = torch.ones(
        score.batch.positive.shape[0],
        pieces.candidates,
        device=score.batch.positive.device,
    )
    with torch.no_grad():
        positive, negative = score.selected(gates)
    positive_margin = _margin_from_selected(positive, score.batch.labels, score.distractors)
    negative_margin = _margin_from_selected(negative, score.batch.labels, score.distractors)
    gap = positive_margin - negative_margin
    return {
        "fixed_gap_mean": float(gap.mean()),
        "clean_target_accuracy": float((positive.argmax(-1) == score.batch.labels).float().mean()),
        "eligible_fraction": float(
            ((gap > min_target_gap) & (positive.argmax(-1) == score.batch.labels)).float().mean()
        ),
    }


def _coalition_marginals(
    score: GateScore,
    pieces: FixedPieceCollection,
    candidate_ids: Tensor,
    prompt_batch: int,
    samples: int,
    pattern_chunk: int,
    seed: int,
) -> list[dict[str, object]]:
    generator = torch.Generator(device=candidate_ids.device).manual_seed(seed)
    ids, on, off = paired_random_coalitions(pieces.candidates, candidate_ids, samples, generator)
    effects = []
    for start in range(0, ids.numel(), pattern_chunk):
        stop = min(ids.numel(), start + pattern_chunk)
        local_on = (
            on[start:stop, None]
            .expand(-1, prompt_batch, -1)
            .reshape((stop - start) * prompt_batch, pieces.candidates)
        )
        local_off = (
            off[start:stop, None]
            .expand_as(on[start:stop, None].expand(-1, prompt_batch, -1))
            .reshape((stop - start) * prompt_batch, pieces.candidates)
        )
        with torch.no_grad():
            on_score = score(local_on).reshape(stop - start, prompt_batch)
            off_score = score(local_off).reshape(stop - start, prompt_batch)
        effects.append(on_score - off_score)
    effects_tensor = torch.cat(effects)
    rows = []
    for candidate in candidate_ids.tolist():
        values = effects_tensor[ids == candidate]
        coalition_means = values.mean(-1)
        prompt_means = values.mean(0)
        flattened = values.flatten()
        rows.append(
            {
                "candidate_id": candidate,
                "mean": float(flattened.mean()),
                "median": float(flattened.median()),
                "mean_ci95_across_prompts_unadjusted": _bootstrap_mean_ci(
                    prompt_means, seed + candidate * 1013
                ),
                "coalition_mean_std": float(coalition_means.std(unbiased=False)),
                "prompt_mean_std": float(prompt_means.std(unbiased=False)),
                "positive_fraction": float((flattened > 0).float().mean()),
                "positive_prompt_mean_fraction": float((prompt_means > 0).float().mean()),
                "q10": float(flattened.quantile(0.1)),
                "q90": float(flattened.quantile(0.9)),
                "coalition_sample_means": coalition_means.tolist(),
                "per_prompt_means": prompt_means.tolist(),
            }
        )
    return rows


def _fingerprint_neighbors(
    candidate_ids: Tensor,
    fingerprints: Tensor,
    pieces: FixedPieceCollection,
) -> list[dict[str, object]]:
    normalized = fingerprints / fingerprints.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-1)
    rows = []
    for local, candidate in enumerate(candidate_ids.tolist()):
        metadata = pieces.metadata[candidate]
        cross_layer = torch.tensor(
            [
                item.layer != metadata.layer
                for item in (pieces.metadata[index] for index in candidate_ids.tolist())
            ],
            device=similarity.device,
        )
        if not bool(cross_layer.any()):
            rows.append(
                {
                    "candidate_id": candidate,
                    "neighbor_id": None,
                    "cosine": None,
                }
            )
            continue
        allowed = similarity[local].masked_fill(~cross_layer, -1)
        neighbor_local = int(allowed.argmax())
        rows.append(
            {
                "candidate_id": candidate,
                "neighbor_id": int(candidate_ids[neighbor_local]),
                "cosine": float(allowed[neighbor_local]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_HF_MODEL)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--geometry", choices=["euclidean", "diag_kfac"], default="euclidean")
    parser.add_argument("--projection", choices=["paired", "cartesian"], default="paired")
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--discovery_pairs", type=int, default=4)
    parser.add_argument(
        "--discovery_variants",
        default="standard,short_lag,long_lag,two_demonstrations",
    )
    parser.add_argument("--screen_pairs", type=int, default=16)
    parser.add_argument("--confirm_pairs", type=int, default=32)
    parser.add_argument("--unfiltered_pairs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--min_target_gap", type=float, default=0.05)
    parser.add_argument("--integration_steps", type=int, default=16)
    parser.add_argument("--screen_top", type=int, default=32)
    parser.add_argument("--screen_per_layer", type=int, default=1)
    parser.add_argument("--intervention_chunk", type=int, default=1)
    parser.add_argument("--impact_top", type=int, default=16)
    parser.add_argument("--coalition_samples", type=int, default=32)
    parser.add_argument("--coalition_chunk", type=int, default=2)
    parser.add_argument("--head_ablation_chunk", type=int, default=4)
    parser.add_argument("--skip_head_ablation", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--model_revision")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="pilot")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Pythia token carving requires CUDA")
    positive_arguments = (
        args.rank,
        args.discovery_pairs,
        args.screen_pairs,
        args.confirm_pairs,
        args.unfiltered_pairs,
        args.integration_steps,
        args.screen_top,
        args.intervention_chunk,
        args.impact_top,
        args.coalition_samples,
        args.coalition_chunk,
        args.head_ablation_chunk,
    )
    if min(positive_arguments) < 1 or args.screen_per_layer < 0:
        raise ValueError("pair counts, ranks, chunks, and shortlist sizes must be positive")
    if args.impact_top > args.screen_top:
        raise ValueError("impact_top cannot exceed screen_top")

    slug = args.model_name.rsplit("/", 1)[-1].lower()
    out_dir = OUT_ROOT / (
        f"{slug}_token_carving_{args.tag}_{args.geometry}_{args.projection}_r{args.rank}"
    )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing run: {out_dir}")

    run_start = time.perf_counter()
    phase_seconds = {}
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    phase_start = time.perf_counter()
    target = load_carving_target(
        "hf", args.model_name, revision=args.model_revision
    ).float().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    module_paths = candidate_linear_paths(target, "hf")
    layers = int(target.config.num_hidden_layers)
    token_vocabulary = vocab_size(target)
    torch.cuda.synchronize()
    phase_seconds["model_load"] = time.perf_counter() - phase_start
    print(
        json.dumps(
            {
                "phase": "loaded",
                "model": args.model_name,
                "modules": len(module_paths),
                "layers": layers,
                "geometry": args.geometry,
                "projection": args.projection,
            }
        ),
        flush=True,
    )

    phase_start = time.perf_counter()
    usages, discovery_batches, discovery_selection = _discovery_usages(
        target, module_paths, args, device
    )
    pieces, extraction = extract_fixed_pieces(
        target,
        module_paths,
        usages,
        args.rank,
        args.geometry,
        args.projection,
        args.damping,
        args.seed + 99_991,
    )
    standard_usage = usages["standard"]
    discovery_credit_rows = []
    discovery_position = None
    for name, (positive, negative) in usages.items():
        credits, position_credit = contrast_candidate_credits(pieces, positive, negative)
        discovery_credit_rows.append(credits)
        if name == "standard":
            discovery_position = position_credit
    discovery_credits = torch.cat(discovery_credit_rows)
    shares, squared_attribution = attribution_shares(discovery_credits)
    if discovery_position is None:
        raise RuntimeError("standard discovery position attribution is missing")
    position_regions = _position_regions(pieces, discovery_batches["standard"], discovery_position)
    discovery_importance = shares.mean(0)
    attribution_ids = _attribution_shortlist(
        pieces,
        discovery_importance,
        args.screen_top,
        args.screen_per_layer,
    )
    loo_screen_ids = attribution_ids
    with _autocast(device, args.bf16):
        head_positive, _ = attention_head_credits(
            target,
            module_paths,
            standard_usage[0],
            int(target.config.num_attention_heads),
        )
        head_negative, _ = attention_head_credits(
            target,
            module_paths,
            standard_usage[1],
            int(target.config.num_attention_heads),
        )
    head_contrast = head_positive - head_negative
    head_layer_attribution = head_contrast.square().mean(0).sum(-1)
    head_layer_attribution /= head_layer_attribution.sum().clamp_min(1e-12)
    torch.cuda.synchronize()
    phase_seconds["discovery_and_extraction"] = time.perf_counter() - phase_start
    print(
        json.dumps(
            {
                "phase": "extracted",
                "candidates": pieces.candidates,
                "active_candidates": sum(item.active for item in pieces.metadata),
                "attribution_shortlist": loo_screen_ids.numel(),
            }
        ),
        flush=True,
    )

    eval_names = (
        "standard",
        "short_lag",
        "long_lag",
        "two_demonstrations",
        "counterfactual_label",
        *STRESS_VARIANTS,
        *ABSENT_VARIANTS,
    )
    screen_standard, screen_distractors, screen_selection = _fixed_target_correct_batch(
        target,
        batch_size=args.screen_pairs,
        seq_len=args.seq_len,
        token_vocabulary=token_vocabulary,
        device=device,
        seed=args.seed + 8_000_003,
        min_target_gap=args.min_target_gap,
        use_bf16=args.bf16,
        variant_name="standard",
    )
    screen_variants = make_functional_variants(
        screen_standard, token_vocabulary, args.seed + 9_000_007
    )
    confirm_standard, confirm_distractors, confirm_selection = _fixed_target_correct_batch(
        target,
        batch_size=args.confirm_pairs,
        seq_len=args.seq_len,
        token_vocabulary=token_vocabulary,
        device=device,
        seed=args.seed + 18_000_013,
        min_target_gap=args.min_target_gap,
        use_bf16=args.bf16,
        variant_name="standard",
    )
    confirm_variants = make_functional_variants(
        confirm_standard, token_vocabulary, args.seed + 19_000_019
    )
    unfiltered_batch = make_matched_induction_batch(
        args.unfiltered_pairs,
        args.seq_len,
        token_vocabulary,
        device,
        args.seed + 28_000_027,
    )
    unfiltered_distractors, unfiltered_fallbacks = _pair_distractors(
        unfiltered_batch, token_vocabulary
    )

    editor = TokenCarvingEditor(target, pieces, module_paths)

    def joint_for_variants(variants, pair_count):
        summaries = {}
        tensors = {}
        for name in eval_names:
            batch = variants[name]
            distractors, fallback_count = _pair_distractors(batch, token_vocabulary)
            score = GateScore(target, editor, batch, distractors, False)
            start = torch.zeros(pair_count, pieces.candidates, device=device)
            end = torch.ones_like(start)
            result = integrate_gate_path(score, start, end, args.integration_steps)
            editor.masks = None
            tensors[name] = result.contributions.detach()
            summaries[name] = {
                "full_gap_mean": float(result.end_scores.mean()),
                "all_piece_residual_gap_mean": float(result.start_scores.mean()),
                "endpoint_difference_mean": float((result.end_scores - result.start_scores).mean()),
                "integrated_sum_mean": float(result.contributions.sum(-1).mean()),
                "completeness_mae": float(result.completeness_error.abs().mean()),
                "distractor_fallback_count": fallback_count,
                "per_candidate_mean": result.contributions.mean(0).tolist(),
                "per_candidate_abs_mean": result.contributions.abs().mean(0).tolist(),
            }
        return summaries, tensors

    def shortlist_joint(score, candidate_ids, pair_count):
        start = torch.ones(pair_count, pieces.candidates, device=device)
        start[:, candidate_ids] = 0
        end = torch.ones_like(start)
        result = integrate_gate_path(score, start, end, args.integration_steps)
        editor.masks = None
        return {
            "full_gap_mean": float(result.end_scores.mean()),
            "shortlist_removed_gap_mean": float(result.start_scores.mean()),
            "endpoint_difference_mean": float((result.end_scores - result.start_scores).mean()),
            "integrated_sum_mean": float(result.contributions.sum(-1).mean()),
            "completeness_mae": float(result.completeness_error.abs().mean()),
            "per_candidate_mean": result.contributions.mean(0).tolist(),
        }

    try:
        phase_start = time.perf_counter()
        screen_joint, screen_joint_tensors = joint_for_variants(screen_variants, args.screen_pairs)
        screen_score = GateScore(
            target,
            editor,
            screen_standard,
            screen_distractors,
            False,
        )
        screen_integrated = _independent_integrals(
            screen_score,
            pieces,
            loo_screen_ids,
            args.screen_pairs,
            args.integration_steps,
            args.intervention_chunk,
        )
        screen_integrated.sort(key=lambda row: abs(row["endpoint_mean"]), reverse=True)
        impact_rows = screen_integrated[: args.impact_top]
        impact_ids = torch.tensor(
            [row["candidate_id"] for row in impact_rows],
            device=device,
            dtype=torch.long,
        )
        screen_shortlist_joint = shortlist_joint(screen_score, impact_ids, args.screen_pairs)
        screen_supporting_ids = torch.tensor(
            [row["candidate_id"] for row in impact_rows if row["endpoint_mean"] > 0],
            device=device,
            dtype=torch.long,
        )
        screen_suppressive_ids = torch.tensor(
            [row["candidate_id"] for row in impact_rows if row["endpoint_mean"] < 0],
            device=device,
            dtype=torch.long,
        )
        torch.cuda.synchronize()
        phase_seconds["screen"] = time.perf_counter() - phase_start
        print(
            json.dumps(
                {
                    "phase": "screened",
                    "impact_shortlist": impact_ids.numel(),
                    "supporting_on_screen": sum(row["endpoint_mean"] > 0 for row in impact_rows),
                    "suppressive_on_screen": sum(row["endpoint_mean"] < 0 for row in impact_rows),
                }
            ),
            flush=True,
        )

        phase_start = time.perf_counter()
        confirm_joint, confirm_joint_tensors = joint_for_variants(
            confirm_variants, args.confirm_pairs
        )
        confirm_score = GateScore(
            target,
            editor,
            confirm_standard,
            confirm_distractors,
            False,
        )
        confirm_shortlist_joint = shortlist_joint(confirm_score, impact_ids, args.confirm_pairs)
        confirm_screen_supporting_joint = shortlist_joint(
            confirm_score, screen_supporting_ids, args.confirm_pairs
        )
        confirm_screen_suppressive_joint = shortlist_joint(
            confirm_score, screen_suppressive_ids, args.confirm_pairs
        )
        confirm_integrated = _independent_integrals(
            confirm_score,
            pieces,
            impact_ids,
            args.confirm_pairs,
            args.integration_steps,
            args.intervention_chunk,
        )
        confirm_endpoint = _endpoint_diagnostics(
            confirm_score,
            pieces,
            impact_ids,
            args.confirm_pairs,
            args.intervention_chunk,
            args.seed + 30_000_031,
        )
        coalitions = _coalition_marginals(
            confirm_score,
            pieces,
            impact_ids,
            args.confirm_pairs,
            args.coalition_samples,
            args.coalition_chunk,
            args.seed + 31_000_033,
        )

        variant_endpoint = {}
        for variant_index, name in enumerate(eval_names):
            batch = confirm_variants[name]
            distractors, fallback_count = _pair_distractors(batch, token_vocabulary)
            score = GateScore(target, editor, batch, distractors, False)
            variant_endpoint[name] = {
                "distractor_fallback_count": fallback_count,
                "candidates": _endpoint_diagnostics(
                    score,
                    pieces,
                    impact_ids,
                    args.confirm_pairs,
                    args.intervention_chunk,
                    args.seed + 32_000_039 + variant_index * 100_003,
                ),
            }

        unfiltered_score = GateScore(
            target,
            editor,
            unfiltered_batch,
            unfiltered_distractors,
            False,
        )
        unfiltered_population = _population_summary(unfiltered_score, pieces, args.min_target_gap)
        unfiltered_endpoint = _endpoint_diagnostics(
            unfiltered_score,
            pieces,
            impact_ids,
            args.unfiltered_pairs,
            args.intervention_chunk,
            args.seed + 40_000_043,
        )
        fingerprint_matrix = torch.stack(
            [
                torch.tensor(
                    [
                        variant_endpoint[name]["candidates"][candidate]["contrast_damage"]["mean"]
                        for name in eval_names
                    ],
                    device=device,
                )
                for candidate in impact_ids.tolist()
            ]
        )
        neighbors = _fingerprint_neighbors(impact_ids, fingerprint_matrix, pieces)
        torch.cuda.synchronize()
        phase_seconds["confirmation_and_controls"] = time.perf_counter() - phase_start
    finally:
        editor.restore()

    confirmed_supporting = []
    confirmed_suppressive = []
    inconclusive = []
    for candidate in impact_ids.tolist():
        low, high = confirm_endpoint[candidate]["contrast_damage"]["mean_ci95_unadjusted"]
        if low > 0:
            confirmed_supporting.append(candidate)
        elif high < 0:
            confirmed_suppressive.append(candidate)
        else:
            inconclusive.append(candidate)

    region_by_id = {row["candidate_id"]: row for row in position_regions}
    screen_standard_joint = screen_joint_tensors["standard"]
    confirm_standard_joint = confirm_joint_tensors["standard"]
    candidate_rows = []
    for item in pieces.metadata:
        candidate = item.candidate_id
        weight_norm = (
            target.get_submodule(item.module_path).weight.detach().float().norm().clamp_min(1e-20)
        )
        row = asdict(item)
        row.update(
            {
                "piece_frobenius_sq_over_module_weight_frobenius_sq": (
                    item.piece_frobenius_norm / float(weight_norm)
                )
                ** 2,
                "discovery_signed_credit_mean": float(discovery_credits[:, candidate].mean()),
                "discovery_abs_credit_mean": float(discovery_credits[:, candidate].abs().mean()),
                "discovery_gate_share_mean": float(shares[:, candidate].mean()),
                "discovery_squared_attribution_mean": float(
                    squared_attribution[:, candidate].mean()
                ),
                "screen_joint_ig_mean": float(screen_standard_joint[:, candidate].mean()),
                "screen_joint_ig_abs_mean": float(screen_standard_joint[:, candidate].abs().mean()),
                "confirmation_joint_ig_mean": float(confirm_standard_joint[:, candidate].mean()),
                "confirmation_joint_ig_abs_mean": float(
                    confirm_standard_joint[:, candidate].abs().mean()
                ),
                "largest_attribution_region": region_by_id[candidate]["largest_attribution_region"],
                "position_region_fraction": region_by_id[candidate]["region_fraction"],
                "position_region_enrichment": region_by_id[candidate]["region_enrichment"],
            }
        )
        candidate_rows.append(row)

    impact_layer_ig = torch.zeros(layers, device=device)
    attention_layer_ig = torch.zeros(layers, device=device)
    impact_layer_weight = torch.zeros(layers, device=device)
    supporting_layer_effect = torch.zeros(layers, device=device)
    suppressive_layer_effect = torch.zeros(layers, device=device)
    supporting_attention_layer_effect = torch.zeros(layers, device=device)
    for candidate in impact_ids.tolist():
        item = pieces.metadata[candidate]
        ig_mass = confirm_standard_joint[:, candidate].abs().mean()
        exact_effect = float(confirm_endpoint[candidate]["contrast_damage"]["mean"])
        impact_layer_ig[item.layer] += ig_mass
        impact_layer_weight[item.layer] += item.piece_frobenius_norm**2
        if exact_effect > 0:
            supporting_layer_effect[item.layer] += exact_effect
            if item.module_kind.startswith("attention"):
                supporting_attention_layer_effect[item.layer] += exact_effect
        elif exact_effect < 0:
            suppressive_layer_effect[item.layer] += -exact_effect
        if item.module_kind.startswith("attention"):
            attention_layer_ig[item.layer] += ig_mass
    impact_layer_ig /= impact_layer_ig.sum().clamp_min(1e-12)
    attention_layer_ig /= attention_layer_ig.sum().clamp_min(1e-12)
    impact_layer_weight /= impact_layer_weight.sum().clamp_min(1e-12)
    supporting_layer_effect /= supporting_layer_effect.sum().clamp_min(1e-12)
    suppressive_layer_effect /= suppressive_layer_effect.sum().clamp_min(1e-12)
    supporting_attention_layer_effect /= supporting_attention_layer_effect.sum().clamp_min(1e-12)
    attribution_layer_cosine = F.cosine_similarity(
        impact_layer_ig, head_layer_attribution, dim=0
    )

    matched_head_reference = None
    if not args.skip_head_ablation:
        phase_start = time.perf_counter()
        matched_head_reference, reference_mass = _direct_head_ablation_reference(
            target,
            confirm_standard,
            confirm_distractors,
            module_paths,
            head_contrast,
            int(target.config.num_attention_heads),
            args.head_ablation_chunk,
            args.seed + 50_000_057,
        )
        reference_layers = set(reference_mass.nonzero(as_tuple=False).flatten().tolist())
        top_count = min(5, layers)
        carved_top = set(torch.topk(impact_layer_ig, top_count).indices.tolist())
        reference_top = set(torch.topk(reference_mass, top_count).indices.tolist())
        matched_head_reference["layer_comparison"] = {
            "warning": (
                "Absolute joint-IG mass mixes supporting and suppressive pieces; "
                "use the sign-separated exact-LOO profiles for causal interpretation."
            ),
            "all_impact_joint_ig_layer_cosine": float(
                F.cosine_similarity(impact_layer_ig, reference_mass, dim=0)
            ),
            "attention_only_joint_ig_layer_cosine": float(
                F.cosine_similarity(attention_layer_ig, reference_mass, dim=0)
            ),
            "supporting_exact_loo_layer_cosine": float(
                F.cosine_similarity(supporting_layer_effect, reference_mass, dim=0)
            ),
            "suppressive_exact_loo_layer_cosine": float(
                F.cosine_similarity(suppressive_layer_effect, reference_mass, dim=0)
            ),
            "supporting_attention_exact_loo_layer_cosine": float(
                F.cosine_similarity(
                    supporting_attention_layer_effect, reference_mass, dim=0
                )
            ),
            "weight_sq_layer_cosine": float(
                F.cosine_similarity(impact_layer_weight, reference_mass, dim=0)
            ),
            "top5_all_impact_layer_overlap_count": len(carved_top & reference_top),
            "all_impact_fraction_on_reference_layers": float(
                impact_layer_ig[list(reference_layers)].sum()
            )
            if reference_layers
            else 0.0,
            "supporting_effect_fraction_on_reference_layers": float(
                supporting_layer_effect[list(reference_layers)].sum()
            )
            if reference_layers
            else 0.0,
        }
        torch.cuda.synchronize()
        phase_seconds["matched_head_ablation"] = time.perf_counter() - phase_start

    torch.cuda.synchronize()
    phase_seconds["total_before_serialization"] = time.perf_counter() - run_start
    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "objective": "dictionary-free-token-conditioned-weight-carving",
        "model_parameters": sum(parameter.numel() for parameter in target.parameters()),
        "resolved_model_revision": getattr(target.config, "_commit_hash", None),
        "software": {
            "torch": str(torch.__version__),
            "transformers": importlib.metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": torch.cuda.get_device_name(device),
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "module_paths": module_paths,
        "candidates": pieces.candidates,
        "candidate_scope": (
            "module-local rank-one atoms; the shortlist spans layers but individual "
            "candidate IDs are not cross-layer bundles"
        ),
        "physical_gate_semantics": "one-intact-zero-remove",
        "attribution_gate_semantics": ("sum-normalized-squared-ranking-share-only"),
        "selection_protocol": (
            "discovery attribution -> disjoint screen LOO -> frozen "
            "shortlist -> disjoint confirmation"
        ),
        "discovery_precision": "bf16" if args.bf16 else "fp32",
        "gated_evaluation_precision": "fp32",
        "score": ("clean-minus-corrupt fixed target-vs-demonstrated-distractor margin"),
    }
    results = {
        "config": config,
        "cost": {
            "phase_seconds": phase_seconds,
            "peak_cuda_memory_gib": peak_memory_gib,
            "optimizer_steps": 0,
        },
        "data_splits": {
            "discovery": discovery_selection,
            "screen": screen_selection,
            "confirmation": confirm_selection,
            "unfiltered": {
                "pairs": args.unfiltered_pairs,
                "distractor_fallback_count": unfiltered_fallbacks,
                **unfiltered_population,
            },
        },
        "extraction": {
            "total_piece_sum_norm": extraction.total_piece_sum_norm,
            "target_weight_norm": extraction.target_weight_norm,
            "piece_norm_l1_over_global_weight_l2": (
                extraction.total_piece_sum_norm / extraction.target_weight_norm
            ),
            "norm_ratio_warning": (
                "This L1-over-L2 ratio is not a fraction of model weight mass."
            ),
            "active_candidates": sum(item.active for item in pieces.metadata),
            "module_geometry": [asdict(row) for row in extraction.module_geometry],
        },
        "discovery_attribution_gating": {
            "effective_components_mean": float(effective_components(shares).mean()),
            "effective_components_median": float(effective_components(shares).median()),
            "candidate_ids_ranked_with_layer_coverage": (attribution_ids.tolist()),
            "candidate_ids_ranked_by_discovery_share": torch.argsort(discovery_importance, descending=True).tolist(),
            "loo_screen_candidate_ids": loo_screen_ids.tolist(),
            "head_layer_attribution": head_layer_attribution.tolist(),
        },
        "screen": {
            "all_piece_joint_integrated_attribution": screen_joint,
            "impact_shortlist_joint_integrated_attribution": (screen_shortlist_joint),
            "intact_context_integrated_loo_effects": screen_integrated,
            "frozen_screen_supporting_ids": screen_supporting_ids.tolist(),
            "frozen_screen_suppressive_ids": screen_suppressive_ids.tolist(),
            "frozen_impact_shortlist_ids": impact_ids.tolist(),
        },
        "confirmation": {
            "all_piece_joint_integrated_attribution": confirm_joint,
            "impact_shortlist_joint_integrated_attribution": (confirm_shortlist_joint),
            "screen_supporting_group_joint_integrated_attribution": (
                confirm_screen_supporting_joint
            ),
            "screen_suppressive_group_joint_integrated_attribution": (
                confirm_screen_suppressive_joint
            ),
            "intact_context_integrated_loo_effects": confirm_integrated,
            "exact_standard_loo_diagnostics": confirm_endpoint,
            "confirmed_supporting_ids_ci95_unadjusted": confirmed_supporting,
            "confirmed_suppressive_ids_ci95_unadjusted": confirmed_suppressive,
            "inconclusive_ids_ci95_unadjusted": inconclusive,
            "stochastic_coalition_marginals": coalitions,
            "variant_exact_loo_diagnostics": variant_endpoint,
            "unfiltered_exact_loo_diagnostics": unfiltered_endpoint,
        },
        "exact_loo_variant_fingerprint_neighbors": neighbors,
        "first_order_layer_overlap": {
            "warning": (
                "This first-order comparison is not a causal circuit test and absolute "
                "mass mixes supporting with suppressive pieces."
            ),
            "supporting_exact_loo_mass": supporting_layer_effect.tolist(),
            "suppressive_exact_loo_mass": suppressive_layer_effect.tolist(),
            "impact_candidate_ig_mass": impact_layer_ig.tolist(),
            "attention_only_candidate_ig_mass": (attention_layer_ig.tolist()),
            "discovery_head_attribution_mass": (head_layer_attribution.tolist()),
            "cosine": float(attribution_layer_cosine),
        },
        "matched_empirical_head_ablation_reference": matched_head_reference,
        "candidates": candidate_rows,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in pieces.state_dict().items()},
            "metadata": [asdict(item) for item in pieces.metadata],
            "module_paths": module_paths,
            "model_name": args.model_name,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": getattr(target.config, "_commit_hash", None),
        },
        out_dir / "pieces.pt",
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "out_dir": str(out_dir),
                "candidates": pieces.candidates,
                "active_candidates": results["extraction"]["active_candidates"],
                "effective_components": results["discovery_attribution_gating"][
                    "effective_components_mean"
                ],
                "full_gap": confirm_joint["standard"]["full_gap_mean"],
                "all_piece_residual_gap": confirm_joint["standard"]["all_piece_residual_gap_mean"],
                "shortlist_removed_gap": confirm_shortlist_joint["shortlist_removed_gap_mean"],
                "confirmed_supporting_ids": confirmed_supporting,
                "confirmed_suppressive_ids": confirmed_suppressive,
                "inconclusive_ids": inconclusive,
                "peak_cuda_memory_gib": peak_memory_gib,
                "phase_seconds": phase_seconds,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
