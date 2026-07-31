"""Controlled functional interventions on matched random-token induction pairs."""

from __future__ import annotations

import torch
from torch import Tensor

from nano_apd.carving import MatchedBatch

PRESENT_VARIANTS = (
    "standard",
    "short_lag",
    "long_lag",
    "two_demonstrations",
    "counterfactual_label",
)
STRESS_VARIANTS = ("recent_conflict",)
ABSENT_VARIANTS = ("source_cue_corrupt", "destination_cue_corrupt")


def source_positions(batch: MatchedBatch) -> Tensor:
    """Find the contrasted earlier cue->label demonstration in each pair."""
    result = []
    for positive, negative, position in zip(
        batch.positive, batch.negative, batch.positions, strict=True
    ):
        pos = int(position)
        cue = positive[pos]
        cue_match = positive[:pos] == cue
        next_token_differs = positive[1:pos + 1] != negative[1:pos + 1]
        matches = (cue_match & next_token_differs).nonzero(
            as_tuple=False
        ).flatten()
        if not matches.numel():
            raise ValueError("matched induction pair has no contrasted earlier cue")
        result.append(matches[-1])
    return torch.stack(result).to(batch.positions.device)


def _different(values: Tensor, vocab_size: int, generator: torch.Generator) -> Tensor:
    offset = torch.randint(
        1, vocab_size, values.shape, device=values.device, generator=generator
    )
    return (values + offset) % vocab_size


def _batch(
    positive: Tensor,
    negative: Tensor,
    positions: Tensor,
    labels: Tensor,
) -> MatchedBatch:
    result = MatchedBatch(positive, negative, positions, labels)
    result.validate()
    return result


def _corrupt_every_source_cue(
    positive: Tensor,
    negative: Tensor,
    positions: Tensor,
    vocab_size: int,
    generator: torch.Generator,
) -> None:
    """Break every earlier match, including the two-demonstration variant."""
    for row, position in enumerate(positions.tolist()):
        cue = positive[row, position]
        matches = (positive[row, :position] == cue).nonzero(
            as_tuple=False
        ).flatten()
        if not matches.numel():
            continue
        replacements = _different(
            positive[row, matches], vocab_size, generator
        )
        positive[row, matches] = replacements
        negative[row, matches] = replacements


def make_functional_variants(
    standard: MatchedBatch,
    vocab_size: int,
    seed: int,
) -> dict[str, MatchedBatch]:
    """Create paired interventions while holding the selected output format fixed.

    Present-mechanism variants retain a valid cue->label demonstration but change lag,
    repetition, or label identity.  Absent-mechanism controls corrupt the source or
    destination cue in both halves, so changing the earlier label should cease to matter.
    The recent-conflict variant inserts a nearer cue->distractor demonstration.
    """
    standard.validate()
    device = standard.positive.device
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = torch.arange(standard.positive.shape[0], device=device)
    source = source_positions(standard)
    destination = standard.positions
    cue = standard.positive[rows, destination]
    label = standard.labels
    distractor = standard.negative[rows, source + 1]
    replacement_cue = _different(cue, vocab_size, generator)

    variants: dict[str, MatchedBatch] = {"standard": standard}

    # Move the only valid demonstration close to or far from the destination.
    for name, new_source in (
        ("short_lag", destination - 4),
        ("long_lag", torch.ones_like(destination)),
    ):
        positive, negative = standard.positive.clone(), standard.negative.clone()
        positive[rows, source] = replacement_cue
        negative[rows, source] = replacement_cue
        negative[rows, source + 1] = positive[rows, source + 1]
        positive[rows, new_source] = cue
        negative[rows, new_source] = cue
        positive[rows, new_source + 1] = label
        negative[rows, new_source + 1] = distractor
        variants[name] = _batch(
            positive, negative, destination.clone(), label.clone()
        )

    # Add a second consistent or conflicting demonstration near the query.
    repeat_source = destination - 6
    positive, negative = standard.positive.clone(), standard.negative.clone()
    positive[rows, repeat_source] = cue
    negative[rows, repeat_source] = cue
    positive[rows, repeat_source + 1] = label
    negative[rows, repeat_source + 1] = distractor
    variants["two_demonstrations"] = _batch(
        positive, negative, destination.clone(), label.clone()
    )

    conflict_source = destination - 4
    positive, negative = standard.positive.clone(), standard.negative.clone()
    positive[rows, conflict_source] = cue
    negative[rows, conflict_source] = cue
    positive[rows, conflict_source + 1] = distractor
    negative[rows, conflict_source + 1] = distractor
    variants["recent_conflict"] = _batch(
        positive, negative, destination.clone(), label.clone()
    )

    # Break either end of the induction match in both pair halves.
    positive, negative = standard.positive.clone(), standard.negative.clone()
    _corrupt_every_source_cue(
        positive, negative, destination, vocab_size, generator
    )
    variants["source_cue_corrupt"] = _batch(
        positive, negative, destination.clone(), label.clone()
    )

    positive, negative = standard.positive.clone(), standard.negative.clone()
    positive[rows, destination] = replacement_cue
    negative[rows, destination] = replacement_cue
    variants["destination_cue_corrupt"] = _batch(
        positive, negative, destination.clone(), label.clone()
    )

    # Change the demonstrated and requested label together: token-identity control.
    counter_label = _different(label, vocab_size, generator)
    counter_distractor = _different(counter_label, vocab_size, generator)
    positive, negative = standard.positive.clone(), standard.negative.clone()
    positive[rows, source + 1] = counter_label
    negative[rows, source + 1] = counter_distractor
    positive[rows, destination + 1] = counter_label
    negative[rows, destination + 1] = counter_label
    variants["counterfactual_label"] = _batch(
        positive, negative, destination.clone(), counter_label
    )
    return variants


def select_rows(batch: MatchedBatch, rows: Tensor) -> MatchedBatch:
    return MatchedBatch(
        batch.positive.index_select(0, rows),
        batch.negative.index_select(0, rows),
        batch.positions.index_select(0, rows),
        batch.labels.index_select(0, rows),
    )
