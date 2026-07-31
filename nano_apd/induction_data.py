"""Task-conditioned data selection for induction component training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from nano_apd.carving import MatchedBatch, make_matched_induction_batch, selected_logits
from nano_apd.induction_variants import PRESENT_VARIANTS, make_functional_variants


def _score(logits: Tensor, labels: Tensor, kind: str) -> Tensor:
    correct = logits.float().gather(-1, labels[:, None]).squeeze(-1)
    if kind == "target_logit":
        return correct
    if kind == "logprob":
        return F.log_softmax(logits.float(), -1).gather(
            -1, labels[:, None]
        ).squeeze(-1)
    competitors = logits.float().scatter(-1, labels[:, None], float("-inf"))
    return correct - competitors.amax(-1)


def target_correct_induction_batch(
    target,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
    score_type: str,
    min_target_gap: float,
    include_target_incorrect: bool,
    use_bf16: bool,
    variant_name: str = "standard",
    oversample: int = 4,
    max_rounds: int = 8,
) -> MatchedBatch:
    """Rejection-sample pairs on which the frozen target exhibits induction.

    Random-token induction is only about 30% top-1 correct for Pythia-410M.  Letting
    each tiny DDP shard contain zero or two useful rows makes dataset-wide component
    coverage almost impossible.  This function spends cheap no-grad forwards to
    return a full local batch of target-correct, positive-gap examples.  Held-out
    evaluation still reports the eligible fraction on unfiltered pairs.
    """
    if variant_name not in PRESENT_VARIANTS:
        raise ValueError(f"training variant {variant_name!r} is not induction-present")
    selected: list[MatchedBatch] = []
    count = 0
    for round_index in range(max_rounds):
        round_seed = seed + round_index * 1_000_003
        candidates = make_matched_induction_batch(
            batch_size * oversample,
            seq_len,
            vocab_size,
            device,
            round_seed,
        )
        if variant_name != "standard":
            candidates = make_functional_variants(
                candidates, vocab_size, round_seed + 500_009
            )[variant_name]
        tokens = torch.cat([candidates.positive, candidates.negative])
        positions = candidates.positions.repeat(2)
        with torch.no_grad(), torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            logits = target(tokens)
            chosen = selected_logits(logits, positions)
        positive, negative = chosen.split(candidates.positive.shape[0])
        gap = (
            _score(positive, candidates.labels, score_type)
            - _score(negative, candidates.labels, score_type)
        )
        keep = gap > min_target_gap
        if not include_target_incorrect:
            keep &= positive.argmax(-1) == candidates.labels
        index = keep.nonzero(as_tuple=False).flatten()
        if index.numel():
            need = batch_size - count
            index = index[:need]
            selected.append(MatchedBatch(
                candidates.positive.index_select(0, index),
                candidates.negative.index_select(0, index),
                candidates.positions.index_select(0, index),
                candidates.labels.index_select(0, index),
            ))
            count += index.numel()
        if count >= batch_size:
            break
    if count < batch_size:
        raise RuntimeError(
            f"found only {count}/{batch_size} eligible induction pairs after "
            f"{max_rounds} oversampled rounds"
        )
    batch = MatchedBatch(
        torch.cat([item.positive for item in selected]),
        torch.cat([item.negative for item in selected]),
        torch.cat([item.positions for item in selected]),
        torch.cat([item.labels for item in selected]),
    )
    batch.validate()
    return batch
