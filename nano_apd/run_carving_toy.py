"""Ground-truth recovery test for contrastive parameter carving.

The toy language model has two known rank-one context-to-token mechanisms in one weight
matrix. Positive/negative examples share their selected cue and next-token label; they
differ only in whether the relevant source token appeared earlier. The experiment asks
whether an overcomplete bank recovers the known pieces, not merely whether its sum edits
the output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.carving import (
    CarvingEditor,
    MatchedBatch,
    bank_for,
    bank_mass,
    build_banks,
    capture_selected_usage,
    completeness_loss,
    component_credits,
    component_usage_sketch,
    geometry_loss,
    kl_per_position,
    make_coordinate_samples,
    normalized_codes,
    participation_ratio,
    pearson_correlation,
    sample_subset_masks,
    selected_scores,
    sketch_reconstruction_loss,
    target_total_credit,
    usage_sketch,
)

OUT_ROOT = Path(__file__).parent / "out"


class ContextSumLM(nn.Module):
    """A frozen causal model whose only learned computation is one linear weight."""

    def __init__(self, vocab_size: int = 12, strength: float = 8.0, seed: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(vocab_size, vocab_size, bias=False)
        generator = torch.Generator().manual_seed(seed)
        input_output = torch.randn(vocab_size, vocab_size, generator=generator) * 0.015
        self.sources = (1, 2)
        self.labels = (8, 9)
        self.true_pieces = []
        for source, label in zip(self.sources, self.labels, strict=True):
            piece = torch.zeros_like(input_output)
            piece[source, label] = strength
            input_output += piece
            self.true_pieces.append(piece)
        self.proj.weight.data.copy_(input_output.t())
        self.proj.weight.requires_grad_(False)

    def forward(self, tokens: Tensor) -> Tensor:
        one_hot = F.one_hot(tokens, self.vocab_size).float()
        context = one_hot.cumsum(1)
        return self.proj(context)


@dataclass(frozen=True)
class ToyPairs:
    matched: MatchedBatch
    modes: Tensor


def make_toy_pairs(model: ContextSumLM, batch_size: int, seq_len: int, seed: int,
                   device: str) -> ToyPairs:
    generator = torch.Generator(device=device).manual_seed(seed)
    modes = torch.randint(0, 2, (batch_size,), generator=generator, device=device)
    tokens = torch.randint(4, 8, (batch_size, seq_len), generator=generator, device=device)
    position = torch.full((batch_size,), seq_len - 2, device=device, dtype=torch.long)
    rows = torch.arange(batch_size, device=device)
    source_pos = torch.randint(1, seq_len // 2, (batch_size,), generator=generator,
                               device=device)
    source_values = torch.tensor(model.sources, device=device)[modes]
    labels = torch.tensor(model.labels, device=device)[modes]
    tokens[rows, source_pos] = source_values
    tokens[rows, position] = 0
    tokens[rows, position + 1] = labels
    positive = tokens
    negative = tokens.clone()
    negative[rows, source_pos] = 3
    matched = MatchedBatch(positive, negative, position, labels)
    matched.validate()
    return ToyPairs(matched, modes)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].mean() if mask.any() else value.new_zeros(())


def _cosine_matrix(learned: Tensor, truth: Tensor) -> Tensor:
    learned = F.normalize(learned.flatten(1).float(), dim=-1)
    truth = F.normalize(truth.flatten(1).float(), dim=-1)
    return truth @ learned.t()


def run(seed: int, args) -> dict:
    torch.manual_seed(seed)
    target = ContextSumLM(seed=10_000 + seed).to(args.device)
    banks = build_banks(target, ["proj"], args.C, args.rank, args.bank_type).to(args.device)
    editor = CarvingEditor(target, banks, ["proj"])
    optimizer = torch.optim.AdamW(banks.parameters(), lr=args.lr, weight_decay=0)
    samples = make_coordinate_samples(
        target, ["proj"], target.vocab_size**2, seed + 100
    )
    fixed_eval = make_toy_pairs(target, 256, args.seq_len, seed + 900_000, args.device)

    for step in range(args.steps):
        toy = make_toy_pairs(
            target, args.batch_size, args.seq_len, seed * 100_000 + step, args.device
        )
        pairs = toy.matched
        positive_usage = capture_selected_usage(
            target, editor, pairs.positive, pairs.positions, pairs.labels, "target_logit"
        )
        negative_usage = capture_selected_usage(
            target, editor, pairs.negative, pairs.positions, pairs.labels, "target_logit"
        )
        positive_credit = component_credits(target, banks, ["proj"], positive_usage)
        negative_credit = component_credits(target, banks, ["proj"], negative_usage)
        contrast_credit = positive_credit - negative_credit
        target_credit = (
            target_total_credit(target, ["proj"], positive_usage)
            - target_total_credit(target, ["proj"], negative_usage)
        )
        sketch = (
            usage_sketch(target, ["proj"], positive_usage, samples)
            - usage_sketch(target, ["proj"], negative_usage, samples)
        )
        learned_sketch = (
            component_usage_sketch(target, banks, ["proj"], positive_usage, samples)
            - component_usage_sketch(target, banks, ["proj"], negative_usage, samples)
        )
        codes = normalized_codes(contrast_credit)
        loss_usage = (
            completeness_loss(contrast_credit, target_credit)
            + sketch_reconstruction_loss(learned_sketch, sketch)
        )
        loss_geometry = geometry_loss(contrast_credit, sketch)
        loss_code = ((participation_ratio(codes) - args.code_budget) / args.C).square().mean()

        retain = torch.randint(4, 8, (args.batch_size, args.seq_len), device=args.device)
        tokens = torch.cat([pairs.positive, pairs.negative, retain])
        with torch.no_grad():
            editor.masks = None
            reference = target(tokens)
        editor.masks = torch.zeros(tokens.shape[0], 1, args.C, device=args.device)
        edited = target(tokens)
        editor.masks = None
        batch = args.batch_size
        original_gap = positive_usage.scores - negative_usage.scores
        edited_gap = (
            selected_scores(edited[:batch], pairs.positions, pairs.labels, "target_logit")
            - selected_scores(
                edited[batch:2 * batch], pairs.positions, pairs.labels, "target_logit"
            )
        )
        loss_forget = F.relu(edited_gap).mean()
        preserve = kl_per_position(edited, reference)
        mask = torch.ones_like(preserve, dtype=torch.bool)
        mask[torch.arange(batch, device=args.device), pairs.positions] = False
        loss_retain = preserve[mask].mean()

        subset_masks, removed_mass = sample_subset_masks(codes, 0.5)
        editor.masks = torch.cat([subset_masks, subset_masks])
        subset = target(torch.cat([pairs.positive, pairs.negative]))
        editor.masks = None
        subset_gap = (
            selected_scores(subset[:batch], pairs.positions, pairs.labels, "target_logit")
            - selected_scores(subset[batch:], pairs.positions, pairs.labels, "target_logit")
        )
        desired = (original_gap * (1 - removed_mass)).detach()
        loss_subset = F.smooth_l1_loss(subset_gap, desired)
        loss_reg = bank_mass(target, banks, ["proj"]) / target.proj.weight.numel()
        loss = (
            loss_forget + loss_retain + args.usage_w * loss_usage
            + args.geometry_w * loss_geometry + args.subset_w * loss_subset
            + args.code_w * loss_code + args.reg_w * loss_reg
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    pairs, modes = fixed_eval.matched, fixed_eval.modes
    positive_usage = capture_selected_usage(
        target, editor, pairs.positive, pairs.positions, pairs.labels
    )
    negative_usage = capture_selected_usage(
        target, editor, pairs.negative, pairs.positions, pairs.labels
    )
    credit = component_credits(target, banks, ["proj"], positive_usage) - component_credits(
        target, banks, ["proj"], negative_usage
    )
    assignments = []
    for mode in range(2):
        assignments.append(credit[modes == mode].abs().mean(0).argmax().item())
    bank = bank_for(banks, "proj")
    A, B = bank.factors(target.proj.weight.detach().t())
    learned = torch.einsum("cir,cro->cio", A, B).detach()
    truth = torch.stack(target.true_pieces).to(args.device)
    cosine = _cosine_matrix(learned, truth)

    pair_tokens = torch.cat([pairs.positive, pairs.negative])
    with torch.no_grad():
        editor.masks = None
        reference = target(pair_tokens)
        editor.masks = torch.zeros(pair_tokens.shape[0], 1, args.C, device=args.device)
        edited = target(pair_tokens)
        editor.masks = None
    batch = pairs.positive.shape[0]
    target_gap = (
        selected_scores(reference[:batch], pairs.positions, pairs.labels, "target_logit")
        - selected_scores(reference[batch:], pairs.positions, pairs.labels, "target_logit")
    )
    edit_gap = (
        selected_scores(edited[:batch], pairs.positions, pairs.labels, "target_logit")
        - selected_scores(edited[batch:], pairs.positions, pairs.labels, "target_logit")
    )
    single_by_mode = torch.zeros(2, args.C, device=args.device)
    for component in range(args.C):
        masks = torch.ones(2 * batch, 1, args.C, device=args.device)
        masks[..., component] = 0
        with torch.no_grad():
            editor.masks = masks
            single_logits = target(pair_tokens)
            editor.masks = None
        single_gap = (
            selected_scores(
                single_logits[:batch], pairs.positions, pairs.labels, "target_logit"
            )
            - selected_scores(
                single_logits[batch:], pairs.positions, pairs.labels, "target_logit"
            )
        )
        for mode in range(2):
            single_by_mode[mode, component] = (
                target_gap[modes == mode] - single_gap[modes == mode]
            ).mean()

    generator = torch.Generator(device=args.device).manual_seed(seed + 990_000)
    actual_subset, predicted_subset = [], []
    for _ in range(16):
        remove = torch.rand(args.C, generator=generator, device=args.device) < 0.5
        if not remove.any():
            remove[0] = True
        masks = (~remove).float().view(1, 1, -1).expand(2 * batch, 1, -1)
        with torch.no_grad():
            editor.masks = masks
            subset_logits = target(pair_tokens)
            editor.masks = None
        subset_gap = (
            selected_scores(
                subset_logits[:batch], pairs.positions, pairs.labels, "target_logit"
            )
            - selected_scores(
                subset_logits[batch:], pairs.positions, pairs.labels, "target_logit"
            )
        )
        actual_subset.append((target_gap - subset_gap).mean())
        predicted_subset.append(single_by_mode[:, remove].sum(-1).mean())
    actual_subset_t = torch.stack(actual_subset)
    predicted_subset_t = torch.stack(predicted_subset)

    result = {
        "seed": seed,
        "bank_type": args.bank_type,
        "C": args.C,
        "rank": args.rank,
        "target_gap": target_gap.mean().item(),
        "edited_gap": edit_gap.mean().item(),
        "gap_removed": (target_gap - edit_gap).mean().item(),
        "mode_assignments": assignments,
        "mode_separation": float(assignments[0] != assignments[1]),
        "single_effect_by_mode": single_by_mode.tolist(),
        "subset_pearson": pearson_correlation(
            actual_subset_t, predicted_subset_t
        ).item(),
        "subset_mae": (actual_subset_t - predicted_subset_t).abs().mean().item(),
        "truth_best_cosine": cosine.max(-1).values.tolist(),
        "truth_mean_best_cosine": cosine.max(-1).values.mean().item(),
        "cosine_matrix": cosine.tolist(),
    }
    editor.restore()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank_type", choices=["free", "projected"], default="free")
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--usage_w", type=float, default=0.5)
    parser.add_argument("--geometry_w", type=float, default=0.5)
    parser.add_argument("--subset_w", type=float, default=0.5)
    parser.add_argument("--code_w", type=float, default=0.01)
    parser.add_argument("--code_budget", type=float, default=1.0)
    parser.add_argument("--reg_w", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="ground_truth")
    parser.add_argument("--skip_rank_matched", action="store_true")
    args = parser.parse_args()
    results, baselines = [], []
    for seed in range(args.seeds):
        result = run(seed, args)
        results.append(result)
        print(json.dumps({"decomposition": result}), flush=True)
        if not args.skip_rank_matched and args.C > 1:
            baseline_values = vars(args).copy()
            baseline_values["rank"] = args.C * args.rank
            baseline_values["C"] = 1
            baseline_values["skip_rank_matched"] = True
            baseline = run(seed, argparse.Namespace(**baseline_values))
            baselines.append(baseline)
            print(json.dumps({"rank_matched_baseline": baseline}), flush=True)
    aggregate = {
        "config": vars(args),
        "runs": results,
        "rank_matched_baselines": baselines,
        "mean_gap_removed": sum(x["gap_removed"] for x in results) / len(results),
        "mean_mode_separation": sum(x["mode_separation"] for x in results) / len(results),
        "mean_truth_cosine": sum(x["truth_mean_best_cosine"] for x in results) / len(results),
        "baseline_mean_gap_removed": (
            sum(x["gap_removed"] for x in baselines) / len(baselines) if baselines else None
        ),
    }
    out_dir = OUT_ROOT / f"carving_toy_{args.tag}_{args.bank_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
