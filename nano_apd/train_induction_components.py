"""Train sparse, cross-layer components for one induction output family.

This is deliberately task-conditioned.  It does not attempt to dictionary-learn the
whole language model.  A small component bank is carved out of every transformer
matrix while an implicit residual absorbs everything else.  Attribution gates are
computed coherently across all layers and sum-normalized per matched induction pair.

The stochastic-ablation objective mixes attribution-routed random coalitions on owner
examples with intact-model leave-one-out on random eligible examples.  The first trains
conditional marginal value; the second directly trains task-wide necessity.  Held-out
intact and routed ablations are reported separately because they answer different
questions.

Example (two H100s):

    torchrun --standalone --nproc_per_node=2 \
      -m nano_apd.train_induction_components --C 16 --bf16 --tag c16
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from nano_apd.carving import (
    MatchedBatch,
    build_banks,
    capture_selected_usage,
    component_credits,
    make_matched_induction_batch,
    selected_logits,
    target_total_credit,
)
from nano_apd.induction_components import (
    coverage_loss,
    effective_components,
    rotating_components,
    selected_distribution_kl,
    stochastic_coalitions,
    sum_norm_loss,
    sum_normalized_gates,
)
from nano_apd.induction_data import target_correct_induction_batch
from nano_apd.induction_editor import InductionEditor
from nano_apd.induction_variants import (
    ABSENT_VARIANTS,
    PRESENT_VARIANTS,
    make_functional_variants,
)
from nano_apd.lm_target import (
    DEFAULT_HF_MODEL,
    candidate_linear_paths,
    load_carving_target,
    vocab_size,
)
from nano_param_decomp.pile_4L import make_loader

OUT_ROOT = Path(__file__).parent / "out"


def _init_distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        return dist.get_rank(), world, device
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    return 0, 1, torch.device("cuda")


def _average_gradients(parameters, world: int) -> None:
    if world == 1:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def _broadcast_parameters(parameters, world: int) -> None:
    if world > 1:
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)


def _score_selected(logits: Tensor, labels: Tensor, kind: str) -> Tensor:
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


def _selected_forward(target, tokens: Tensor, positions: Tensor) -> Tensor:
    logits = target(tokens)
    return selected_logits(logits, positions)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return values[mask].mean() if mask.any() else values.sum() * 0


class PairSource:
    def __init__(self, args, device: torch.device, local_batch: int, rank: int, vocab: int):
        self.args = args
        self.device = device
        self.local_batch = local_batch
        self.rank = rank
        self.vocab = vocab

    def batch(self, step: int) -> MatchedBatch:
        return make_matched_induction_batch(
            self.local_batch,
            self.args.seq_len,
            self.vocab,
            self.device,
            self.args.seed * 1_000_003 + step * 10_007 + self.rank,
        )


class RetainSource:
    def __init__(self, args, device: torch.device, rank: int, world: int):
        self.device = device
        self.local_batch = args.batch_retain // world if args.batch_retain else 0
        self.loader = (
            make_loader(
                args.batch_retain, args.seq_len, rank, world, "train",
                args.seed + 4_000_000,
            )
            if self.local_batch else None
        )

    def batch(self) -> Tensor | None:
        if self.loader is None:
            return None
        return next(self.loader).to(self.device)

    def close(self) -> None:
        close = getattr(self.loader, "close", None)
        if close is not None:
            close()


def _load_target(args, rank: int, world: int, device: torch.device):
    target = None
    if world > 1:
        if rank == 0:
            target = load_carving_target("hf", args.model_name)
        dist.barrier()
        if rank != 0:
            target = load_carving_target("hf", args.model_name)
        dist.barrier()
    else:
        target = load_carving_target("hf", args.model_name)
    assert target is not None
    target = target.float().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_HF_MODEL)
    parser.add_argument("--C", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument("--rank_m", type=int, default=1, dest="rank")
    parser.add_argument("--bank_type", choices=["free", "projected"], default="free")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--batch_pairs", type=int, default=32,
                        help="global matched-pair batch, split across ranks")
    parser.add_argument("--retain_components", type=int, default=8,
                        help="rotating components individually checked on natural text")
    parser.add_argument("--batch_retain", type=int, default=8,
                        help="global natural-text retain batch, split across ranks; 0 disables")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument(
        "--train_variants",
        default=",".join(PRESENT_VARIANTS),
        help="comma-separated induction-present variants cycled across steps",
    )
    parser.add_argument("--score_type", choices=["logit_margin", "target_logit", "logprob"],
                        default="logit_margin")
    parser.add_argument("--min_target_gap", type=float, default=0.05)
    parser.add_argument("--include_target_incorrect", action="store_true")
    parser.add_argument("--gap_floor", type=float, default=0.0)
    parser.add_argument("--coalition_components", type=int, default=8,
                        help="rotating components receiving random-coalition marginals per step")
    parser.add_argument("--necessity_fraction", type=float, default=0.12,
                        help="required marginal as this fraction * ownership * target gap")
    parser.add_argument("--full_model_loo_probability", type=float, default=0.5,
                        help="fraction of coalitions that are intact-model leave-one-out")
    parser.add_argument(
        "--full_model_necessity_fraction", type=float, default=0.02,
        help="intact-model ablation damage floor as a fraction of target gap",
    )
    parser.add_argument("--control_components", type=int, default=8,
                        help="rotating components individually ablated on controls")
    parser.add_argument("--l0_ramp_start", type=float, default=0.4,
                        help="training fraction before per-example sparsity ramps in")
    parser.add_argument("--necessity_ramp_start", type=float, default=0.2,
                        help="training fraction before coalition necessity ramps in")
    parser.add_argument("--l0_target", type=float, default=1.5)
    parser.add_argument("--minimum_owner_share", type=float, default=0.12)
    parser.add_argument("--carve_ramp_frac", type=float, default=0.2,
                        help="fraction of training used to ramp residual carving from zero")
    parser.add_argument("--carve_w", type=float, default=1.0)
    parser.add_argument("--recon_w", type=float, default=1.0)
    parser.add_argument("--gap_recon_w", type=float, default=0.2)
    parser.add_argument("--specificity_w", type=float, default=0.5)
    parser.add_argument("--retain_w", type=float, default=0.25)
    parser.add_argument("--retain_individual_w", type=float, default=1.0,
                        help="per-component natural-text preservation penalty")
    parser.add_argument("--completeness_w", type=float, default=0.05)
    parser.add_argument("--necessity_w", type=float, default=0.5)
    parser.add_argument("--l0_w", type=float, default=0.02)
    parser.add_argument("--balance_w", type=float, default=0.0,
                        help="dataset-level marginal-use balance; pair with L0 sparsity")
    parser.add_argument("--coverage_w", type=float, default=0.1)
    parser.add_argument("--sum_norm_w", type=float, default=0.01)
    parser.add_argument("--control_collective_w", type=float, default=1.0)
    parser.add_argument("--control_individual_w", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="induction_attr")
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--ckpt_every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    training_variants = tuple(
        name.strip() for name in args.train_variants.split(",") if name.strip()
    )
    unknown_variants = sorted(set(training_variants) - set(PRESENT_VARIANTS))
    if not training_variants or unknown_variants:
        raise ValueError(
            f"train_variants must be drawn from {PRESENT_VARIANTS}; "
            f"unknown={unknown_variants}"
        )

    rank, world, device = _init_distributed()
    if args.batch_pairs < world or args.batch_pairs % world:
        raise ValueError("batch_pairs must be positive and divisible by world size")
    if args.batch_retain and (
        args.batch_retain < world or args.batch_retain % world
    ):
        raise ValueError("batch_retain must be zero or divisible by world size")
    if args.steps < 1 or args.rank < 1:
        raise ValueError("steps and rank_m must be positive")
    if not 1 <= args.coalition_components <= args.C:
        raise ValueError("coalition_components must be in [1, C]")
    if not 1 <= args.retain_components <= args.C:
        raise ValueError("retain_components must be in [1, C]")
    if not 1 <= args.control_components <= args.C:
        raise ValueError("control_components must be in [1, C]")
    if not 0 <= args.full_model_loo_probability <= 1:
        raise ValueError("full_model_loo_probability must lie in [0, 1]")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    target = _load_target(args, rank, world, device)
    module_paths = candidate_linear_paths(target, "hf")
    expected_layers = int(target.config.num_hidden_layers)
    seen_layers = {int(path.split(".layers.")[1].split(".")[0]) for path in module_paths}
    if seen_layers != set(range(expected_layers)):
        raise RuntimeError("cross-layer bank does not cover every transformer layer")

    local_pairs = args.batch_pairs // world
    pair_source = PairSource(args, device, local_pairs, rank, vocab_size(target))
    retain_source = RetainSource(args, device, rank, world)

    # Cache timing can consume RNG differently across ranks; reset before the bank.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    banks = build_banks(target, module_paths, args.C, args.rank, args.bank_type).to(device)
    editor = InductionEditor(target, banks, module_paths)
    optimizer = torch.optim.AdamW(banks.parameters(), lr=args.lr, weight_decay=0)
    _broadcast_parameters(banks.parameters(), world)

    slug = args.model_name.rsplit("/", 1)[-1].lower()
    out_dir = OUT_ROOT / f"{slug}_induction_components_{args.tag}_C{args.C}"
    checkpoint_path = out_dir / "ckpt.pt"
    start_step = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        if checkpoint["module_paths"] != module_paths:
            raise ValueError("checkpoint module list differs from the current target")
        banks.load_state_dict(checkpoint["banks"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        if rank == 0:
            print(f"resumed {checkpoint_path} at step {start_step}", flush=True)

    config = vars(args) | {
        "target": "hf",
        "module_paths": module_paths,
        "n_modules": len(module_paths),
        "n_layers": expected_layers,
        "world_size": world,
        "local_batch_pairs": local_pairs,
        "objective": "task-conditioned-cross-layer-attribution-components",
        "gate_normalization": "sum",
        "residual": "implicit-exact",
        "stochastic_ablation": "routed-random-plus-intact-leave-one-out",
        "training_variant_names": training_variants,
        "control_variant_names": ABSENT_VARIANTS,
        "sum_norm_scope": "whole-cross-layer-component",
    }
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(json.dumps({
            "out_dir": str(out_dir),
            "components": args.C,
            "rank": args.rank,
            "modules": len(module_paths),
            "layers": expected_layers,
            "bank_parameters": sum(p.numel() for p in banks.parameters()),
        }), flush=True)
    if world > 1:
        dist.barrier()

    coalition_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_001 + rank
    )
    for step in range(start_step, args.steps + 1):
        phase = min(step, args.steps) / args.steps
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * phase))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        variant_name = training_variants[step % len(training_variants)]
        control_name = ABSENT_VARIANTS[
            (step // len(training_variants)) % len(ABSENT_VARIANTS)
        ]
        pairs = target_correct_induction_batch(
            target, batch_size=local_pairs, seq_len=args.seq_len,
            vocab_size=pair_source.vocab, device=device,
            seed=args.seed * 1_000_003 + step * 10_007 + rank,
            score_type=args.score_type, min_target_gap=args.min_target_gap,
            include_target_incorrect=args.include_target_incorrect,
            use_bf16=args.bf16, variant_name=variant_name,
        )
        control_pairs = make_functional_variants(
            pairs, pair_source.vocab,
            args.seed * 2_000_003 + step * 20_011 + rank,
        )[control_name]
        retain = retain_source.batch()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            positive_usage = capture_selected_usage(
                target, editor, pairs.positive, pairs.positions, pairs.labels,
                args.score_type,
            )
            negative_usage = capture_selected_usage(
                target, editor, pairs.negative, pairs.positions, pairs.labels,
                args.score_type,
            )
            credits = (
                component_credits(target, banks, module_paths, positive_usage)
                - component_credits(target, banks, module_paths, negative_usage)
            )
            target_credit = (
                target_total_credit(target, module_paths, positive_usage)
                - target_total_credit(target, module_paths, negative_usage)
            )
            gates, attribution = sum_normalized_gates(credits)
            target_gap_capture = positive_usage.scores - negative_usage.scores
            eligible = target_gap_capture > args.min_target_gap
            if not args.include_target_incorrect:
                eligible &= positive_usage.predictions == pairs.labels

            pair_tokens = torch.cat([pairs.positive, pairs.negative])
            pair_positions = torch.cat([pairs.positions, pairs.positions])
            pair_gates = torch.cat([gates, gates]).unsqueeze(1)
            batch = pairs.positive.shape[0]

            editor.masks = torch.ones(
                2 * batch, 1, args.C, device=device, dtype=gates.dtype
            )
            with torch.no_grad():
                reference_selected = _selected_forward(
                    target, pair_tokens, pair_positions
                )

            editor.masks = torch.zeros_like(pair_gates)
            residual_selected = _selected_forward(target, pair_tokens, pair_positions)
            editor.masks = pair_gates
            routed_selected = _selected_forward(target, pair_tokens, pair_positions)
            editor.masks = None

            reference_pos, reference_neg = reference_selected.split(batch)
            residual_pos, residual_neg = residual_selected.split(batch)
            routed_pos, routed_neg = routed_selected.split(batch)
            reference_gap = (
                _score_selected(reference_pos, pairs.labels, args.score_type)
                - _score_selected(reference_neg, pairs.labels, args.score_type)
            )
            residual_gap = (
                _score_selected(residual_pos, pairs.labels, args.score_type)
                - _score_selected(residual_neg, pairs.labels, args.score_type)
            )
            routed_gap = (
                _score_selected(routed_pos, pairs.labels, args.score_type)
                - _score_selected(routed_neg, pairs.labels, args.score_type)
            )

            loss_carve = _masked_mean(
                F.relu(residual_gap - args.gap_floor), eligible
            )
            routed_kl = selected_distribution_kl(
                routed_selected, reference_selected
            )
            loss_recon = routed_kl.mean()
            gap_scale = reference_gap.detach().abs().mean().clamp_min(0.1)
            loss_gap_recon = _masked_mean(
                F.smooth_l1_loss(
                    routed_gap, reference_gap.detach(), reduction="none"
                ) / gap_scale,
                eligible,
            )
            loss_specificity = selected_distribution_kl(
                residual_neg, reference_neg
            ).mean()

            credit_scale = target_credit[eligible].square().mean().clamp_min(1e-6)
            loss_completeness = (
                (credits[eligible].sum(-1) - target_credit[eligible].detach())
                .square().mean() / credit_scale
                if eligible.any() else credits.sum() * 0
            )
            l0 = effective_components(attribution)
            loss_l0 = _masked_mean((l0 - args.l0_target).square(), eligible)
            mean_gate = gates[eligible].mean(0) if eligible.any() else gates.mean(0)
            loss_balance = args.C * mean_gate.square().sum() - 1.0
            loss_coverage = coverage_loss(
                gates, eligible, args.minimum_owner_share
            )
            loss_sum_norm = sum_norm_loss(target, banks, module_paths)

            component_ids = rotating_components(
                step, args.C, args.coalition_components
            )
            coalition = stochastic_coalitions(
                gates, eligible, component_ids, coalition_generator,
                args.full_model_loo_probability,
            )
            if coalition is None:
                loss_necessity = credits.sum() * 0
                coalition_marginal = credits.new_zeros(())
            else:
                owners = coalition.owner_rows
                pos = pairs.positive.index_select(0, owners)
                neg = pairs.negative.index_select(0, owners)
                positions = pairs.positions.index_select(0, owners)
                labels = pairs.labels.index_select(0, owners)
                k = owners.numel()
                coalition_tokens = torch.cat([pos, neg, pos, neg])
                coalition_positions = positions.repeat(4)
                coalition_masks = torch.cat([
                    coalition.gates_on,
                    coalition.gates_on,
                    coalition.gates_off,
                    coalition.gates_off,
                ]).unsqueeze(1)
                editor.masks = coalition_masks
                coalition_selected = _selected_forward(
                    target, coalition_tokens, coalition_positions
                )
                editor.masks = None
                on_pos, on_neg, off_pos, off_neg = coalition_selected.split(k)
                gap_on = (
                    _score_selected(on_pos, labels, args.score_type)
                    - _score_selected(on_neg, labels, args.score_type)
                )
                gap_off = (
                    _score_selected(off_pos, labels, args.score_type)
                    - _score_selected(off_neg, labels, args.score_type)
                )
                marginal = gap_on - gap_off
                target_scale = (
                    reference_gap.index_select(0, owners).detach().clamp_min(0.5)
                )
                routed_desired = (
                    args.necessity_fraction
                    * coalition.owner_shares.detach()
                    * target_scale
                )
                intact_desired = (
                    args.full_model_necessity_fraction * target_scale
                )
                desired = torch.where(
                    coalition.intact_leave_one_out,
                    intact_desired, routed_desired,
                )
                loss_necessity = F.relu(desired - marginal).mean()
                coalition_marginal = marginal.detach().mean()

            control_tokens = torch.cat([
                control_pairs.positive, control_pairs.negative
            ])
            control_positions = control_pairs.positions.repeat(2)
            editor.masks = None
            with torch.no_grad():
                control_reference = _selected_forward(
                    target, control_tokens, control_positions
                )
            editor.masks = torch.zeros(
                2 * batch, 1, args.C, device=device, dtype=gates.dtype
            )
            control_residual = _selected_forward(
                target, control_tokens, control_positions
            )
            loss_control_collective = selected_distribution_kl(
                control_residual, control_reference
            ).mean()

            control_ids = rotating_components(
                step, args.C, args.control_components
            )
            control_count = control_ids.numel()
            control_masks = torch.ones(
                control_count * batch, args.C, device=device, dtype=gates.dtype
            )
            for offset, component in enumerate(control_ids.tolist()):
                control_masks[
                    offset * batch:(offset + 1) * batch, component
                ] = 0
            editor.masks = torch.cat([
                control_masks, control_masks
            ]).unsqueeze(1)
            control_ablation_tokens = torch.cat([
                control_pairs.positive.repeat(control_count, 1),
                control_pairs.negative.repeat(control_count, 1),
            ])
            control_ablation_positions = control_pairs.positions.repeat(
                2 * control_count
            )
            control_ablated = _selected_forward(
                target, control_ablation_tokens, control_ablation_positions
            )
            control_reference_pos, control_reference_neg = (
                control_reference.split(batch)
            )
            control_reference_tiled = torch.cat([
                control_reference_pos.repeat(control_count, 1),
                control_reference_neg.repeat(control_count, 1),
            ])
            loss_control_individual = selected_distribution_kl(
                control_ablated, control_reference_tiled
            ).mean()
            editor.masks = None

            loss_retain = credits.sum() * 0
            loss_retain_individual = credits.sum() * 0
            if retain is not None:
                retain_positions = torch.full(
                    (retain.shape[0],), retain.shape[1] - 2,
                    device=device, dtype=torch.long,
                )
                keep = torch.ones(
                    retain.shape[0], 1, args.C, device=device, dtype=gates.dtype
                )
                editor.masks = keep
                with torch.no_grad():
                    retain_reference = _selected_forward(
                        target, retain, retain_positions
                    )
                editor.masks = torch.zeros_like(keep)
                retain_residual = _selected_forward(target, retain, retain_positions)
                editor.masks = None
                loss_retain = selected_distribution_kl(
                    retain_residual, retain_reference
                ).mean()
                retain_ids = rotating_components(
                    step, args.C, args.retain_components
                )
                retain_count = retain_ids.numel()
                retain_masks = torch.ones(
                    retain_count * retain.shape[0], args.C, device=device,
                    dtype=gates.dtype,
                )
                for offset, component in enumerate(retain_ids.tolist()):
                    retain_masks[
                        offset * retain.shape[0]:(offset + 1) * retain.shape[0],
                        component,
                    ] = 0
                editor.masks = retain_masks.unsqueeze(1)
                retain_ablated = _selected_forward(
                    target, retain.repeat(retain_count, 1),
                    retain_positions.repeat(retain_count),
                )
                editor.masks = None
                loss_retain_individual = selected_distribution_kl(
                    retain_ablated, retain_reference.repeat(retain_count, 1)
                ).mean()

            train_fraction = step / args.steps
            l0_ramp = max(
                0.0, min(1.0, (train_fraction - args.l0_ramp_start)
                             / max(1e-6, 1.0 - args.l0_ramp_start))
            )
            necessity_ramp = max(
                0.0, min(1.0, (train_fraction - args.necessity_ramp_start)
                             / max(1e-6, 1.0 - args.necessity_ramp_start))
            )
            carve_ramp = min(
                1.0, step / max(1.0, args.carve_ramp_frac * args.steps)
            )
            loss = (
                args.carve_w * carve_ramp * loss_carve
                + args.recon_w * loss_recon
                + args.gap_recon_w * loss_gap_recon
                + args.specificity_w * loss_specificity
                + args.retain_w * loss_retain
                + args.retain_individual_w * loss_retain_individual
                + args.completeness_w * loss_completeness
                + args.necessity_w * necessity_ramp * loss_necessity
                + args.l0_w * l0_ramp * loss_l0
                + args.balance_w * loss_balance
                + args.coverage_w * loss_coverage
                + args.sum_norm_w * loss_sum_norm
                + args.control_collective_w * loss_control_collective
                + args.control_individual_w * loss_control_individual
            )

        if step % args.eval_every == 0:
            strongest = gates[eligible].amax(0) if eligible.any() else gates.amax(0) * 0
            owner = gates[eligible].argmax(-1) if eligible.any() else None
            owners = torch.bincount(owner, minlength=args.C).float() if owner is not None else strongest
            metrics = torch.stack([
                loss.detach().float(),
                loss_carve.detach().float(),
                loss_recon.detach().float(),
                loss_gap_recon.detach().float(),
                loss_specificity.detach().float(),
                loss_retain.detach().float(),
                loss_completeness.detach().float(),
                loss_necessity.detach().float(),
                loss_l0.detach().float(),
                loss_coverage.detach().float(),
                loss_sum_norm.detach().float(),
                l0[eligible].mean().detach().float() if eligible.any() else l0.mean() * 0,
                gates[eligible].amax(-1).mean().detach().float() if eligible.any() else gates.mean() * 0,
                eligible.float().mean(),
                reference_gap[eligible].mean().detach().float() if eligible.any() else reference_gap.mean() * 0,
                routed_gap[eligible].mean().detach().float() if eligible.any() else routed_gap.mean() * 0,
                residual_gap[eligible].mean().detach().float() if eligible.any() else residual_gap.mean() * 0,
                coalition_marginal.float(),
                (strongest >= args.minimum_owner_share).float().mean(),
                (owners > 0).float().mean(),
                loss_balance.detach().float(),
                loss_control_collective.detach().float(),
                loss_control_individual.detach().float(),
                loss_retain_individual.detach().float(),
            ])
            if world > 1:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics.div_(world)
            if rank == 0:
                values = metrics.tolist()
                record = {
                    "step": step,
                    "loss": round(values[0], 5),
                    "carve": round(values[1], 5),
                    "carve_ramp": round(carve_ramp, 4),
                    "l0_ramp": round(l0_ramp, 4),
                    "necessity_ramp": round(necessity_ramp, 4),
                    "recon_kl": round(values[2], 5),
                    "gap_recon": round(values[3], 5),
                    "negative_kl": round(values[4], 5),
                    "retain_kl": round(values[5], 5),
                    "completeness": round(values[6], 4),
                    "necessity": round(values[7], 4),
                    "l0_loss": round(values[8], 4),
                    "coverage_loss": round(values[9], 4),
                    "sum_norm": round(values[10], 5),
                    "l0_effective": round(values[11], 3),
                    "winner_share": round(values[12], 3),
                    "eligible_frac": round(values[13], 3),
                    "target_gap": round(values[14], 4),
                    "routed_gap": round(values[15], 4),
                    "residual_gap": round(values[16], 4),
                    "coalition_marginal": round(values[17], 4),
                    "components_above_owner_floor": round(values[18], 3),
                    "components_owning_batch_examples": round(values[19], 3),
                    "balance": round(values[20], 4),
                    "control_collective_kl": round(values[21], 5),
                    "control_individual_kl": round(values[22], 5),
                    "retain_individual_kl": round(values[23], 5),
                    "training_variant": variant_name,
                    "control_variant": control_name,
                    "lr": round(lr, 8),
                }
                print(json.dumps(record), flush=True)

        if step != args.steps:
            loss.backward()
            _average_gradients(banks.parameters(), world)
            torch.nn.utils.clip_grad_norm_(banks.parameters(), 10.0)
            optimizer.step()
            if (
                rank == 0 and args.ckpt_every > 0 and step > 0
                and step % args.ckpt_every == 0
            ):
                tmp = checkpoint_path.with_suffix(".tmp")
                torch.save({
                    "banks": banks.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "module_paths": module_paths,
                }, tmp)
                tmp.replace(checkpoint_path)

    if rank == 0:
        torch.save(banks.state_dict(), out_dir / "banks.pt")
        checkpoint_path.unlink(missing_ok=True)
        print(f"saved to {out_dir}", flush=True)
        print("INDUCTION_COMPONENT_TRAINING_DONE", flush=True)
    if world > 1:
        dist.barrier()
    retain_source.close()
    editor.restore()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
