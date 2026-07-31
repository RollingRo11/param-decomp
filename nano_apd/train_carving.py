"""Train a contrastive selected-token parameter carving on a frozen causal LM.

The trainer learns a small residual-plus-behavior bank. It differs from a conventional
retain/forget LoRA in two ways:

* supervision is one explicit output position across matched positive/negative contexts;
* components must explain the unedited target's contrastive parameter-usage fingerprints,
  and random component subsets are trained to have predictable partial effects.

Example:

    torchrun --standalone --nproc_per_node=2 -m nano_apd.train_carving \
        --target hf --model_name EleutherAI/pythia-410m --bf16 \
        --tag induction_v1 --auto_modules 8 --C 16

Use ``--bank_type projected`` as the stricter control whose pieces are two-sided
projections of the existing target weights.
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
    CarvingEditor,
    MatchedBatch,
    bank_mass,
    build_banks,
    capture_selected_usage,
    completeness_loss,
    component_credits,
    component_usage_sketch,
    geometry_loss,
    kl_per_position,
    load_matched_batch,
    make_coordinate_samples,
    make_matched_induction_batch,
    module_contrast_scores,
    normalized_codes,
    participation_ratio,
    sample_subset_masks,
    selected_accuracy,
    selected_scores,
    sketch_reconstruction_loss,
    target_total_credit,
    usage_sketch,
)
from nano_apd.lm_target import (
    DEFAULT_HF_MODEL,
    candidate_linear_paths,
    load_carving_target,
    vocab_size,
)
from nano_param_decomp.pile_4L import make_loader

OUT_ROOT = Path(__file__).parent / "out"


def _init_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        return rank, world, local_rank, device
    if not torch.cuda.is_available():
        raise RuntimeError("train_carving requires CUDA")
    return 0, 1, 0, torch.device("cuda")


def _average_gradients(parameters, world: int) -> None:
    """Synchronize the small banks without wrapping the patched target in DDP."""
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



class PairSource:
    def __init__(
        self, args, device: torch.device, model_vocab_size: int, local_batch: int, rank: int
    ):
        self.args = args
        self.device = device
        self.vocab_size = model_vocab_size
        self.local_batch = local_batch
        self.rank = rank
        self.saved = load_matched_batch(args.pairs_file, "cpu") if args.pairs_file else None
        self.generator = torch.Generator().manual_seed(args.seed + 81_337 + rank)

    def batch(
        self, step: int, *, eval_seed: int | None = None, batch_size: int | None = None
    ) -> MatchedBatch:
        size = batch_size or self.local_batch
        if self.saved is None:
            seed = (
                eval_seed
                if eval_seed is not None
                else self.args.seed * 1_000_003 + step * 10_007 + self.rank
            )
            return make_matched_induction_batch(
                size,
                self.args.seq_len,
                self.vocab_size,
                self.device,
                seed,
            )
        count = self.saved.positive.shape[0]
        if eval_seed is None:
            index = torch.randint(count, (size,), generator=self.generator)
        else:
            generator = torch.Generator().manual_seed(eval_seed)
            index = torch.randint(count, (size,), generator=generator)
        return MatchedBatch(
            self.saved.positive[index],
            self.saved.negative[index],
            self.saved.positions[index],
            self.saved.labels[index],
        ).to(self.device)


class RetainSource:
    def __init__(
        self, args, device: torch.device, local_batch: int, rank: int, world: int
    ):
        self.device = device
        self.saved: Tensor | None = None
        self.generator = torch.Generator().manual_seed(args.seed + 91_337 + rank)
        if args.retain_file:
            obj = torch.load(args.retain_file, weights_only=True, map_location="cpu")
            self.saved = obj["input_ids"] if isinstance(obj, dict) else obj
            self.saved = self.saved[:, :args.seq_len].long()
            self.loader = None
        elif local_batch > 0:
            self.loader = make_loader(
                args.batch_retain,
                args.seq_len,
                rank,
                world,
                "train",
                args.seed + 3_000_000,
            )
        else:
            self.loader = None
        self.batch_size = local_batch

    def batch(self) -> Tensor | None:
        if self.batch_size == 0:
            return None
        if self.saved is not None:
            index = torch.randint(
                self.saved.shape[0], (self.batch_size,), generator=self.generator
            )
            return self.saved[index].to(self.device)
        assert self.loader is not None
        return next(self.loader).to(self.device)

    def close(self) -> None:
        if self.loader is not None:
            close = getattr(self.loader, "close", None)
            if close is not None:
                close()


def _filtered_modules(paths: list[str], patterns: str) -> list[str]:
    if not patterns:
        return paths
    keep = [item.strip() for item in patterns.split(",") if item.strip()]
    result = [path for path in paths if any(pattern in path for pattern in keep)]
    if not result:
        raise ValueError(f"--modules_filter={patterns!r} selected no modules")
    return result


def _localize_modules(
    target,
    candidates: list[str],
    pairs: MatchedBatch,
    topk: int,
    score_type: str,
    min_target_gap: float,
    include_target_incorrect: bool,
) -> list[str]:
    # Capture does not access banks while masks are None, so an empty bank is sufficient.
    editor = CarvingEditor(target, torch.nn.ModuleDict(), candidates)
    positive = capture_selected_usage(
        target, editor, pairs.positive, pairs.positions, pairs.labels, score_type
    )
    negative = capture_selected_usage(
        target, editor, pairs.negative, pairs.positions, pairs.labels, score_type
    )
    eligible = positive.scores - negative.scores > min_target_gap
    if not include_target_incorrect:
        eligible &= positive.predictions == pairs.labels
    if not eligible.any():
        editor.restore()
        raise RuntimeError(
            "module localization found no eligible pairs; increase --localize_pairs, "
            "lower --min_target_gap, or inspect the pair construction"
        )
    scores = module_contrast_scores(
        target, candidates, positive, negative, example_mask=eligible
    )
    editor.restore()
    ordered = sorted(candidates, key=scores.get, reverse=True)
    selected = ordered[:topk]
    shown = ordered[:max(16, topk)]
    print(json.dumps({
        "localized_modules": selected,
        "eligible_localization_frac": round(eligible.float().mean().item(), 4),
        "top_module_scores": {path: round(scores[path], 8) for path in shown},
    }), flush=True)
    return selected


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return values[mask].mean() if mask.any() else values.new_zeros(())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["hf", "pile4l"], default="hf")
    parser.add_argument("--model_name", default=DEFAULT_HF_MODEL,
                        help="Hugging Face causal LM id when --target=hf")
    parser.add_argument("--C", type=int, default=16)
    parser.add_argument("--rank_m", type=int, default=4, dest="rank")
    parser.add_argument("--bank_type", choices=["free", "projected"], default="free")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_pairs", type=int, default=16,
                        help="global matched-pair batch (split across DDP ranks)")
    parser.add_argument("--batch_retain", type=int, default=32,
                        help="global retain batch (split across DDP ranks)")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--pairs_file", default="",
                        help="optional .pt with positive/negative/positions[/labels]")
    parser.add_argument("--retain_file", default="",
                        help="optional .pt token tensor or {'input_ids': tensor}")
    parser.add_argument("--modules_filter", default="")
    parser.add_argument("--auto_modules", type=int, default=8,
                        help="keep this many modules by target-use contrast; 0 disables")
    parser.add_argument("--localize_pairs", type=int, default=16,
                        help="rank-0 matched batch used for automatic module localization")
    parser.add_argument("--sketch_per_module", type=int, default=256)
    parser.add_argument("--forget_w", type=float, default=1.0)
    parser.add_argument("--retain_w", type=float, default=1.0)
    parser.add_argument("--usage_w", type=float, default=0.2)
    parser.add_argument("--geometry_w", type=float, default=0.2)
    parser.add_argument("--subset_w", type=float, default=0.5)
    parser.add_argument("--code_w", type=float, default=0.01)
    parser.add_argument("--code_budget", type=float, default=4.0)
    parser.add_argument("--reg_w", type=float, default=0.0)
    parser.add_argument("--score_type", choices=["logit_margin", "target_logit", "logprob"],
                        default="logit_margin")
    parser.add_argument("--gap_floor", type=float, default=0.0,
                        help="maximum edited positive-minus-negative selected-score gap")
    parser.add_argument("--min_target_gap", type=float, default=0.05,
                        help="ignore supervision where the target has no positive advantage")
    parser.add_argument("--include_target_incorrect", action="store_true",
                        help="also supervise pairs where the target misses the chosen token")
    parser.add_argument("--subset_remove_p", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="induction_v1")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    rank, world, _local_rank, device = _init_distributed()
    if args.C < 1 or args.rank < 1 or args.steps < 1 or args.localize_pairs < 1:
        raise ValueError("C, rank_m, steps, and localize_pairs must be positive")
    if args.batch_pairs < world or args.batch_pairs % world:
        raise ValueError("--batch_pairs must be positive and divisible by WORLD_SIZE")
    if args.batch_retain < 0 or (
        args.batch_retain > 0
        and (args.batch_retain < world or args.batch_retain % world)
    ):
        raise ValueError("--batch_retain must be zero or positive and divisible by WORLD_SIZE")
    local_pairs = args.batch_pairs // world
    local_retain = args.batch_retain // world if args.batch_retain else 0
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Let rank 0 populate the HF cache before the other workers open the same files.
    target = None
    if args.target == "hf" and world > 1:
        if rank == 0:
            target = load_carving_target(args.target, args.model_name)
        dist.barrier()
        if rank != 0:
            target = load_carving_target(args.target, args.model_name)
        dist.barrier()
    else:
        target = load_carving_target(args.target, args.model_name)
    assert target is not None
    target = target.float().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    pairs_source = PairSource(args, device, vocab_size(target), local_pairs, rank)
    retain_source = RetainSource(args, device, local_retain, rank, world)

    target_slug = (
        "pile4l" if args.target == "pile4l"
        else args.model_name.rsplit("/", 1)[-1].lower().replace("_", "-")
    )
    out_dir = OUT_ROOT / f"{target_slug}_carving_{args.tag}"
    checkpoint_path = out_dir / "ckpt.pt"
    checkpoint = None
    candidates = _filtered_modules(
        candidate_linear_paths(target, args.target), args.modules_filter
    )
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        module_paths = checkpoint["module_paths"]
    elif args.auto_modules > 0 and args.auto_modules < len(candidates):
        module_paths = None
        if rank == 0:
            module_paths = _localize_modules(
                target,
                candidates,
                pairs_source.batch(
                    0,
                    eval_seed=args.seed + 700_001,
                    batch_size=args.localize_pairs,
                ),
                args.auto_modules,
                args.score_type,
                args.min_target_gap,
                args.include_target_incorrect,
            )
        if world > 1:
            selected_object = [module_paths]
            dist.broadcast_object_list(selected_object, src=0)
            module_paths = selected_object[0]
        assert module_paths is not None
        torch.cuda.empty_cache()
    else:
        module_paths = candidates

    # Model construction can consume RNG differently on cache-hit/miss; reset before banks.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    banks = build_banks(target, module_paths, args.C, args.rank, args.bank_type).to(device)
    editor = CarvingEditor(target, banks, module_paths)
    optimizer = torch.optim.AdamW(banks.parameters(), lr=args.lr, weight_decay=0.0)
    start_step = 0
    if checkpoint is not None:
        if checkpoint["bank_type"] != args.bank_type:
            raise ValueError("checkpoint bank_type disagrees with command line")
        if checkpoint.get("target", args.target) != args.target:
            raise ValueError("checkpoint target disagrees with command line")
        if checkpoint.get("model_name", args.model_name) != args.model_name:
            raise ValueError("checkpoint model_name disagrees with command line")
        banks.load_state_dict(checkpoint["banks"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        if rank == 0:
            print(f"resumed {checkpoint_path} at step {start_step}", flush=True)
    _broadcast_parameters(banks.parameters(), world)

    samples = make_coordinate_samples(
        target, module_paths, args.sketch_per_module, args.seed + 40_001
    )
    n_weight_params = sum(target.get_submodule(path).weight.numel() for path in module_paths)
    config = vars(args) | {
        "module_paths": module_paths,
        "rank_total": args.C * args.rank,
        "world_size": world,
        "local_batch_pairs": local_pairs,
        "local_batch_retain": local_retain,
        "objective": "contrastive-selected-token-parameter-carving",
    }
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    if world > 1:
        dist.barrier()

    wb = None
    if args.wandb and rank == 0:
        import wandb

        wb = wandb.init(project="nano-apd", name=f"carving_{args.tag}", config=config)

    for step in range(start_step, args.steps + 1):
        phase = min(step, args.steps) / args.steps
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * phase))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        pairs = pairs_source.batch(step)
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
            positive_credit = component_credits(
                target, banks, module_paths, positive_usage
            )
            negative_credit = component_credits(
                target, banks, module_paths, negative_usage
            )
            contrast_credit = positive_credit - negative_credit
            target_credit = (
                target_total_credit(target, module_paths, positive_usage)
                - target_total_credit(target, module_paths, negative_usage)
            )
            target_sketch = (
                usage_sketch(target, module_paths, positive_usage, samples)
                - usage_sketch(target, module_paths, negative_usage, samples)
            )
            learned_sketch = (
                component_usage_sketch(
                    target, banks, module_paths, positive_usage, samples
                )
                - component_usage_sketch(
                    target, banks, module_paths, negative_usage, samples
                )
            )
            usage_active = (
                positive_usage.scores - negative_usage.scores > args.min_target_gap
            )
            if not args.include_target_incorrect:
                usage_active &= positive_usage.predictions == pairs.labels
            if usage_active.any():
                loss_usage = (
                    completeness_loss(
                        contrast_credit[usage_active], target_credit[usage_active]
                    )
                    + sketch_reconstruction_loss(
                        learned_sketch[usage_active], target_sketch[usage_active]
                    )
                )
                loss_geometry = geometry_loss(
                    contrast_credit[usage_active], target_sketch[usage_active]
                )
            else:
                # Keep a zero-valued graph on every bank when a small local DDP batch
                # happens to contain no target-correct examples.
                loss_usage = (
                    contrast_credit.sum() + learned_sketch.sum()
                ) * 0
                loss_geometry = contrast_credit.sum() * 0
            codes = normalized_codes(contrast_credit)
            code_pr = participation_ratio(codes)
            loss_code = _masked_mean(
                ((code_pr - args.code_budget) / args.C).square(), usage_active
            )

            sequences = [pairs.positive, pairs.negative]
            if retain is not None:
                sequences.append(retain)
            all_tokens = torch.cat(sequences)
            with torch.no_grad():
                # A keep-all mask executes the same bank/output-allocation path as an
                # edit. This matters in BF16: different GEMM storage/alignment paths can
                # otherwise create a measurable false KL even for an exactly zero bank.
                editor.masks = torch.ones(
                    all_tokens.shape[0], 1, args.C, device=device
                )
                reference_logits = target(all_tokens)
            editor.masks = torch.zeros(all_tokens.shape[0], 1, args.C, device=device)
            edited_logits = target(all_tokens)
            editor.masks = None

            batch = pairs.positive.shape[0]
            pos_edit = selected_scores(
                edited_logits[:batch], pairs.positions, pairs.labels, args.score_type
            )
            neg_edit = selected_scores(
                edited_logits[batch:2 * batch], pairs.positions, pairs.labels, args.score_type
            )
            # Use the same concatenated reference pass as the edited pass. In BF16,
            # GEMM accumulation can differ slightly between separate B-sized passes and
            # one 2B-sized pass even when the bank is an exact no-op.
            original_gap = (
                selected_scores(
                    reference_logits[:batch],
                    pairs.positions,
                    pairs.labels,
                    args.score_type,
                )
                - selected_scores(
                    reference_logits[batch:2 * batch],
                    pairs.positions,
                    pairs.labels,
                    args.score_type,
                )
            )
            full_gap = pos_edit - neg_edit
            rows = torch.arange(batch, device=device)
            target_correct = (
                reference_logits[:batch][rows, pairs.positions].argmax(-1)
                == pairs.labels
            )
            active = original_gap > args.min_target_gap
            if not args.include_target_incorrect:
                active &= target_correct
            loss_forget = _masked_mean(F.relu(full_gap - args.gap_floor), active)

            preserve_kl = kl_per_position(edited_logits, reference_logits)
            preserve = torch.ones_like(preserve_kl, dtype=torch.bool)
            preserve[rows, pairs.positions] = False
            loss_retain = preserve_kl[preserve].mean()

            subset_masks, removed_mass = sample_subset_masks(
                codes, args.subset_remove_p
            )
            pair_tokens = torch.cat([pairs.positive, pairs.negative])
            editor.masks = torch.cat([subset_masks, subset_masks])
            subset_logits = target(pair_tokens)
            editor.masks = None
            subset_gap = (
                selected_scores(
                    subset_logits[:batch], pairs.positions, pairs.labels, args.score_type
                )
                - selected_scores(
                    subset_logits[batch:], pairs.positions, pairs.labels, args.score_type
                )
            )
            desired_gap = (
                original_gap * (1 - removed_mass) + args.gap_floor * removed_mass
            ).detach()
            gap_scale = original_gap.detach().abs().mean().clamp_min(0.1)
            loss_subset_effect = _masked_mean(
                F.smooth_l1_loss(subset_gap, desired_gap, reduction="none") / gap_scale,
                active,
            )
            negative_subset_kl = kl_per_position(
                subset_logits[batch:], reference_logits[batch:2 * batch]
            ).mean()
            loss_subset = loss_subset_effect + negative_subset_kl
            loss_reg = bank_mass(target, banks, module_paths) / n_weight_params

            loss = (
                args.forget_w * loss_forget
                + args.retain_w * loss_retain
                + args.usage_w * loss_usage
                + args.geometry_w * loss_geometry
                + args.subset_w * loss_subset
                + args.code_w * loss_code
                + args.reg_w * loss_reg
            )

        if step % args.eval_every == 0:
            metrics = torch.stack([
                loss.detach().float(),
                loss_forget.detach().float(),
                loss_retain.detach().float(),
                loss_usage.detach().float(),
                loss_geometry.detach().float(),
                loss_subset.detach().float(),
                code_pr.detach().float().mean(),
                active.float().mean(),
                usage_active.float().mean(),
                original_gap.detach().float().mean(),
                full_gap.detach().float().mean(),
                selected_accuracy(
                    reference_logits[:batch], pairs.positions, pairs.labels
                ),
                selected_accuracy(edited_logits[:batch], pairs.positions, pairs.labels),
                loss_reg.detach().float(),
            ])
            if world > 1:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics.div_(world)
            if rank == 0:
                values = metrics.tolist()
                rec = {
                    "step": step,
                    "loss": round(values[0], 5),
                    "forget": round(values[1], 5),
                    "retain_kl": round(values[2], 6),
                    "usage": round(values[3], 5),
                    "geometry": round(values[4], 5),
                    "subset": round(values[5], 5),
                    "code_pr": round(values[6], 3),
                    "active_frac": round(values[7], 3),
                    "usage_active_frac": round(values[8], 3),
                    "target_gap": round(values[9], 4),
                    "edited_gap": round(values[10], 4),
                    "target_acc": round(values[11], 3),
                    "edited_acc": round(values[12], 3),
                    "mass_per_weight": round(values[13], 8),
                    "lr": round(lr, 8),
                }
                print(json.dumps(rec), flush=True)
                if wb is not None:
                    wb.log(rec, step=step)

        if step != args.steps:
            loss.backward()
            _average_gradients(banks.parameters(), world)
            torch.nn.utils.clip_grad_norm_(banks.parameters(), 10.0)
            optimizer.step()
            if (
                rank == 0
                and args.ckpt_every > 0
                and step > 0
                and step % args.ckpt_every == 0
            ):
                tmp = checkpoint_path.with_suffix(".tmp")
                torch.save({
                    "banks": banks.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "module_paths": module_paths,
                    "bank_type": args.bank_type,
                    "target": args.target,
                    "model_name": args.model_name,
                }, tmp)
                tmp.replace(checkpoint_path)

    if rank == 0:
        torch.save(banks.state_dict(), out_dir / "banks.pt")
        checkpoint_path.unlink(missing_ok=True)
    if world > 1:
        dist.barrier()
    retain_source.close()
    editor.restore()
    if rank == 0:
        print(f"saved to {out_dir}", flush=True)
        print("CARVING_DONE", flush=True)
        if wb is not None:
            wb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
