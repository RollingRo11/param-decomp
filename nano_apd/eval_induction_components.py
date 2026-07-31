"""Held-out causal and interpretability evaluation for induction components.

The report distinguishes four claims:

* routing sparsity: few components receive attribution on each example;
* causal relevance: ablating a component changes the routed output;
* causal necessity: its held-out marginal is reliably positive on examples it owns;
* circuit correspondence: its head/layer footprint agrees with a post-hoc target-model
  reference made from both attribution and direct head ablation.

The expensive target circuit reference is intentionally run last and only with
``--ground_truth``.  It is an empirical reference, not literal mechanistic ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.carving import (
    MatchedBatch,
    build_banks,
    capture_selected_usage,
    component_credits,
    make_matched_induction_batch,
    selected_logits,
)
from nano_apd.induction_components import (
    attention_head_credits,
    component_attention_head_credits,
    component_module_credits,
    component_piece_masses,
    effective_components,
    layer_index,
    selected_distribution_kl,
    sum_normalized_gates,
    target_module_credits,
)
from nano_apd.induction_editor import InductionEditor
from nano_apd.lm_target import load_carving_target, vocab_size

OUT_ROOT = Path(__file__).parent / "out"


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


def _selected_forward(target, tokens: Tensor, positions: Tensor) -> Tensor:
    return selected_logits(target(tokens), positions)


def _gap(pos: Tensor, neg: Tensor, labels: Tensor, kind: str) -> Tensor:
    return _score(pos, labels, kind) - _score(neg, labels, kind)


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def _bootstrap_interval(values: Tensor, seed: int, samples: int = 1000) -> list[float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return [float("nan"), float("nan")]
    if values.numel() == 1:
        value = round(values.item(), 6)
        return [value, value]
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[index].mean(-1)
    return [
        round(torch.quantile(means, 0.025).item(), 6),
        round(torch.quantile(means, 0.975).item(), 6),
    ]


def _mean(values: Tensor) -> float:
    return round(values.float().mean().item(), 6) if values.numel() else float("nan")



def _json_safe(value):
    """Replace non-finite floats so the report is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value

def _layer_profile(values: Tensor, module_paths: list[str], n_layers: int) -> Tensor:
    """Sum final module axis into layers, preserving all leading dimensions."""
    result = values.new_zeros(*values.shape[:-1], n_layers)
    for module, path in enumerate(module_paths):
        result[..., layer_index(path)] += values[..., module]
    return result


class HeadAblator:
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
                x = args[0]
                d_head = x.shape[-1] // self.n_heads
                mask = self.masks[path].to(x.dtype).view(
                    x.shape[0], 1, self.n_heads, 1
                )
                x = (x.view(
                    x.shape[0], x.shape[1], self.n_heads, d_head
                ) * mask).reshape_as(x)
                return (x,) + args[1:]

            self.handles.append(module.register_forward_pre_hook(hook))

    def clear(self) -> None:
        self.masks = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _head_ground_truth(
    target,
    editor,
    pairs: MatchedBatch,
    eligible: Tensor,
    reference_pos: Tensor,
    reference_neg: Tensor,
    target_head_contrast: Tensor,
    dense_paths: list[str],
    n_heads: int,
    score_type: str,
    chunk: int,
    use_bf16: bool,
    seed: int,
) -> dict:
    """Run direct single-head ablations after every learned-component metric."""
    batch, n_layers = pairs.positive.shape[0], len(dense_paths)
    reference_gap = _gap(reference_pos, reference_neg, pairs.labels, score_type)
    ablator = HeadAblator(target, dense_paths, n_heads)
    damage = torch.empty(batch, n_layers, n_heads, device=pairs.positive.device)
    negative_kl = torch.empty_like(damage)
    heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    editor.masks = None
    try:
        for start in range(0, len(heads), chunk):
            chosen = heads[start:start + chunk]
            k = len(chosen)
            ablator.masks = {
                path: torch.ones(k * 2 * batch, n_heads, device=pairs.positive.device)
                for path in dense_paths
            }
            for variant, (layer, head) in enumerate(chosen):
                rows_pos = slice(variant * batch, (variant + 1) * batch)
                rows_neg = slice(
                    k * batch + variant * batch,
                    k * batch + (variant + 1) * batch,
                )
                ablator.masks[dense_paths[layer]][rows_pos, head] = 0
                ablator.masks[dense_paths[layer]][rows_neg, head] = 0
            tokens = torch.cat([
                pairs.positive.repeat(k, 1), pairs.negative.repeat(k, 1)
            ])
            positions = pairs.positions.repeat(2 * k)
            with torch.no_grad(), _autocast(use_bf16):
                selected = _selected_forward(target, tokens, positions)
            pos, neg = selected.split(k * batch)
            pos = pos.view(k, batch, -1)
            neg = neg.view(k, batch, -1)
            for variant, (layer, head) in enumerate(chosen):
                ablated_gap = _gap(
                    pos[variant], neg[variant], pairs.labels, score_type
                )
                damage[:, layer, head] = reference_gap - ablated_gap
                negative_kl[:, layer, head] = selected_distribution_kl(
                    neg[variant], reference_neg
                )
    finally:
        ablator.close()

    rows = []
    attr = target_head_contrast.abs()
    for layer, head in heads:
        d = damage[eligible, layer, head]
        a = attr[eligible, layer, head]
        nk = negative_kl[eligible, layer, head]
        ci = _bootstrap_interval(d, seed + layer * n_heads + head)
        rows.append({
            "layer": layer,
            "head": head,
            "mean_gap_damage": _mean(d),
            "gap_damage_ci95": ci,
            "mean_abs_attribution": _mean(a),
            "matched_negative_kl": _mean(nk),
            "positive_causal_ci": bool(ci[0] > 0),
        })
    rows.sort(key=lambda row: row["mean_gap_damage"], reverse=True)
    attr_order = sorted(rows, key=lambda row: row["mean_abs_attribution"], reverse=True)
    attr_top = {(row["layer"], row["head"]) for row in attr_order[:32]}
    convergent = [
        row for row in rows
        if row["positive_causal_ci"] and (row["layer"], row["head"]) in attr_top
    ]
    return {
        "warning": (
            "Empirical post-hoc circuit reference, not literal ground truth. "
            "A head can be shared infrastructure or interact non-additively."
        ),
        "n_heads_tested": len(rows),
        "top_by_ablation": rows[:32],
        "top_by_attribution": attr_order[:32],
        "attribution_and_positive_ablation": convergent[:32],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--n_pairs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=81_001)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--component_chunk", type=int, default=4)
    parser.add_argument("--owner_share", type=float, default=0.12)
    parser.add_argument("--necessity_margin", type=float, default=0.05)
    parser.add_argument("--min_owned_examples", type=int, default=5)
    parser.add_argument("--ground_truth", action="store_true")
    parser.add_argument("--head_chunk", type=int, default=8)
    args = parser.parse_args()
    if args.n_pairs < 1 or args.batch_size < 1:
        raise ValueError("n_pairs and batch_size must be positive")

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
    n_heads = int(target.config.num_attention_heads)
    n_layers = int(target.config.num_hidden_layers)

    storage: dict[str, list[Tensor]] = defaultdict(list)
    all_pairs: list[MatchedBatch] = []
    with torch.enable_grad():
        for start in range(0, args.n_pairs, args.batch_size):
            size = min(args.batch_size, args.n_pairs - start)
            pairs = make_matched_induction_batch(
                size, int(config["seq_len"]), vocab_size(target), device,
                args.seed + start,
            )
            all_pairs.append(MatchedBatch(
                pairs.positive.cpu(), pairs.negative.cpu(), pairs.positions.cpu(),
                pairs.labels.cpu(),
            ))
            with _autocast(args.use_bf16):
                pos_usage = capture_selected_usage(
                    target, editor, pairs.positive, pairs.positions, pairs.labels,
                    score_type,
                )
                neg_usage = capture_selected_usage(
                    target, editor, pairs.negative, pairs.positions, pairs.labels,
                    score_type,
                )
                credit = (
                    component_credits(target, banks, module_paths, pos_usage)
                    - component_credits(target, banks, module_paths, neg_usage)
                )
                gates, attribution = sum_normalized_gates(credit)
                module_credit = (
                    component_module_credits(target, banks, module_paths, pos_usage)
                    - component_module_credits(target, banks, module_paths, neg_usage)
                )
                target_module = (
                    target_module_credits(target, module_paths, pos_usage)
                    - target_module_credits(target, module_paths, neg_usage)
                )
                head_pos, dense_paths = attention_head_credits(
                    target, module_paths, pos_usage, n_heads
                )
                head_neg, _ = attention_head_credits(
                    target, module_paths, neg_usage, n_heads
                )
                comp_head_pos, _ = component_attention_head_credits(
                    target, banks, module_paths, pos_usage, n_heads
                )
                comp_head_neg, _ = component_attention_head_credits(
                    target, banks, module_paths, neg_usage, n_heads
                )

            tokens = torch.cat([pairs.positive, pairs.negative])
            positions = pairs.positions.repeat(2)
            editor.masks = torch.ones(
                2 * size, 1, components, device=device, dtype=gates.dtype
            )
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
            ref_pos, ref_neg = reference.split(size)
            routed_pos, routed_neg = routed.split(size)
            residual_pos, residual_neg = residual.split(size)
            reference_gap = _gap(ref_pos, ref_neg, pairs.labels, score_type)
            routed_gap = _gap(routed_pos, routed_neg, pairs.labels, score_type)
            residual_gap = _gap(residual_pos, residual_neg, pairs.labels, score_type)
            eligible = reference_gap > float(config["min_target_gap"])
            if not config.get("include_target_incorrect", False):
                eligible &= ref_pos.argmax(-1) == pairs.labels

            loo_damage = torch.empty(size, components, device=device)
            loo_negative_kl = torch.empty_like(loo_damage)
            loo_pos_correct = torch.empty(
                size, components, dtype=torch.bool, device=device
            )
            for cstart in range(0, components, args.component_chunk):
                ids = list(range(cstart, min(components, cstart + args.component_chunk)))
                k = len(ids)
                masks = gates.repeat(k, 1)
                for variant, component in enumerate(ids):
                    masks[variant * size:(variant + 1) * size, component] = 0
                tiled_tokens = torch.cat([
                    pairs.positive.repeat(k, 1), pairs.negative.repeat(k, 1)
                ])
                tiled_positions = pairs.positions.repeat(2 * k)
                editor.masks = torch.cat([masks, masks]).unsqueeze(1)
                with torch.no_grad(), _autocast(args.use_bf16):
                    ablated = _selected_forward(
                        target, tiled_tokens, tiled_positions
                    )
                pos, neg = ablated.split(k * size)
                pos, neg = pos.view(k, size, -1), neg.view(k, size, -1)
                for variant, component in enumerate(ids):
                    ablated_gap = _gap(
                        pos[variant], neg[variant], pairs.labels, score_type
                    )
                    loo_damage[:, component] = routed_gap - ablated_gap
                    loo_negative_kl[:, component] = selected_distribution_kl(
                        neg[variant], ref_neg
                    )
                    loo_pos_correct[:, component] = (
                        pos[variant].argmax(-1) == pairs.labels
                    )
            editor.masks = None

            values = {
                "eligible": eligible,
                "gates": gates,
                "attribution": attribution,
                "reference_gap": reference_gap,
                "routed_gap": routed_gap,
                "residual_gap": residual_gap,
                "reference_pos_correct": ref_pos.argmax(-1) == pairs.labels,
                "routed_pos_correct": routed_pos.argmax(-1) == pairs.labels,
                "residual_pos_correct": residual_pos.argmax(-1) == pairs.labels,
                "routed_kl": selected_distribution_kl(routed, reference),
                "residual_negative_kl": selected_distribution_kl(residual_neg, ref_neg),
                "loo_damage": loo_damage,
                "loo_negative_kl": loo_negative_kl,
                "loo_pos_correct": loo_pos_correct,
                "component_module_credit": module_credit,
                "target_module_credit": target_module,
                "target_head_credit": head_pos - head_neg,
                "component_head_credit": comp_head_pos - comp_head_neg,
            }
            for key, value in values.items():
                storage[key].append(value.detach().cpu())

    merged = {key: torch.cat(values) for key, values in storage.items()}
    eligible = merged["eligible"].bool()
    if not eligible.any():
        raise RuntimeError("held-out evaluation has no target-correct induction pairs")
    gates = merged["gates"]
    owners = gates.argmax(-1)
    l0 = effective_components(merged["attribution"])
    component_rows = []
    for component in range(components):
        owned = eligible & (
            (owners == component) | (gates[:, component] >= args.owner_share)
        )
        damage = merged["loo_damage"][owned, component]
        ci = _bootstrap_interval(damage, args.seed + component)
        routed_correct = merged["routed_pos_correct"][owned]
        ablated_correct = merged["loo_pos_correct"][owned, component]
        necessity_rate = (
            (damage > args.necessity_margin).float().mean().item()
            if damage.numel() else float("nan")
        )
        flip_rate = (
            (routed_correct & ~ablated_correct).float().mean().item()
            if damage.numel() else float("nan")
        )
        component_rows.append({
            "component": component,
            "owned_examples": int(owned.sum()),
            "mean_share_on_owned": _mean(gates[owned, component]),
            "max_share": round(gates[eligible, component].max().item(), 6),
            "mean_gap_damage": _mean(damage),
            "gap_damage_ci95": ci,
            "necessity_rate": round(necessity_rate, 4),
            "prediction_flip_rate": round(flip_rate, 4),
            "matched_negative_kl": _mean(
                merged["loo_negative_kl"][owned, component]
            ),
            "passes_necessity_rule": bool(
                owned.sum() >= args.min_owned_examples
                and ci[0] > 0
                and necessity_rate >= 0.5
            ),
        })

    masses = component_piece_masses(target, banks, module_paths).detach().cpu()
    mass_layers = _layer_profile(masses, module_paths, n_layers)
    credit_modules = merged["component_module_credit"][eligible].abs().mean(0)
    credit_layers = _layer_profile(credit_modules, module_paths, n_layers)
    component_heads = merged["component_head_credit"][eligible].abs().mean(0)
    for component, row in enumerate(component_rows):
        top_modules = credit_modules[component].topk(5).indices.tolist()
        top_layers = credit_layers[component].topk(5).indices.tolist()
        flat_heads = component_heads[component].flatten()
        top_heads = flat_heads.topk(8).indices.tolist()
        row["top_credit_modules"] = [module_paths[index] for index in top_modules]
        row["top_credit_layers"] = top_layers
        row["top_attention_heads"] = [
            {"layer": index // n_heads, "head": index % n_heads}
            for index in top_heads
        ]
        row["piece_mass_effective_layers"] = round(
            effective_components(mass_layers[component]).item(), 4
        )

    target_module = merged["target_module_credit"][eligible].abs().mean(0)
    target_head = merged["target_head_credit"]
    target_module_order = target_module.argsort(descending=True).tolist()
    target_head_mean = target_head[eligible].abs().mean(0)
    target_head_order = target_head_mean.flatten().argsort(descending=True).tolist()
    report = {
        "artifact": str(args.artifact),
        "precision": "bf16" if args.use_bf16 else "fp32",
        "n_pairs": args.n_pairs,
        "eligible_count": int(eligible.sum()),
        "model": config["model_name"],
        "components": components,
        "rank": int(config["rank"]),
        "cross_layer": {
            "modules": len(module_paths),
            "layers": n_layers,
            "component_index_shared_across_modules": True,
        },
        "routing": {
            "normalization": "sum",
            "mean_effective_components": _mean(l0[eligible]),
            "mean_winner_share": _mean(gates[eligible].amax(-1)),
            "components_owning_examples": int(torch.unique(owners[eligible]).numel()),
        },
        "behavior": {
            "target_gap": _mean(merged["reference_gap"][eligible]),
            "routed_gap": _mean(merged["routed_gap"][eligible]),
            "residual_gap": _mean(merged["residual_gap"][eligible]),
            "target_positive_accuracy": _mean(
                merged["reference_pos_correct"][eligible].float()
            ),
            "routed_positive_accuracy": _mean(
                merged["routed_pos_correct"][eligible].float()
            ),
            "residual_positive_accuracy": _mean(
                merged["residual_pos_correct"][eligible].float()
            ),
            "routed_selected_kl": _mean(
                merged["routed_kl"]
            ),
            "residual_matched_negative_kl": _mean(
                merged["residual_negative_kl"][eligible]
            ),
        },
        "necessity_rule": {
            "owned": f"top owner or share >= {args.owner_share}",
            "minimum_owned_examples": args.min_owned_examples,
            "requirements": (
                "bootstrap CI lower bound > 0 and gap-damage rate above "
                f"{args.necessity_margin} >= 50%"
            ),
            "components_passing": sum(
                row["passes_necessity_rule"] for row in component_rows
            ),
        },
        "components_detail": component_rows,
        "target_attribution_reference": {
            "top_modules": [module_paths[index] for index in target_module_order[:16]],
            "top_heads": [
                {"layer": index // n_heads, "head": index % n_heads}
                for index in target_head_order[:32]
            ],
        },
    }

    # The user requested ablation + attribution ground truth only after component training
    # and ordinary held-out evaluation.  Keep it literally last in the evaluator.
    if args.ground_truth:
        pairs = MatchedBatch(
            torch.cat([pair.positive for pair in all_pairs]).to(device),
            torch.cat([pair.negative for pair in all_pairs]).to(device),
            torch.cat([pair.positions for pair in all_pairs]).to(device),
            torch.cat([pair.labels for pair in all_pairs]).to(device),
        )
        with torch.no_grad(), _autocast(args.use_bf16):
            reference = _selected_forward(
                target, torch.cat([pairs.positive, pairs.negative]),
                pairs.positions.repeat(2),
            )
        ref_pos, ref_neg = reference.split(args.n_pairs)
        gt_eligible = eligible.to(device)
        target_head_contrast = merged["target_head_credit"].to(device)
        report["posthoc_head_ground_truth"] = _head_ground_truth(
            target, editor, pairs, gt_eligible, ref_pos, ref_neg,
            target_head_contrast, dense_paths, n_heads, score_type,
            args.head_chunk, args.use_bf16, args.seed + 900_000,
        )

    report = _json_safe(report)

    suffix = "bf16" if args.use_bf16 else "fp32"
    if args.ground_truth:
        suffix += "_ground_truth"
    output_path = args.artifact / f"induction_eval_{suffix}.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {output_path}", flush=True)
    editor.restore()


if __name__ == "__main__":
    main()
