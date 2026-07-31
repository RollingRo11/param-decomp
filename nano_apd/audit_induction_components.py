"""Audit context selectivity and distributed support of induction components.

Necessity alone cannot distinguish a compact true circuit from winner-take-all
optimization.  This audit measures each component along three independent axes:

1. context breadth: which induction lags/positions and examples it owns;
2. mechanistic breadth: how many modules, layers, and attention heads carry credit;
3. causal breadth: which individual layer pieces of the cross-layer component matter.

Layer/head breadth describes where a component is implemented; it is not evidence of
polysemanticity. Synthetic random-token induction has no semantic labels, so the script
reports observable context strata and examples rather than inventing names.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from nano_apd.carving import (
    MatchedBatch,
    build_banks,
    capture_selected_usage,
    component_credits,
    make_matched_induction_batch,
    selected_logits,
)
from nano_apd.induction_components import (
    component_attention_head_credits,
    component_module_credits,
    component_piece_masses,
    effective_components,
    layer_index,
    sum_normalized_gates,
)
from nano_apd.induction_editor import InductionEditor
from nano_apd.lm_target import load_carving_target, vocab_size


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


def _gap(pos: Tensor, neg: Tensor, labels: Tensor, kind: str) -> Tensor:
    return _score(pos, labels, kind) - _score(neg, labels, kind)


def _selected_forward(target, tokens: Tensor, positions: Tensor) -> Tensor:
    return selected_logits(target(tokens), positions)


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    return value


def _mean(values: Tensor) -> float:
    return values.float().mean().item() if values.numel() else float("nan")


def _profile(values: Tensor) -> dict:
    values = values.float().clamp_min(0)
    total = values.sum().clamp_min(1e-12)
    shares = values / total
    return {
        "effective_count": round(effective_components(values).item(), 4),
        "top1_share": round(shares.max().item(), 4),
        "top5_share": round(shares.topk(min(5, shares.numel())).values.sum().item(), 4),
    }


def _layer_profile(values: Tensor, paths: list[str], n_layers: int) -> Tensor:
    result = values.new_zeros(*values.shape[:-1], n_layers)
    for module, path in enumerate(paths):
        result[..., layer_index(path)] += values[..., module]
    return result


def _lags(pairs: MatchedBatch) -> tuple[Tensor, Tensor]:
    lags, sources = [], []
    for tokens, position in zip(pairs.positive, pairs.positions, strict=True):
        pos = int(position)
        cue = tokens[pos]
        matches = (tokens[:pos] == cue).nonzero(as_tuple=False).flatten()
        source = int(matches[-1]) if matches.numel() else -1
        sources.append(source)
        lags.append(pos - source if source >= 0 else -1)
    return torch.tensor(lags), torch.tensor(sources)


def _binned_profile(values: Tensor, observations: Tensor, edges: list[int]) -> dict:
    bins = torch.bucketize(observations, torch.tensor(edges))
    mass = torch.stack([
        values[bins == index].sum() for index in range(len(edges) + 1)
    ])
    profile = _profile(mass)
    profile["mass_by_bin"] = [round(item, 4) for item in (
        mass / mass.sum().clamp_min(1e-12)
    ).tolist()]
    profile["bin_upper_edges"] = edges
    return profile


def _decode_examples(
    pairs: MatchedBatch,
    gates: Tensor,
    eligible: Tensor,
    component: int,
    count: int,
    tokenizer,
) -> list[dict]:
    score = gates[:, component].clone()
    score[~eligible] = -1
    indices = score.topk(min(count, int(eligible.sum()))).indices.tolist()
    lags, sources = _lags(pairs)
    rows = []
    for index in indices:
        position = int(pairs.positions[index])
        source = int(sources[index])
        cue = int(pairs.positive[index, position])
        label = int(pairs.labels[index])
        lo = max(0, source - 2)
        hi = min(pairs.positive.shape[1], position + 2)
        row = {
            "example": index,
            "share": round(gates[index, component].item(), 5),
            "source_position": source,
            "prediction_position": position,
            "lag": int(lags[index]),
            "cue_token_id": cue,
            "label_token_id": label,
        }
        if tokenizer is not None:
            row |= {
                "cue": tokenizer.decode([cue]),
                "label": tokenizer.decode([label]),
                "snippet": tokenizer.decode(pairs.positive[index, lo:hi].tolist()),
            }
        rows.append(row)
    return rows


def _layer_piece_damage(
    target,
    editor: InductionEditor,
    pairs: MatchedBatch,
    gates: Tensor,
    base_gap: Tensor,
    component: int,
    module_paths: list[str],
    n_layers: int,
    score_type: str,
    chunk: int,
    use_bf16: bool,
) -> Tensor:
    """Ablate one component only in one layer, with every other piece unchanged."""
    batch = pairs.positive.shape[0]
    result = torch.empty(batch, n_layers, device=pairs.positive.device)
    for start in range(0, n_layers, chunk):
        layers = list(range(start, min(n_layers, start + chunk)))
        k = len(layers)
        base_masks = gates.repeat(k, 1)
        editor.masks = torch.cat([base_masks, base_masks]).unsqueeze(1)
        editor.mask_overrides = {}
        for path in module_paths:
            layer = layer_index(path)
            if layer not in layers:
                continue
            variant = layers.index(layer)
            override = editor.masks.clone()
            pos_rows = slice(variant * batch, (variant + 1) * batch)
            neg_rows = slice(
                k * batch + variant * batch,
                k * batch + (variant + 1) * batch,
            )
            override[pos_rows, :, component] = 0
            override[neg_rows, :, component] = 0
            editor.mask_overrides[path] = override
        tokens = torch.cat([
            pairs.positive.repeat(k, 1), pairs.negative.repeat(k, 1)
        ])
        positions = pairs.positions.repeat(2 * k)
        with torch.no_grad(), _autocast(use_bf16):
            selected = _selected_forward(target, tokens, positions)
        pos, neg = selected.split(k * batch)
        pos, neg = pos.view(k, batch, -1), neg.view(k, batch, -1)
        for variant, layer in enumerate(layers):
            ablated_gap = _gap(
                pos[variant], neg[variant], pairs.labels, score_type
            )
            result[:, layer] = base_gap - ablated_gap
    editor.masks = None
    editor.mask_overrides = {}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--n_pairs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=81_001)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--top_examples", type=int, default=5)
    parser.add_argument("--audit_components", type=int, default=4)
    parser.add_argument("--layer_examples", type=int, default=16)
    parser.add_argument("--layer_chunk", type=int, default=4)
    args = parser.parse_args()

    config = json.loads((args.artifact / "config.json").read_text())
    device = torch.device("cuda")
    target = load_carving_target("hf", config["model_name"]).float().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    paths = config["module_paths"]
    components = int(config["C"])
    n_layers = int(target.config.num_hidden_layers)
    n_heads = int(target.config.num_attention_heads)
    banks = build_banks(
        target, paths, components, int(config["rank"]), config["bank_type"]
    ).to(device)
    banks.load_state_dict(torch.load(
        args.artifact / "banks.pt", weights_only=True, map_location=device
    ))
    editor = InductionEditor(target, banks, paths)
    score_type = config["score_type"]

    saved_pairs, saved = [], {
        key: [] for key in (
            "eligible", "gates", "routed_gap", "loo_damage",
            "module_credit", "head_credit",
        )
    }
    for start in range(0, args.n_pairs, args.batch_size):
        size = min(args.batch_size, args.n_pairs - start)
        pairs = make_matched_induction_batch(
            size, int(config["seq_len"]), vocab_size(target), device,
            args.seed + start,
        )
        saved_pairs.append(pairs.to("cpu"))
        with torch.enable_grad(), _autocast(args.use_bf16):
            pos_usage = capture_selected_usage(
                target, editor, pairs.positive, pairs.positions, pairs.labels, score_type
            )
            neg_usage = capture_selected_usage(
                target, editor, pairs.negative, pairs.positions, pairs.labels, score_type
            )
            credit = (
                component_credits(target, banks, paths, pos_usage)
                - component_credits(target, banks, paths, neg_usage)
            )
            gates, _ = sum_normalized_gates(credit)
            module_credit = (
                component_module_credits(target, banks, paths, pos_usage)
                - component_module_credits(target, banks, paths, neg_usage)
            )
            hp, _ = component_attention_head_credits(
                target, banks, paths, pos_usage, n_heads
            )
            hn, _ = component_attention_head_credits(
                target, banks, paths, neg_usage, n_heads
            )

        tokens = torch.cat([pairs.positive, pairs.negative])
        positions = pairs.positions.repeat(2)
        editor.masks = torch.ones(2 * size, 1, components, device=device)
        with torch.no_grad(), _autocast(args.use_bf16):
            reference = _selected_forward(target, tokens, positions)
        editor.masks = torch.cat([gates, gates]).unsqueeze(1)
        with torch.no_grad(), _autocast(args.use_bf16):
            routed = _selected_forward(target, tokens, positions)
        editor.masks = None
        ref_pos, ref_neg = reference.split(size)
        routed_pos, routed_neg = routed.split(size)
        reference_gap = _gap(ref_pos, ref_neg, pairs.labels, score_type)
        routed_gap = _gap(routed_pos, routed_neg, pairs.labels, score_type)
        eligible = reference_gap > float(config["min_target_gap"])
        if not config.get("include_target_incorrect", False):
            eligible &= ref_pos.argmax(-1) == pairs.labels

        loo = torch.empty(size, components, device=device)
        for component in range(components):
            masks = gates.clone()
            masks[:, component] = 0
            editor.masks = torch.cat([masks, masks]).unsqueeze(1)
            with torch.no_grad(), _autocast(args.use_bf16):
                ablated = _selected_forward(target, tokens, positions)
            ap, an = ablated.split(size)
            loo[:, component] = routed_gap - _gap(
                ap, an, pairs.labels, score_type
            )
        editor.masks = None
        for key, value in {
            "eligible": eligible,
            "gates": gates,
            "routed_gap": routed_gap,
            "loo_damage": loo,
            "module_credit": module_credit,
            "head_credit": hp - hn,
        }.items():
            saved[key].append(value.detach().cpu())

    pairs = MatchedBatch(*[
        torch.cat([getattr(item, field) for item in saved_pairs])
        for field in ("positive", "negative", "positions", "labels")
    ])
    values = {key: torch.cat(rows) for key, rows in saved.items()}
    eligible = values["eligible"].bool()
    if not eligible.any():
        raise RuntimeError("no eligible held-out induction examples")
    gates = values["gates"]
    owners = gates.argmax(-1)
    lags, _ = _lags(pairs)
    positions = pairs.positions
    module_profile = values["module_credit"][eligible].abs().mean(0)
    layer_profile = _layer_profile(module_profile, paths, n_layers)
    head_profile = values["head_credit"][eligible].abs().mean(0)
    masses = component_piece_masses(target, banks, paths).detach().cpu()
    mass_layers = _layer_profile(masses, paths, n_layers)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    except Exception:
        tokenizer = None

    rows = []
    for component in range(components):
        owned = eligible & (owners == component)
        row = {
            "component": component,
            "owner_count": int(owned.sum()),
            "mean_share": round(_mean(gates[eligible, component]), 5),
            "max_share": round(gates[eligible, component].max().item(), 5),
            "mean_loo_gap_damage_owned": round(
                _mean(values["loo_damage"][owned, component]), 5
            ),
            "positive_damage_rate_owned": round(
                _mean((values["loo_damage"][owned, component] > 0.05).float()), 4
            ),
            "context_lag_profile": _binned_profile(
                gates[eligible, component], lags[eligible], [12, 18, 24]
            ),
            "prediction_position_profile": _binned_profile(
                gates[eligible, component], positions[eligible], [19, 23, 27]
            ),
            "credit_module_breadth": _profile(module_profile[component]),
            "credit_layer_breadth": _profile(layer_profile[component]),
            "credit_head_breadth": _profile(head_profile[component].flatten()),
            "piece_mass_layer_breadth": _profile(mass_layers[component]),
            "top_modules": [
                paths[index] for index in module_profile[component].topk(8).indices.tolist()
            ],
            "top_heads": [
                {"layer": index // n_heads, "head": index % n_heads}
                for index in head_profile[component].flatten().topk(8).indices.tolist()
            ],
            "top_examples": _decode_examples(
                pairs, gates, eligible, component, args.top_examples, tokenizer
            ),
        }
        broad_context = row["context_lag_profile"]["effective_count"] >= 3
        broad_layers = row["credit_layer_breadth"]["effective_count"] >= 8
        broad_heads = row["credit_head_breadth"]["effective_count"] >= 24
        row["breadth_descriptors"] = {
            "broad_lag_support": broad_context,
            "broad_layer_credit": broad_layers,
            "broad_head_credit": broad_heads,
            "broad_on_all_measured_axes": broad_context and broad_layers and broad_heads,
        }
        rows.append(row)

    active_ids = [
        component for component in range(components)
        if (eligible & (owners == component)).any()
    ]
    causal_order = sorted(
        active_ids,
        key=lambda component: _mean(
            values["loo_damage"][eligible & (owners == component), component]
        ),
        reverse=True,
    )
    audit_ids = causal_order[:args.audit_components]
    selected_rows = eligible.nonzero(as_tuple=False).flatten()[:args.layer_examples]
    layer_pairs = MatchedBatch(
        pairs.positive.index_select(0, selected_rows).to(device),
        pairs.negative.index_select(0, selected_rows).to(device),
        pairs.positions.index_select(0, selected_rows).to(device),
        pairs.labels.index_select(0, selected_rows).to(device),
    )
    layer_gates = gates.index_select(0, selected_rows).to(device)
    layer_base_gap = values["routed_gap"].index_select(0, selected_rows).to(device)
    layer_audit = {}
    for component in audit_ids:
        damage = _layer_piece_damage(
            target, editor, layer_pairs, layer_gates, layer_base_gap, component,
            paths, n_layers, score_type, args.layer_chunk, args.use_bf16,
        ).cpu()
        mean_damage = damage.mean(0)
        order = mean_damage.abs().argsort(descending=True).tolist()
        layer_audit[str(component)] = {
            "mean_damage_by_layer": [round(value, 5) for value in mean_damage.tolist()],
            "effective_causal_layers": round(
                effective_components(mean_damage.abs()).item(), 4
            ),
            "top_layers": [
                {"layer": layer, "mean_gap_damage": round(mean_damage[layer].item(), 5)}
                for layer in order[:8]
            ],
        }

    active = [row for row in rows if row["owner_count"] > 0]
    report = _safe({
        "artifact": str(args.artifact),
        "eligible_count": int(eligible.sum()),
        "components": components,
        "components_owning_any_example": len(active),
        "interpretation_warning": (
            "Random-token induction supports lag/position and mechanistic breadth audits, "
            "not semantic labels. Balance pressure can create artificial context partitions."
        ),
        "summary": {
            "owner_counts": {str(row["component"]): row["owner_count"] for row in active},
            "causal_owner_components": audit_ids,
            "broad_context_components": [
                row["component"] for row in rows
                if row["breadth_descriptors"]["broad_lag_support"]
            ],
        },
        "components_detail": rows,
        "causal_layer_piece_audit": layer_audit,
    })
    output = args.artifact / (
        "absorption_audit_bf16.json" if args.use_bf16 else "absorption_audit_fp32.json"
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print(f"saved {output}", flush=True)
    editor.restore()


if __name__ == "__main__":
    main()
