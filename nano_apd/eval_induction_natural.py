"""Measure whether induction components are also broad natural-text machinery.

The controlled induction interventions in ``eval_induction_roles`` establish task
consistency, but they cannot by themselves rule out a component that simply carries
generally useful language-model computation.  This evaluator removes each component
from the otherwise intact model on held-out Pile positions and measures distributional
KL, next-token log-probability damage, and prediction flips.

Layer breadth is intentionally absent from this test.  A component is treated as one
cross-layer object; only its function on task-present, task-absent, and natural inputs
is compared.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from nano_apd.carving import build_banks, selected_logits
from nano_apd.induction_components import selected_distribution_kl
from nano_apd.induction_editor import InductionEditor
from nano_apd.lm_target import load_carving_target
from nano_param_decomp.pile_4L import make_loader


def _autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def _mean(values: Tensor) -> float:
    return round(values.detach().float().mean().item(), 7)


def _bootstrap_interval(
    values: Tensor, seed: int, samples: int = 1000
) -> list[float]:
    values = values.detach().float().cpu()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[indices].mean(-1)
    return [
        round(torch.quantile(means, 0.025).item(), 7),
        round(torch.quantile(means, 0.975).item(), 7),
    ]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    value = numerator / max(denominator, 1e-9)
    return round(value, 4) if math.isfinite(value) else None


def _load_natural_batch(args, seq_len: int) -> tuple[Tensor, str]:
    try:
        loader = make_loader(
            args.n_sequences, seq_len, 0, 1, "val", args.seed
        )
        return next(loader), "val"
    except Exception:
        if not args.allow_train_fallback:
            raise
        loader = make_loader(
            args.n_sequences, seq_len, 0, 1, "train", args.seed + 8_000_000
        )
        return next(loader), "train-fallback"


def _induction_kl(artifact: Path, suffix: str, components: int) -> list[dict] | None:
    path = artifact / f"induction_roles_{suffix}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    rows = report.get("component_functional_signatures", [])
    if len(rows) != components:
        raise ValueError(f"{path} has {len(rows)} components, expected {components}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--n_sequences", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--component_chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=271_003)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--allow_train_fallback", action="store_true")
    args = parser.parse_args()
    if min(args.n_sequences, args.batch_size, args.component_chunk) < 1:
        raise ValueError("sequence, batch, and component counts must be positive")

    config = json.loads((args.artifact / "config.json").read_text())
    seq_len = int(config["seq_len"])
    components = int(config["C"])
    device = torch.device("cuda")
    target = load_carving_target("hf", config["model_name"]).float().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    paths = config["module_paths"]
    banks = build_banks(
        target, paths, components, int(config["rank"]), config["bank_type"]
    ).to(device)
    banks.load_state_dict(torch.load(
        args.artifact / "banks.pt", weights_only=True, map_location=device
    ))
    banks.eval()
    editor = InductionEditor(target, banks, paths)

    natural, split = _load_natural_batch(args, seq_len)
    if natural.shape[0] != args.n_sequences:
        raise RuntimeError(
            f"natural loader returned {natural.shape[0]} rows, expected {args.n_sequences}"
        )
    generator = torch.Generator().manual_seed(args.seed + 1)
    # Vary the selected prediction location across rows while retaining enough prefix
    # for meaningful natural-language context.  labels are the next tokens.
    positions = torch.randint(8, seq_len - 1, (args.n_sequences,), generator=generator)
    labels = natural.gather(1, (positions + 1)[:, None]).squeeze(1)

    storage = {
        key: [[] for _ in range(components)]
        for key in ("kl", "logprob_damage", "prediction_flip")
    }
    residual_kl: list[Tensor] = []
    try:
        for start in range(0, args.n_sequences, args.batch_size):
            stop = min(start + args.batch_size, args.n_sequences)
            tokens = natural[start:stop].to(device)
            selected_positions = positions[start:stop].to(device)
            selected_labels = labels[start:stop].to(device)
            batch = stop - start
            editor.masks = None
            with torch.no_grad(), _autocast(args.use_bf16):
                reference = selected_logits(target(tokens), selected_positions)
            reference_lp = F.log_softmax(reference.float(), -1)
            reference_label_lp = reference_lp.gather(
                -1, selected_labels[:, None]
            ).squeeze(-1)
            reference_prediction = reference.argmax(-1)

            editor.masks = torch.zeros(
                batch, 1, components, device=device, dtype=reference.dtype
            )
            with torch.no_grad(), _autocast(args.use_bf16):
                residual = selected_logits(target(tokens), selected_positions)
            residual_kl.append(selected_distribution_kl(residual, reference).cpu())

            for first_component in range(0, components, args.component_chunk):
                ids = list(range(
                    first_component,
                    min(first_component + args.component_chunk, components),
                ))
                count = len(ids)
                masks = torch.ones(
                    count * batch,
                    components,
                    device=device,
                    dtype=reference.dtype,
                )
                for offset, component in enumerate(ids):
                    masks[offset * batch:(offset + 1) * batch, component] = 0
                editor.masks = masks.unsqueeze(1)
                with torch.no_grad(), _autocast(args.use_bf16):
                    ablated = selected_logits(
                        target(tokens.repeat(count, 1)),
                        selected_positions.repeat(count),
                    ).view(count, batch, -1)
                for offset, component in enumerate(ids):
                    logits = ablated[offset]
                    label_lp = F.log_softmax(logits.float(), -1).gather(
                        -1, selected_labels[:, None]
                    ).squeeze(-1)
                    storage["kl"][component].append(
                        selected_distribution_kl(logits, reference).cpu()
                    )
                    storage["logprob_damage"][component].append(
                        (reference_label_lp - label_lp).cpu()
                    )
                    storage["prediction_flip"][component].append(
                        (logits.argmax(-1) != reference_prediction).float().cpu()
                    )
            editor.masks = None
            print(f"natural rows {stop}/{args.n_sequences}", flush=True)
    finally:
        editor.restore()

    suffix = "bf16" if args.use_bf16 else "fp32"
    induction_rows = _induction_kl(args.artifact, suffix, components)
    rows = []
    for component in range(components):
        kl = torch.cat(storage["kl"][component])
        logprob_damage = torch.cat(storage["logprob_damage"][component])
        flip = torch.cat(storage["prediction_flip"][component])
        row = {
            "component": component,
            "natural_selected_kl": _mean(kl),
            "natural_selected_kl_ci95": _bootstrap_interval(
                kl, args.seed + component
            ),
            "natural_next_token_logprob_damage": _mean(logprob_damage),
            "natural_abs_next_token_logprob_damage": _mean(logprob_damage.abs()),
            "natural_prediction_flip_rate": _mean(flip),
        }
        if induction_rows is not None:
            induction_kl = float(
                induction_rows[component]["present_variant_mean_selected_kl"]
            )
            control_kl = float(
                induction_rows[component]["absent_control_mean_selected_kl"]
            )
            row |= {
                "induction_present_selected_kl": induction_kl,
                "synthetic_absent_control_selected_kl": control_kl,
                "induction_to_natural_kl_ratio": _safe_ratio(
                    induction_kl, row["natural_selected_kl"]
                ),
                "induction_to_max_control_kl_ratio": _safe_ratio(
                    induction_kl,
                    max(control_kl, row["natural_selected_kl"]),
                ),
            }
        rows.append(row)

    by_natural = sorted(rows, key=lambda row: row["natural_selected_kl"], reverse=True)
    report = {
        "artifact": str(args.artifact),
        "purpose": (
            "Check whether intact-model component ablations broadly perturb held-out "
            "natural text, which would be evidence against induction specificity."
        ),
        "cross_layer_policy": (
            "Each component is ablated as one cross-layer object; layer breadth is not "
            "penalized or treated as polysemanticity."
        ),
        "natural_split": split,
        "n_sequences": args.n_sequences,
        "position_sampling": "uniform prediction positions 8 through seq_len-2",
        "precision": suffix,
        "residual_natural_selected_kl": _mean(torch.cat(residual_kl)),
        "component_summary": {
            "median_natural_selected_kl": round(torch.tensor([
                row["natural_selected_kl"] for row in rows
            ]).median().item(), 7),
            "max_natural_selected_kl": by_natural[0]["natural_selected_kl"],
            "max_natural_impact_component": by_natural[0]["component"],
        },
        "components": rows,
        "interpretation_guardrail": (
            "Low average natural-text impact plus high controlled induction impact is "
            "evidence of task selectivity, not a proof of full natural-text "
            "monosemanticity; rare unmeasured functions may remain."
        ),
    }
    output = args.artifact / f"induction_natural_{suffix}.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
