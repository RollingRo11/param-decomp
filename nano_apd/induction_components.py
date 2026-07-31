"""Shared math for task-conditioned, cross-layer induction components.

The target weight at module ``m`` is represented exactly as

    W_m = R_m + sum_c P_{m,c}

where ``R_m`` is implicit.  A routed forward uses
``R_m + sum_c gate_c P_{m,c}``, so all-one gates are the untouched model and
all-zero gates are the residual.  Component index ``c`` is shared across every
selected transformer matrix; it is therefore one cross-layer object rather than
an unrelated per-matrix atom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.carving import CapturedUsage, bank_for


def sum_normalized_gates(credits: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Convert signed coherent credits to sum-normalized attribution gates.

    Squaring happens only after contributions have been summed across modules.
    The returned attribution is pre-normalization and the gate rows sum to one,
    including the exactly-zero initialization (which falls back to uniform).
    """
    attribution = credits.float().square()
    gates = (attribution + eps) / (
        attribution.sum(-1, keepdim=True) + eps * attribution.shape[-1]
    )
    return gates.to(credits.dtype), attribution


def effective_components(values: Tensor, eps: float = 1e-12) -> Tensor:
    """Participation ratio along the final dimension."""
    values = values.float().clamp_min(0)
    return values.sum(-1).square() / values.square().sum(-1).clamp_min(eps)


def selected_distribution_kl(edited: Tensor, target: Tensor) -> Tensor:
    """Forward KL(target || edited) for selected-position logits [B, vocab]."""
    target_lp = F.log_softmax(target.float().detach(), -1)
    edited_lp = F.log_softmax(edited.float(), -1)
    return (target_lp.exp() * (target_lp - edited_lp)).sum(-1)


def component_module_credits(
    target: nn.Module,
    banks: nn.ModuleDict,
    module_paths: list[str],
    usage: CapturedUsage,
) -> Tensor:
    """Signed credit before cross-module aggregation, shape [B, C, M]."""
    rows = []
    for path in module_paths:
        linear = target.get_submodule(path)
        A, B = bank_for(banks, path).factors(linear.weight.detach().t())
        read = torch.einsum("btd,cdr->btcr", usage.pre[path], A)
        write = torch.einsum("cro,bto->btcr", B, usage.gpost[path])
        rows.append((read * write).sum((1, 3)))
    if not rows:
        raise ValueError("component_module_credits needs at least one module")
    return torch.stack(rows, -1)


def target_module_credits(
    target: nn.Module, module_paths: list[str], usage: CapturedUsage
) -> Tensor:
    """Gradient-times-weight target credit for each example and module [B, M]."""
    rows = []
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        rows.append(torch.einsum(
            "bti,io,bto->b", usage.pre[path], weight, usage.gpost[path]
        ))
    if not rows:
        raise ValueError("target_module_credits needs at least one module")
    return torch.stack(rows, -1)


def attention_head_credits(
    target: nn.Module,
    module_paths: list[str],
    usage: CapturedUsage,
    n_heads: int,
) -> tuple[Tensor, list[str]]:
    """Credit at attention-head outputs, using inputs to each attention dense.

    For GPT-NeoX, the input dimension of ``attention.dense`` is the concatenation
    of head outputs.  Backpropagating the captured post-linear gradient through
    the dense weight gives an exact first-order credit for each head slice.
    Returns [B, layers, heads] and the corresponding dense module paths.
    """
    dense_paths = [path for path in module_paths if path.endswith("attention.dense")]
    layers = []
    for path in dense_paths:
        pre = usage.pre[path]
        weight = target.get_submodule(path).weight.detach()  # [out, in]
        grad_in = torch.einsum("bto,oi->bti", usage.gpost[path], weight)
        if pre.shape[-1] % n_heads:
            raise ValueError(f"{path} input width is not divisible by {n_heads}")
        d_head = pre.shape[-1] // n_heads
        credit = (pre * grad_in).reshape(
            pre.shape[0], pre.shape[1], n_heads, d_head
        ).sum((1, 3))
        layers.append(credit)
    if not layers:
        raise ValueError("no attention.dense modules found")
    return torch.stack(layers, 1), dense_paths


def component_attention_head_credits(
    target: nn.Module,
    banks: nn.ModuleDict,
    module_paths: list[str],
    usage: CapturedUsage,
    n_heads: int,
) -> tuple[Tensor, list[str]]:
    """Per-component credit through attention-head output coordinates.

    Returns [B, C, layers, heads].  Summing heads and layers gives the part of
    the component's coherent credit carried by attention output projections.
    """
    dense_paths = [path for path in module_paths if path.endswith("attention.dense")]
    layers = []
    for path in dense_paths:
        linear = target.get_submodule(path)
        weight = linear.weight.detach().t()
        A, B = bank_for(banks, path).factors(weight)
        pre = usage.pre[path]
        gp = usage.gpost[path]
        if pre.shape[-1] % n_heads:
            raise ValueError(f"{path} input width is not divisible by {n_heads}")
        d_head = pre.shape[-1] // n_heads
        # Coordinate-wise expansion of <g, h A B>, grouped by input head.
        write = torch.einsum("cro,bto->btcr", B, gp)
        coord = torch.einsum("bti,cir,btcr->btci", pre, A, write)
        credit = coord.reshape(
            pre.shape[0], pre.shape[1], A.shape[0], n_heads, d_head
        ).sum((1, 4))
        layers.append(credit)
    if not layers:
        raise ValueError("no attention.dense modules found")
    return torch.stack(layers, 2), dense_paths


def component_piece_masses(
    target: nn.Module, banks: nn.ModuleDict, module_paths: list[str]
) -> Tensor:
    """Squared Frobenius masses [C, M] without materializing component matrices."""
    rows = []
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        rows.append(bank_for(banks, path).piece_sq_mass(weight).float())
    return torch.stack(rows, -1)


def cross_layer_component_norms(piece_sq_masses: Tensor) -> Tensor:
    """Frobenius norm of each whole cross-layer component.

    Moving fixed mass between layers leaves this value unchanged, so the regularizer
    does not mistake distributed implementation for polysemanticity.
    """
    if piece_sq_masses.ndim != 2:
        raise ValueError("piece_sq_masses must have shape [components, modules]")
    return piece_sq_masses.sum(-1).clamp_min(1e-16).sqrt()


def sum_norm_loss(
    target: nn.Module, banks: nn.ModuleDict, module_paths: list[str]
) -> Tensor:
    """Sum-norm sparsity over whole cross-layer components, not layer pieces."""
    masses = component_piece_masses(target, banks, module_paths)
    numerator = cross_layer_component_norms(masses).sum()
    target_sq_mass = torch.stack([
        target.get_submodule(path).weight.detach().float().square().sum()
        for path in module_paths
    ]).sum()
    return numerator / target_sq_mass.clamp_min(1e-12).sqrt()


def rotating_components(step: int, components: int, count: int) -> Tensor:
    """Deterministically cover every component without making seed a hyperparameter."""
    count = min(max(1, count), components)
    start = (step * count) % components
    return (torch.arange(count) + start) % components


@dataclass(frozen=True)
class CoalitionBatch:
    component_ids: Tensor
    owner_rows: Tensor
    gates_on: Tensor
    gates_off: Tensor
    owner_shares: Tensor
    intact_leave_one_out: Tensor


def stochastic_coalitions(
    gates: Tensor,
    eligible: Tensor,
    component_ids: Tensor,
    generator: torch.Generator | None = None,
    full_model_probability: float = 0.5,
) -> CoalitionBatch | None:
    """Pair routed random coalitions with intact-model leave-one-out tests.

    Every on/off pair differs in exactly the named component. Random rows retain the
    learned sum-normalized gates. Full rows use all-one gates on random eligible task
    examples, training task-wide necessity rather than an attribution-owned shortcut.
    """
    if not 0 <= full_model_probability <= 1:
        raise ValueError("full_model_probability must lie in [0, 1]")
    eligible_rows = eligible.nonzero(as_tuple=False).flatten()
    if eligible_rows.numel() == 0:
        return None
    ids = component_ids.to(gates.device)
    eligible_gates = gates.index_select(0, eligible_rows)
    owner_local = eligible_gates.index_select(1, ids).argmax(0)
    owners = eligible_rows.index_select(0, owner_local)
    k = ids.numel()
    # Intact-model necessity is a task-wide claim, so test it on random eligible
    # examples rather than the component's favorite attribution-owned example.
    full = (
        torch.rand(k, 1, device=gates.device, generator=generator)
        < full_model_probability
    )
    if full.any():
        random_local = torch.randint(
            eligible_rows.numel(), (k,), device=gates.device, generator=generator
        )
        random_owners = eligible_rows.index_select(0, random_local)
        owners = torch.where(full.squeeze(-1), random_owners, owners)
    base = gates.index_select(0, owners)
    _, components = base.shape
    rho = torch.rand(k, 1, device=gates.device, generator=generator)
    keep = (
        torch.rand(k, components, device=gates.device, generator=generator) < rho
    ).to(base.dtype)
    arange = torch.arange(k, device=gates.device)
    keep[arange, ids] = 0
    off = base * keep
    on = off.clone()
    on[arange, ids] = base[arange, ids]
    if full.any():
        full_rows = full.squeeze(-1)
        on[full_rows] = 1
        off[full_rows] = 1
        off[full_rows, ids[full_rows]] = 0
    return CoalitionBatch(
        ids, owners, on, off, base[arange, ids], full.squeeze(-1)
    )


def coverage_loss(gates: Tensor, eligible: Tensor, minimum_share: float) -> Tensor:
    """Require every component to own at least one eligible example in the batch."""
    if not eligible.any():
        return gates.sum() * 0
    strongest = gates[eligible].amax(0)
    return F.relu(minimum_share - strongest).square().mean()


def layer_index(path: str) -> int:
    parts = path.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    for part in parts:
        if part.isdigit():
            return int(part)
    raise ValueError(f"cannot infer layer from {path!r}")
