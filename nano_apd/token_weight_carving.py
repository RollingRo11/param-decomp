"""Dictionary-free, token-conditioned carving of existing weight matrices.

For one selected-token contrast, every linear has a low-rank parameter gradient

    G = sum_t delta_t x_t^T.

This module uses the leading singular read/write subspaces of that gradient to project
the *existing* target weight into fixed low-rank pieces.  No component parameters are
trained.  Physical gates satisfy one=intact and zero=remove; sum-normalized attribution
shares are kept separate and are used only for ranking and sparsity diagnostics.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nano_apd.carving import CapturedUsage, selected_logits
from nano_apd.induction_components import layer_index, sum_normalized_gates


@dataclass(frozen=True)
class GradientFactors:
    """Factorization ``G = left @ right.T`` for one module's mean contrast gradient."""

    left: Tensor  # [d_out, samples]
    right: Tensor  # [d_in, samples]

    def validate(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("gradient factors must be matrices")
        if self.left.shape[1] != self.right.shape[1]:
            raise ValueError("gradient factors must have the same sample dimension")


@dataclass(frozen=True)
class CandidateMetadata:
    candidate_id: int
    module_path: str
    layer: int
    module_kind: str
    local_index: int
    geometry: str
    projection: str
    gradient_singular_value: float | None
    active: bool
    weight_coefficient: float
    first_order_credit: float
    piece_frobenius_norm: float


@dataclass(frozen=True)
class ModuleGeometry:
    module_path: str
    singular_values: tuple[float, ...]
    variant_bilinear_capture: dict[str, float]
    gradient_frobenius_norm: float
    active_rank: int
    retained_gradient_energy_fraction: float


@dataclass(frozen=True)
class ExtractionReport:
    module_geometry: tuple[ModuleGeometry, ...]
    total_piece_sum_norm: float
    target_weight_norm: float


class FixedModulePieces(nn.Module):
    """Fixed rank-one pieces ``output[k] input[k]^T`` for one target linear."""

    def __init__(
        self,
        input_directions: Tensor,
        output_directions: Tensor,
        candidate_ids: Tensor,
    ):
        super().__init__()
        if input_directions.ndim != 2 or output_directions.ndim != 2:
            raise ValueError("piece directions must be matrices")
        if input_directions.shape[0] != output_directions.shape[0]:
            raise ValueError("input and output directions need the same piece count")
        if candidate_ids.shape != (input_directions.shape[0],):
            raise ValueError("candidate_ids must have one entry per piece")
        self.register_buffer("input_directions", input_directions)
        self.register_buffer("output_directions", output_directions)
        self.register_buffer("candidate_ids", candidate_ids.long())

    @property
    def pieces(self) -> int:
        return self.input_directions.shape[0]

    def dense(self) -> Tensor:
        return torch.einsum("ko,ki->koi", self.output_directions, self.input_directions)


class FixedPieceCollection(nn.Module):
    """A module-local collection with one global gate index per rank-one piece."""

    def __init__(
        self,
        by_path: dict[str, FixedModulePieces],
        metadata: tuple[CandidateMetadata, ...],
    ):
        super().__init__()
        self.by_path = nn.ModuleDict(
            {path.replace(".", "/"): value for path, value in by_path.items()}
        )
        self.metadata = metadata
        ids = sorted(item.candidate_id for item in metadata)
        if ids != list(range(len(metadata))):
            raise ValueError("candidate ids must be contiguous and globally unique")

    @property
    def candidates(self) -> int:
        return len(self.metadata)

    def for_path(self, path: str) -> FixedModulePieces:
        return self.by_path[path.replace(".", "/")]


@dataclass(frozen=True)
class LoadedPieceCheckpoint:
    """A reconstructed fixed-piece collection and its target-module order."""

    pieces: FixedPieceCollection
    module_paths: tuple[str, ...]


def load_fixed_piece_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> LoadedPieceCheckpoint:
    """Load a ``pieces.pt`` checkpoint written by ``run_token_weight_carving``."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("fixed-piece checkpoint must contain a dictionary")
    try:
        state_dict = payload["state_dict"]
        metadata_rows = payload["metadata"]
        raw_module_paths = payload["module_paths"]
    except KeyError as error:
        raise ValueError(f"fixed-piece checkpoint is missing {error.args[0]!r}") from error
    if not isinstance(state_dict, dict):
        raise ValueError("fixed-piece checkpoint state_dict must be a dictionary")
    if not isinstance(metadata_rows, list) or not all(
        isinstance(row, dict) for row in metadata_rows
    ):
        raise ValueError("fixed-piece checkpoint metadata must be a list of dictionaries")
    if not isinstance(raw_module_paths, (list, tuple)) or not all(
        isinstance(path, str) for path in raw_module_paths
    ):
        raise ValueError("fixed-piece checkpoint module_paths must be a sequence of strings")

    module_paths = tuple(raw_module_paths)
    if len(set(module_paths)) != len(module_paths):
        raise ValueError("fixed-piece checkpoint module_paths must be unique")
    try:
        metadata = tuple(CandidateMetadata(**row) for row in metadata_rows)
    except TypeError as error:
        raise ValueError("fixed-piece checkpoint contains invalid candidate metadata") from error

    by_path: dict[str, FixedModulePieces] = {}
    for path in module_paths:
        prefix = f"by_path.{path.replace('.', '/')}"
        keys = {
            "input": f"{prefix}.input_directions",
            "output": f"{prefix}.output_directions",
            "ids": f"{prefix}.candidate_ids",
        }
        missing = [key for key in keys.values() if key not in state_dict]
        if missing:
            raise ValueError(f"fixed-piece checkpoint is missing tensor {missing[0]!r}")
        module = FixedModulePieces(
            state_dict[keys["input"]],
            state_dict[keys["output"]],
            state_dict[keys["ids"]],
        )
        expected_ids = tuple(
            item.candidate_id
            for item in sorted(
                (item for item in metadata if item.module_path == path),
                key=lambda item: item.local_index,
            )
        )
        if tuple(module.candidate_ids.tolist()) != expected_ids:
            raise ValueError(f"candidate metadata does not match saved tensors for {path!r}")
        by_path[path] = module

    if {item.module_path for item in metadata} != set(module_paths):
        raise ValueError("candidate metadata and module_paths name different modules")
    return LoadedPieceCheckpoint(FixedPieceCollection(by_path, metadata), module_paths)


class TokenCarvingEditor:
    """Apply exact-residual physical gates without constructing per-example weights."""

    def __init__(
        self,
        target: nn.Module,
        pieces: FixedPieceCollection,
        module_paths: list[str],
    ):
        self.target = target
        self.pieces = pieces
        self.module_paths = list(module_paths)
        self.masks: Tensor | None = None
        self._original: dict[str, object] = {}
        for path in self.module_paths:
            linear = target.get_submodule(path)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"{path} is not nn.Linear")
            self._original[path] = linear.forward
            linear.forward = self._make_forward(path, linear)

    def _make_forward(self, path: str, linear: nn.Linear):
        def forward(x: Tensor) -> Tensor:
            if x.ndim != 3:
                raise ValueError(f"token carving expects [batch, token, hidden] at {path}")
            out = F.linear(x, linear.weight, linear.bias)
            if self.masks is None:
                return out
            masks = self.masks
            if masks.ndim == 2:
                masks = masks.unsqueeze(1)
            if masks.ndim != 3 or masks.shape[0] != x.shape[0]:
                raise ValueError(
                    "physical masks must have shape [batch, components] or [batch, token, components]"
                )
            module = self.pieces.for_path(path)
            local_masks = masks.index_select(-1, module.candidate_ids)
            read = torch.einsum("bti,ki->btk", x, module.input_directions)
            removal = read * (1.0 - local_masks)
            return out - torch.einsum("btk,ko->bto", removal, module.output_directions)

        return forward

    def restore(self) -> None:
        for path, forward in self._original.items():
            self.target.get_submodule(path).forward = forward
        self._original.clear()


@dataclass(frozen=True)
class IntegratedPathAttribution:
    contributions: Tensor
    start_scores: Tensor
    end_scores: Tensor
    completeness_error: Tensor


def fixed_margin_scores(
    logits: Tensor,
    positions: Tensor,
    labels: Tensor,
    distractors: Tensor,
) -> Tensor:
    """Selected target logit minus a fixed, matched distractor logit."""
    chosen = selected_logits(logits, positions).float()
    target = chosen.gather(-1, labels[:, None]).squeeze(-1)
    distractor = chosen.gather(-1, distractors[:, None]).squeeze(-1)
    return target - distractor


def capture_fixed_margin_usage(
    target: nn.Module,
    editor,
    tokens: Tensor,
    positions: Tensor,
    labels: Tensor,
    distractors: Tensor,
) -> CapturedUsage:
    """Capture one score gradient per sequence using a fixed smooth readout."""
    if labels.shape != positions.shape or distractors.shape != positions.shape:
        raise ValueError("positions, labels, and distractors must have shape [batch]")
    editor.masks = None
    editor.start_capture()
    logits = target(tokens)
    scores = fixed_margin_scores(logits, positions, labels, distractors)
    posts = [editor.cache[path]["post"] for path in editor.module_paths]
    grads = torch.autograd.grad(scores.sum(), posts, allow_unused=True)
    pre = {path: editor.cache[path]["pre"].detach() for path in editor.module_paths}
    gpost = {
        path: grad.detach() if grad is not None else torch.zeros_like(posts[index])
        for index, (path, grad) in enumerate(zip(editor.module_paths, grads, strict=True))
    }
    predictions = selected_logits(logits.detach(), positions).argmax(-1)
    editor.stop_capture()
    editor.cache = {}
    return CapturedUsage(pre, gpost, scores.detach(), predictions)


def contrast_gradient_factors(
    positive: CapturedUsage,
    negative: CapturedUsage,
    path: str,
) -> GradientFactors:
    """Factor the mean clean-minus-corrupt parameter gradient without materializing it."""
    if positive.pre[path].shape[0] != negative.pre[path].shape[0]:
        raise ValueError("positive and negative usage batches must match")
    batch = positive.pre[path].shape[0]
    scale = batch**-0.5
    positive_delta = positive.gpost[path].float().reshape(-1, positive.gpost[path].shape[-1])
    negative_delta = negative.gpost[path].float().reshape(-1, negative.gpost[path].shape[-1])
    positive_input = positive.pre[path].float().reshape(-1, positive.pre[path].shape[-1])
    negative_input = negative.pre[path].float().reshape(-1, negative.pre[path].shape[-1])
    factors = GradientFactors(
        torch.cat([positive_delta.T, -negative_delta.T], -1) * scale,
        torch.cat([positive_input.T, negative_input.T], -1) * scale,
    )
    factors.validate()
    return factors


def concatenate_gradient_factors(factors: list[GradientFactors]) -> GradientFactors:
    """Average equally weighted prompt-family gradients in factorized form."""
    if not factors:
        raise ValueError("at least one gradient factorization is required")
    family_scale = len(factors) ** -0.5
    result = GradientFactors(
        torch.cat([item.left * family_scale for item in factors], -1),
        torch.cat([item.right * family_scale for item in factors], -1),
    )
    result.validate()
    return result


def factorized_matmul_right(factors: GradientFactors, value: Tensor) -> Tensor:
    """Compute ``G @ value`` for ``G = left @ right.T``."""
    return factors.left @ (factors.right.T @ value)


def factorized_matmul_left(factors: GradientFactors, value: Tensor) -> Tensor:
    """Compute ``G.T @ value`` for ``G = left @ right.T``."""
    return factors.right @ (factors.left.T @ value)


def factorized_inner_product(
    factors: GradientFactors, output_direction: Tensor, input_direction: Tensor
) -> Tensor:
    return (factors.left.T @ output_direction * (factors.right.T @ input_direction)).sum()


def factorized_frobenius_sq(factors: GradientFactors) -> Tensor:
    """Exact Frobenius mass using only the sample-sized Gram matrices."""
    left_gram = factors.left.T @ factors.left
    right_gram = factors.right.T @ factors.right
    return (left_gram * right_gram).sum()


def rank_one_frobenius_norms(
    input_directions: Tensor, output_directions: Tensor
) -> Tensor:
    """Frobenius norms of outer products without materializing dense matrices."""
    if input_directions.ndim != 2 or output_directions.ndim != 2:
        raise ValueError("rank-one directions must be matrices")
    if input_directions.shape[0] != output_directions.shape[0]:
        raise ValueError("rank-one directions need the same piece count")
    return torch.linalg.vector_norm(input_directions, dim=1) * torch.linalg.vector_norm(
        output_directions, dim=1
    )


def randomized_factor_svd(
    factors: GradientFactors,
    rank: int,
    oversample: int,
    power_iterations: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Randomized truncated SVD of a factorized gradient."""
    factors.validate()
    d_out, d_in = factors.left.shape[0], factors.right.shape[0]
    if rank < 1 or rank > min(d_out, d_in):
        raise ValueError("rank must lie within the matrix dimensions")
    q = min(rank + max(0, oversample), d_out, d_in)
    omega = torch.randn(d_in, q, device=factors.left.device, generator=generator)
    q_out = torch.linalg.qr(factorized_matmul_right(factors, omega), mode="reduced").Q
    for _ in range(power_iterations):
        q_in = torch.linalg.qr(factorized_matmul_left(factors, q_out), mode="reduced").Q
        q_out = torch.linalg.qr(factorized_matmul_right(factors, q_in), mode="reduced").Q
    small = (q_out.T @ factors.left) @ factors.right.T
    small_u, singular, vh = torch.linalg.svd(small, full_matrices=False)
    return q_out @ small_u[:, :rank], singular[:rank], vh[:rank].T


def diagonal_geometry_scales(
    usages: list[tuple[CapturedUsage, CapturedUsage]],
    path: str,
    damping: float,
) -> tuple[Tensor, Tensor]:
    if damping <= 0:
        raise ValueError("diagonal geometry damping must be positive")
    if not usages:
        raise ValueError("at least one usage pair is required")
    input_squares = []
    output_squares = []
    for positive, negative in usages:
        input_squares.extend(
            [positive.pre[path].float().square(), negative.pre[path].float().square()]
        )
        output_squares.extend(
            [positive.gpost[path].float().square(), negative.gpost[path].float().square()]
        )
    input_second = torch.stack([value.mean((0, 1)) for value in input_squares]).mean(0)
    output_second = torch.stack([value.mean((0, 1)) for value in output_squares]).mean(0)
    input_scale = (input_second + damping * input_second.mean()).clamp_min(1e-20).sqrt()
    output_scale = (output_second + damping * output_second.mean()).clamp_min(1e-20).sqrt()
    return input_scale, output_scale


def _module_kind(path: str) -> str:
    if path.endswith("attention.query_key_value"):
        return "attention_qkv"
    if path.endswith("attention.dense"):
        return "attention_output"
    if path.endswith("mlp.dense_h_to_4h"):
        return "mlp_input"
    if path.endswith("mlp.dense_4h_to_h"):
        return "mlp_output"
    return path.rsplit(".", 1)[-1]


def extract_fixed_pieces(
    target: nn.Module,
    module_paths: list[str],
    usages_by_variant: dict[str, tuple[CapturedUsage, CapturedUsage]],
    rank: int,
    geometry: str,
    projection: str,
    damping: float,
    seed: int,
) -> tuple[FixedPieceCollection, ExtractionReport]:
    """Extract analytic rank-one atoms from every selected target matrix."""
    if geometry not in {"euclidean", "diag_kfac"}:
        raise ValueError("geometry must be 'euclidean' or 'diag_kfac'")
    if projection not in {"paired", "cartesian"}:
        raise ValueError("projection must be 'paired' or 'cartesian'")
    if not usages_by_variant:
        raise ValueError("at least one discovery variant is required")

    by_path: dict[str, FixedModulePieces] = {}
    metadata: list[CandidateMetadata] = []
    geometry_rows: list[ModuleGeometry] = []
    total_piece_sum_norm = 0.0
    target_weight_sq = 0.0
    generator = torch.Generator(device=next(target.parameters()).device).manual_seed(seed)

    for path in module_paths:
        variant_factors = {
            name: contrast_gradient_factors(positive, negative, path)
            for name, (positive, negative) in usages_by_variant.items()
        }
        factors = concatenate_gradient_factors(list(variant_factors.values()))
        if geometry == "diag_kfac":
            input_scale, output_scale = diagonal_geometry_scales(
                list(usages_by_variant.values()), path, damping
            )
        else:
            linear = target.get_submodule(path)
            input_scale = torch.ones(linear.in_features, device=linear.weight.device)
            output_scale = torch.ones(linear.out_features, device=linear.weight.device)

        whitened = GradientFactors(
            factors.left / output_scale[:, None],
            factors.right / input_scale[:, None],
        )
        u, singular, v = randomized_factor_svd(
            whitened, rank, oversample=4, power_iterations=2, generator=generator
        )
        singular_threshold = torch.maximum(singular.new_tensor(1e-12), singular.amax() * 1e-6)
        active_modes = singular > singular_threshold
        u = u * active_modes[None, :]
        v = v * active_modes[None, :]
        retained_singular = singular * active_modes
        weight = target.get_submodule(path).weight.detach().float()
        whitened_weight = output_scale[:, None] * weight * input_scale[None, :]
        core = u.T @ whitened_weight @ v

        if projection == "paired":
            coefficients = core.diagonal()
            white_output = u * coefficients[None, :]
            white_input = v
            mode_singular = [float(value) for value in singular.tolist()]
        else:
            core_u, coefficients, core_vh = torch.linalg.svd(core, full_matrices=False)
            white_output = (u @ core_u) * coefficients[None, :]
            white_input = v @ core_vh.T
            mode_singular = [None] * rank

        input_directions = (white_input / input_scale[:, None]).T.contiguous()
        output_directions = (white_output / output_scale[:, None]).T.contiguous()
        ids = torch.arange(len(metadata), len(metadata) + rank, device=weight.device)
        by_path[path] = FixedModulePieces(input_directions, output_directions, ids)

        actual_credits = torch.stack(
            [
                factorized_inner_product(factors, output_directions[index], input_directions[index])
                for index in range(rank)
            ]
        )
        expected_total = (retained_singular * core.diagonal()).sum()
        torch.testing.assert_close(actual_credits.sum(), expected_total, rtol=5e-3, atol=5e-4)
        piece_norms = rank_one_frobenius_norms(
            input_directions.float(), output_directions.float()
        )
        total_piece_sum_norm += piece_norms.sum().item()
        target_weight_sq += weight.square().sum().item()

        projected_variant_capture = {}
        for name, variant in variant_factors.items():
            white_variant = GradientFactors(
                variant.left / output_scale[:, None],
                variant.right / input_scale[:, None],
            )
            small = (u.T @ white_variant.left) @ (white_variant.right.T @ v)
            denominator = factorized_frobenius_sq(white_variant).clamp_min(1e-20)
            if projection == "paired":
                numerator = small.diagonal().square().sum()
            else:
                rotated_small = core_u.T @ small @ core_vh.T
                numerator = rotated_small.diagonal().square().sum()
            projected_variant_capture[name] = (numerator / denominator).item()

        gradient_frobenius_sq = factorized_frobenius_sq(whitened)
        retained_energy = retained_singular.square().sum() / gradient_frobenius_sq.clamp_min(1e-20)

        geometry_rows.append(
            ModuleGeometry(
                module_path=path,
                singular_values=tuple(float(value) for value in singular.tolist()),
                variant_bilinear_capture=projected_variant_capture,
                gradient_frobenius_norm=float(gradient_frobenius_sq.sqrt().item()),
                active_rank=int(active_modes.sum()),
                retained_gradient_energy_fraction=float(retained_energy),
            )
        )
        for local_index in range(rank):
            metadata.append(
                CandidateMetadata(
                    candidate_id=int(ids[local_index]),
                    module_path=path,
                    layer=layer_index(path),
                    module_kind=_module_kind(path),
                    local_index=local_index,
                    geometry=geometry,
                    projection=projection,
                    gradient_singular_value=mode_singular[local_index],
                    active=bool(piece_norms[local_index] > 0),
                    weight_coefficient=float(coefficients[local_index]),
                    first_order_credit=float(actual_credits[local_index]),
                    piece_frobenius_norm=float(piece_norms[local_index]),
                )
            )

    collection = FixedPieceCollection(by_path, tuple(metadata))
    report = ExtractionReport(tuple(geometry_rows), total_piece_sum_norm, target_weight_sq**0.5)
    return collection, report


def contrast_candidate_credits(
    pieces: FixedPieceCollection,
    positive: CapturedUsage,
    negative: CapturedUsage,
) -> tuple[Tensor, Tensor]:
    """Return per-example signed credits [B,C] and per-position credits [B,C,T]."""
    first_path = pieces.metadata[0].module_path
    batch, tokens = positive.pre[first_path].shape[:2]
    credits = torch.zeros(batch, pieces.candidates, device=positive.pre[first_path].device)
    positions = torch.zeros(batch, pieces.candidates, tokens, device=credits.device)
    for path in {item.module_path for item in pieces.metadata}:
        module = pieces.for_path(path)
        positive_read = torch.einsum(
            "bti,ki->btk",
            positive.pre[path].float(),
            module.input_directions.float(),
        )
        positive_write = torch.einsum(
            "bto,ko->btk",
            positive.gpost[path].float(),
            module.output_directions.float(),
        )
        negative_read = torch.einsum(
            "bti,ki->btk",
            negative.pre[path].float(),
            module.input_directions.float(),
        )
        negative_write = torch.einsum(
            "bto,ko->btk",
            negative.gpost[path].float(),
            module.output_directions.float(),
        )
        local_positions = positive_read * positive_write - negative_read * negative_write
        positions[:, module.candidate_ids] = local_positions.transpose(1, 2)
        credits[:, module.candidate_ids] = local_positions.sum(1)
    return credits, positions


def attribution_shares(credits: Tensor) -> tuple[Tensor, Tensor]:
    """Sum-normalized squared credit shares; these are not physical gates."""
    return sum_normalized_gates(credits)


def gauss_legendre_unit_interval(steps: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Gauss-Legendre nodes and weights on [0, 1] without a NumPy dependency."""
    if steps < 1:
        raise ValueError("integration steps must be positive")
    if steps == 1:
        return torch.tensor([0.5], device=device), torch.tensor([1.0], device=device)
    index = torch.arange(1, steps, device=device, dtype=torch.float64)
    off_diagonal = index / torch.sqrt(4 * index.square() - 1)
    jacobi = torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(jacobi)
    nodes = (eigenvalues + 1) / 2
    weights = eigenvectors[0].square()
    return nodes.float(), weights.float()


def integrate_gate_path(
    score_fn: Callable[[Tensor], Tensor],
    start: Tensor,
    end: Tensor,
    steps: int,
) -> IntegratedPathAttribution:
    """Integrate per-gate derivatives along a straight physical-gate path."""
    if start.shape != end.shape or start.ndim != 2:
        raise ValueError("path endpoints must have matching [batch, candidates] shape")
    nodes, weights = gauss_legendre_unit_interval(steps, start.device)
    delta = end - start
    contributions = torch.zeros_like(start, dtype=torch.float32)
    for node, weight in zip(nodes, weights, strict=True):
        gates = (start + node * delta).detach().requires_grad_(True)
        scores = score_fn(gates)
        if scores.shape != (start.shape[0],):
            raise ValueError("score_fn must return one scalar per gate row")
        gradient = torch.autograd.grad(scores.sum(), gates)[0]
        contributions += weight * gradient.float() * delta.float()
    with torch.no_grad():
        start_scores = score_fn(start).float()
        end_scores = score_fn(end).float()
    endpoint_difference = end_scores - start_scores
    completeness = contributions.sum(-1) - endpoint_difference
    return IntegratedPathAttribution(contributions, start_scores, end_scores, completeness)


def paired_random_coalitions(
    candidates: int,
    candidate_ids: Tensor,
    samples_per_candidate: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """Paired on/off coalitions differing only in the named candidate."""
    if samples_per_candidate < 1:
        raise ValueError("samples_per_candidate must be positive")
    ids = candidate_ids.repeat_interleave(samples_per_candidate)
    rows = ids.numel()
    rho = torch.rand(rows, 1, device=ids.device, generator=generator)
    base = torch.ones(rows, candidates, device=ids.device)
    unique_ids = candidate_ids.unique(sorted=True)
    subset = (
        torch.rand(rows, unique_ids.numel(), device=ids.device, generator=generator) < rho
    ).float()
    base[:, unique_ids] = subset
    arange = torch.arange(rows, device=ids.device)
    off = base.clone()
    off[arange, ids] = 0
    on = off.clone()
    on[arange, ids] = 1
    return ids, on, off
