"""Reusable primitives for contrastive, selected-token parameter carving.

A carving run learns a small bank of low-rank pieces for one supervised behavior and
leaves the rest of every target weight as an implicit residual. Unlike ``targeted.py``,
the bank is also supervised by how the *unedited target* uses its weights for matched
positive/negative examples at one chosen prediction position.

The module deliberately separates reusable math from the pile-4L command-line trainer in
``train_carving.py`` so the important invariants can be tested on tiny CPU models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

BankKind = Literal["free", "projected"]


@dataclass(frozen=True)
class MatchedBatch:
    """Matched contexts with one supervised next-token position per sequence."""

    positive: Tensor  # [B, T]
    negative: Tensor  # [B, T]
    positions: Tensor  # [B], logit position which predicts ``labels``
    labels: Tensor  # [B]

    def validate(self) -> None:
        if self.positive.ndim != 2 or self.negative.shape != self.positive.shape:
            raise ValueError("positive and negative must both have shape [B, T]")
        batch, length = self.positive.shape
        if self.positions.shape != (batch,) or self.labels.shape != (batch,):
            raise ValueError("positions and labels must both have shape [B]")
        if ((self.positions < 0) | (self.positions >= length - 1)).any():
            raise ValueError("positions must predict an in-sequence next token")
        rows = torch.arange(batch, device=self.positive.device)
        if not torch.equal(self.positive[rows, self.positions + 1], self.labels):
            raise ValueError("positive labels do not match the selected next tokens")
        if not torch.equal(self.negative[rows, self.positions + 1], self.labels):
            raise ValueError("negative labels do not match the selected next tokens")

    def to(self, device: torch.device | str) -> MatchedBatch:
        return MatchedBatch(*(x.to(device) for x in (
            self.positive, self.negative, self.positions, self.labels
        )))


def make_matched_induction_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device | str,
    seed: int,
) -> MatchedBatch:
    """Create a controlled induction contrast with the same local context and label.

    Positive and negative sequences differ at one earlier token only. In the positive,
    an earlier ``cue -> label`` bigram makes the selected later ``cue -> label`` prediction
    copyable. In the negative, the earlier cue predicts a distractor. The selected cue,
    selected label, and all local context are identical, preventing a token-identity edit
    from satisfying the contrast by itself.
    """
    if batch_size < 1 or seq_len < 8 or vocab_size < 4:
        raise ValueError("need batch_size>=1, seq_len>=8, and vocab_size>=4")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    base = torch.randint(0, vocab_size, (batch_size, seq_len), device=device,
                         generator=generator)
    rows = torch.arange(batch_size, device=device)
    source = torch.randint(1, max(2, seq_len // 3), (batch_size,), device=device,
                           generator=generator)
    position = torch.randint(max(seq_len // 2, 3), seq_len - 1, (batch_size,),
                             device=device, generator=generator)
    cue = torch.randint(0, vocab_size, (batch_size,), device=device, generator=generator)
    label = torch.randint(0, vocab_size, (batch_size,), device=device, generator=generator)
    offset = torch.randint(1, vocab_size, (batch_size,), device=device, generator=generator)
    distractor = (label + offset) % vocab_size

    positive = base.clone()
    positive[rows, source] = cue
    positive[rows, source + 1] = label
    positive[rows, position] = cue
    positive[rows, position + 1] = label
    negative = positive.clone()
    negative[rows, source + 1] = distractor
    batch = MatchedBatch(positive, negative, position, label)
    batch.validate()
    return batch


def load_matched_batch(path: str | Path, device: torch.device | str) -> MatchedBatch:
    """Load ``positive``, ``negative``, ``positions``, and optional ``labels`` tensors."""
    obj = torch.load(path, weights_only=True, map_location=device)
    positive = obj["positive"].long()
    negative = obj["negative"].long()
    positions = obj["positions"].long()
    rows = torch.arange(positive.shape[0], device=positive.device)
    labels = obj.get("labels", positive[rows, positions + 1]).long()
    batch = MatchedBatch(positive, negative, positions, labels)
    batch.validate()
    return batch


def selected_logits(logits: Tensor, positions: Tensor) -> Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, positions]


def selected_log_probs(logits: Tensor, positions: Tensor, labels: Tensor) -> Tensor:
    return F.log_softmax(selected_logits(logits, positions).float(), -1).gather(
        -1, labels[:, None]
    ).squeeze(-1)


def selected_scores(
    logits: Tensor,
    positions: Tensor,
    labels: Tensor,
    kind: Literal["logit_margin", "target_logit", "logprob"] = "logit_margin",
) -> Tensor:
    """Scalar whose target-weight gradient defines selected-token usage.

    Logit margin avoids the vanishing gradient of a saturated correct-token log-probability.
    Log-probability remains available for compatibility and output reporting.
    """
    if kind == "logprob":
        return selected_log_probs(logits, positions, labels)
    chosen = selected_logits(logits, positions).float()
    correct = chosen.gather(-1, labels[:, None]).squeeze(-1)
    if kind == "target_logit":
        return correct
    if kind != "logit_margin":
        raise ValueError(f"unknown selected score kind {kind!r}")
    competitors = chosen.scatter(-1, labels[:, None], float("-inf"))
    return correct - competitors.amax(-1)


def selected_accuracy(logits: Tensor, positions: Tensor, labels: Tensor) -> Tensor:
    return (selected_logits(logits, positions).argmax(-1) == labels).float().mean()


def kl_per_position(edited: Tensor, target: Tensor) -> Tensor:
    """Forward KL(target || edited), retaining all leading token dimensions.

    Computing target probabilities and then taking their logarithm creates a noticeable
    positive self-KL floor for 50k-token vocabularies. Reusing target log-probabilities
    makes identical logits exactly zero and is also the direct KL formula.
    """
    target_log_probs = F.log_softmax(target.float().detach(), -1)
    edited_log_probs = F.log_softmax(edited.float(), -1)
    return (target_log_probs.exp() * (target_log_probs - edited_log_probs)).sum(-1)


class FreeComponentBank(nn.Module):
    """An unconstrained LoRA-scale bank; the critical rank-matched edit baseline."""

    def __init__(self, d_in: int, d_out: int, components: int, rank: int):
        super().__init__()
        self.components = components
        self.rank = rank
        self.A = nn.Parameter(torch.empty(components, d_in, rank))
        self.B = nn.Parameter(torch.zeros(components, rank, d_out))
        nn.init.xavier_normal_(self.A)

    def factors(self, _target_weight: Tensor) -> tuple[Tensor, Tensor]:
        return self.A, self.B

    def piece_sq_mass(self, target_weight: Tensor) -> Tensor:
        return factor_sq_mass(*self.factors(target_weight))


class ProjectedComponentBank(nn.Module):
    """Low-rank pieces obtained by two-sided projection of the existing target weight.

    For input/output bases Q_i and Q_o, a piece is
    ``Q_i Q_i^T W Q_o Q_o^T``. This arm is intentionally stricter than a free edit: if
    only the free bank succeeds, the evidence supports counter-programming more strongly
    than carving an existing weight subspace.
    """

    def __init__(self, d_in: int, d_out: int, components: int, rank: int):
        super().__init__()
        self.components = components
        self.rank = rank
        self.q_in = nn.Parameter(torch.randn(components, d_in, rank) / d_in**0.5)
        self.q_out = nn.Parameter(torch.randn(components, d_out, rank) / d_out**0.5)
        # A near-zero softplus scale also kills gradients into both learned bases.
        # Starting each random projection at 1/C keeps the sum controlled without
        # saturating the scale parameter before it can become a meaningful control.
        initial_scale = 1.0 / components
        inverse_softplus = math.log(math.expm1(initial_scale))
        self.log_scale = nn.Parameter(
            torch.full((components,), inverse_softplus)
        )

    @staticmethod
    def _orthonormal(x: Tensor) -> Tensor:
        # QR in fp32 is stable and supported even when the surrounding pass uses bf16.
        return torch.linalg.qr(x.float(), mode="reduced").Q.to(x.dtype)

    def factors(self, target_weight: Tensor) -> tuple[Tensor, Tensor]:
        q_in = self._orthonormal(self.q_in)
        q_out = self._orthonormal(self.q_out)
        core = torch.einsum("cir,io,cos->crs", q_in, target_weight, q_out)
        B = torch.einsum("crs,cos->cro", core, q_out)
        B = B * F.softplus(self.log_scale).view(-1, 1, 1)
        return q_in, B

    def piece_sq_mass(self, target_weight: Tensor) -> Tensor:
        return factor_sq_mass(*self.factors(target_weight))


def factor_sq_mass(A: Tensor, B: Tensor) -> Tensor:
    left = torch.einsum("cir,cis->crs", A.float(), A.float())
    right = torch.einsum("cro,cso->crs", B.float(), B.float())
    return (left @ right).diagonal(dim1=-2, dim2=-1).sum(-1)


def build_banks(
    target: nn.Module,
    module_paths: list[str],
    components: int,
    rank: int,
    kind: BankKind = "free",
) -> nn.ModuleDict:
    cls = FreeComponentBank if kind == "free" else ProjectedComponentBank
    banks = nn.ModuleDict()
    for path in module_paths:
        linear = target.get_submodule(path)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{path} is not nn.Linear")
        d_out, d_in = linear.weight.shape
        banks[path.replace(".", "/")] = cls(d_in, d_out, components, rank)
    return banks


def bank_for(banks: nn.ModuleDict, path: str) -> FreeComponentBank | ProjectedComponentBank:
    return banks[path.replace(".", "/")]


class CarvingEditor:
    """Instrument selected linears for target capture, subtraction, and relearning."""

    def __init__(self, target: nn.Module, banks: nn.ModuleDict, module_paths: list[str]):
        self.target = target
        self.banks = banks
        self.module_paths = list(module_paths)
        self.masks: Tensor | None = None  # 1=keep, 0=remove; [B,1,C] or [B,T,C]
        self.capture = False
        self.cache: dict[str, dict[str, Tensor]] = {}
        self.recovery_banks: nn.ModuleDict | None = None
        self._original: dict[str, object] = {}
        for path in self.module_paths:
            linear = target.get_submodule(path)
            self._original[path] = linear.forward
            linear.forward = self._make_forward(path, linear)

    def _bank_output(self, x: Tensor, bank: nn.Module, weight: Tensor) -> Tensor:
        A, B = bank.factors(weight.detach().t())
        inner = torch.einsum("btd,cdr->btcr", x, A)
        return torch.einsum("btcr,cro->btco", inner, B)

    def _make_forward(self, path: str, linear: nn.Linear):
        def forward(x: Tensor) -> Tensor:
            if x.ndim != 3:
                raise ValueError(f"carving expects [batch, token, hidden] at {path}")
            out = F.linear(x, linear.weight, linear.bias)
            if self.masks is not None:
                pieces = self._bank_output(x, bank_for(self.banks, path), linear.weight)
                removal = 1.0 - self.masks
                out = out - (pieces * removal.unsqueeze(-1)).sum(-2)
            if self.recovery_banks is not None:
                recovery = self._bank_output(
                    x, bank_for(self.recovery_banks, path), linear.weight
                )
                out = out + recovery.sum(-2)
            if self.capture:
                # With a frozen target the first captured projection has no graph.
                # Making its output a leaf is sufficient for downstream activation grads.
                if not out.requires_grad and torch.is_grad_enabled():
                    out.requires_grad_(True)
                self.cache[path] = {"pre": x, "post": out}
            return out

        return forward

    def start_capture(self) -> None:
        self.capture = True
        self.cache = {}

    def stop_capture(self) -> None:
        self.capture = False

    def set_recovery(self, banks: nn.ModuleDict | None) -> None:
        self.recovery_banks = banks

    def restore(self) -> None:
        for path, forward in self._original.items():
            self.target.get_submodule(path).forward = forward
        self._original.clear()


@dataclass(frozen=True)
class CapturedUsage:
    pre: dict[str, Tensor]
    gpost: dict[str, Tensor]
    scores: Tensor
    predictions: Tensor


def capture_selected_usage(
    target: nn.Module,
    editor: CarvingEditor,
    tokens: Tensor,
    positions: Tensor,
    labels: Tensor,
    score_kind: Literal["logit_margin", "target_logit", "logprob"] = "logit_margin",
) -> CapturedUsage:
    """One backward obtains per-example activation gradients for one selected output.

    Summing the selected scores across a batch is safe: separate sequences have separate
    computation graphs, so each cached activation receives only its own example's gradient.
    """
    editor.masks = None
    editor.set_recovery(None)
    editor.start_capture()
    logits = target(tokens)
    scores = selected_scores(logits, positions, labels, score_kind)
    posts = [editor.cache[path]["post"] for path in editor.module_paths]
    grads = torch.autograd.grad(scores.sum(), posts, allow_unused=True)
    pre = {path: editor.cache[path]["pre"].detach() for path in editor.module_paths}
    gpost = {
        path: (grad.detach() if grad is not None else torch.zeros_like(posts[i]))
        for i, (path, grad) in enumerate(zip(editor.module_paths, grads, strict=True))
    }
    predictions = selected_logits(logits.detach(), positions).argmax(-1)
    editor.stop_capture()
    editor.cache = {}
    return CapturedUsage(pre, gpost, scores.detach(), predictions)


def component_credits(
    target: nn.Module,
    banks: nn.ModuleDict,
    module_paths: list[str],
    usage: CapturedUsage,
) -> Tensor:
    """Signed first-order credit of every cross-module component, shape [B, C]."""
    total = None
    for path in module_paths:
        linear = target.get_submodule(path)
        A, B = bank_for(banks, path).factors(linear.weight.detach().t())
        read = torch.einsum("btd,cdr->btcr", usage.pre[path], A)
        write_grad = torch.einsum("cro,bto->btcr", B, usage.gpost[path])
        value = (read * write_grad).sum((1, 3))
        total = value if total is None else total + value
    if total is None:
        raise ValueError("component credits need at least one module")
    return total


def target_total_credit(
    target: nn.Module, module_paths: list[str], usage: CapturedUsage
) -> Tensor:
    """Credit from scaling all selected original weights together, shape [B]."""
    total = None
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        value = torch.einsum(
            "bti,io,bto->b", usage.pre[path], weight, usage.gpost[path]
        )
        total = value if total is None else total + value
    if total is None:
        raise ValueError("target credit needs at least one module")
    return total


def module_contrast_scores(
    target: nn.Module,
    module_paths: list[str],
    positive: CapturedUsage,
    negative: CapturedUsage,
    example_mask: Tensor | None = None,
) -> dict[str, float]:
    """Size-normalized target-use contrast for cheap automatic module localization."""
    if example_mask is not None and not example_mask.any():
        raise ValueError("module localization has no eligible examples")
    scores = {}
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        p = torch.einsum("bti,io,bto->b", positive.pre[path], weight,
                         positive.gpost[path])
        n = torch.einsum("bti,io,bto->b", negative.pre[path], weight,
                         negative.gpost[path])
        contrast = p - n
        if example_mask is not None:
            contrast = contrast[example_mask]
        scores[path] = (
            contrast.square().mean().sqrt() / weight.numel()**0.5
        ).item()
    return scores


@dataclass(frozen=True)
class CoordinateSamples:
    entries: dict[str, tuple[Tensor, Tensor, float]]


def make_coordinate_samples(
    target: nn.Module, module_paths: list[str], per_module: int, seed: int
) -> CoordinateSamples:
    """Fixed random weight coordinates used as a cheap parameter-usage sketch."""
    if per_module < 1:
        raise ValueError("per_module must be positive")
    generator = torch.Generator().manual_seed(seed)
    entries = {}
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        count = min(per_module, weight.numel())
        if count == weight.numel():
            flat = torch.randperm(weight.numel(), generator=generator)
            i, o = flat // weight.shape[1], flat % weight.shape[1]
        else:
            i = torch.randint(weight.shape[0], (count,), generator=generator)
            o = torch.randint(weight.shape[1], (count,), generator=generator)
        scale = (weight.numel() / count) ** 0.5
        entries[path] = (i, o, scale)
    return CoordinateSamples(entries)


def usage_sketch(
    target: nn.Module,
    module_paths: list[str],
    usage: CapturedUsage,
    samples: CoordinateSamples,
) -> Tensor:
    """Random-coordinate sketch of ``(d score / dW) * W`` for each example."""
    chunks = []
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        i_cpu, o_cpu, scale = samples.entries[path]
        i, o = i_cpu.to(weight.device), o_cpu.to(weight.device)
        grad_entries = (
            usage.pre[path].index_select(-1, i)
            * usage.gpost[path].index_select(-1, o)
        ).sum(1)
        chunks.append(grad_entries * weight[i, o] * scale)
    return torch.cat(chunks, -1)



def component_usage_sketch(
    target: nn.Module,
    banks: nn.ModuleDict,
    module_paths: list[str],
    usage: CapturedUsage,
    samples: CoordinateSamples,
) -> Tensor:
    """Sampled usage fingerprints of each learned component, shape [B, C, D]."""
    chunks = []
    for path in module_paths:
        linear = target.get_submodule(path)
        weight = linear.weight.detach().t()
        i_cpu, o_cpu, scale = samples.entries[path]
        i, o = i_cpu.to(weight.device), o_cpu.to(weight.device)
        grad_entries = (
            usage.pre[path].index_select(-1, i)
            * usage.gpost[path].index_select(-1, o)
        ).sum(1)
        A, B = bank_for(banks, path).factors(weight)
        piece_entries = (A.index_select(1, i) * B.index_select(2, o).transpose(1, 2)).sum(-1)
        chunks.append(grad_entries[:, None, :] * piece_entries[None, :, :] * scale)
    return torch.cat(chunks, -1)


def sketch_reconstruction_loss(component_sketch: Tensor, target_sketch: Tensor) -> Tensor:
    """Make the bank explain the target contrast coordinate-by-coordinate."""
    scale = target_sketch.detach().square().mean().clamp_min(1e-8)
    return (
        (component_sketch.sum(1) - target_sketch.detach()).square().mean() / scale
    )

def geometry_loss(component_usage: Tensor, target_sketch: Tensor) -> Tensor:
    """Match off-diagonal example geometry; diagonal self-similarity is uninformative."""
    if component_usage.shape[0] < 2:
        return component_usage.new_zeros(())
    comp = F.normalize(component_usage.float(), dim=-1)
    target = F.normalize(target_sketch.float(), dim=-1)
    comp_gram, target_gram = comp @ comp.t(), target @ target.t()
    offdiag = ~torch.eye(comp.shape[0], dtype=torch.bool, device=comp.device)
    return ((comp_gram - target_gram.detach())[offdiag] ** 2).mean()


def completeness_loss(component_credit: Tensor, target_credit: Tensor) -> Tensor:
    """Require the small bank to explain the target-use contrast's total signed credit."""
    scale = target_credit.detach().square().mean().clamp_min(1e-8)
    return ((component_credit.sum(-1) - target_credit.detach()) ** 2).mean() / scale


def normalized_codes(component_credit: Tensor, eps: float = 1e-8) -> Tensor:
    values = component_credit.abs()
    return values / values.sum(-1, keepdim=True).clamp_min(eps)


def participation_ratio(codes: Tensor) -> Tensor:
    values = codes.clamp_min(0)
    return values.sum(-1).square() / values.square().sum(-1).clamp_min(1e-8)


def sample_subset_masks(codes: Tensor, remove_probability: float) -> tuple[Tensor, Tensor]:
    """Sample component removals and return keep-masks plus removed attribution mass."""
    if not 0 < remove_probability < 1:
        raise ValueError("remove_probability must lie strictly between zero and one")
    detached = codes.detach()
    remove = torch.rand_like(detached) < remove_probability
    empty = ~remove.any(-1)
    if empty.any():
        remove[empty, detached[empty].argmax(-1)] = True
    masks = (~remove).to(codes.dtype).unsqueeze(1)
    removed_mass = (detached * remove).sum(-1) / detached.sum(-1).clamp_min(1e-8)
    return masks, removed_mass


def bank_mass(
    target: nn.Module, banks: nn.ModuleDict, module_paths: list[str]
) -> Tensor:
    values = []
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        values.append(bank_for(banks, path).piece_sq_mass(weight).sum())
    return torch.stack(values).sum()


def materialize_bank_sum(
    target: nn.Module, banks: nn.ModuleDict, module_paths: list[str]
) -> dict[str, Tensor]:
    """Materialize the shipped edit for export or equivalence checks, never for training."""
    result = {}
    for path in module_paths:
        weight = target.get_submodule(path).weight.detach().t()
        A, B = bank_for(banks, path).factors(weight)
        result[path] = torch.einsum("cir,cro->io", A, B)
    return result


def pearson_correlation(x: Tensor, y: Tensor) -> Tensor:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return (x * y).sum() / denom.clamp_min(1e-8)
