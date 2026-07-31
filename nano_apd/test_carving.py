"""CPU tests for the selected-token parameter-carving invariants."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nano_apd.carving import (
    CarvingEditor,
    ProjectedComponentBank,
    build_banks,
    capture_selected_usage,
    completeness_loss,
    component_credits,
    component_usage_sketch,
    geometry_loss,
    kl_per_position,
    make_coordinate_samples,
    make_matched_induction_batch,
    normalized_codes,
    participation_ratio,
    sample_subset_masks,
    selected_log_probs,
    sketch_reconstruction_loss,
    usage_sketch,
)
from nano_apd.eval_lm import normalize_gates
from nano_apd.lm_target import candidate_linear_paths, one_loader_batch, vocab_size


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 13, hidden: int = 7):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=True)
        self.unembed = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens):
        hidden = torch.tanh(self.proj(self.embedding(tokens)))
        return self.unembed(hidden)


def frozen_tiny(seed: int = 0):
    torch.manual_seed(seed)
    model = TinyLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_matched_induction_controls_token_identity():
    batch = make_matched_induction_batch(8, 16, 97, "cpu", seed=3)
    batch.validate()
    rows = torch.arange(8)
    assert torch.equal(batch.positive[rows, batch.positions],
                       batch.negative[rows, batch.positions])
    assert torch.equal(batch.positive[rows, batch.positions + 1], batch.labels)
    assert torch.equal(batch.negative[rows, batch.positions + 1], batch.labels)
    differences = (batch.positive != batch.negative).sum(-1)
    assert torch.equal(differences, torch.ones_like(differences))
    assert (batch.positive[rows, batch.positions] != batch.labels).any()


def test_free_bank_starts_as_exact_noop_and_matches_manual_edit():
    model = frozen_tiny()
    tokens = torch.randint(0, 13, (4, 6))
    banks = build_banks(model, ["proj"], components=3, rank=2, kind="free")
    editor = CarvingEditor(model, banks, ["proj"])
    with torch.no_grad():
        reference = model(tokens)
        editor.masks = torch.zeros(4, 1, 3)
        initial_edit = model(tokens)
        assert torch.allclose(reference, initial_edit, atol=1e-7)
        bank = banks["proj"]
        bank.B.normal_(0, 0.1)
        edited = model(tokens)
        embedding = model.embedding(tokens)
        pieces = torch.einsum("btd,cdr,cro->btco", embedding, bank.A, bank.B)
        hidden = torch.tanh(F.linear(embedding, model.proj.weight, model.proj.bias)
                            - pieces.sum(-2))
        manual = model.unembed(hidden)
        assert torch.allclose(edited, manual, atol=1e-6)
    editor.restore()


def test_projected_bank_is_scaled_two_sided_weight_projection():
    torch.manual_seed(1)
    bank = ProjectedComponentBank(7, 5, components=2, rank=2)
    weight = torch.randn(7, 5)
    A, B = bank.factors(weight)
    q_in = bank._orthonormal(bank.q_in)
    q_out = bank._orthonormal(bank.q_out)
    scales = F.softplus(bank.log_scale)
    assert torch.allclose(scales, torch.full_like(scales, 0.5))
    for component in range(2):
        piece = A[component] @ B[component]
        expected = (
            q_in[component] @ q_in[component].T @ weight
            @ q_out[component] @ q_out[component].T * scales[component]
        )
        assert torch.allclose(piece, expected, atol=1e-5)
        assert torch.linalg.matrix_rank(piece, atol=1e-5) <= 2


def test_component_credit_matches_weight_directional_derivative():
    model = frozen_tiny(seed=2)
    tokens = torch.randint(0, 13, (3, 7))
    positions = torch.tensor([2, 3, 4])
    labels = torch.tensor([1, 5, 8])
    banks = build_banks(model, ["proj"], components=2, rank=2, kind="free")
    with torch.no_grad():
        banks["proj"].B.normal_(0, 0.1)
    editor = CarvingEditor(model, banks, ["proj"])
    usage = capture_selected_usage(
        model, editor, tokens, positions, labels, score_kind="logprob"
    )
    credit = component_credits(model, banks, ["proj"], usage)[:, 0]
    A, B = banks["proj"].factors(model.proj.weight.detach().t())
    direction = (A[0] @ B[0]).t()
    epsilon = 2e-3
    with torch.no_grad():
        base = selected_log_probs(model(tokens), positions, labels)
        model.proj.weight.add_(direction * epsilon)
        shifted = selected_log_probs(model(tokens), positions, labels)
        model.proj.weight.sub_(direction * epsilon)
    finite_difference = (shifted - base) / epsilon
    assert torch.allclose(credit, finite_difference, atol=2e-3, rtol=3e-2)
    editor.restore()


def test_usage_sketch_matches_its_sampled_coordinates():
    model = frozen_tiny(seed=4)
    tokens = torch.randint(0, 13, (2, 6))
    positions = torch.tensor([2, 4])
    labels = torch.tensor([3, 7])
    editor = CarvingEditor(model, nn.ModuleDict(), ["proj"])
    usage = capture_selected_usage(model, editor, tokens, positions, labels)
    samples = make_coordinate_samples(model, ["proj"], per_module=11, seed=5)
    sketch = usage_sketch(model, ["proj"], usage, samples)
    i, o, scale = samples.entries["proj"]
    weight = model.proj.weight.detach().t()
    expected = (
        usage.pre["proj"].index_select(-1, i)
        * usage.gpost["proj"].index_select(-1, o)
    ).sum(1) * weight[i, o] * scale
    assert torch.allclose(sketch, expected, atol=1e-7)
    banks = build_banks(model, ["proj"], components=2, rank=2, kind="free")
    with torch.no_grad():
        banks["proj"].B.normal_(0, 0.1)
    component_sketch = component_usage_sketch(
        model, banks, ["proj"], usage, samples
    )
    assert component_sketch.shape == (2, 2, sketch.shape[-1])
    reconstruction = sketch_reconstruction_loss(component_sketch, sketch)
    reconstruction.backward()
    assert banks["proj"].B.grad is not None
    editor.restore()


def test_usage_losses_and_subset_masks_are_differentiable_and_bounded():
    torch.manual_seed(6)
    credit = torch.randn(5, 7, requires_grad=True)
    target_credit = torch.randn(5)
    sketch = torch.randn(5, 19)
    loss = completeness_loss(credit, target_credit) + geometry_loss(credit, sketch)
    loss.backward()
    assert credit.grad is not None and torch.isfinite(credit.grad).all()
    codes = normalized_codes(credit.detach())
    assert torch.allclose(codes.sum(-1), torch.ones(5), atol=1e-6)
    pr = participation_ratio(codes)
    assert ((pr >= 1) & (pr <= 7 + 1e-5)).all()
    masks, removed_mass = sample_subset_masks(codes, 0.5)
    assert masks.shape == (5, 1, 7)
    assert ((removed_mass > 0) & (removed_mass <= 1)).all()


def test_eval_gate_normalization_uses_saved_recipe():
    attribution = torch.tensor([[[1.0, 2.0, 3.0]]])
    max_gate = normalize_gates(attribution, {"gate_norm": "max", "tau": 1.0})
    sum_gate = normalize_gates(attribution, {"gate_norm": "sum", "tau": 1.0})
    assert torch.allclose(max_gate, torch.tensor([[[1 / 3, 2 / 3, 1.0]]]))
    assert torch.allclose(sum_gate, torch.tensor([[[1 / 6, 2 / 6, 3 / 6]]]))
    assert not torch.allclose(max_gate, sum_gate)
    with pytest.raises(ValueError, match="gate_thresh"):
        normalize_gates(attribution, {"gate_thresh": 1.0})

def test_hf_module_discovery_excludes_output_head():
    class MockHF(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"vocab_size": 101})()
            self.gpt_neox = nn.Module()
            self.gpt_neox.layers = nn.ModuleList([nn.Module()])
            self.gpt_neox.layers[0].attention = nn.Module()
            self.gpt_neox.layers[0].attention.query_key_value = nn.Linear(7, 21)
            self.gpt_neox.layers[0].attention.dense = nn.Linear(7, 7)
            self.lm_head = nn.Linear(7, 101)

    model = MockHF()
    assert candidate_linear_paths(model, "hf") == [
        "gpt_neox.layers.0.attention.query_key_value",
        "gpt_neox.layers.0.attention.dense",
    ]
    assert vocab_size(model) == 101

def test_kl_self_comparison_is_exactly_zero_for_large_vocab():
    torch.manual_seed(9)
    logits = torch.randn(2, 3, 50_277)
    self_kl = kl_per_position(logits, logits.clone())
    assert torch.equal(self_kl, torch.zeros_like(self_kl))

def test_one_loader_batch_closes_stream():
    closed = []

    def stream():
        try:
            yield torch.tensor([3])
        finally:
            closed.append(True)

    assert torch.equal(one_loader_batch(stream()), torch.tensor([3]))
    assert closed == [True]
