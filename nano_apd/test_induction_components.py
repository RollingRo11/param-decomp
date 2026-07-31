import torch
from torch import nn

from nano_apd.carving import CapturedUsage, build_banks, component_credits
from nano_apd.induction_components import (
    component_module_credits,
    effective_components,
    rotating_components,
    stochastic_coalitions,
    sum_normalized_gates,
)
from nano_apd.induction_editor import InductionEditor


class TinySequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 4)

    def forward(self, x):
        return self.proj(x)


def test_sum_normalized_gates_zero_fallback_and_sparsity():
    gates, attribution = sum_normalized_gates(torch.zeros(3, 4))
    torch.testing.assert_close(gates, torch.full((3, 4), 0.25))
    torch.testing.assert_close(attribution, torch.zeros_like(attribution))
    sparse, _ = sum_normalized_gates(torch.tensor([[0.0, 0.0, 3.0, 0.0]]))
    torch.testing.assert_close(sparse.sum(-1), torch.ones(1))
    torch.testing.assert_close(effective_components(sparse), torch.ones(1))


def test_cross_layer_credit_is_coherent_sum_of_module_rows():
    torch.manual_seed(0)
    target = TinySequenceModel()
    banks = build_banks(target, ["proj"], components=3, rank=2)
    with torch.no_grad():
        banks["proj"].B.normal_()
    pre = torch.randn(2, 3, 5)
    gpost = torch.randn(2, 3, 4)
    usage = CapturedUsage(
        pre={"proj": pre},
        gpost={"proj": gpost},
        scores=torch.zeros(2),
        predictions=torch.zeros(2, dtype=torch.long),
    )
    rows = component_module_credits(target, banks, ["proj"], usage)
    total = component_credits(target, banks, ["proj"], usage)
    torch.testing.assert_close(rows.sum(-1), total)


def test_editor_all_one_is_exact_and_random_gate_matches_formula():
    torch.manual_seed(1)
    target = TinySequenceModel()
    banks = build_banks(target, ["proj"], components=3, rank=2)
    with torch.no_grad():
        banks["proj"].B.normal_(std=0.1)
    x = torch.randn(2, 4, 5)
    original = target(x)
    editor = InductionEditor(target, banks, ["proj"])
    editor.masks = torch.ones(2, 1, 3)
    torch.testing.assert_close(target(x), original)

    gates = torch.rand(2, 1, 3)
    editor.masks = gates
    actual = target(x)
    bank = banks["proj"]
    pieces = torch.einsum("btd,cdr,cro->btco", x, bank.A, bank.B)
    expected = original - (pieces * (1 - gates).unsqueeze(-1)).sum(-2)
    torch.testing.assert_close(actual, expected)
    editor.restore()


def test_rotating_schedule_covers_every_component():
    seen = torch.cat([rotating_components(step, 16, 5) for step in range(16)])
    assert set(seen.tolist()) == set(range(16))


def test_stochastic_coalitions_differ_only_in_named_component():
    gates = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    result = stochastic_coalitions(
        gates,
        torch.ones(3, dtype=torch.bool),
        torch.arange(3),
        torch.Generator().manual_seed(3),
    )
    assert result is not None
    difference = result.gates_on - result.gates_off
    for row, component in enumerate(result.component_ids.tolist()):
        nonzero = difference[row].nonzero(as_tuple=False).flatten().tolist()
        assert nonzero == [component]
        if not result.intact_leave_one_out[row]:
            assert result.owner_rows[row].item() == component
