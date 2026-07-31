import torch

from nano_apd.induction_components import (
    cross_layer_component_norms,
    stochastic_coalitions,
)


def test_cross_layer_sum_norm_does_not_charge_for_layer_breadth():
    concentrated = torch.tensor([[4.0, 0.0], [0.0, 9.0]])
    distributed = torch.tensor([[2.0, 2.0], [4.5, 4.5]])
    torch.testing.assert_close(
        cross_layer_component_norms(concentrated),
        cross_layer_component_norms(distributed),
    )


def test_full_model_coalition_is_true_leave_one_out():
    gates = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    result = stochastic_coalitions(
        gates,
        torch.ones(3, dtype=torch.bool),
        torch.arange(3),
        torch.Generator().manual_seed(41),
        full_model_probability=1.0,
    )
    assert result is not None
    torch.testing.assert_close(result.gates_on, torch.ones_like(result.gates_on))
    for row, component in enumerate(result.component_ids.tolist()):
        expected = torch.ones(3)
        expected[component] = 0
        torch.testing.assert_close(result.gates_off[row], expected)


def test_full_model_probability_is_validated():
    gates = torch.ones(1, 2)
    try:
        stochastic_coalitions(
            gates, torch.ones(1, dtype=torch.bool), torch.arange(1),
            full_model_probability=1.1,
        )
    except ValueError as error:
        assert "full_model_probability" in str(error)
    else:
        raise AssertionError("expected invalid probability to raise")
