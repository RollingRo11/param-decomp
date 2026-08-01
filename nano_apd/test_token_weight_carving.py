import copy
from dataclasses import asdict

import torch
from torch import nn

from nano_apd.carving import CapturedUsage, CarvingEditor
from nano_apd.run_token_weight_carving import _attribution_shortlist
from nano_apd.token_weight_carving import (
    TokenCarvingEditor,
    attribution_shares,
    capture_fixed_margin_usage,
    contrast_candidate_credits,
    contrast_gradient_factors,
    extract_fixed_pieces,
    fixed_margin_scores,
    gauss_legendre_unit_interval,
    integrate_gate_path,
    load_fixed_piece_checkpoint,
    paired_random_coalitions,
    rank_one_frobenius_norms,
)


class TinyTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 4), nn.Linear(4, 5)])

    def forward(self, x):
        return self.layers[1](torch.tanh(self.layers[0](x)))


def _usages():
    torch.manual_seed(3)
    target = TinyTokenModel()
    paths = ["layers.0", "layers.1"]
    positive = torch.randn(3, 4, 5)
    negative = positive.clone()
    negative[:, 0] += torch.randn(3, 5)
    positions = torch.tensor([1, 2, 3])
    labels = torch.tensor([0, 1, 2])
    distractors = torch.tensor([3, 3, 4])
    capture = CarvingEditor(target, nn.ModuleDict(), paths)
    positive_usage = capture_fixed_margin_usage(
        target, capture, positive, positions, labels, distractors
    )
    negative_usage = capture_fixed_margin_usage(
        target, capture, negative, positions, labels, distractors
    )
    capture.restore()
    return (
        target,
        paths,
        positive,
        negative,
        positions,
        labels,
        distractors,
        positive_usage,
        negative_usage,
    )


def test_factorized_contrast_gradient_matches_parameter_autograd():
    (
        target,
        paths,
        positive,
        negative,
        positions,
        labels,
        distractors,
        positive_usage,
        negative_usage,
    ) = _usages()
    for path in paths:
        weight = target.get_submodule(path).weight
        positive_score = fixed_margin_scores(
            target(positive), positions, labels, distractors
        ).mean()
        positive_gradient = torch.autograd.grad(positive_score, weight, retain_graph=True)[0]
        negative_score = fixed_margin_scores(
            target(negative), positions, labels, distractors
        ).mean()
        negative_gradient = torch.autograd.grad(negative_score, weight)[0]
        factors = contrast_gradient_factors(positive_usage, negative_usage, path)
        torch.testing.assert_close(
            factors.left @ factors.right.T,
            positive_gradient - negative_gradient,
            rtol=2e-5,
            atol=2e-6,
        )


def test_null_gradient_modes_produce_exactly_zero_pieces():
    target, paths, *_, positive_usage, negative_usage = _usages()

    def zero_gradient(usage):
        return CapturedUsage(
            usage.pre,
            {path: torch.zeros_like(value) for path, value in usage.gpost.items()},
            usage.scores,
            usage.predictions,
        )

    variants = {"standard": (zero_gradient(positive_usage), zero_gradient(negative_usage))}
    for projection in ("paired", "cartesian"):
        pieces, report = extract_fixed_pieces(
            target,
            paths,
            variants,
            rank=2,
            geometry="euclidean",
            projection=projection,
            damping=1e-3,
            seed=13,
        )
        for path in paths:
            torch.testing.assert_close(
                pieces.for_path(path).dense(),
                torch.zeros_like(pieces.for_path(path).dense()),
                rtol=0,
                atol=0,
            )
        assert all(not item.active for item in pieces.metadata)
        assert all(row.active_rank == 0 for row in report.module_geometry)


def test_paired_and_cartesian_projection_have_equal_first_order_credit():
    target, paths, *_, positive_usage, negative_usage = _usages()
    variants = {"standard": (positive_usage, negative_usage)}
    paired, _ = extract_fixed_pieces(
        target,
        paths,
        variants,
        rank=2,
        geometry="euclidean",
        projection="paired",
        damping=1e-3,
        seed=5,
    )
    cartesian, _ = extract_fixed_pieces(
        target,
        paths,
        variants,
        rank=2,
        geometry="euclidean",
        projection="cartesian",
        damping=1e-3,
        seed=5,
    )
    paired_credit = sum(item.first_order_credit for item in paired.metadata)
    cartesian_credit = sum(item.first_order_credit for item in cartesian.metadata)
    torch.testing.assert_close(
        torch.tensor(paired_credit),
        torch.tensor(cartesian_credit),
        rtol=2e-4,
        atol=2e-5,
    )
    for path in paths:
        assert torch.linalg.matrix_rank(paired.for_path(path).dense().sum(0)) <= 2
        assert torch.linalg.matrix_rank(cartesian.for_path(path).dense().sum(0)) <= 2


def test_attribution_shortlist_reserves_layer_coverage_before_global_fill():
    target, paths, *_, positive_usage, negative_usage = _usages()
    pieces, _ = extract_fixed_pieces(
        target,
        paths,
        {"standard": (positive_usage, negative_usage)},
        rank=2,
        geometry="euclidean",
        projection="paired",
        damping=1e-3,
        seed=19,
    )
    scores = torch.tensor([100.0, 90.0, 1.0, 0.0])
    selected = _attribution_shortlist(
        pieces,
        scores,
        total_count=3,
        per_layer=1,
    )
    assert selected.tolist() == [0, 1, 2]
    assert {pieces.metadata[index].layer for index in selected.tolist()} == {0, 1}


def test_rank_one_factor_norm_product_matches_dense_frobenius_norm():
    torch.manual_seed(41)
    input_directions = torch.randn(3, 7)
    output_directions = torch.randn(3, 5)
    expected = (
        torch.einsum("ko,ki->koi", output_directions, input_directions)
        .square()
        .sum((1, 2))
        .sqrt()
    )
    actual = rank_one_frobenius_norms(input_directions, output_directions)
    torch.testing.assert_close(actual, expected)


def test_cartesian_capture_excludes_off_diagonal_rotated_subspace_mass():
    torch.manual_seed(29)
    target = TinyTokenModel()
    paths = ["layers.0"]
    batch, tokens = 4, 3

    def random_usage():
        return CapturedUsage(
            {"layers.0": torch.randn(batch, tokens, 5)},
            {"layers.0": torch.randn(batch, tokens, 4)},
            torch.zeros(batch),
            torch.zeros(batch, dtype=torch.long),
        )

    positive_usage = random_usage()
    negative_usage = random_usage()
    pieces, report = extract_fixed_pieces(
        target,
        paths,
        {"standard": (positive_usage, negative_usage)},
        rank=2,
        geometry="euclidean",
        projection="cartesian",
        damping=1e-3,
        seed=5,
    )
    reports = {row.module_path: row for row in report.module_geometry}
    saw_off_diagonal_mass = False
    for path in paths:
        module = pieces.for_path(path)
        gradient_factors = contrast_gradient_factors(
            positive_usage, negative_usage, path
        )
        gradient = gradient_factors.left @ gradient_factors.right.T
        output_norms = torch.linalg.vector_norm(module.output_directions, dim=1)
        assert bool((output_norms > 0).all())
        output_basis = module.output_directions / output_norms[:, None]
        coordinates = output_basis @ gradient @ module.input_directions.T
        denominator = gradient.square().sum().clamp_min(1e-20)
        diagonal_capture = coordinates.diagonal().square().sum() / denominator
        full_subspace_capture = coordinates.square().sum() / denominator
        torch.testing.assert_close(
            torch.tensor(reports[path].variant_bilinear_capture["standard"]),
            diagonal_capture,
            rtol=2e-5,
            atol=2e-6,
        )
        saw_off_diagonal_mass |= bool(full_subspace_capture > diagonal_capture + 1e-6)
    assert saw_off_diagonal_mass


def test_fixed_piece_checkpoint_round_trip(tmp_path):
    target, paths, *_, positive_usage, negative_usage = _usages()
    pieces, _ = extract_fixed_pieces(
        target,
        paths,
        {"standard": (positive_usage, negative_usage)},
        rank=2,
        geometry="diag_kfac",
        projection="cartesian",
        damping=1e-3,
        seed=17,
    )
    checkpoint_path = tmp_path / "pieces.pt"
    torch.save(
        {
            "state_dict": pieces.state_dict(),
            "metadata": [asdict(item) for item in pieces.metadata],
            "module_paths": paths,
        },
        checkpoint_path,
    )

    loaded = load_fixed_piece_checkpoint(checkpoint_path)
    assert loaded.module_paths == tuple(paths)
    assert loaded.pieces.metadata == pieces.metadata
    assert list(loaded.pieces.state_dict()) == list(pieces.state_dict())
    for key, expected in pieces.state_dict().items():
        torch.testing.assert_close(loaded.pieces.state_dict()[key], expected.cpu())


def test_physical_editor_is_exact_and_credit_matches_gate_gradient():
    (
        target,
        paths,
        positive,
        negative,
        positions,
        labels,
        distractors,
        positive_usage,
        negative_usage,
    ) = _usages()
    pieces, _ = extract_fixed_pieces(
        target,
        paths,
        {"standard": (positive_usage, negative_usage)},
        rank=2,
        geometry="euclidean",
        projection="paired",
        damping=1e-3,
        seed=7,
    )
    untouched = copy.deepcopy(target)
    editor = TokenCarvingEditor(target, pieces, paths)
    all_tokens = torch.cat([positive, negative])
    batch = positive.shape[0]
    editor.masks = torch.ones(2 * batch, pieces.candidates)
    torch.testing.assert_close(target(all_tokens), untouched(all_tokens))

    gates = torch.ones(batch, pieces.candidates, requires_grad=True)
    editor.masks = torch.cat([gates, gates])
    logits = target(all_tokens)
    positive_logits, negative_logits = logits.split(batch)
    gap = fixed_margin_scores(
        positive_logits, positions, labels, distractors
    ) - fixed_margin_scores(negative_logits, positions, labels, distractors)
    gate_gradient = torch.autograd.grad(gap.sum(), gates)[0]
    analytic, _ = contrast_candidate_credits(pieces, positive_usage, negative_usage)
    torch.testing.assert_close(gate_gradient, analytic, rtol=2e-5, atol=2e-6)

    editor.masks = torch.zeros(2 * batch, pieces.candidates)
    actual_residual = target(all_tokens)
    residual = copy.deepcopy(untouched)
    with torch.no_grad():
        for path in paths:
            residual.get_submodule(path).weight.sub_(pieces.for_path(path).dense().sum(0))
    torch.testing.assert_close(actual_residual, residual(all_tokens), rtol=2e-5, atol=2e-6)
    editor.restore()


def test_joint_gate_integral_is_complete_for_nonlinear_score():
    start = torch.zeros(4, 3)
    end = torch.ones(4, 3)

    def score(gates):
        return torch.sin(gates[:, 0]) + gates[:, 1].square() * gates[:, 2]

    result = integrate_gate_path(score, start, end, steps=8)
    torch.testing.assert_close(
        result.contributions.sum(-1),
        result.end_scores - result.start_scores,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(result.completeness_error, torch.zeros(4), atol=2e-6, rtol=0)
    nodes, weights = gauss_legendre_unit_interval(8, torch.device("cpu"))
    assert bool(((nodes > 0) & (nodes < 1)).all())
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))


def test_attribution_shares_are_not_physical_gates_and_coalitions_are_paired():
    credits = torch.tensor([[4.0, -2.0, 0.0]])
    shares, _ = attribution_shares(credits)
    torch.testing.assert_close(shares.sum(-1), torch.ones(1))
    assert not torch.equal(shares, torch.ones_like(shares))

    ids, on, off = paired_random_coalitions(
        candidates=5,
        candidate_ids=torch.tensor([1, 3]),
        samples_per_candidate=4,
        generator=torch.Generator().manual_seed(11),
    )
    difference = on - off
    for row, candidate in enumerate(ids.tolist()):
        assert difference[row].nonzero(as_tuple=False).flatten().tolist() == [candidate]
