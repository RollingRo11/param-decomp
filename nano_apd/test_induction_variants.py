import torch

from nano_apd.carving import make_matched_induction_batch
from nano_apd.induction_variants import (
    ABSENT_VARIANTS,
    PRESENT_VARIANTS,
    STRESS_VARIANTS,
    make_functional_variants,
    source_positions,
)


def test_functional_variants_are_valid_and_complete():
    standard = make_matched_induction_batch(8, 32, 101, "cpu", 7)
    variants = make_functional_variants(standard, 101, 11)
    assert set(variants) == set(PRESENT_VARIANTS + STRESS_VARIANTS + ABSENT_VARIANTS)
    for batch in variants.values():
        batch.validate()
        torch.testing.assert_close(batch.positions, standard.positions)


def test_present_variants_keep_a_source_to_destination_match():
    standard = make_matched_induction_batch(8, 32, 101, "cpu", 13)
    variants = make_functional_variants(standard, 101, 17)
    rows = torch.arange(standard.positive.shape[0])
    for name in PRESENT_VARIANTS:
        batch = variants[name]
        source = source_positions(batch)
        cue = batch.positive[rows, batch.positions]
        torch.testing.assert_close(batch.positive[rows, source], cue)
        torch.testing.assert_close(batch.positive[rows, source + 1], batch.labels)


def test_absent_controls_break_the_relevant_cue_match():
    standard = make_matched_induction_batch(8, 32, 101, "cpu", 19)
    variants = make_functional_variants(standard, 101, 23)
    rows = torch.arange(standard.positive.shape[0])

    source_control = variants["source_cue_corrupt"]
    for row, position in enumerate(source_control.positions.tolist()):
        cue = source_control.positive[row, position]
        assert not (source_control.positive[row, :position] == cue).any()

    destination_control = variants["destination_cue_corrupt"]
    source = source_positions(standard)
    source_cue = destination_control.positive[rows, source]
    destination_cue = destination_control.positive[rows, destination_control.positions]
    assert not torch.equal(source_cue, destination_cue)
    assert (source_cue != destination_cue).all()


def test_source_corruption_breaks_both_consistent_demonstrations():
    standard = make_matched_induction_batch(8, 32, 101, "cpu", 29)
    twice = make_functional_variants(standard, 101, 31)["two_demonstrations"]
    control = make_functional_variants(twice, 101, 37)["source_cue_corrupt"]
    for row, position in enumerate(control.positions.tolist()):
        cue = control.positive[row, position]
        assert not (control.positive[row, :position] == cue).any()
