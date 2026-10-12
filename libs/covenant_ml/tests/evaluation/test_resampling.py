"""Tests for unit assignment and the unit bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from covenant_ml.evaluation.resampling import bootstrap_weights, build_resampling, unit_sums
from covenant_ml.evaluation.types import (
    ERR_INVALID_BLOCK,
    ERR_INVALID_RESAMPLES,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_UNITS,
    ResamplingUnit,
)

from ._fixtures import at, floats, ints


def test_row_unit_makes_every_row_its_own_unit() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 4, ints([9, 9, 9, 9]), 1)
    assert resampling.n_units == 4
    assert np.array_equal(resampling.groups, ints([0, 1, 2, 3]))


def test_cluster_unit_densifies_the_codes() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, 5, ints([40, 7, 40, 99, 7]), 1)
    assert resampling.n_units == 3
    assert np.array_equal(resampling.groups, ints([1, 0, 1, 2, 0]))


def test_block_unit_cuts_runs_of_consecutive_rows() -> None:
    resampling = build_resampling(ResamplingUnit.BLOCK, 7, ints([0] * 7), 3)
    assert np.array_equal(resampling.groups, ints([0, 0, 0, 1, 1, 1, 2]))
    assert resampling.n_units == 3
    assert resampling.block_length == 3


def test_a_block_of_one_row_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_BLOCK):
        build_resampling(ResamplingUnit.BLOCK, 4, ints([0, 0, 0, 0]), 1)


@pytest.mark.parametrize("unit", [ResamplingUnit.ROW, ResamplingUnit.CLUSTER])
def test_a_block_length_off_the_block_unit_is_refused(unit: ResamplingUnit) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_BLOCK):
        build_resampling(unit, 4, ints([0, 1, 2, 3]), 2)


def test_codes_that_do_not_cover_the_rows_are_refused() -> None:
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        build_resampling(ResamplingUnit.ROW, 4, ints([0, 1, 2]), 1)


def test_one_cluster_is_too_few_units() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_UNITS):
        build_resampling(ResamplingUnit.CLUSTER, 3, ints([5, 5, 5]), 1)


def test_no_rows_is_too_few_units() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_UNITS):
        build_resampling(ResamplingUnit.ROW, 0, ints([]), 1)


def test_unit_sums_add_within_each_unit() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, 4, ints([1, 2, 1, 2]), 1)
    sums = unit_sums(floats([1.0, 10.0, 2.0, 20.0]), resampling)
    assert np.array_equal(sums, floats([3.0, 30.0]))


def test_unit_sums_refuse_misaligned_values() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 3, ints([0, 1, 2]), 1)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        unit_sums(floats([0.0, 0.0]), resampling)


def test_each_resample_weights_rows_by_their_units_draw_count() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, 6, ints([0, 0, 0, 1, 1, 2]), 1)
    for weights in bootstrap_weights(resampling, 25, seed=4):
        # All rows of one unit share a weight, and the units drawn total three.
        assert at(weights, 0) == at(weights, 1) == at(weights, 2)
        assert at(weights, 3) == at(weights, 4)
        assert at(weights, 0) + at(weights, 3) + at(weights, 5) == 3.0


def test_the_bootstrap_is_a_function_of_its_seed() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 8, ints([0] * 8), 1)
    first = list(bootstrap_weights(resampling, 5, seed=1))
    again = list(bootstrap_weights(resampling, 5, seed=1))
    other = list(bootstrap_weights(resampling, 5, seed=2))
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))


def test_one_resample_is_refused() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 3, ints([0, 0, 0]), 1)
    with pytest.raises(ValueError, match=ERR_INVALID_RESAMPLES):
        list(bootstrap_weights(resampling, 1, seed=0))
