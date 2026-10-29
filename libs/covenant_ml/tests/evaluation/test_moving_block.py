"""Overlapping block construction and paired inference with independent oracles."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.evaluation.codec_records import decode_comparison, encode_comparison
from covenant_ml.evaluation.paired import compare_moving_block
from covenant_ml.evaluation.resampling import moving_block_weights
from covenant_ml.evaluation.types import IntervalMethod

from ._fixtures import at, floats, ints


def test_all_overlapping_blocks_are_reachable_without_wraparound() -> None:
    actual = {
        tuple(int(at(draw, index)) for index in range(4))
        for draw in moving_block_weights(4, 2, 1000, 7)
    }
    # Starts 0,1,2 give blocks [0,1], [1,2], [2,3]. Enumerate every unordered
    # pair, including the middle block which fixed nonoverlapping draws omit.
    assert actual == {
        (2, 2, 0, 0),
        (1, 2, 1, 0),
        (1, 1, 1, 1),
        (0, 2, 2, 0),
        (0, 1, 2, 1),
        (0, 0, 2, 2),
    }
    first = tuple(moving_block_weights(5, 2, 20, 71))
    second = tuple(moving_block_weights(5, 2, 20, 71))
    for left, right in zip(first, second, strict=True):
        assert float(np.sum(left)) == 5.0
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(("rows", "length", "draws"), [(4, 1, 2), (3, 2, 2), (4, 2, 1)])
def test_invalid_block_design_is_refused(rows: int, length: int, draws: int) -> None:
    with pytest.raises(ValueError, match="CVML-EVAL"):
        tuple(moving_block_weights(rows, length, draws, 7))


def test_bootstrap_matches_enumerated_small_sample_distribution() -> None:
    times = ints([24001, 24002, 24003, 24004])
    result = compare_moving_block(
        "candidate",
        "reference",
        floats([0, 1, 4, 9]),
        floats([0, 0, 0, 0]),
        times,
        times,
        block_length=2,
        n_resamples=10000,
        seed=7,
        confidence=0.95,
    )
    interval = result["difference"]
    # Each block mean is .5, 2.5 or 6.5 with probability 1/3. The mean of
    # two independent block draws has variance (56/9)/2 = 28/9. Each
    # extreme has mass 1/9, so the 2.5 and 97.5 percentiles are .5 and 6.5.
    assert interval["estimate"] == 3.5
    assert interval["ci_low"] == 0.5
    assert interval["ci_high"] == 6.5
    assert interval["std_error"] == pytest.approx(math.sqrt(28 / 9), rel=0.02)
    assert interval["method"] is IntervalMethod.MOVING_BLOCK_BOOTSTRAP
    assert interval["n_units"] == 2
    assert result["n_rows"] == 4
    assert decode_comparison(encode_comparison(result)) == result


def test_shared_market_movements_cancel_in_a_paired_interval() -> None:
    reference = floats([0, 1, -1, 2, -2, 3])
    arm: NDArray[np.float64] = reference + 0.125
    times = ints(list(range(6)))
    result = compare_moving_block(
        "a",
        "b",
        arm,
        reference,
        times,
        times,
        block_length=2,
        n_resamples=100,
        seed=7,
        confidence=0.95,
    )
    interval = result["difference"]
    assert interval["estimate"] == 0.125
    assert interval["ci_low"] == interval["ci_high"] == 0.125
    assert interval["std_error"] == 0.0


@pytest.mark.parametrize("case", ["values", "times", "shape", "reference_shape", "identity", "gap"])
def test_mispairing_missing_months_and_reordering_are_refused(case: str) -> None:
    arm = floats([0, 1, 2, 3])
    reference = arm.copy()
    times = ints([1, 2, 3, 4])
    reference_times = times.copy()
    if case == "values":
        reference = reference[:-1]
    elif case == "times":
        times = times[:-1]
    elif case == "shape":
        times = times.reshape(4, 1)
    elif case == "reference_shape":
        reference_times = reference_times.reshape(4, 1)
    elif case == "identity":
        reference_times = reference_times[::-1]
    else:
        times = reference_times = ints([1, 2, 4, 5])
    with pytest.raises(ValueError, match="CVML-EVAL"):
        compare_moving_block(
            "a",
            "b",
            arm,
            reference,
            times,
            reference_times,
            block_length=2,
            n_resamples=10,
            seed=7,
            confidence=0.95,
        )
