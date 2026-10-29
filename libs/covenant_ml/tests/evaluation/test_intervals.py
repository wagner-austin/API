"""Tests for interval construction."""

from __future__ import annotations

import math
from statistics import NormalDist, stdev

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.power_distributions import t_critical, two_sided_t_survival

from covenant_ml.evaluation.intervals import (
    normal_interval,
    percentile_interval,
    pool_across_seeds,
    require_confidence,
    t_interval,
)
from covenant_ml.evaluation.types import (
    ERR_INVALID_CONFIDENCE,
    ERR_INVALID_RESAMPLES,
    ERR_LENGTH_MISMATCH,
    ERR_TOO_FEW_SEEDS,
    IntervalMethod,
)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
def test_confidence_outside_the_open_unit_interval_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_CONFIDENCE):
        require_confidence(confidence)


def test_t_interval_uses_the_shared_t_quantile_and_survival() -> None:
    interval = t_interval(IntervalMethod.PAIRED_T, 0.3, 0.1, 9, 0.95, 10)
    critical = t_critical(9, 0.05)
    assert interval["ci_low"] == pytest.approx(0.3 - critical * 0.1)
    assert interval["ci_high"] == pytest.approx(0.3 + critical * 0.1)
    assert interval["p_value"] == pytest.approx(two_sided_t_survival(3.0, 9))
    assert interval["method"] is IntervalMethod.PAIRED_T
    assert interval["n_units"] == 10
    assert interval["confidence"] == 0.95


def test_a_negative_estimate_gets_the_same_p_as_its_mirror() -> None:
    positive = t_interval(IntervalMethod.PAIRED_T, 0.3, 0.1, 9, 0.95, 10)
    negative = t_interval(IntervalMethod.PAIRED_T, -0.3, 0.1, 9, 0.95, 10)
    assert negative["p_value"] == positive["p_value"]
    assert negative["ci_high"] == pytest.approx(-positive["ci_low"])


def test_zero_spread_collapses_the_interval_onto_a_certain_estimate() -> None:
    interval = t_interval(IntervalMethod.PAIRED_T, 0.25, 0.0, 4, 0.9, 5)
    assert (interval["ci_low"], interval["ci_high"]) == (0.25, 0.25)
    assert interval["p_value"] == 0.0
    assert interval["std_error"] == 0.0


def test_zero_spread_around_zero_is_no_evidence() -> None:
    interval = normal_interval(IntervalMethod.DELONG, 0.0, 0.0, 0.9, 5)
    assert interval["p_value"] == 1.0


def test_t_interval_refuses_a_bad_confidence() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_CONFIDENCE):
        t_interval(IntervalMethod.PAIRED_T, 0.3, 0.1, 9, 1.0, 10)


def test_normal_interval_matches_the_standard_normal() -> None:
    interval = normal_interval(IntervalMethod.NEWEY_WEST, 0.2, 0.1, 0.95, 50)
    critical = NormalDist().inv_cdf(0.975)
    assert interval["ci_low"] == pytest.approx(0.2 - critical * 0.1)
    assert interval["p_value"] == pytest.approx(2.0 * (1.0 - NormalDist().cdf(2.0)))
    assert interval["method"] is IntervalMethod.NEWEY_WEST


def test_normal_interval_refuses_a_bad_confidence() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_CONFIDENCE):
        normal_interval(IntervalMethod.DELONG, 0.2, 0.1, 0.0, 50)


def _draws(values: list[float]) -> NDArray[np.float64]:
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return array


def test_percentile_bounds_are_linear_quantiles() -> None:
    values = [0.5, -0.1, 0.2, 0.9, 0.3, 0.0, 0.7, 0.4, 0.1, 0.6]
    interval = percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.35, _draws(values), 0.8, 30)
    # Sorted: -0.1, 0.0, 0.1, ..., 0.7, 0.9. The 10th percentile sits at
    # position 0.9 (between -0.1 and 0.0), the 90th at 8.1 (0.7 to 0.9).
    assert interval["ci_low"] == pytest.approx(-0.01)
    assert interval["ci_high"] == pytest.approx(0.72)
    assert interval["estimate"] == 0.35
    assert interval["method"] is IntervalMethod.UNIT_BOOTSTRAP
    assert interval["std_error"] == pytest.approx(stdev(values))


def test_percentile_p_value_counts_the_smaller_tail_with_add_one() -> None:
    # Two of ten draws are at or below zero (-0.1 and 0.0), nine at or above.
    draws = _draws([0.5, -0.1, 0.2, 0.9, 0.3, 0.0, 0.7, 0.4, 0.1, 0.6])
    interval = percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.35, draws, 0.95, 30)
    assert interval["p_value"] == pytest.approx(2.0 * 3 / 11)


def test_percentile_p_value_never_reaches_zero() -> None:
    draws = _draws([0.1, 0.2, 0.3, 0.4])
    interval = percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.25, draws, 0.95, 4)
    assert interval["p_value"] == pytest.approx(2.0 / 5)


def test_percentile_p_value_caps_at_one() -> None:
    draws = _draws([0.0, 0.0, 0.0])
    assert percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.0, draws, 0.95, 3)["p_value"] == 1.0


def test_percentile_needs_two_draws() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_RESAMPLES):
        percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.1, _draws([0.1]), 0.95, 3)


def test_percentile_refuses_a_bad_confidence() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_CONFIDENCE):
        percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.1, _draws([0.1, 0.2]), 2.0, 3)


def test_pooling_applies_the_nadeau_bengio_correction() -> None:
    per_seed = [0.01, 0.03, 0.02, 0.04]
    mean = sum(per_seed) / 4
    variance = sum((value - mean) ** 2 for value in per_seed) / 3
    interval = pool_across_seeds(per_seed, [0.0, 0.0, 0.0, 0.0], 0.25, 0.95)
    assert interval["estimate"] == pytest.approx(mean)
    assert interval["std_error"] == pytest.approx(math.sqrt((0.25 + 0.25) * variance))
    assert interval["method"] is IntervalMethod.ACROSS_SEEDS
    assert interval["n_units"] == 4


def test_pooling_is_never_narrower_than_the_seeds_own_intervals() -> None:
    # Two seeds that happen to agree to four decimals: the case the floor is for.
    interval = pool_across_seeds([0.0024, 0.0025], [0.0035, 0.0039], 0.2, 0.95)
    assert interval["std_error"] == pytest.approx(0.0037)
    assert interval["p_value"] > 0.5


def test_pooling_needs_two_seeds() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_SEEDS):
        pool_across_seeds([0.1], [0.01], 0.2, 0.95)


def test_pooling_needs_one_error_per_seed() -> None:
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        pool_across_seeds([0.1, 0.2], [0.01], 0.2, 0.95)
