"""Independent numerical oracles and refusals for evaluation boundaries."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.errors import AppError

from covenant_ml.evaluation.auc import delong_auc_difference
from covenant_ml.evaluation.intervals import (
    normal_interval,
    percentile_interval,
    pool_across_seeds,
    t_interval,
)
from covenant_ml.evaluation.paired import (
    compare_per_row,
    compare_probabilities,
    paired_mean_difference,
)
from covenant_ml.evaluation.resampling import Resampling, build_resampling
from covenant_ml.evaluation.types import (
    ERR_INVALID_ARGUMENT,
    ERR_TOO_FEW_UNITS,
    ComparisonMetric,
    IntervalMethod,
    ResamplingUnit,
)

from ._fixtures import floats, ints


def test_t_interval_matches_analytic_cauchy_distribution() -> None:
    # Student t with one degree of freedom is Cauchy, with elementary CDF
    # 1/2 + atan(x)/pi. No shared quantile or survival helper in this oracle.
    interval = t_interval(IntervalMethod.PAIRED_T, 2.0, 0.5, 1, 0.9, 2)
    radius = 0.5 * math.tan(math.pi * 0.45)
    assert interval["ci_low"] == pytest.approx(2.0 - radius)
    assert interval["ci_high"] == pytest.approx(2.0 + radius)
    assert interval["p_value"] == pytest.approx(1.0 - 2.0 * math.atan(4.0) / math.pi)


def test_unequal_clusters_keep_row_weighted_estimand() -> None:
    # Mean 4; cluster residual totals [-3, -2, 5]. G/(G-1)*38/36 = 19/12.
    differences = floats([1, 2, 4, 3, 6, 8])
    clusters = ints([9, 4, 4, 7, 7, 7])
    plan = build_resampling(ResamplingUnit.CLUSTER, 6, clusters, 1)
    interval = paired_mean_difference(differences, plan, 0.95)
    assert interval["estimate"] == 4.0
    assert interval["std_error"] == pytest.approx(math.sqrt(19 / 12))
    assert interval["n_units"] == 3


@pytest.mark.parametrize("error", [-0.1, math.nan, math.inf, -math.inf])
def test_interval_errors_must_be_finite_and_nonnegative(error: float) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        normal_interval(IntervalMethod.DELONG, 0.1, error, 0.95, 3)
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        t_interval(IntervalMethod.PAIRED_T, 0.1, error, 2, 0.95, 3)


@pytest.mark.parametrize("estimate", [math.nan, math.inf, -math.inf])
def test_nonfinite_estimates_are_refused_even_at_zero_error(estimate: float) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        normal_interval(IntervalMethod.DELONG, estimate, 0.0, 0.95, 3)
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        t_interval(IntervalMethod.PAIRED_T, estimate, 0.0, 2, 0.95, 3)


@pytest.mark.parametrize("n_units", [0, 1])
def test_interval_requires_enough_units(n_units: int) -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_UNITS):
        normal_interval(IntervalMethod.DELONG, 0.1, 0.0, 0.95, n_units)


def test_zero_error_cannot_bypass_degrees_of_freedom_validation() -> None:
    with pytest.raises(AppError, match="degrees of freedom"):
        t_interval(IntervalMethod.PAIRED_T, 0.1, 0.0, 0, 0.95, 2)


@pytest.mark.parametrize("ratio", [-0.1, math.nan, math.inf])
def test_pooling_requires_a_finite_nonnegative_sampling_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        pool_across_seeds([0.1, 0.2], [0.01, 0.02], ratio, 0.95)


def test_pooling_rejects_invalid_seed_values_and_errors() -> None:
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        pool_across_seeds([0.1, math.nan], [0.01, 0.02], 0.2, 0.95)
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        pool_across_seeds([0.1, 0.2], [0.01, -0.02], 0.2, 0.95)


@pytest.mark.parametrize("invalid", [math.nan, math.inf])
def test_paired_and_bootstrap_observations_must_be_finite(invalid: float) -> None:
    values = floats([0.1, invalid])
    groups: NDArray[np.int64] = np.arange(2, dtype=np.int64)
    plan = build_resampling(ResamplingUnit.ROW, 2, groups, 1)
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        paired_mean_difference(values, plan, 0.95)
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 0.1, values, 0.95, 2)


def test_pairing_refuses_broadcasting_before_subtraction() -> None:
    vector = floats([1.0, 2.0, 3.0])
    column: NDArray[np.float64] = vector.reshape(3, 1)
    groups: NDArray[np.int64] = np.arange(3, dtype=np.int64)
    plan = build_resampling(ResamplingUnit.ROW, 3, groups, 1)
    for arm, reference in ((vector, column), (column, vector)):
        with pytest.raises(ValueError, match="one-dimensional"):
            compare_per_row(ComparisonMetric.UTILITY, "a", "b", arm, reference, plan, 0.95)
    with pytest.raises(ValueError, match="one-dimensional"):
        percentile_interval(IntervalMethod.UNIT_BOOTSTRAP, 2.0, column, 0.95, 3)


@pytest.mark.parametrize("n_rows", [0, 1])
def test_paired_mean_refuses_too_few_rows_before_dividing(n_rows: int) -> None:
    values: NDArray[np.float64] = np.zeros(n_rows, dtype=np.float64)
    groups: NDArray[np.int64] = np.arange(n_rows, dtype=np.int64)
    plan = Resampling(ResamplingUnit.ROW, groups, n_rows, 1)
    with pytest.raises(ValueError, match=ERR_TOO_FEW_UNITS):
        compare_per_row(ComparisonMetric.UTILITY, "a", "b", values, values, plan, 0.95)


@pytest.mark.parametrize("invalid", [-1, 2])
def test_auc_rejects_nonbinary_labels(invalid: int) -> None:
    labels = ints([0, 0, 1, 1, invalid])
    scores: NDArray[np.float64] = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError, match="binary values"):
        delong_auc_difference(labels, scores, scores, 0.95)


def test_auc_rejects_multidimensional_labels() -> None:
    labels: NDArray[np.int64] = ints([0, 0, 1, 1]).reshape(4, 1)
    scores: NDArray[np.float64] = np.arange(4, dtype=np.float64)
    with pytest.raises(ValueError, match="binary values"):
        delong_auc_difference(labels, scores, scores, 0.95)


@pytest.mark.parametrize("invalid", [-0.01, 1.01, math.nan])
def test_probability_comparison_does_not_clip_invalid_predictions(invalid: float) -> None:
    labels = ints([0, 0, 1, 1])
    scores = floats([0.1, 0.2, 0.8, invalid])
    plan = build_resampling(ResamplingUnit.ROW, 4, labels, 1)
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        compare_probabilities(labels, scores, scores, "a", "b", plan, 2, 7, 0.95)
