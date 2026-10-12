"""Tests for paired mean differences and the per-metric comparison router."""

from __future__ import annotations

import math

import pytest
from platform_core.power_distributions import t_critical

from covenant_ml.evaluation.paired import (
    compare_per_row,
    compare_probabilities,
    paired_mean_difference,
)
from covenant_ml.evaluation.resampling import build_resampling
from covenant_ml.evaluation.types import (
    ERR_LENGTH_MISMATCH,
    ComparisonMetric,
    IntervalMethod,
    ResamplingUnit,
)
from covenant_ml.metrics import compute_brier_score, compute_log_loss

from ._fixtures import floats, ints

DIFFS = [0.4, -0.1, 0.3, 0.2, 0.5, 0.0]


def test_row_unit_is_the_paired_t() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 6, ints([0] * 6), 1)
    interval = paired_mean_difference(floats(DIFFS), resampling, 0.95)
    mean = sum(DIFFS) / 6
    std_error = math.sqrt(sum((d - mean) ** 2 for d in DIFFS) / 5 / 6)
    assert interval["method"] is IntervalMethod.PAIRED_T
    assert interval["estimate"] == pytest.approx(mean)
    assert interval["std_error"] == pytest.approx(std_error)
    assert interval["ci_high"] == pytest.approx(mean + t_critical(5, 0.05) * std_error)
    assert interval["n_units"] == 6


def test_clusters_of_one_row_reduce_to_the_paired_t() -> None:
    # With every row its own cluster, the sandwich estimator's G/(G-1) factor
    # is exactly the n/(n-1) of the sample variance.
    row = build_resampling(ResamplingUnit.ROW, 6, ints([0] * 6), 1)
    singletons = build_resampling(ResamplingUnit.CLUSTER, 6, ints([5, 4, 3, 2, 1, 0]), 1)
    by_row = paired_mean_difference(floats(DIFFS), row, 0.95)
    by_cluster = paired_mean_difference(floats(DIFFS), singletons, 0.95)
    assert by_cluster["method"] is IntervalMethod.CLUSTER_ROBUST
    assert by_cluster["std_error"] == pytest.approx(by_row["std_error"])


def test_cluster_robust_error_sums_within_clusters_first() -> None:
    resampling = build_resampling(ResamplingUnit.CLUSTER, 6, ints([0, 0, 1, 1, 2, 2]), 1)
    interval = paired_mean_difference(floats(DIFFS), resampling, 0.95)
    mean = sum(DIFFS) / 6
    residuals = [
        DIFFS[0] + DIFFS[1] - 2 * mean,
        DIFFS[2] + DIFFS[3] - 2 * mean,
        DIFFS[4] + DIFFS[5] - 2 * mean,
    ]
    variance = 3 / 2 * sum(r * r for r in residuals) / 36
    assert interval["std_error"] == pytest.approx(math.sqrt(variance))
    assert interval["n_units"] == 3


def test_block_unit_is_newey_west() -> None:
    # Worked by hand: mean 3.5, gamma0 = 17.5/6, gamma1 = 8.75/6, one lag
    # with Bartlett weight 1/2, so the long-run variance is 4.375.
    resampling = build_resampling(ResamplingUnit.BLOCK, 6, ints([0] * 6), 2)
    interval = paired_mean_difference(floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), resampling, 0.95)
    assert interval["method"] is IntervalMethod.NEWEY_WEST
    assert interval["std_error"] == pytest.approx(math.sqrt(4.375 / 6))
    assert interval["n_units"] == 3


def test_differences_must_cover_the_rows() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 6, ints([0] * 6), 1)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        paired_mean_difference(floats(DIFFS[:-1]), resampling, 0.95)


def test_compare_per_row_differences_arm_minus_reference() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 3, ints([0] * 3), 1)
    comparison = compare_per_row(
        ComparisonMetric.UTILITY,
        "cleargbm",
        "lightgbm",
        floats([1.0, 2.0, 3.0]),
        floats([0.5, 2.5, 2.0]),
        resampling,
        0.9,
    )
    assert comparison["value_arm"] == pytest.approx(2.0)
    assert comparison["value_reference"] == pytest.approx(5.0 / 3)
    assert comparison["difference"]["estimate"] == pytest.approx(1.0 / 3)
    assert comparison["n_rows"] == 3
    assert (comparison["arm"], comparison["reference"]) == ("cleargbm", "lightgbm")


def test_compare_per_row_needs_the_same_rows() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 3, ints([0] * 3), 1)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        compare_per_row(
            ComparisonMetric.BRIER,
            "a",
            "b",
            floats([1.0, 2.0, 3.0]),
            floats([1.0, 2.0]),
            resampling,
            0.9,
        )


LABELS = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
PROBA_A = [0.9, 0.3, 0.8, 0.45, 0.5, 0.1, 0.7, 0.2, 0.65, 0.55, 0.35, 0.05]
PROBA_B = [0.6, 0.4, 0.9, 0.2, 0.3, 0.25, 0.5, 0.1, 0.7, 0.8, 0.45, 0.35]


def test_independent_rows_get_delong_and_the_two_losses() -> None:
    resampling = build_resampling(ResamplingUnit.ROW, 12, ints([0] * 12), 1)
    comparisons = compare_probabilities(
        ints(LABELS), floats(PROBA_A), floats(PROBA_B), "a", "b", resampling, 20, 1, 0.95
    )
    assert [c["metric"] for c in comparisons] == [
        ComparisonMetric.AUC,
        ComparisonMetric.LOG_LOSS,
        ComparisonMetric.BRIER,
    ]
    assert comparisons[0]["difference"]["method"] is IntervalMethod.DELONG
    assert comparisons[1]["value_arm"] == pytest.approx(
        compute_log_loss(ints(LABELS), floats(PROBA_A))
    )
    assert comparisons[2]["value_reference"] == pytest.approx(
        compute_brier_score(ints(LABELS), floats(PROBA_B))
    )


def test_clustered_rows_get_the_unit_bootstrap_for_auc() -> None:
    clusters = ints([index // 2 for index in range(12)])
    resampling = build_resampling(ResamplingUnit.CLUSTER, 12, clusters, 1)
    comparisons = compare_probabilities(
        ints(LABELS), floats(PROBA_A), floats(PROBA_B), "a", "b", resampling, 30, 1, 0.95
    )
    assert comparisons[0]["difference"]["method"] is IntervalMethod.UNIT_BOOTSTRAP
    assert comparisons[1]["difference"]["method"] is IntervalMethod.CLUSTER_ROBUST
    assert all(c["n_rows"] == 12 for c in comparisons)
