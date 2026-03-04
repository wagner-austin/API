"""Tests for McKinnon-style rank-trend hypothesis testing functions.

Tests cover all 9 public functions in _netcdf_trend_testing.py: OLS slope
computation, rank conversion, composite rank averaging, latitude weighting,
weighted spatial mean, spatial DOF estimation, Monte Carlo null distribution
generation, p-value computation, and the full analysis orchestrator.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._netcdf_trend_testing import (
    compute_latitude_weights,
    compute_ols_slope,
    compute_trend_pvalue,
    compute_weighted_spatial_mean,
    estimate_spatial_dof,
    generate_null_trend_slopes,
    rank_heat_metrics,
    rank_metric_series,
    run_rank_trend_analysis,
)
from covenant_ml.datasets.testing import create_synthetic_trending_metrics
from covenant_ml.datasets.types import (
    make_rank_trend_config,
)

# --- Typed helpers (avoid mypy Any from numpy indexing) ---


def _f64(values: list[float]) -> NDArray[np.float64]:
    """Create float64 array from typed list."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for idx, v in enumerate(values):
        result[idx] = v
    return result


def _f64_2d(values: list[list[float]]) -> NDArray[np.float64]:
    """Create 2D float64 array from nested list."""
    rows = len(values)
    cols = len(values[0])
    result: NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            result[i, j] = v
    return result


def _f64_3d(values: list[list[list[float]]]) -> NDArray[np.float64]:
    """Create 3D float64 array from nested list."""
    d0 = len(values)
    d1 = len(values[0])
    d2 = len(values[0][0])
    result: NDArray[np.float64] = np.zeros((d0, d1, d2), dtype=np.float64)
    for i in range(d0):
        for j in range(d1):
            for k in range(d2):
                result.flat[i * d1 * d2 + j * d2 + k] = values[i][j][k]
    return result


def _flat(arr: NDArray[np.float64], idx: int) -> float:
    """Extract a typed float from flat index."""
    return float(arr.flat[idx])


def _val2(arr: NDArray[np.float64], i: int, j: int) -> float:
    """Extract a typed float from a 2D NDArray."""
    return float(arr.flat[i * int(arr.shape[1]) + j])


# ===================================================================
# TestComputeOlsSlope
# ===================================================================


class TestComputeOlsSlope:
    """Tests for compute_ols_slope: manual OLS regression slope."""

    def test_perfect_positive_slope(self) -> None:
        """Perfect linear relationship y = 2x + 1 gives slope 2."""
        x = _f64([0.0, 1.0, 2.0, 3.0, 4.0])
        y = _f64([1.0, 3.0, 5.0, 7.0, 9.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope - 2.0) < 1e-10

    def test_perfect_negative_slope(self) -> None:
        """Perfect linear y = -0.5x + 10 gives slope -0.5."""
        x = _f64([0.0, 2.0, 4.0, 6.0])
        y = _f64([10.0, 9.0, 8.0, 7.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope - (-0.5)) < 1e-10

    def test_zero_slope(self) -> None:
        """Constant y gives slope 0."""
        x = _f64([1.0, 2.0, 3.0, 4.0, 5.0])
        y = _f64([5.0, 5.0, 5.0, 5.0, 5.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope) < 1e-10

    def test_constant_x_returns_zero(self) -> None:
        """Constant x (zero denominator) returns 0."""
        x = _f64([3.0, 3.0, 3.0])
        y = _f64([1.0, 2.0, 3.0])
        slope = compute_ols_slope(x, y)
        assert slope == 0.0

    def test_two_points(self) -> None:
        """Minimum case: 2 points."""
        x = _f64([0.0, 1.0])
        y = _f64([0.0, 3.0])
        slope = compute_ols_slope(x, y)
        assert abs(slope - 3.0) < 1e-10

    def test_mismatched_lengths_raises(self) -> None:
        """Different length arrays raise ValueError."""
        x = _f64([1.0, 2.0])
        y = _f64([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="x length 2 != y length 3"):
            compute_ols_slope(x, y)

    def test_single_point_raises(self) -> None:
        """Fewer than 2 points raises ValueError."""
        x = _f64([1.0])
        y = _f64([2.0])
        with pytest.raises(ValueError, match="Need at least 2 points"):
            compute_ols_slope(x, y)


# ===================================================================
# TestRankMetricSeries
# ===================================================================


class TestRankMetricSeries:
    """Tests for rank_metric_series: 1D value-to-rank conversion."""

    def test_ascending_values_no_negate(self) -> None:
        """Ascending values without negation: rank 1 = smallest."""
        values = _f64([10.0, 20.0, 30.0, 40.0])
        ranks = rank_metric_series(values, negate=False)
        assert int(_flat(ranks, 0)) == 1
        assert int(_flat(ranks, 1)) == 2
        assert int(_flat(ranks, 2)) == 3
        assert int(_flat(ranks, 3)) == 4

    def test_descending_values_no_negate(self) -> None:
        """Descending values without negation: rank 1 = smallest."""
        values = _f64([40.0, 30.0, 20.0, 10.0])
        ranks = rank_metric_series(values, negate=False)
        assert int(_flat(ranks, 0)) == 4
        assert int(_flat(ranks, 1)) == 3
        assert int(_flat(ranks, 2)) == 2
        assert int(_flat(ranks, 3)) == 1

    def test_with_negate(self) -> None:
        """With negate=True: rank 1 = largest original value."""
        values = _f64([10.0, 20.0, 30.0, 40.0])
        ranks = rank_metric_series(values, negate=True)
        # Negated: [-10, -20, -30, -40], argsort gives [-40,-30,-20,-10]
        # so original 40 (index 3) gets rank 1
        assert int(_flat(ranks, 3)) == 1
        assert int(_flat(ranks, 0)) == 4

    def test_output_dtype_and_shape(self) -> None:
        """Output is float64 with same shape as input."""
        values = _f64([5.0, 3.0, 1.0])
        ranks = rank_metric_series(values, negate=False)
        assert ranks.dtype == np.float64
        assert int(ranks.shape[0]) == 3

    def test_all_ranks_present(self) -> None:
        """All ranks from 1 to n are assigned exactly once."""
        values = _f64([7.0, 2.0, 5.0, 9.0, 1.0])
        ranks = rank_metric_series(values, negate=False)
        sorted_ranks: list[int] = sorted(int(_flat(ranks, i)) for i in range(5))
        assert sorted_ranks == [1, 2, 3, 4, 5]


# ===================================================================
# TestRankHeatMetrics
# ===================================================================


class TestRankHeatMetrics:
    """Tests for rank_heat_metrics: multi-metric ranking with sign conventions."""

    def test_hot_metric_negated(self) -> None:
        """HOT metric (seasonal_max) is negated: rank 1 = largest value."""
        # 3 years, 1 metric (seasonal_max), 1 location
        # Values increasing: [10, 20, 30] → negated: [-10, -20, -30]
        # argsort of negated: [-30, -20, -10] → index 2 gets rank 1
        metrics = _f64_3d(
            [
                [[10.0]],
                [[20.0]],
                [[30.0]],
            ]
        )
        ranked, _names = rank_heat_metrics(metrics, ("seasonal_max",))
        # seasonal_max is HOT → negated → rank 1 = year with value 30
        year2_rank = float(ranked.flat[2 * 3 * 1 + 0 * 1 + 0])
        assert int(year2_rank) == 1

    def test_cold_metric_direct(self) -> None:
        """COLD metric (seasonal_min) is ranked directly: rank 1 = smallest."""
        # 3 years, 1 metric (seasonal_min), 1 location
        # Values: [10, 5, 15] → rank 1 = year with value 5 (index 1)
        metrics = _f64_3d(
            [
                [[10.0]],
                [[5.0]],
                [[15.0]],
            ]
        )
        ranked, _names = rank_heat_metrics(metrics, ("seasonal_min",))
        year1_rank = float(ranked.flat[1 * 3 * 1 + 0 * 1 + 0])
        assert int(year1_rank) == 1

    def test_extended_names_include_composites(self) -> None:
        """Output names include avg_across_metrics_hot and avg_across_metrics_cold."""
        # 3 years, 2 metrics, 1 location
        metrics = _f64_3d(
            [
                [[10.0], [5.0]],
                [[20.0], [3.0]],
                [[30.0], [7.0]],
            ]
        )
        _, names = rank_heat_metrics(metrics, ("seasonal_max", "seasonal_min"))
        assert names[-2] == "avg_across_metrics_hot"
        assert names[-1] == "avg_across_metrics_cold"
        assert len(names) == 4  # 2 original + 2 composites

    def test_multi_location_ranking(self) -> None:
        """Each location is ranked independently."""
        # 3 years, 1 metric (seasonal_max), 2 locations
        metrics = _f64_3d(
            [
                [[30.0, 10.0]],  # year 0: loc0=30(high), loc1=10(low)
                [[20.0, 20.0]],  # year 1
                [[10.0, 30.0]],  # year 2: loc0=10(low), loc1=30(high)
            ]
        )
        ranked, _ = rank_heat_metrics(metrics, ("seasonal_max",))
        n_ext = int(ranked.shape[1])
        n_loc = int(ranked.shape[2])
        # seasonal_max is HOT: rank 1 = largest
        # loc0: [30, 20, 10] → negated [-30, -20, -10] → rank 1 = year 0
        yr0_loc0 = float(ranked.flat[0 * n_ext * n_loc + 0 * n_loc + 0])
        assert int(yr0_loc0) == 1
        # loc1: [10, 20, 30] → negated [-10, -20, -30] → rank 1 = year 2
        yr2_loc1 = float(ranked.flat[2 * n_ext * n_loc + 0 * n_loc + 1])
        assert int(yr2_loc1) == 1

    def test_mismatched_names_raises(self) -> None:
        """Metric names length mismatch raises ValueError."""
        metrics = _f64_3d([[[1.0]], [[2.0]]])
        with pytest.raises(ValueError, match="metric_names length"):
            rank_heat_metrics(metrics, ("a", "b"))

    def test_composite_hot_average(self) -> None:
        """avg_across_metrics_hot is mean rank of hot metrics (excluding ar1)."""
        # 3 years, 2 hot metrics, 1 location
        # seasonal_max: [10, 20, 30] → negated ranks: [3, 2, 1]
        # ndays_excess_hot: [30, 20, 10] → negated ranks: [1, 2, 3]
        metrics = _f64_3d(
            [
                [[10.0], [30.0]],
                [[20.0], [20.0]],
                [[30.0], [10.0]],
            ]
        )
        ranked, names = rank_heat_metrics(metrics, ("seasonal_max", "ndays_excess_hot"))
        hot_idx = names.index("avg_across_metrics_hot")
        n_ext = int(ranked.shape[1])
        # Year 0: seasonal_max rank=3, ndays_excess_hot rank=1 → avg = 2.0
        yr0_avg = float(ranked.flat[0 * n_ext * 1 + hot_idx * 1 + 0])
        assert abs(yr0_avg - 2.0) < 1e-10


# ===================================================================
# TestComputeLatitudeWeights
# ===================================================================


class TestComputeLatitudeWeights:
    """Tests for compute_latitude_weights: area-based latitude weighting."""

    def test_equator_has_max_weight(self) -> None:
        """Equatorial location has cos(0) = 1, the maximum weight."""
        lats = _f64([0.0, 45.0, 60.0])
        weights = compute_latitude_weights(lats)
        # Equator should have the largest weight
        w0 = _flat(weights, 0)
        w1 = _flat(weights, 1)
        w2 = _flat(weights, 2)
        assert w0 > w1
        assert w1 > w2

    def test_weights_sum_to_one(self) -> None:
        """Weights are normalized to sum to 1."""
        lats = _f64([0.0, 30.0, 60.0, -30.0, -60.0])
        weights = compute_latitude_weights(lats)
        total: float = 0.0
        for i in range(5):
            total += _flat(weights, i)
        assert abs(total - 1.0) < 1e-10

    def test_single_location(self) -> None:
        """Single location gets weight 1.0."""
        lats = _f64([45.0])
        weights = compute_latitude_weights(lats)
        assert abs(_flat(weights, 0) - 1.0) < 1e-10

    def test_symmetric_latitudes(self) -> None:
        """Symmetric latitudes get equal weights."""
        lats = _f64([-45.0, 45.0])
        weights = compute_latitude_weights(lats)
        assert abs(_flat(weights, 0) - _flat(weights, 1)) < 1e-10

    def test_poles_have_equal_tiny_weights(self) -> None:
        """Pole latitudes (cos≈0) still get normalized equal weights."""
        lats = _f64([90.0, -90.0])
        weights = compute_latitude_weights(lats)
        # Both cos(90°) ≈ 6.1e-17, so weights should be ~0.5 each
        assert abs(_flat(weights, 0) - 0.5) < 1e-10
        assert abs(_flat(weights, 1) - 0.5) < 1e-10

    def test_empty_raises(self) -> None:
        """Empty latitudes raise ValueError."""
        lats: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        with pytest.raises(ValueError, match="latitudes must not be empty"):
            compute_latitude_weights(lats)


# ===================================================================
# TestComputeWeightedSpatialMean
# ===================================================================


class TestComputeWeightedSpatialMean:
    """Tests for compute_weighted_spatial_mean: weighted average across locations."""

    def test_uniform_weights(self) -> None:
        """Uniform weights give arithmetic mean."""
        # 3 years, 2 locations
        values = _f64_2d(
            [
                [10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0],
            ]
        )
        weights = _f64([0.5, 0.5])
        result = compute_weighted_spatial_mean(values, weights)
        assert abs(_flat(result, 0) - 15.0) < 1e-10
        assert abs(_flat(result, 1) - 35.0) < 1e-10
        assert abs(_flat(result, 2) - 55.0) < 1e-10

    def test_unequal_weights(self) -> None:
        """Non-uniform weights weight locations differently."""
        values = _f64_2d(
            [
                [10.0, 30.0],
            ]
        )
        weights = _f64([0.75, 0.25])
        result = compute_weighted_spatial_mean(values, weights)
        # 0.75 * 10 + 0.25 * 30 = 7.5 + 7.5 = 15.0
        assert abs(_flat(result, 0) - 15.0) < 1e-10

    def test_single_location(self) -> None:
        """Single location with weight 1 returns that location's values."""
        values = _f64_2d(
            [
                [5.0],
                [10.0],
            ]
        )
        weights = _f64([1.0])
        result = compute_weighted_spatial_mean(values, weights)
        assert abs(_flat(result, 0) - 5.0) < 1e-10
        assert abs(_flat(result, 1) - 10.0) < 1e-10

    def test_mismatched_weights_raises(self) -> None:
        """Weights length mismatch raises ValueError."""
        values = _f64_2d([[1.0, 2.0]])
        weights = _f64([1.0])
        with pytest.raises(ValueError, match="weights length"):
            compute_weighted_spatial_mean(values, weights)


# ===================================================================
# TestEstimateSpatialDof
# ===================================================================


class TestEstimateSpatialDof:
    """Tests for estimate_spatial_dof: Bretherton et al. (1999) DOF estimation."""

    def test_single_location_returns_one(self) -> None:
        """Single location always gives DOF = 1."""
        ranks = _f64_2d(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
            ]
        )
        weights = _f64([1.0])
        dof = estimate_spatial_dof(ranks, weights)
        assert dof == 1

    def test_uncorrelated_locations_high_dof(self) -> None:
        """Independent locations yield DOF close to n_locations."""
        rng = np.random.default_rng(42)
        n_years = 50
        n_locations = 10
        ranks: NDArray[np.float64] = np.zeros((n_years, n_locations), dtype=np.float64)
        for loc in range(n_locations):
            perm = rng.permutation(n_years)
            for yr in range(n_years):
                ranks.flat[yr * n_locations + loc] = float(perm.flat[yr]) + 1.0
        weights = _f64([1.0 / n_locations] * n_locations)
        dof = estimate_spatial_dof(ranks, weights)
        # Independent → DOF should be close to n_locations
        assert dof >= 5

    def test_perfectly_correlated_low_dof(self) -> None:
        """Identical locations yield DOF = 1."""
        n_years = 20
        n_locations = 5
        ranks: NDArray[np.float64] = np.zeros((n_years, n_locations), dtype=np.float64)
        # All locations have the same rank series
        for yr in range(n_years):
            for loc in range(n_locations):
                ranks.flat[yr * n_locations + loc] = float(yr + 1)
        weights = _f64([0.2, 0.2, 0.2, 0.2, 0.2])
        dof = estimate_spatial_dof(ranks, weights)
        assert dof == 1

    def test_minimum_dof_is_one(self) -> None:
        """DOF is always at least 1."""
        ranks = _f64_2d(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
        weights = _f64([0.5, 0.5])
        dof = estimate_spatial_dof(ranks, weights)
        assert dof >= 1

    def test_mismatched_weights_raises(self) -> None:
        """Weights length mismatch raises ValueError."""
        ranks = _f64_2d([[1.0, 2.0], [3.0, 4.0]])
        weights = _f64([1.0])  # 1 weight but 2 locations
        with pytest.raises(ValueError, match="weights length"):
            estimate_spatial_dof(ranks, weights)

    def test_zero_frobenius_norm_returns_one(self) -> None:
        """Zero Frobenius norm (constant covariance) returns DOF = 1."""
        # All locations have identical constant values → cov = 0 matrix
        n_years = 5
        n_locations = 3
        ranks: NDArray[np.float64] = np.zeros((n_years, n_locations), dtype=np.float64)
        # All values identical → zero variance → zero covariance → zero Frobenius
        for yr in range(n_years):
            for loc in range(n_locations):
                ranks.flat[yr * n_locations + loc] = 1.0
        weights = _f64([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        dof = estimate_spatial_dof(ranks, weights)
        assert dof == 1

    def test_too_few_years_raises(self) -> None:
        """Fewer than 2 years raises ValueError."""
        ranks = _f64_2d([[1.0, 2.0]])
        weights = _f64([0.5, 0.5])
        with pytest.raises(ValueError, match="Need at least 2 years"):
            estimate_spatial_dof(ranks, weights)


# ===================================================================
# TestGenerateNullTrendSlopes
# ===================================================================


class TestGenerateNullTrendSlopes:
    """Tests for generate_null_trend_slopes: Monte Carlo null distribution."""

    def test_output_shape(self) -> None:
        """Output has shape (n_samples,)."""
        slopes = generate_null_trend_slopes(dof=3, n_years=10, n_samples=100, seed=42)
        assert int(slopes.shape[0]) == 100

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        s1 = generate_null_trend_slopes(dof=3, n_years=10, n_samples=50, seed=99)
        s2 = generate_null_trend_slopes(dof=3, n_years=10, n_samples=50, seed=99)
        for i in range(50):
            assert _flat(s1, i) == _flat(s2, i)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different results."""
        s1 = generate_null_trend_slopes(dof=3, n_years=10, n_samples=50, seed=1)
        s2 = generate_null_trend_slopes(dof=3, n_years=10, n_samples=50, seed=2)
        any_different = False
        for i in range(50):
            if _flat(s1, i) != _flat(s2, i):
                any_different = True
                break
        assert any_different

    def test_distribution_centered_near_zero(self) -> None:
        """Null slopes should be centered near zero (no trend under H0)."""
        slopes = generate_null_trend_slopes(dof=5, n_years=20, n_samples=500, seed=42)
        total: float = 0.0
        for i in range(500):
            total += _flat(slopes, i)
        mean_slope = total / 500.0
        assert abs(mean_slope) < 0.5

    def test_invalid_dof_raises(self) -> None:
        """dof < 1 raises ValueError."""
        with pytest.raises(ValueError, match="dof must be >= 1"):
            generate_null_trend_slopes(dof=0, n_years=10, n_samples=10, seed=42)

    def test_invalid_n_years_raises(self) -> None:
        """n_years < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_years must be >= 2"):
            generate_null_trend_slopes(dof=3, n_years=1, n_samples=10, seed=42)

    def test_invalid_n_samples_raises(self) -> None:
        """n_samples < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            generate_null_trend_slopes(dof=3, n_years=10, n_samples=0, seed=42)


# ===================================================================
# TestComputeTrendPvalue
# ===================================================================


class TestComputeTrendPvalue:
    """Tests for compute_trend_pvalue: two-sided p-value from null distribution."""

    def test_extreme_slope_gives_small_pvalue(self) -> None:
        """Observed slope far beyond null gives p-value near 0."""
        null_slopes = _f64([0.1, -0.1, 0.05, -0.05, 0.2, -0.2])
        p = compute_trend_pvalue(10.0, null_slopes)
        assert p == 0.0

    def test_zero_slope_gives_large_pvalue(self) -> None:
        """Observed slope of 0 gives p-value = 1.0 (all null |slopes| >= 0)."""
        null_slopes = _f64([0.1, -0.1, 0.05, -0.05])
        p = compute_trend_pvalue(0.0, null_slopes)
        assert abs(p - 1.0) < 1e-10

    def test_boundary_case(self) -> None:
        """Slopes exactly at the boundary are counted."""
        null_slopes = _f64([0.5, -0.5, 0.3, -0.3])
        p = compute_trend_pvalue(0.5, null_slopes)
        # |null| >= 0.5: 0.5 and -0.5 → 2/4 = 0.5
        assert abs(p - 0.5) < 1e-10

    def test_negative_observed_uses_absolute(self) -> None:
        """Negative observed slope is compared by absolute value."""
        null_slopes = _f64([0.5, -0.5, 0.1, -0.1])
        p_pos = compute_trend_pvalue(0.5, null_slopes)
        p_neg = compute_trend_pvalue(-0.5, null_slopes)
        assert abs(p_pos - p_neg) < 1e-10

    def test_empty_null_raises(self) -> None:
        """Empty null slopes raise ValueError."""
        null_slopes: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        with pytest.raises(ValueError, match="null_slopes must not be empty"):
            compute_trend_pvalue(1.0, null_slopes)


# ===================================================================
# TestRunRankTrendAnalysis
# ===================================================================


class TestRunRankTrendAnalysis:
    """Tests for run_rank_trend_analysis: full orchestrator."""

    def test_trending_data_is_significant(self) -> None:
        """Strong trends in synthetic data are detected as significant."""
        data = create_synthetic_trending_metrics(
            n_years=30,
            n_locations=5,
            seed=42,
            trend_slope=2.0,
            noise_std=0.1,
        )
        config = make_rank_trend_config(n_null_samples=200, random_seed=42)
        result = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        # Find seasonal_max result — should be significant
        found_significant = False
        for mr in result["metric_results"]:
            if mr["metric_name"] == "seasonal_max":
                found_significant = mr["is_significant"]
                break
        assert found_significant

    def test_flat_data_not_significant(self) -> None:
        """No trend in flat data should not be significant."""
        data = create_synthetic_trending_metrics(
            n_years=20,
            n_locations=3,
            seed=42,
            trend_slope=0.0,
            noise_std=1.0,
        )
        config = make_rank_trend_config(n_null_samples=200, random_seed=42)
        result = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        # With zero trend and high noise, most metrics should not be significant
        n_significant = sum(1 for mr in result["metric_results"] if mr["is_significant"])
        # Allow at most 1 false positive out of 4 metrics
        assert n_significant <= 1

    def test_result_structure(self) -> None:
        """Result has correct structure and fields."""
        data = create_synthetic_trending_metrics(
            n_years=10,
            n_locations=2,
            seed=42,
        )
        config = make_rank_trend_config(n_null_samples=50, random_seed=42)
        result = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        assert result["n_null_samples"] == 50
        assert result["random_seed"] == 42
        # 2 original metrics + 2 composites = 4
        assert len(result["metric_results"]) == 4

    def test_all_expected_metrics_present(self) -> None:
        """Result contains entries for all original + composite metrics."""
        data = create_synthetic_trending_metrics(
            n_years=10,
            n_locations=2,
            seed=42,
        )
        config = make_rank_trend_config(n_null_samples=50, random_seed=42)
        result = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        names = {mr["metric_name"] for mr in result["metric_results"]}
        assert "seasonal_max" in names
        assert "seasonal_min" in names
        assert "avg_across_metrics_hot" in names
        assert "avg_across_metrics_cold" in names

    def test_metric_result_fields_valid(self) -> None:
        """Each metric result has valid field values."""
        data = create_synthetic_trending_metrics(
            n_years=10,
            n_locations=2,
            seed=42,
        )
        config = make_rank_trend_config(n_null_samples=50, random_seed=42)
        result = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        for mr in result["metric_results"]:
            assert 0.0 <= mr["p_value"] <= 1.0
            assert mr["n_years"] == 10
            assert mr["spatial_dof"] >= 1
            assert mr["is_significant"] in (True, False)
            assert mr["observed_slope"] == mr["observed_slope"]  # not NaN

    def test_reproducible_with_same_config(self) -> None:
        """Same inputs and config produce identical results."""
        data = create_synthetic_trending_metrics(
            n_years=10,
            n_locations=2,
            seed=42,
        )
        config = make_rank_trend_config(n_null_samples=50, random_seed=42)
        r1 = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        r2 = run_rank_trend_analysis(
            metrics=data["metrics"],
            metric_names=data["metric_names"],
            latitudes=data["latitudes"],
            config=config,
        )
        for m1, m2 in zip(r1["metric_results"], r2["metric_results"], strict=True):
            assert m1["metric_name"] == m2["metric_name"]
            assert m1["observed_slope"] == m2["observed_slope"]
            assert m1["p_value"] == m2["p_value"]
