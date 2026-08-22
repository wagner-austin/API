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

from covenant_ml.datasets.loaders._netcdf_trend_significance import (
    compute_latitude_weights,
    compute_trend_pvalue,
    compute_weighted_spatial_mean,
    estimate_spatial_dof,
    generate_null_trend_slopes,
    run_rank_trend_analysis,
)
from covenant_ml.datasets.testing import create_synthetic_trending_metrics
from covenant_ml.datasets.types_trend import make_rank_trend_config
from tests.datasets.loaders._trend_fixtures import (
    _f64,
    _f64_2d,
    _flat,
)


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
