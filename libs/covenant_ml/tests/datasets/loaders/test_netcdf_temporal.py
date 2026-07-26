"""Tests for McKinnon-style temporal feature extraction functions.

Tests cover all 9 public functions in _netcdf_temporal.py: Fourier seasonal
cycle fitting/removal, within-season median computation, residual computation,
tail threshold fitting, heat metric computation, fit/transform orchestration,
and feature name building.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._netcdf_temporal import (
    build_temporal_feature_names,
    compute_heat_metrics,
    compute_residuals,
    compute_within_season_medians,
    fit_seasonal_cycle,
    fit_tail_thresholds,
    fit_temporal_features,
    remove_seasonal_cycle,
    select_season,
    transform_temporal_features,
)
from covenant_ml.datasets.testing import create_synthetic_daily_timeseries
from covenant_ml.datasets.types import (
    HEAT_METRIC_NAMES,
    HEAT_METRIC_NAMES_NO_AR1,
    TailThresholds,
    TemporalFeatureConfig,
)


def _val(arr: NDArray[np.float64], i: int, j: int, k: int) -> float:
    """Extract a typed float from a 3D NDArray (avoids mypy Any from indexing)."""
    row: NDArray[np.float64] = arr[i, j]
    return float(row.flat[k])


def _val2(arr: NDArray[np.float64], i: int, j: int) -> float:
    """Extract a typed float from a 2D NDArray (avoids mypy Any from indexing)."""
    row: NDArray[np.float64] = arr[i]
    return float(row.flat[j])


def _i64(values: list[int]) -> NDArray[np.int64]:
    """Create int64 array from typed list (avoids mypy list[Any] error)."""
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for idx, v in enumerate(values):
        result[idx] = v
    return result


def _f64(values: list[float]) -> NDArray[np.float64]:
    """Create float64 array from typed list (avoids mypy list[Any] error)."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for idx, v in enumerate(values):
        result[idx] = v
    return result


def _f64_2d(values: list[list[float]]) -> NDArray[np.float64]:
    """Create 2D float64 array from nested list (avoids mypy list[Any] error)."""
    rows = len(values)
    cols = len(values[0])
    result: NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            result[i, j] = v
    return result


def _repeat_i64(segments: list[tuple[int, int]]) -> NDArray[np.int64]:
    """Create int64 array by repeating values (avoids np.full/np.concatenate Any).

    Args:
        segments: List of (value, count) pairs.

    Returns:
        1D int64 array with each value repeated count times.
    """
    total = sum(count for _, count in segments)
    result: NDArray[np.int64] = np.zeros(total, dtype=np.int64)
    offset = 0
    for value, count in segments:
        for i in range(count):
            result[offset + i] = value
        offset += count
    return result


def _max_abs(arr: NDArray[np.float64]) -> float:
    """Compute max absolute value (avoids mypy Any from np.abs/np.max)."""
    flat: NDArray[np.float64] = arr.ravel()
    max_val: float = 0.0
    for i in range(int(flat.shape[0])):
        val = abs(float(flat.flat[i]))
        if val > max_val:
            max_val = val
    return max_val


def _variance(arr: NDArray[np.float64]) -> float:
    """Compute variance (avoids mypy Any from .var())."""
    flat: NDArray[np.float64] = arr.ravel()
    n = int(flat.shape[0])
    total: float = 0.0
    for i in range(n):
        total += float(flat.flat[i])
    mean = total / n
    sq_sum: float = 0.0
    for i in range(n):
        diff = float(flat.flat[i]) - mean
        sq_sum += diff * diff
    return sq_sum / n


def _has_nan(arr: NDArray[np.float64]) -> bool:
    """Check if array has any NaN values (avoids mypy Any from np.isnan)."""
    flat: NDArray[np.float64] = arr.ravel()
    for i in range(int(flat.shape[0])):
        val = float(flat.flat[i])
        if val != val:  # NaN != NaN
            return True
    return False


def _make_config(compute_ar1: bool = True) -> TemporalFeatureConfig:
    """Create a temporal feature config for testing."""
    return TemporalFeatureConfig(
        n_fourier_harmonics=3,
        hot_cutoff_percentile=95.0,
        cold_cutoff_percentile=5.0,
        season="warm",
        season_months=(6, 7, 8),
        compute_ar1=compute_ar1,
    )


class TestFitSeasonalCycle:
    """Tests for fit_seasonal_cycle function."""

    def test_recovers_known_coefficients(self) -> None:
        """fit_seasonal_cycle recovers known Fourier coefficients from clean data."""
        data = create_synthetic_daily_timeseries(
            n_years=10,
            n_locations=2,
            n_harmonics=3,
            seed=42,
            noise_std=0.0,
        )

        coeffs = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], n_harmonics=3)

        assert coeffs["n_harmonics"] == 3
        assert coeffs["n_days_per_year"] == 365
        for j in range(2):
            assert abs(coeffs["mean"][j] - data["true_mean"][j]) < 0.01
            for k in range(3):
                assert (
                    abs(coeffs["cos_coefficients"][k][j] - data["true_cos_coefficients"][k][j])
                    < 0.01
                )
                assert (
                    abs(coeffs["sin_coefficients"][k][j] - data["true_sin_coefficients"][k][j])
                    < 0.01
                )

    def test_returns_correct_structure(self) -> None:
        """fit_seasonal_cycle returns SeasonalCycleCoefficients with all fields."""
        data = create_synthetic_daily_timeseries(
            n_years=3,
            n_locations=2,
            n_harmonics=2,
            seed=42,
        )

        coeffs = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], n_harmonics=2)

        assert len(coeffs["cos_coefficients"]) == 2
        assert len(coeffs["sin_coefficients"]) == 2
        assert len(coeffs["cos_coefficients"][0]) == 2  # n_locations
        assert len(coeffs["mean"]) == 2

    def test_custom_n_days_per_year(self) -> None:
        """fit_seasonal_cycle accepts custom n_days_per_year."""
        data = create_synthetic_daily_timeseries(
            n_years=3,
            n_locations=1,
            n_harmonics=2,
            seed=42,
        )

        coeffs = fit_seasonal_cycle(
            data["daily_values"],
            data["day_of_year"],
            n_harmonics=2,
            n_days_per_year=366,
        )

        assert coeffs["n_days_per_year"] == 366

    def test_raises_on_1d_input(self) -> None:
        """fit_seasonal_cycle raises ValueError on 1D input."""
        values = np.ones(100, dtype=np.float64)
        doy = np.ones(100, dtype=np.int64)

        with pytest.raises(ValueError, match="must be 2D"):
            fit_seasonal_cycle(values, doy, n_harmonics=3)

    def test_raises_on_shape_mismatch(self) -> None:
        """fit_seasonal_cycle raises ValueError on mismatched array lengths."""
        values = np.ones((100, 2), dtype=np.float64)
        doy = np.ones(50, dtype=np.int64)

        with pytest.raises(ValueError, match="Shape mismatch"):
            fit_seasonal_cycle(values, doy, n_harmonics=3)

    def test_raises_on_empty(self) -> None:
        """fit_seasonal_cycle raises ValueError on empty arrays."""
        values = np.zeros((0, 2), dtype=np.float64)
        doy = np.zeros(0, dtype=np.int64)

        with pytest.raises(ValueError, match="Cannot fit seasonal cycle to empty"):
            fit_seasonal_cycle(values, doy, n_harmonics=3)

    def test_single_harmonic(self) -> None:
        """fit_seasonal_cycle works with a single harmonic."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=1,
            n_harmonics=1,
            seed=42,
            noise_std=0.0,
        )

        coeffs = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], n_harmonics=1)

        assert len(coeffs["cos_coefficients"]) == 1
        assert abs(coeffs["cos_coefficients"][0][0] - data["true_cos_coefficients"][0][0]) < 0.01


class TestRemoveSeasonalCycle:
    """Tests for remove_seasonal_cycle function."""

    def test_zero_residuals_for_clean_data(self) -> None:
        """Removing seasonal cycle from noise-free data gives near-zero anomalies."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
            noise_std=0.0,
        )

        coeffs = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], n_harmonics=3)
        anomalies = remove_seasonal_cycle(data["daily_values"], data["day_of_year"], coeffs)

        assert _max_abs(anomalies) < 0.01

    def test_preserves_shape(self) -> None:
        """remove_seasonal_cycle output has same shape as input."""
        data = create_synthetic_daily_timeseries(
            n_years=3,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )

        coeffs = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], n_harmonics=3)
        anomalies = remove_seasonal_cycle(data["daily_values"], data["day_of_year"], coeffs)

        assert anomalies.shape == data["daily_values"].shape

    def test_reduces_variance(self) -> None:
        """Deseasonalized data has lower variance than raw data."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
            noise_std=1.0,
        )

        coeffs = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], n_harmonics=3)
        anomalies = remove_seasonal_cycle(data["daily_values"], data["day_of_year"], coeffs)

        var_anom = _variance(anomalies)
        var_raw = _variance(data["daily_values"])
        assert var_anom < var_raw


class TestComputeWithinSeasonMedians:
    """Tests for compute_within_season_medians function."""

    def test_correct_shape(self) -> None:
        """compute_within_season_medians returns (n_years, n_locations) medians."""
        anomalies = np.zeros((6, 2), dtype=np.float64)
        anomalies[:3, 0] = [1.0, 3.0, 5.0]
        anomalies[3:, 0] = [10.0, 20.0, 30.0]
        anomalies[:3, 1] = [2.0, 4.0, 6.0]
        anomalies[3:, 1] = [11.0, 21.0, 31.0]
        year_labels = _i64([2000, 2000, 2000, 2001, 2001, 2001])

        medians, unique_years = compute_within_season_medians(anomalies, year_labels)

        assert medians.shape == (2, 2)
        assert int(unique_years.shape[0]) == 2

    def test_known_values(self) -> None:
        """compute_within_season_medians gives correct medians for known data."""
        anomalies = np.zeros((6, 2), dtype=np.float64)
        anomalies[:3, 0] = [1.0, 3.0, 5.0]
        anomalies[3:, 0] = [10.0, 20.0, 30.0]
        anomalies[:3, 1] = [2.0, 4.0, 6.0]
        anomalies[3:, 1] = [11.0, 21.0, 31.0]
        year_labels = _i64([2000, 2000, 2000, 2001, 2001, 2001])

        medians, unique_years = compute_within_season_medians(anomalies, year_labels)

        assert int(unique_years.flat[0]) == 2000
        assert abs(_val2(medians, 0, 0) - 3.0) < 1e-10
        assert abs(_val2(medians, 0, 1) - 4.0) < 1e-10
        assert abs(_val2(medians, 1, 0) - 20.0) < 1e-10
        assert abs(_val2(medians, 1, 1) - 21.0) < 1e-10

    def test_sorted_years(self) -> None:
        """Unique years are sorted in ascending order."""
        anomalies = np.zeros((6, 1), dtype=np.float64)
        year_labels = _i64([2002, 2002, 2000, 2000, 2001, 2001])

        _, unique_years = compute_within_season_medians(anomalies, year_labels)

        years_list = [int(unique_years.flat[i]) for i in range(3)]
        assert years_list == [2000, 2001, 2002]


class TestComputeResiduals:
    """Tests for compute_residuals function."""

    def test_subtracts_median(self) -> None:
        """compute_residuals subtracts the correct year's median per location."""
        anomalies = np.zeros((6, 2), dtype=np.float64)
        anomalies[:3, 0] = [1.0, 3.0, 5.0]
        anomalies[3:, 0] = [10.0, 20.0, 30.0]
        anomalies[:3, 1] = [2.0, 4.0, 6.0]
        anomalies[3:, 1] = [11.0, 21.0, 31.0]
        year_labels = _i64([2000, 2000, 2000, 2001, 2001, 2001])
        medians = _f64_2d([[3.0, 4.0], [20.0, 21.0]])
        unique_years = _i64([2000, 2001])

        residuals = compute_residuals(anomalies, year_labels, medians, unique_years)

        expected_0 = _f64([-2.0, 0.0, 2.0, -10.0, 0.0, 10.0])
        expected_1 = _f64([-2.0, 0.0, 2.0, -10.0, 0.0, 10.0])
        np.testing.assert_allclose(residuals[:, 0], expected_0, atol=1e-10)
        np.testing.assert_allclose(residuals[:, 1], expected_1, atol=1e-10)

    def test_preserves_shape(self) -> None:
        """compute_residuals output has same shape as anomalies."""
        anomalies = np.zeros((10, 3), dtype=np.float64)
        year_labels = _repeat_i64([(2000, 5), (2001, 5)])
        medians = np.zeros((2, 3), dtype=np.float64)
        unique_years = _i64([2000, 2001])

        residuals = compute_residuals(anomalies, year_labels, medians, unique_years)

        assert residuals.shape == (10, 3)

    def test_does_not_mutate_input(self) -> None:
        """compute_residuals does not modify the input anomalies array."""
        anomalies = _f64_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        original = anomalies.copy()
        year_labels = _i64([2000, 2000, 2000])
        medians = _f64_2d([[3.0, 4.0]])
        unique_years = _i64([2000])

        compute_residuals(anomalies, year_labels, medians, unique_years)

        np.testing.assert_array_equal(anomalies, original)


class TestFitTailThresholds:
    """Tests for fit_tail_thresholds function."""

    def test_correct_structure(self) -> None:
        """fit_tail_thresholds returns TailThresholds with per-location values."""
        residuals = np.zeros((100, 2), dtype=np.float64)
        residuals[:, 0] = np.arange(100, dtype=np.float64)
        residuals[:, 1] = np.arange(100, dtype=np.float64) * 2.0

        thresholds = fit_tail_thresholds(residuals, 95.0, 5.0)

        assert len(thresholds["hot_threshold"]) == 2
        assert len(thresholds["cold_threshold"]) == 2
        assert thresholds["hot_percentile"] == 95.0
        assert thresholds["cold_percentile"] == 5.0

    def test_hot_greater_than_cold(self) -> None:
        """Hot threshold is greater than cold threshold at each location."""
        rng = np.random.default_rng(42)
        residuals: NDArray[np.float64] = rng.standard_normal((1000, 3)).astype(np.float64)

        thresholds = fit_tail_thresholds(residuals, 95.0, 5.0)

        for j in range(3):
            assert thresholds["hot_threshold"][j] > thresholds["cold_threshold"][j]

    def test_location_independence(self) -> None:
        """Each location gets its own threshold based on its own distribution."""
        residuals = np.zeros((100, 2), dtype=np.float64)
        residuals[:, 0] = np.arange(100, dtype=np.float64)
        residuals[:, 1] = np.arange(100, dtype=np.float64) * 2.0

        thresholds = fit_tail_thresholds(residuals, 95.0, 5.0)

        assert thresholds["hot_threshold"][1] > thresholds["hot_threshold"][0] * 1.5


class TestComputeHeatMetrics:
    """Tests for compute_heat_metrics function."""

    def test_correct_shape_with_ar1(self) -> None:
        """compute_heat_metrics returns (n_years, n_locations, 9) with ar1."""
        residuals = np.zeros((200, 2), dtype=np.float64)
        residuals[:100, :] = np.arange(100, dtype=np.float64).reshape(-1, 1)
        residuals[100:, :] = np.arange(100, dtype=np.float64).reshape(-1, 1) + 50.0
        year_labels = _repeat_i64([(2000, 100), (2001, 100)])
        thresholds = TailThresholds(
            hot_threshold=(90.0, 90.0),
            cold_threshold=(10.0, 10.0),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=True)

        assert result.shape == (2, 2, 9)

    def test_correct_shape_without_ar1(self) -> None:
        """compute_heat_metrics returns (n_years, n_locations, 8) without ar1."""
        residuals = np.arange(100, dtype=np.float64).reshape(-1, 1)
        year_labels = np.full(100, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(90.0,),
            cold_threshold=(10.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert result.shape == (1, 1, 8)

    def test_seasonal_max_min(self) -> None:
        """compute_heat_metrics correctly computes seasonal max and min."""
        residuals = _f64([-5.0, 0.0, 3.0, 10.0, -2.0]).reshape(-1, 1)
        year_labels = np.full(5, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(100.0,),
            cold_threshold=(-100.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert abs(_val(result, 0, 0, 0) - 10.0) < 1e-10
        assert abs(_val(result, 0, 0, 1) - (-5.0)) < 1e-10

    def test_hot_excess(self) -> None:
        """compute_heat_metrics correctly computes hot-tail excess metrics."""
        residuals = _f64([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
        year_labels = np.full(5, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(3.5,),
            cold_threshold=(-100.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert abs(_val(result, 0, 0, 2) - 9.0) < 1e-10
        assert abs(_val(result, 0, 0, 3) - 4.5) < 1e-10
        assert abs(_val(result, 0, 0, 4) - 2.0) < 1e-10

    def test_cold_excess(self) -> None:
        """compute_heat_metrics correctly computes cold-tail excess metrics."""
        residuals = _f64([-5.0, -3.0, 0.0, 2.0, 4.0]).reshape(-1, 1)
        year_labels = np.full(5, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(100.0,),
            cold_threshold=(-2.5,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert abs(_val(result, 0, 0, 5) - (-8.0)) < 1e-10
        assert abs(_val(result, 0, 0, 6) - (-4.0)) < 1e-10
        assert abs(_val(result, 0, 0, 7) - 2.0) < 1e-10

    def test_no_hot_days(self) -> None:
        """compute_heat_metrics returns zeros when no days exceed hot threshold."""
        residuals = _f64([1.0, 2.0, 3.0]).reshape(-1, 1)
        year_labels = np.full(3, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(100.0,),
            cold_threshold=(-100.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert _val(result, 0, 0, 2) == 0.0
        assert _val(result, 0, 0, 3) == 0.0
        assert _val(result, 0, 0, 4) == 0.0

    def test_no_cold_days(self) -> None:
        """compute_heat_metrics returns zeros when no days below cold threshold."""
        residuals = _f64([1.0, 2.0, 3.0]).reshape(-1, 1)
        year_labels = np.full(3, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(100.0,),
            cold_threshold=(-100.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert _val(result, 0, 0, 5) == 0.0
        assert _val(result, 0, 0, 6) == 0.0
        assert _val(result, 0, 0, 7) == 0.0

    def test_ar1_positive_correlation(self) -> None:
        """AR(1) is positive for monotonically increasing series."""
        residuals = np.arange(100, dtype=np.float64).reshape(-1, 1)
        year_labels = np.full(100, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(1000.0,),
            cold_threshold=(-1000.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=True)

        assert _val(result, 0, 0, 8) > 0.9

    def test_ar1_short_series(self) -> None:
        """AR(1) returns 0.0 for series with fewer than 3 values."""
        residuals = _f64([1.0, 2.0]).reshape(-1, 1)
        year_labels = np.full(2, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(1000.0,),
            cold_threshold=(-1000.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=True)

        assert _val(result, 0, 0, 8) == 0.0

    def test_ar1_constant_series(self) -> None:
        """AR(1) returns 0.0 for constant series (zero variance)."""
        residuals = np.full((10, 1), 5.0, dtype=np.float64)
        year_labels = np.full(10, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(1000.0,),
            cold_threshold=(-1000.0,),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=True)

        assert _val(result, 0, 0, 8) == 0.0

    def test_per_location_thresholds(self) -> None:
        """Each location uses its own threshold for hot/cold classification."""
        residuals = np.zeros((5, 2), dtype=np.float64)
        residuals[:, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]
        residuals[:, 1] = [1.0, 2.0, 3.0, 4.0, 5.0]
        year_labels = np.full(5, 2000, dtype=np.int64)
        thresholds = TailThresholds(
            hot_threshold=(3.5, 1.5),
            cold_threshold=(-100.0, -100.0),
            hot_percentile=95.0,
            cold_percentile=5.0,
        )

        result = compute_heat_metrics(residuals, year_labels, thresholds, compute_ar1=False)

        assert abs(_val(result, 0, 0, 4) - 2.0) < 1e-10
        assert abs(_val(result, 0, 1, 4) - 4.0) < 1e-10


class TestFitTemporalFeatures:
    """Tests for fit_temporal_features orchestrator."""

    def test_returns_complete_state(self) -> None:
        """fit_temporal_features returns TemporalFeatureState with all fields."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        config = _make_config()

        state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )

        assert state["config"] == config
        assert state["n_locations"] == 2
        assert state["seasonal_cycle"]["n_harmonics"] == 3
        assert len(state["thresholds"]["hot_threshold"]) == 2
        assert len(state["median_baseline"]) == 2

    def test_deterministic(self) -> None:
        """fit_temporal_features produces identical state for identical input."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        config = _make_config()

        state1 = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )
        state2 = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )

        assert state1["seasonal_cycle"]["mean"] == state2["seasonal_cycle"]["mean"]
        assert state1["thresholds"]["hot_threshold"] == state2["thresholds"]["hot_threshold"]


class TestTransformTemporalFeatures:
    """Tests for transform_temporal_features function."""

    def test_returns_flattened_matrix(self) -> None:
        """transform returns (n_years * n_locations, n_metrics) matrix."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        config = _make_config()

        state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )
        result = transform_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            state,
        )

        assert result.shape == (5 * 2, 9)

    def test_without_ar1(self) -> None:
        """transform respects compute_ar1=False."""
        data = create_synthetic_daily_timeseries(
            n_years=3,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        config = _make_config(compute_ar1=False)

        state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )
        result = transform_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            state,
        )

        assert result.shape == (3 * 2, 8)

    def test_on_new_data(self) -> None:
        """transform works on different data using fitted state."""
        train = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        test_data = create_synthetic_daily_timeseries(
            n_years=3,
            n_locations=2,
            n_harmonics=3,
            seed=99,
        )
        config = _make_config()

        state = fit_temporal_features(
            train["daily_values"],
            train["day_of_year"],
            train["month_labels"],
            train["year_labels"],
            config,
        )
        result = transform_temporal_features(
            test_data["daily_values"],
            test_data["day_of_year"],
            test_data["month_labels"],
            test_data["year_labels"],
            state,
        )

        assert result.shape == (3 * 2, 9)

    def test_no_nan(self) -> None:
        """transform produces no NaN values."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        config = _make_config()

        state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )
        result = transform_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            state,
        )

        assert not _has_nan(result)


class TestBuildTemporalFeatureNames:
    """Tests for build_temporal_feature_names function."""

    def test_with_ar1(self) -> None:
        """build_temporal_feature_names returns 9 names with AR1."""
        config = _make_config(compute_ar1=True)
        names = build_temporal_feature_names(config)

        assert names == HEAT_METRIC_NAMES
        assert len(names) == 9

    def test_without_ar1(self) -> None:
        """build_temporal_feature_names returns 8 names without AR1."""
        config = _make_config(compute_ar1=False)
        names = build_temporal_feature_names(config)

        assert names == HEAT_METRIC_NAMES_NO_AR1
        assert len(names) == 8


class TestSelectSeason:
    """Tests for select_season, which is the pipeline's one restriction point."""

    def test_selects_only_the_named_months(self) -> None:
        """The mask is true exactly on the configured months."""
        months = _i64([1, 5, 6, 7, 8, 9, 12])

        mask = select_season(months, (6, 7, 8))

        assert [bool(mask.flat[i]) for i in range(int(mask.shape[0]))] == [
            False,
            False,
            True,
            True,
            True,
            False,
            False,
        ]

    def test_a_single_month_season_is_allowed(self) -> None:
        """A season may be one month; nothing requires three."""
        months = _i64([6, 7, 8])

        mask = select_season(months, (7,))

        assert [bool(mask.flat[i]) for i in range(int(mask.shape[0]))] == [False, True, False]

    def test_empty_season_is_rejected(self) -> None:
        """An empty season would fit the thresholds on nothing."""
        with pytest.raises(ValueError, match="season_months is empty"):
            select_season(_i64([6, 7, 8]), ())

    def test_non_month_value_is_rejected(self) -> None:
        """A month outside 1-12 is a mistake, not an empty selection."""
        with pytest.raises(ValueError, match="non-months"):
            select_season(_i64([6, 7, 8]), (6, 13))

    def test_season_matching_no_observed_month_is_rejected(self) -> None:
        """Winter months against summer data must fail, not select nothing.

        Silently selecting nothing would reach fit_tail_thresholds with an
        empty array and produce NaN thresholds, which compare false against
        everything and so never flag an extreme.
        """
        with pytest.raises(ValueError, match="match none of the observed"):
            select_season(_i64([6, 7, 8]), (12, 1, 2))


class TestSeasonalCycleConditioning:
    """A cycle that the observations cannot determine must be refused."""

    def test_narrow_window_is_rejected(self) -> None:
        """Summer days alone cannot determine an annual Fourier basis.

        Over a 92-day window the harmonics of a 365-day period are nearly
        collinear, so the solve returns coefficients around 1e5 that cancel
        inside the window and diverge outside it. The fit looks excellent
        exactly where it was fitted, so nothing downstream can notice.
        """
        summer_doy = _i64(list(range(152, 244)))
        values = _f64_2d([[20.0 + 0.1 * day] for day in range(152, 244)])

        with pytest.raises(ValueError, match="not determined by these observations"):
            fit_seasonal_cycle(values, summer_doy, 5)

    def test_rejection_names_the_span_and_the_remedy(self) -> None:
        """The message must say what was too narrow and what to do instead."""
        summer_doy = _i64(list(range(152, 244)))
        values = _f64_2d([[20.0 + 0.1 * day] for day in range(152, 244)])

        with pytest.raises(ValueError, match=r"span 92 of 365 days"):
            fit_seasonal_cycle(values, summer_doy, 5)

        with pytest.raises(ValueError, match="full year"):
            fit_seasonal_cycle(values, summer_doy, 5)

    def test_full_year_is_accepted(self) -> None:
        """A year of daily observations determines the basis comfortably."""
        data = create_synthetic_daily_timeseries(
            n_years=2,
            n_locations=1,
            n_harmonics=5,
            seed=11,
        )

        cycle = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], 5)

        assert len(cycle["cos_coefficients"]) == 5

    def test_full_year_fit_stays_bounded_all_year(self) -> None:
        """Reconstruction must be physical on every day, not just in season.

        The streaming extractor accepts any day-of-year and evaluates the
        fitted series directly. A fit that only holds inside the season
        turns every off-season observation into an enormous anomaly, and so
        into a critical alert.
        """
        data = create_synthetic_daily_timeseries(
            n_years=3,
            n_locations=1,
            n_harmonics=5,
            seed=12,
            seasonal_amplitude=10.0,
            mean_value=20.0,
        )
        cycle = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], 5)

        every_day = _i64(list(range(1, 366)))
        single_column = _f64_2d([[0.0] for _ in range(365)])
        reconstruction = single_column - remove_seasonal_cycle(single_column, every_day, cycle)

        assert _max_abs(reconstruction) < 100.0


class TestSeasonRestrictionInFit:
    """The season selects which days the thresholds describe."""

    def test_season_months_changes_the_thresholds(self) -> None:
        """Two configs differing only in season must not agree.

        season_months was declared, validated and serialized while no
        computation read it, so this assertion held trivially in reverse:
        every season produced identical thresholds.
        """
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        summer = TemporalFeatureConfig(
            n_fourier_harmonics=3,
            hot_cutoff_percentile=95.0,
            cold_cutoff_percentile=5.0,
            season="warm",
            season_months=(6, 7, 8),
            compute_ar1=False,
        )
        winter = TemporalFeatureConfig(
            n_fourier_harmonics=3,
            hot_cutoff_percentile=95.0,
            cold_cutoff_percentile=5.0,
            season="cold",
            season_months=(12, 1, 2),
            compute_ar1=False,
        )

        summer_state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            summer,
        )
        winter_state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            winter,
        )

        assert (
            summer_state["thresholds"]["hot_threshold"]
            != winter_state["thresholds"]["hot_threshold"]
        )

    def test_seasonal_cycle_does_not_depend_on_the_season(self) -> None:
        """The cycle is fitted on the whole year, so the season cannot move it."""
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        summer = _make_config(compute_ar1=False)
        winter = TemporalFeatureConfig(
            n_fourier_harmonics=3,
            hot_cutoff_percentile=95.0,
            cold_cutoff_percentile=5.0,
            season="cold",
            season_months=(12, 1, 2),
            compute_ar1=False,
        )

        summer_state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            summer,
        )
        winter_state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            winter,
        )

        assert summer_state["seasonal_cycle"] == winter_state["seasonal_cycle"]

    def test_thresholds_match_a_hand_restricted_fit(self) -> None:
        """The orchestrator restricts exactly where the primitives would.

        Spelling the chain out by hand pins which stages see the whole year
        and which see one season, so a future edit that moves the boundary
        fails here rather than shifting every threshold quietly.
        """
        data = create_synthetic_daily_timeseries(
            n_years=5,
            n_locations=2,
            n_harmonics=3,
            seed=42,
        )
        config = _make_config(compute_ar1=False)

        state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )

        cycle = fit_seasonal_cycle(data["daily_values"], data["day_of_year"], 3)
        anomalies = remove_seasonal_cycle(data["daily_values"], data["day_of_year"], cycle)
        in_season = select_season(data["month_labels"], (6, 7, 8))
        medians, years = compute_within_season_medians(
            anomalies[in_season], data["year_labels"][in_season]
        )
        residuals = compute_residuals(
            anomalies[in_season], data["year_labels"][in_season], medians, years
        )
        expected: TailThresholds = fit_tail_thresholds(residuals, 95.0, 5.0)

        assert state["thresholds"] == expected

    def test_season_absent_from_the_data_is_rejected(self) -> None:
        """A season the observations never cover cannot be fitted."""
        data = create_synthetic_daily_timeseries(
            n_years=2,
            n_locations=1,
            n_harmonics=3,
            seed=7,
        )
        january_only = _i64([1] * int(data["month_labels"].shape[0]))
        config = _make_config(compute_ar1=False)

        with pytest.raises(ValueError, match="match none of the observed"):
            fit_temporal_features(
                data["daily_values"],
                data["day_of_year"],
                january_only,
                data["year_labels"],
                config,
            )


class TestEndToEndPipeline:
    """Integration tests for the full temporal feature pipeline."""

    def test_full_pipeline_known_seasonal_recovery(self) -> None:
        """Full pipeline recovers known seasonal cycle and produces valid metrics."""
        data = create_synthetic_daily_timeseries(
            n_years=10,
            n_locations=2,
            n_harmonics=3,
            seed=42,
            noise_std=0.5,
        )
        config = _make_config()

        state = fit_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            config,
        )

        cycle = state["seasonal_cycle"]
        for j in range(2):
            for k in range(3):
                assert (
                    abs(cycle["cos_coefficients"][k][j] - data["true_cos_coefficients"][k][j]) < 0.5
                )
                assert (
                    abs(cycle["sin_coefficients"][k][j] - data["true_sin_coefficients"][k][j]) < 0.5
                )

        result = transform_temporal_features(
            data["daily_values"],
            data["day_of_year"],
            data["month_labels"],
            data["year_labels"],
            state,
        )

        assert result.shape == (10 * 2, 9)
        assert not _has_nan(result)
