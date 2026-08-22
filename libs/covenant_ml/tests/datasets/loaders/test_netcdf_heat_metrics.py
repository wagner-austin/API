"""Tests for McKinnon-style temporal feature extraction functions.

Tests cover all 9 public functions in _netcdf_temporal.py: Fourier seasonal
cycle fitting/removal, within-season median computation, residual computation,
tail threshold fitting, heat metric computation, fit/transform orchestration,
and feature name building.
"""

from __future__ import annotations

import numpy as np

from covenant_ml.datasets.loaders._netcdf_heat_metrics import (
    build_temporal_feature_names,
    compute_heat_metrics,
    fit_temporal_features,
    transform_temporal_features,
)
from covenant_ml.datasets.testing import create_synthetic_daily_timeseries
from covenant_ml.datasets.types_temporal import (
    HEAT_METRIC_NAMES,
    HEAT_METRIC_NAMES_NO_AR1,
    TailThresholds,
)
from tests.datasets.loaders._netcdf_fixtures import (
    _f64,
    _has_nan,
    _make_config,
    _repeat_i64,
    _val,
)


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
