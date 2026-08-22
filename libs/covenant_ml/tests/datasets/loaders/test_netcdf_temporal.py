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
    compute_residuals,
    compute_within_season_medians,
    fit_seasonal_cycle,
    fit_tail_thresholds,
    remove_seasonal_cycle,
    select_season,
)
from covenant_ml.datasets.testing import create_synthetic_daily_timeseries
from tests.datasets.loaders._netcdf_fixtures import (
    _f64,
    _f64_2d,
    _i64,
    _max_abs,
    _repeat_i64,
    _val2,
    _variance,
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
