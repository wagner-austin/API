"""Tests for McKinnon-style temporal feature extraction functions.

Tests cover all 9 public functions in _netcdf_temporal.py: Fourier seasonal
cycle fitting/removal, within-season median computation, residual computation,
tail threshold fitting, heat metric computation, fit/transform orchestration,
and feature name building.
"""

from __future__ import annotations

import pytest

from covenant_ml.datasets.loaders._netcdf_heat_metrics import (
    fit_temporal_features,
    transform_temporal_features,
)
from covenant_ml.datasets.loaders._netcdf_temporal import (
    compute_residuals,
    compute_within_season_medians,
    fit_seasonal_cycle,
    fit_tail_thresholds,
    remove_seasonal_cycle,
    select_season,
)
from covenant_ml.datasets.testing import create_synthetic_daily_timeseries
from covenant_ml.datasets.types_temporal import (
    TailThresholds,
    TemporalFeatureConfig,
)
from tests.datasets.loaders._netcdf_fixtures import (
    _has_nan,
    _i64,
    _make_config,
)


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
