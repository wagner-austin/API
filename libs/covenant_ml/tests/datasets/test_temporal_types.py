"""Tests for temporal TypedDict definitions, require_* validation, and encode/decode.

Tests cover: TemporalFeatureConfig, SeasonalCycleCoefficients, TailThresholds,
TemporalFeatureState, HeatMetricResult, RankTrendConfig, MetricTrendResult,
RankTrendResult, and their encode/decode round-trips.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONValue,
)

from covenant_ml.datasets.types_temporal import (
    DEFAULT_TEMPORAL_FEATURE_CONFIG,
    HEAT_METRIC_NAMES,
    HeatMetricResult,
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
    require_seasonal_cycle_coefficients,
    require_temporal_feature_config,
)


class TestTypedDictStructure:
    """Tests for TypedDict field presence and basic construction."""

    def test_temporal_feature_config_has_all_fields(self) -> None:
        """TemporalFeatureConfig TypedDict has all required keys."""
        config: TemporalFeatureConfig = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": (6, 7, 8),
            "compute_ar1": True,
        }
        assert config["n_fourier_harmonics"] == 5
        assert config["season"] == "warm"

    def test_seasonal_cycle_coefficients_multi_location(self) -> None:
        """SeasonalCycleCoefficients stores per-location coefficients."""
        coeffs: SeasonalCycleCoefficients = {
            "n_harmonics": 2,
            "cos_coefficients": ((1.0, 2.0), (3.0, 4.0)),
            "sin_coefficients": ((5.0, 6.0), (7.0, 8.0)),
            "mean": (20.0, 22.0),
            "n_days_per_year": 365,
        }
        assert len(coeffs["cos_coefficients"]) == 2
        assert len(coeffs["cos_coefficients"][0]) == 2
        assert coeffs["mean"][1] == 22.0

    def test_tail_thresholds_per_location(self) -> None:
        """TailThresholds stores per-location threshold values."""
        thresholds: TailThresholds = {
            "hot_threshold": (2.5, 3.0),
            "cold_threshold": (-2.5, -3.0),
            "hot_percentile": 95.0,
            "cold_percentile": 5.0,
        }
        assert len(thresholds["hot_threshold"]) == 2
        assert thresholds["cold_threshold"][1] == -3.0

    def test_temporal_feature_state_has_n_locations(self) -> None:
        """TemporalFeatureState includes n_locations field."""
        state: TemporalFeatureState = {
            "config": DEFAULT_TEMPORAL_FEATURE_CONFIG,
            "seasonal_cycle": {
                "n_harmonics": 1,
                "cos_coefficients": ((1.0,),),
                "sin_coefficients": ((0.5,),),
                "mean": (20.0,),
                "n_days_per_year": 365,
            },
            "thresholds": {
                "hot_threshold": (2.0,),
                "cold_threshold": (-2.0,),
                "hot_percentile": 95.0,
                "cold_percentile": 5.0,
            },
            "median_baseline": (0.0,),
            "n_locations": 1,
        }
        assert state["n_locations"] == 1

    def test_heat_metric_result_has_all_fields(self) -> None:
        """HeatMetricResult TypedDict has all required keys."""
        result: HeatMetricResult = {
            "entity_id": "loc_0",
            "n_years": 2,
            "metric_names": HEAT_METRIC_NAMES,
            "values": (
                (1.0, -1.0, 0.5, 0.5, 1.0, -0.5, -0.5, 1.0, 0.9),
                (2.0, -2.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0, 0.8),
            ),
        }
        assert result["entity_id"] == "loc_0"
        assert result["n_years"] == 2
        assert len(result["values"]) == 2
        assert len(result["values"][0]) == 9


class TestDefaultConfig:
    """Tests for DEFAULT_TEMPORAL_FEATURE_CONFIG constant."""

    def test_matches_mckinnon_defaults(self) -> None:
        """DEFAULT_TEMPORAL_FEATURE_CONFIG matches McKinnon PNAS 2024 defaults."""
        assert DEFAULT_TEMPORAL_FEATURE_CONFIG["n_fourier_harmonics"] == 5
        assert DEFAULT_TEMPORAL_FEATURE_CONFIG["hot_cutoff_percentile"] == 95.0
        assert DEFAULT_TEMPORAL_FEATURE_CONFIG["cold_cutoff_percentile"] == 5.0
        assert DEFAULT_TEMPORAL_FEATURE_CONFIG["season"] == "warm"
        assert DEFAULT_TEMPORAL_FEATURE_CONFIG["season_months"] == (6, 7, 8)
        assert DEFAULT_TEMPORAL_FEATURE_CONFIG["compute_ar1"] is True


class TestRequireTemporalFeatureConfig:
    """Tests for require_temporal_feature_config validation."""

    def test_valid(self) -> None:
        """Accepts valid data."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6, 7, 8],
            "compute_ar1": True,
        }

        config = require_temporal_feature_config(data, "test")

        assert config["n_fourier_harmonics"] == 5
        assert config["season"] == "warm"

    def test_all_seasons(self) -> None:
        """Accepts all season values."""
        for season in ("warm", "cold", "full_year"):
            data: dict[str, JSONValue] = {
                "n_fourier_harmonics": 3,
                "hot_cutoff_percentile": 90.0,
                "cold_cutoff_percentile": 10.0,
                "season": season,
                "season_months": [1],
                "compute_ar1": False,
            }
            config = require_temporal_feature_config(data, "test")
            assert config["season"] == season

    def test_invalid_harmonics(self) -> None:
        """Raises on non-positive harmonics."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 0,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="must be positive integer"):
            require_temporal_feature_config(data, "test")

    def test_invalid_percentile(self) -> None:
        """Raises on out-of-range percentile."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 100.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="between 0 and 100 exclusive"):
            require_temporal_feature_config(data, "test")

    def test_invalid_season(self) -> None:
        """Raises on invalid season string."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "summer",
            "season_months": [6],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="must be 'warm', 'cold', or 'full_year'"):
            require_temporal_feature_config(data, "test")

    def test_invalid_month(self) -> None:
        """Raises on out-of-range month."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [13],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match=r"must be int in 1\.\.12"):
            require_temporal_feature_config(data, "test")

    def test_empty_months(self) -> None:
        """Raises on empty season_months."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="must be non-empty"):
            require_temporal_feature_config(data, "test")

    def test_bool_not_int(self) -> None:
        """Rejects bool where int expected."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": True,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="must be positive integer"):
            require_temporal_feature_config(data, "test")

    def test_percentile_wrong_type(self) -> None:
        """Raises when percentile is a string instead of numeric."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": "high",
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="must be numeric"):
            require_temporal_feature_config(data, "test")

    def test_percentile_bool(self) -> None:
        """Raises when percentile is bool."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": True,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6],
            "compute_ar1": True,
        }

        with pytest.raises(ValueError, match="must be numeric"):
            require_temporal_feature_config(data, "test")

    def test_compute_ar1_not_bool(self) -> None:
        """Raises when compute_ar1 is not a bool."""
        data: dict[str, JSONValue] = {
            "n_fourier_harmonics": 5,
            "hot_cutoff_percentile": 95.0,
            "cold_cutoff_percentile": 5.0,
            "season": "warm",
            "season_months": [6],
            "compute_ar1": 1,
        }

        with pytest.raises(ValueError, match="must be bool"):
            require_temporal_feature_config(data, "test")


class TestRequireSeasonalCycleCoefficients:
    """Tests for require_seasonal_cycle_coefficients validation."""

    def test_valid(self) -> None:
        """Accepts valid nested data."""
        data: dict[str, JSONValue] = {
            "n_harmonics": 2,
            "cos_coefficients": [[1.0, 2.0], [3.0, 4.0]],
            "sin_coefficients": [[5.0, 6.0], [7.0, 8.0]],
            "mean": [20.0, 22.0],
            "n_days_per_year": 365,
        }

        coeffs = require_seasonal_cycle_coefficients(data, "test", n_locations=2)

        assert coeffs["n_harmonics"] == 2
        assert coeffs["cos_coefficients"] == ((1.0, 2.0), (3.0, 4.0))
        assert coeffs["mean"] == (20.0, 22.0)

    def test_wrong_inner_length(self) -> None:
        """Raises when inner tuple length doesn't match n_locations."""
        data: dict[str, JSONValue] = {
            "n_harmonics": 1,
            "cos_coefficients": [[1.0, 2.0, 3.0]],
            "sin_coefficients": [[5.0, 6.0]],
            "mean": [20.0, 22.0],
            "n_days_per_year": 365,
        }

        with pytest.raises(ValueError, match="length 3 != expected 2"):
            require_seasonal_cycle_coefficients(data, "test", n_locations=2)

    def test_wrong_outer_length(self) -> None:
        """Raises when outer tuple length doesn't match n_harmonics."""
        data: dict[str, JSONValue] = {
            "n_harmonics": 2,
            "cos_coefficients": [[1.0]],
            "sin_coefficients": [[5.0], [6.0]],
            "mean": [20.0],
            "n_days_per_year": 365,
        }

        with pytest.raises(ValueError, match="outer length 1 != expected 2"):
            require_seasonal_cycle_coefficients(data, "test", n_locations=1)

    def test_not_sequence(self) -> None:
        """Raises when cos_coefficients is not a sequence."""
        data: dict[str, JSONValue] = {
            "n_harmonics": 1,
            "cos_coefficients": 1.0,
            "sin_coefficients": [[0.5]],
            "mean": [20.0],
            "n_days_per_year": 365,
        }

        with pytest.raises(ValueError, match="must be nested tuple of floats"):
            require_seasonal_cycle_coefficients(data, "test", n_locations=1)
