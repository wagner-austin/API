"""Tests for temporal TypedDict definitions, require_* validation, and encode/decode.

Tests cover: TemporalFeatureConfig, SeasonalCycleCoefficients, TailThresholds,
TemporalFeatureState, HeatMetricResult, RankTrendConfig, MetricTrendResult,
RankTrendResult, and their encode/decode round-trips.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from covenant_ml.datasets.types import (
    COLD_RANKED_METRICS,
    DEFAULT_TEMPORAL_FEATURE_CONFIG,
    HEAT_METRIC_NAMES,
    HOT_RANKED_METRICS,
    HeatMetricResult,
    MetricTrendResult,
    RankTrendConfig,
    RankTrendResult,
    SeasonalCycleCoefficients,
    TailThresholds,
    TemporalFeatureConfig,
    TemporalFeatureState,
    encode_heat_metric_result,
    encode_metric_trend_result,
    encode_rank_trend_result,
    encode_temporal_feature_state,
    make_metric_trend_result,
    make_rank_trend_config,
    make_rank_trend_result,
    require_heat_metric_result,
    require_metric_trend_result,
    require_rank_trend_config,
    require_rank_trend_result,
    require_seasonal_cycle_coefficients,
    require_tail_thresholds,
    require_temporal_feature_config,
    require_temporal_feature_state,
)


def _make_test_state() -> TemporalFeatureState:
    """Create a test TemporalFeatureState for encode/decode tests."""
    return TemporalFeatureState(
        config=TemporalFeatureConfig(
            n_fourier_harmonics=2,
            hot_cutoff_percentile=95.0,
            cold_cutoff_percentile=5.0,
            season="warm",
            season_months=(6, 7, 8),
            compute_ar1=True,
        ),
        seasonal_cycle=SeasonalCycleCoefficients(
            n_harmonics=2,
            cos_coefficients=((1.5, 2.5), (3.5, 4.5)),
            sin_coefficients=((0.5, 1.5), (2.5, 3.5)),
            mean=(20.0, 22.0),
            n_days_per_year=365,
        ),
        thresholds=TailThresholds(
            hot_threshold=(2.0, 2.5),
            cold_threshold=(-2.0, -2.5),
            hot_percentile=95.0,
            cold_percentile=5.0,
        ),
        median_baseline=(0.3, -0.2),
        n_locations=2,
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


class TestRequireTailThresholds:
    """Tests for require_tail_thresholds validation."""

    def test_valid(self) -> None:
        """Accepts valid per-location data."""
        data: dict[str, JSONValue] = {
            "hot_threshold": [2.5, 3.0],
            "cold_threshold": [-2.5, -3.0],
            "hot_percentile": 95.0,
            "cold_percentile": 5.0,
        }

        thresholds = require_tail_thresholds(data, "test", n_locations=2)

        assert thresholds["hot_threshold"] == (2.5, 3.0)
        assert thresholds["cold_threshold"] == (-2.5, -3.0)

    def test_wrong_length(self) -> None:
        """Raises when threshold tuple length doesn't match n_locations."""
        data: dict[str, JSONValue] = {
            "hot_threshold": [2.5],
            "cold_threshold": [-2.5, -3.0],
            "hot_percentile": 95.0,
            "cold_percentile": 5.0,
        }

        with pytest.raises(ValueError, match="length 1 != expected 2"):
            require_tail_thresholds(data, "test", n_locations=2)

    def test_numeric_wrong_type(self) -> None:
        """Raises when threshold percentile is non-numeric."""
        data: dict[str, JSONValue] = {
            "hot_threshold": [2.5],
            "cold_threshold": [-2.5],
            "hot_percentile": "ninety-five",
            "cold_percentile": 5.0,
        }

        with pytest.raises(ValueError, match="must be numeric"):
            require_tail_thresholds(data, "test", n_locations=1)

    def test_threshold_not_sequence(self) -> None:
        """Raises when hot_threshold is a scalar instead of sequence."""
        data: dict[str, JSONValue] = {
            "hot_threshold": 2.5,
            "cold_threshold": [-2.5],
            "hot_percentile": 95.0,
            "cold_percentile": 5.0,
        }

        with pytest.raises(ValueError, match="must be tuple of floats"):
            require_tail_thresholds(data, "test", n_locations=1)

    def test_threshold_element_bool(self) -> None:
        """Raises when threshold element is bool instead of numeric."""
        data: dict[str, JSONValue] = {
            "hot_threshold": [True],
            "cold_threshold": [-2.5],
            "hot_percentile": 95.0,
            "cold_percentile": 5.0,
        }

        with pytest.raises(ValueError, match="must be numeric"):
            require_tail_thresholds(data, "test", n_locations=1)


class TestRequireTemporalFeatureState:
    """Tests for require_temporal_feature_state validation."""

    def test_valid(self) -> None:
        """Accepts valid complete data."""
        data: dict[str, JSONValue] = {
            "config": {
                "n_fourier_harmonics": 2,
                "hot_cutoff_percentile": 95.0,
                "cold_cutoff_percentile": 5.0,
                "season": "warm",
                "season_months": [6, 7, 8],
                "compute_ar1": True,
            },
            "seasonal_cycle": {
                "n_harmonics": 2,
                "cos_coefficients": [[1.0], [2.0]],
                "sin_coefficients": [[3.0], [4.0]],
                "mean": [20.0],
                "n_days_per_year": 365,
            },
            "thresholds": {
                "hot_threshold": [2.0],
                "cold_threshold": [-2.0],
                "hot_percentile": 95.0,
                "cold_percentile": 5.0,
            },
            "median_baseline": [0.5],
            "n_locations": 1,
        }

        state = require_temporal_feature_state(data)

        assert state["n_locations"] == 1
        assert state["config"]["n_fourier_harmonics"] == 2
        assert state["seasonal_cycle"]["cos_coefficients"] == ((1.0,), (2.0,))
        assert state["thresholds"]["hot_threshold"] == (2.0,)
        assert state["median_baseline"] == (0.5,)

    def test_missing_n_locations(self) -> None:
        """Raises when n_locations is missing."""
        data: dict[str, JSONValue] = {
            "config": {
                "n_fourier_harmonics": 1,
                "hot_cutoff_percentile": 95.0,
                "cold_cutoff_percentile": 5.0,
                "season": "warm",
                "season_months": [6],
                "compute_ar1": False,
            },
            "seasonal_cycle": {
                "n_harmonics": 1,
                "cos_coefficients": [[1.0]],
                "sin_coefficients": [[0.5]],
                "mean": [20.0],
                "n_days_per_year": 365,
            },
            "thresholds": {
                "hot_threshold": [2.0],
                "cold_threshold": [-2.0],
                "hot_percentile": 95.0,
                "cold_percentile": 5.0,
            },
            "median_baseline": [0.0],
        }

        with pytest.raises(ValueError, match="n_locations must be positive integer"):
            require_temporal_feature_state(data)

    def test_missing_config(self) -> None:
        """Raises when config dict is missing."""
        data: dict[str, JSONValue] = {
            "n_locations": 1,
            "seasonal_cycle": {
                "n_harmonics": 1,
                "cos_coefficients": [[1.0]],
                "sin_coefficients": [[0.5]],
                "mean": [20.0],
                "n_days_per_year": 365,
            },
            "thresholds": {
                "hot_threshold": [2.0],
                "cold_threshold": [-2.0],
                "hot_percentile": 95.0,
                "cold_percentile": 5.0,
            },
            "median_baseline": [0.0],
        }

        with pytest.raises(ValueError, match="config must be dictionary"):
            require_temporal_feature_state(data)


class TestRequireHeatMetricResult:
    """Tests for require_heat_metric_result validation."""

    def test_valid(self) -> None:
        """Accepts valid data."""
        data: dict[str, JSONValue] = {
            "entity_id": "loc_0",
            "n_years": 2,
            "metric_names": ["seasonal_max", "seasonal_min"],
            "values": [[10.0, -5.0], [12.0, -3.0]],
        }

        result = require_heat_metric_result(data)

        assert result["entity_id"] == "loc_0"
        assert result["n_years"] == 2
        assert result["metric_names"] == ("seasonal_max", "seasonal_min")
        assert result["values"] == ((10.0, -5.0), (12.0, -3.0))

    def test_missing_entity_id(self) -> None:
        """Raises when entity_id is missing."""
        data: dict[str, JSONValue] = {
            "n_years": 1,
            "metric_names": ["seasonal_max"],
            "values": [[10.0]],
        }

        with pytest.raises(ValueError, match="entity_id must be string"):
            require_heat_metric_result(data)

    def test_wrong_values_shape(self) -> None:
        """Raises when values inner length doesn't match metric_names."""
        data: dict[str, JSONValue] = {
            "entity_id": "loc_0",
            "n_years": 1,
            "metric_names": ["seasonal_max", "seasonal_min"],
            "values": [[10.0]],
        }

        with pytest.raises(ValueError, match="length 1 != expected 2"):
            require_heat_metric_result(data)

    def test_metric_names_not_sequence(self) -> None:
        """Raises when metric_names is not a sequence."""
        data: dict[str, JSONValue] = {
            "entity_id": "loc_0",
            "n_years": 1,
            "metric_names": "seasonal_max",
            "values": [[10.0]],
        }

        with pytest.raises(ValueError, match="must be tuple of strings"):
            require_heat_metric_result(data)

    def test_metric_names_element_not_str(self) -> None:
        """Raises when metric_names element is not a string."""
        data: dict[str, JSONValue] = {
            "entity_id": "loc_0",
            "n_years": 1,
            "metric_names": [42],
            "values": [[10.0]],
        }

        with pytest.raises(ValueError, match="must be string"):
            require_heat_metric_result(data)


class TestEncodeDecodeRoundTrip:
    """Tests for encode/decode round-trip integrity."""

    def test_temporal_feature_state_roundtrip(self) -> None:
        """Encode then decode produces identical TemporalFeatureState."""
        original = _make_test_state()

        encoded = encode_temporal_feature_state(original)
        decoded = require_temporal_feature_state(encoded)

        assert decoded["config"] == original["config"]
        assert decoded["seasonal_cycle"] == original["seasonal_cycle"]
        assert decoded["thresholds"] == original["thresholds"]
        assert decoded["median_baseline"] == original["median_baseline"]
        assert decoded["n_locations"] == original["n_locations"]

    def test_temporal_feature_state_json_serializable(self) -> None:
        """Encoded state is valid JSON."""
        state = _make_test_state()
        encoded = encode_temporal_feature_state(state)

        json_str = dump_json_str(encoded)
        parsed = narrow_json_to_dict(load_json_str(json_str))

        decoded = require_temporal_feature_state(parsed)
        assert decoded["n_locations"] == 2

    def test_roundtrip_byte_identical(self) -> None:
        """encode -> decode -> re-encode produces byte-identical JSON."""
        original = _make_test_state()

        encoded1 = encode_temporal_feature_state(original)
        json1 = dump_json_str(encoded1)

        decoded = require_temporal_feature_state(encoded1)
        encoded2 = encode_temporal_feature_state(decoded)
        json2 = dump_json_str(encoded2)

        assert json1 == json2

    def test_heat_metric_result_roundtrip(self) -> None:
        """Encode then decode produces identical HeatMetricResult."""
        original = HeatMetricResult(
            entity_id="station_42",
            n_years=3,
            metric_names=("seasonal_max", "seasonal_min", "ar1"),
            values=(
                (10.0, -5.0, 0.8),
                (12.0, -3.0, 0.7),
                (11.0, -4.0, 0.75),
            ),
        )

        encoded = encode_heat_metric_result(original)
        decoded = require_heat_metric_result(encoded)

        assert decoded["entity_id"] == original["entity_id"]
        assert decoded["n_years"] == original["n_years"]
        assert decoded["metric_names"] == original["metric_names"]
        assert decoded["values"] == original["values"]

    def test_heat_metric_result_json_serializable(self) -> None:
        """Encoded HeatMetricResult is valid JSON."""
        result = HeatMetricResult(
            entity_id="loc_0",
            n_years=1,
            metric_names=("seasonal_max",),
            values=((10.0,),),
        )
        encoded = encode_heat_metric_result(result)

        json_str = dump_json_str(encoded)
        parsed = narrow_json_to_dict(load_json_str(json_str))

        decoded = require_heat_metric_result(parsed)
        assert decoded["entity_id"] == "loc_0"

    def test_metric_trend_result_roundtrip(self) -> None:
        """Encode then decode produces identical MetricTrendResult."""
        original = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=0.05,
            p_value=0.02,
            is_significant=True,
            n_years=30,
            spatial_dof=15,
        )

        encoded = encode_metric_trend_result(original)
        decoded = require_metric_trend_result(encoded)

        assert decoded["metric_name"] == original["metric_name"]
        assert decoded["observed_slope"] == original["observed_slope"]
        assert decoded["p_value"] == original["p_value"]
        assert decoded["is_significant"] == original["is_significant"]
        assert decoded["n_years"] == original["n_years"]
        assert decoded["spatial_dof"] == original["spatial_dof"]

    def test_rank_trend_result_roundtrip(self) -> None:
        """Encode then decode produces identical RankTrendResult."""
        metric = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=0.05,
            p_value=0.02,
            is_significant=True,
            n_years=30,
            spatial_dof=15,
        )
        original = make_rank_trend_result(
            metric_results=(metric,),
            n_null_samples=1000,
            random_seed=42,
        )

        encoded = encode_rank_trend_result(original)
        decoded = require_rank_trend_result(encoded)

        assert decoded["n_null_samples"] == original["n_null_samples"]
        assert decoded["random_seed"] == original["random_seed"]
        assert len(decoded["metric_results"]) == 1
        assert decoded["metric_results"][0]["metric_name"] == "seasonal_max"

    def test_rank_trend_result_json_serializable(self) -> None:
        """Encoded RankTrendResult is valid JSON."""
        metric = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=0.05,
            p_value=0.02,
            is_significant=True,
            n_years=30,
            spatial_dof=15,
        )
        result = make_rank_trend_result(
            metric_results=(metric,),
            n_null_samples=1000,
            random_seed=42,
        )
        encoded = encode_rank_trend_result(result)

        json_str = dump_json_str(encoded)
        parsed = narrow_json_to_dict(load_json_str(json_str))

        decoded = require_rank_trend_result(parsed)
        assert decoded["n_null_samples"] == 1000


class TestRankingConstants:
    """Tests for HOT_RANKED_METRICS and COLD_RANKED_METRICS constants."""

    def test_hot_ranked_metrics_count(self) -> None:
        """HOT_RANKED_METRICS has 6 metrics."""
        assert len(HOT_RANKED_METRICS) == 6

    def test_cold_ranked_metrics_count(self) -> None:
        """COLD_RANKED_METRICS has 3 metrics."""
        assert len(COLD_RANKED_METRICS) == 3

    def test_all_metrics_covered(self) -> None:
        """Every metric in HEAT_METRIC_NAMES is in HOT or COLD."""
        for name in HEAT_METRIC_NAMES:
            assert name in HOT_RANKED_METRICS or name in COLD_RANKED_METRICS

    def test_ndays_excess_cold_in_hot(self) -> None:
        """ndays_excess_cold is in HOT (higher count = more extreme)."""
        assert "ndays_excess_cold" in HOT_RANKED_METRICS

    def test_ar1_in_hot(self) -> None:
        """ar1 is in HOT (higher autocorrelation = more persistence)."""
        assert "ar1" in HOT_RANKED_METRICS


class TestRankTrendConfig:
    """Tests for RankTrendConfig TypedDict and factory."""

    def test_structure(self) -> None:
        """RankTrendConfig has all required fields."""
        config: RankTrendConfig = {
            "n_null_samples": 1000,
            "random_seed": 42,
        }
        assert config["n_null_samples"] == 1000
        assert config["random_seed"] == 42

    def test_factory_valid(self) -> None:
        """make_rank_trend_config creates valid config."""
        config = make_rank_trend_config(n_null_samples=500, random_seed=0)
        assert config["n_null_samples"] == 500
        assert config["random_seed"] == 0

    def test_factory_rejects_zero_samples(self) -> None:
        """make_rank_trend_config raises on n_null_samples < 1."""
        with pytest.raises(ValueError, match="n_null_samples must be >= 1"):
            make_rank_trend_config(n_null_samples=0, random_seed=42)

    def test_factory_rejects_negative_seed(self) -> None:
        """make_rank_trend_config raises on negative random_seed."""
        with pytest.raises(ValueError, match="random_seed must be >= 0"):
            make_rank_trend_config(n_null_samples=100, random_seed=-1)

    def test_require_valid(self) -> None:
        """require_rank_trend_config accepts valid data."""
        data: dict[str, JSONValue] = {
            "n_null_samples": 1000,
            "random_seed": 42,
        }
        config = require_rank_trend_config(data, "test")
        assert config["n_null_samples"] == 1000

    def test_require_invalid_samples(self) -> None:
        """require_rank_trend_config raises on non-positive samples."""
        data: dict[str, JSONValue] = {
            "n_null_samples": 0,
            "random_seed": 42,
        }
        with pytest.raises(ValueError, match="must be positive integer"):
            require_rank_trend_config(data, "test")


class TestMetricTrendResult:
    """Tests for MetricTrendResult TypedDict and factory."""

    def test_structure(self) -> None:
        """MetricTrendResult has all required fields."""
        result: MetricTrendResult = {
            "metric_name": "seasonal_max",
            "observed_slope": 0.05,
            "p_value": 0.02,
            "is_significant": True,
            "n_years": 30,
            "spatial_dof": 15,
        }
        assert result["metric_name"] == "seasonal_max"
        assert result["is_significant"] is True

    def test_factory_valid(self) -> None:
        """make_metric_trend_result creates valid result."""
        result = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=-0.1,
            p_value=0.5,
            is_significant=False,
            n_years=20,
            spatial_dof=10,
        )
        assert result["observed_slope"] == -0.1

    def test_factory_rejects_empty_name(self) -> None:
        """make_metric_trend_result raises on empty metric_name."""
        with pytest.raises(ValueError, match="metric_name must not be empty"):
            make_metric_trend_result(
                metric_name="",
                observed_slope=0.0,
                p_value=0.5,
                is_significant=False,
                n_years=10,
                spatial_dof=5,
            )

    def test_factory_rejects_invalid_pvalue(self) -> None:
        """make_metric_trend_result raises on p_value out of [0, 1]."""
        with pytest.raises(ValueError, match="p_value must be in"):
            make_metric_trend_result(
                metric_name="seasonal_max",
                observed_slope=0.0,
                p_value=1.5,
                is_significant=False,
                n_years=10,
                spatial_dof=5,
            )

    def test_factory_rejects_too_few_years(self) -> None:
        """make_metric_trend_result raises on n_years < 2."""
        with pytest.raises(ValueError, match="n_years must be >= 2"):
            make_metric_trend_result(
                metric_name="seasonal_max",
                observed_slope=0.0,
                p_value=0.5,
                is_significant=False,
                n_years=1,
                spatial_dof=5,
            )

    def test_factory_rejects_zero_spatial_dof(self) -> None:
        """make_metric_trend_result raises on spatial_dof < 1."""
        with pytest.raises(ValueError, match="spatial_dof must be >= 1"):
            make_metric_trend_result(
                metric_name="seasonal_max",
                observed_slope=0.0,
                p_value=0.5,
                is_significant=False,
                n_years=10,
                spatial_dof=0,
            )


class TestRankTrendResult:
    """Tests for RankTrendResult TypedDict and factory."""

    def test_structure(self) -> None:
        """RankTrendResult has all required fields."""
        metric = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=0.05,
            p_value=0.02,
            is_significant=True,
            n_years=30,
            spatial_dof=15,
        )
        result: RankTrendResult = {
            "metric_results": (metric,),
            "n_null_samples": 1000,
            "random_seed": 42,
        }
        assert len(result["metric_results"]) == 1

    def test_factory_valid(self) -> None:
        """make_rank_trend_result creates valid result."""
        metric = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=0.05,
            p_value=0.02,
            is_significant=True,
            n_years=30,
            spatial_dof=15,
        )
        result = make_rank_trend_result(
            metric_results=(metric,),
            n_null_samples=1000,
            random_seed=42,
        )
        assert result["n_null_samples"] == 1000

    def test_factory_rejects_empty_results(self) -> None:
        """make_rank_trend_result raises on empty metric_results."""
        with pytest.raises(ValueError, match="metric_results must not be empty"):
            make_rank_trend_result(
                metric_results=(),
                n_null_samples=1000,
                random_seed=42,
            )

    def test_factory_rejects_zero_null_samples(self) -> None:
        """make_rank_trend_result raises on n_null_samples < 1."""
        metric = make_metric_trend_result(
            metric_name="seasonal_max",
            observed_slope=0.05,
            p_value=0.02,
            is_significant=True,
            n_years=30,
            spatial_dof=15,
        )
        with pytest.raises(ValueError, match="n_null_samples must be >= 1"):
            make_rank_trend_result(
                metric_results=(metric,),
                n_null_samples=0,
                random_seed=42,
            )

    def test_require_rejects_negative_seed(self) -> None:
        """require_rank_trend_config raises on negative random_seed."""
        data: dict[str, JSONValue] = {
            "n_null_samples": 100,
            "random_seed": -1,
        }
        with pytest.raises(ValueError, match="non-negative integer"):
            require_rank_trend_config(data, "config")

    def test_require_invalid_metric_results_type(self) -> None:
        """require_rank_trend_result raises when metric_results is not a list."""
        data: dict[str, JSONValue] = {
            "metric_results": "not a list",
            "n_null_samples": 100,
            "random_seed": 42,
        }
        with pytest.raises(ValueError, match="metric_results must be a list"):
            require_rank_trend_result(data)

    def test_require_invalid_metric_results_item(self) -> None:
        """require_rank_trend_result raises when item is not a dict."""
        data: dict[str, JSONValue] = {
            "metric_results": ["not a dict"],
            "n_null_samples": 100,
            "random_seed": 42,
        }
        with pytest.raises(ValueError, match="must be a dictionary"):
            require_rank_trend_result(data)
