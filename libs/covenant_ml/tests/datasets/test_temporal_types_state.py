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

from covenant_ml.datasets.types_temporal import (
    HeatMetricResult,
    encode_heat_metric_result,
    encode_temporal_feature_state,
    require_heat_metric_result,
    require_tail_thresholds,
    require_temporal_feature_state,
)
from covenant_ml.datasets.types_trend import (
    encode_metric_trend_result,
    encode_rank_trend_result,
    make_metric_trend_result,
    make_rank_trend_result,
    require_metric_trend_result,
    require_rank_trend_result,
)
from tests.datasets._temporal_types_fixtures import (
    _make_test_state,
)


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
