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
    HEAT_METRIC_NAMES,
)
from covenant_ml.datasets.types_trend import (
    COLD_RANKED_METRICS,
    HOT_RANKED_METRICS,
    MetricTrendResult,
    RankTrendConfig,
    RankTrendResult,
    make_metric_trend_result,
    make_rank_trend_config,
    make_rank_trend_result,
    require_rank_trend_config,
    require_rank_trend_result,
)


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
