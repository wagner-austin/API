"""Rank-trend dataset types: metric trends over ranked heat metrics."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import JSONValue

from covenant_ml.datasets._json_fields import (
    _require_bool_field,
    _require_non_negative_int,
    _require_numeric,
    _require_positive_int,
    _require_str_field,
)

HOT_RANKED_METRICS: tuple[str, ...] = (
    "seasonal_max",
    "cum_excess_hot",
    "avg_excess_hot",
    "ndays_excess_hot",
    "ndays_excess_cold",
    "ar1",
)


COLD_RANKED_METRICS: tuple[str, ...] = (
    "seasonal_min",
    "cum_excess_cold",
    "avg_excess_cold",
)


class RankTrendConfig(TypedDict, total=True):
    """Configuration for rank-trend significance testing.

    Controls the Monte Carlo null distribution generation for
    assessing whether temporal trends in ranked metrics are
    statistically significant.

    Attributes:
        n_null_samples: Number of Monte Carlo permutation samples
            for building the null distribution of trend slopes.
        random_seed: Random seed for reproducibility of null samples.
    """

    n_null_samples: int
    random_seed: int


class MetricTrendResult(TypedDict, total=True):
    """Result of trend significance testing for a single metric.

    Contains the observed OLS slope of spatially-averaged ranks
    over time, the two-sided p-value from the Monte Carlo null
    distribution, and whether the trend is significant at p < 0.05.

    Attributes:
        metric_name: Name of the heat metric tested.
        observed_slope: OLS slope of rank-vs-year regression.
        p_value: Two-sided p-value from null distribution.
        is_significant: Whether p_value < 0.05.
        n_years: Number of years in the time series.
        spatial_dof: Estimated spatial degrees of freedom
            (Bretherton et al. 1999).
    """

    metric_name: str
    observed_slope: float
    p_value: float
    is_significant: bool
    n_years: int
    spatial_dof: int


class RankTrendResult(TypedDict, total=True):
    """Complete result of rank-trend analysis across all metrics.

    Produced by ``run_rank_trend_analysis``. Contains per-metric
    results and the configuration used.

    Attributes:
        metric_results: Per-metric trend test results.
        n_null_samples: Number of Monte Carlo samples used.
        random_seed: Random seed used for reproducibility.
    """

    metric_results: tuple[MetricTrendResult, ...]
    n_null_samples: int
    random_seed: int


def make_rank_trend_config(
    *,
    n_null_samples: int,
    random_seed: int,
) -> RankTrendConfig:
    """Create a validated RankTrendConfig.

    Args:
        n_null_samples: Number of Monte Carlo permutation samples.
        random_seed: Random seed for reproducibility.

    Returns:
        Immutable RankTrendConfig TypedDict.

    Raises:
        ValueError: If n_null_samples < 1 or random_seed < 0.
    """
    if n_null_samples < 1:
        raise ValueError(f"n_null_samples must be >= 1, got {n_null_samples}")
    if random_seed < 0:
        raise ValueError(f"random_seed must be >= 0, got {random_seed}")
    return RankTrendConfig(
        n_null_samples=n_null_samples,
        random_seed=random_seed,
    )


def make_metric_trend_result(
    *,
    metric_name: str,
    observed_slope: float,
    p_value: float,
    is_significant: bool,
    n_years: int,
    spatial_dof: int,
) -> MetricTrendResult:
    """Create a validated MetricTrendResult.

    Args:
        metric_name: Name of the heat metric tested.
        observed_slope: OLS slope of rank-vs-year regression.
        p_value: Two-sided p-value from null distribution.
        is_significant: Whether p_value < 0.05.
        n_years: Number of years in the time series.
        spatial_dof: Estimated spatial degrees of freedom.

    Returns:
        Immutable MetricTrendResult TypedDict.

    Raises:
        ValueError: If metric_name is empty, p_value not in [0, 1],
            n_years < 2, or spatial_dof < 1.
    """
    if not metric_name:
        raise ValueError("metric_name must not be empty")
    if not (0.0 <= p_value <= 1.0):
        raise ValueError(f"p_value must be in [0, 1], got {p_value}")
    if n_years < 2:
        raise ValueError(f"n_years must be >= 2, got {n_years}")
    if spatial_dof < 1:
        raise ValueError(f"spatial_dof must be >= 1, got {spatial_dof}")
    return MetricTrendResult(
        metric_name=metric_name,
        observed_slope=observed_slope,
        p_value=p_value,
        is_significant=is_significant,
        n_years=n_years,
        spatial_dof=spatial_dof,
    )


def make_rank_trend_result(
    *,
    metric_results: tuple[MetricTrendResult, ...],
    n_null_samples: int,
    random_seed: int,
) -> RankTrendResult:
    """Create a validated RankTrendResult.

    Args:
        metric_results: Per-metric trend test results.
        n_null_samples: Number of Monte Carlo samples used.
        random_seed: Random seed used for reproducibility.

    Returns:
        Immutable RankTrendResult TypedDict.

    Raises:
        ValueError: If metric_results is empty or n_null_samples < 1.
    """
    if len(metric_results) == 0:
        raise ValueError("metric_results must not be empty")
    if n_null_samples < 1:
        raise ValueError(f"n_null_samples must be >= 1, got {n_null_samples}")
    return RankTrendResult(
        metric_results=metric_results,
        n_null_samples=n_null_samples,
        random_seed=random_seed,
    )


def require_rank_trend_config(
    data: dict[str, JSONValue],
    key: str,
) -> RankTrendConfig:
    """Validate and extract RankTrendConfig from parsed data.

    Args:
        data: Dictionary containing rank trend configuration.
        key: Key name for error messages.

    Returns:
        Validated RankTrendConfig TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    return make_rank_trend_config(
        n_null_samples=_require_positive_int(data.get("n_null_samples"), f"{key}.n_null_samples"),
        random_seed=_require_non_negative_int(data.get("random_seed"), f"{key}.random_seed"),
    )


def require_metric_trend_result(
    data: dict[str, JSONValue],
) -> MetricTrendResult:
    """Validate and extract MetricTrendResult from parsed data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated MetricTrendResult TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    p_value = _require_numeric(data.get("p_value"), "p_value")
    return make_metric_trend_result(
        metric_name=_require_str_field(data.get("metric_name"), "metric_name"),
        observed_slope=_require_numeric(data.get("observed_slope"), "observed_slope"),
        p_value=p_value,
        is_significant=_require_bool_field(data.get("is_significant"), "is_significant"),
        n_years=_require_positive_int(data.get("n_years"), "n_years"),
        spatial_dof=_require_positive_int(data.get("spatial_dof"), "spatial_dof"),
    )


def require_rank_trend_result(
    data: dict[str, JSONValue],
) -> RankTrendResult:
    """Validate and extract RankTrendResult from parsed data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated RankTrendResult TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    raw_results = data.get("metric_results")
    if not isinstance(raw_results, list):
        raise ValueError("metric_results must be a list")
    results: list[MetricTrendResult] = []
    for i, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise ValueError(f"metric_results[{i}] must be a dictionary")
        results.append(require_metric_trend_result(item))
    return make_rank_trend_result(
        metric_results=tuple(results),
        n_null_samples=_require_positive_int(data.get("n_null_samples"), "n_null_samples"),
        random_seed=_require_non_negative_int(data.get("random_seed"), "random_seed"),
    )


def encode_metric_trend_result(
    result: MetricTrendResult,
) -> dict[str, JSONValue]:
    """Encode MetricTrendResult to JSON-serializable dictionary.

    Args:
        result: Validated result to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    return {
        "metric_name": result["metric_name"],
        "observed_slope": result["observed_slope"],
        "p_value": result["p_value"],
        "is_significant": result["is_significant"],
        "n_years": result["n_years"],
        "spatial_dof": result["spatial_dof"],
    }


def encode_rank_trend_result(
    result: RankTrendResult,
) -> dict[str, JSONValue]:
    """Encode RankTrendResult to JSON-serializable dictionary.

    Args:
        result: Validated result to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    encoded_metrics: list[JSONValue] = []
    for metric in result["metric_results"]:
        encoded_metrics.append(encode_metric_trend_result(metric))
    return {
        "metric_results": encoded_metrics,
        "n_null_samples": result["n_null_samples"],
        "random_seed": result["random_seed"],
    }


__all__ = [
    "COLD_RANKED_METRICS",
    "HOT_RANKED_METRICS",
    "MetricTrendResult",
    "RankTrendConfig",
    "RankTrendResult",
    "encode_metric_trend_result",
    "encode_rank_trend_result",
    "make_metric_trend_result",
    "make_rank_trend_config",
    "make_rank_trend_result",
    "require_metric_trend_result",
    "require_rank_trend_config",
    "require_rank_trend_result",
]
