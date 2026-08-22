"""Temporal-feature dataset types: heat metrics, seasonal cycles, tails."""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import JSONValue

from covenant_ml.datasets._json_fields import (
    SeasonDefinition,
    _require_bool_field,
    _require_float_tuple,
    _require_json_dict,
    _require_month_tuple,
    _require_nested_float_tuple,
    _require_numeric,
    _require_percentile,
    _require_positive_int,
    _require_season,
    _require_str_field,
    _require_str_tuple,
)

HeatMetricName = Literal[
    "seasonal_max",
    "seasonal_min",
    "cum_excess_hot",
    "avg_excess_hot",
    "ndays_excess_hot",
    "cum_excess_cold",
    "avg_excess_cold",
    "ndays_excess_cold",
    "ar1",
]


HEAT_METRIC_NAMES: tuple[str, ...] = (
    "seasonal_max",
    "seasonal_min",
    "cum_excess_hot",
    "avg_excess_hot",
    "ndays_excess_hot",
    "cum_excess_cold",
    "avg_excess_cold",
    "ndays_excess_cold",
    "ar1",
)


HEAT_METRIC_NAMES_NO_AR1: tuple[str, ...] = HEAT_METRIC_NAMES[:-1]


class TemporalFeatureConfig(TypedDict, total=True):
    """Configuration for McKinnon-style temporal feature extraction.

    Controls Fourier deseasonalization, tail-excess thresholds, and
    season bounds for heat metric computation.

    Attributes:
        n_fourier_harmonics: Number of Fourier harmonics for seasonal cycle
            removal. McKinnon uses 5 at annual frequency.
        hot_cutoff_percentile: Percentile for hot-tail threshold (0-100).
        cold_cutoff_percentile: Percentile for cold-tail threshold (0-100).
        season: Which season to analyze.
        season_months: Month numbers defining the season.
        compute_ar1: Whether to compute lag-1 autocorrelation.
    """

    n_fourier_harmonics: int
    hot_cutoff_percentile: float
    cold_cutoff_percentile: float
    season: SeasonDefinition
    season_months: tuple[int, ...]
    compute_ar1: bool


class SeasonalCycleCoefficients(TypedDict, total=True):
    """Fitted Fourier coefficients for seasonal cycle removal.

    Immutable state from fitting step. Stores coefficients for
    reconstructing the seasonal cycle per location via::

        y(doy, loc) = mean[loc] + sum_k(cos_k[loc] * cos(2*pi*k*doy/N)
                                       + sin_k[loc] * sin(2*pi*k*doy/N))

    Attributes:
        n_harmonics: Number of harmonics used.
        cos_coefficients: Cosine coefficients, shape (n_harmonics, n_locations).
        sin_coefficients: Sine coefficients, shape (n_harmonics, n_locations).
        mean: Mean value per location, shape (n_locations,).
        n_days_per_year: Days per year for frequency calculation (365).
    """

    n_harmonics: int
    cos_coefficients: tuple[tuple[float, ...], ...]
    sin_coefficients: tuple[tuple[float, ...], ...]
    mean: tuple[float, ...]
    n_days_per_year: int


class TailThresholds(TypedDict, total=True):
    """Pre-computed percentile thresholds for tail-excess metrics.

    Computed from training residuals per location. Applied to new data
    to identify extreme days and compute cumulative/average exceedances.

    Attributes:
        hot_threshold: Hot-tail threshold per location, shape (n_locations,).
        cold_threshold: Cold-tail threshold per location, shape (n_locations,).
        hot_percentile: Percentile used for hot threshold.
        cold_percentile: Percentile used for cold threshold.
    """

    hot_threshold: tuple[float, ...]
    cold_threshold: tuple[float, ...]
    hot_percentile: float
    cold_percentile: float


class TemporalFeatureState(TypedDict, total=True):
    """Complete fitted state for temporal feature extraction.

    Produced by fit(), consumed by transform(). Fully serializable
    via encode/decode functions.

    Attributes:
        config: Configuration used for fitting.
        seasonal_cycle: Fitted Fourier seasonal cycle coefficients.
        thresholds: Pre-computed tail thresholds from training residuals.
        median_baseline: Mean of within-season medians per location from
            training data, shape (n_locations,). Used by streaming
            extractors to approximate the residual (anomaly minus median)
            when the current season's median is not yet available.
        n_locations: Number of spatial locations (or entities).
    """

    config: TemporalFeatureConfig
    seasonal_cycle: SeasonalCycleCoefficients
    thresholds: TailThresholds
    median_baseline: tuple[float, ...]
    n_locations: int


class HeatMetricResult(TypedDict, total=True):
    """Computed heat metrics for one entity across years.

    Produced by per-entity extraction from the multi-location output.
    Stored as immutable tuple-of-tuples for serializability.

    Attributes:
        entity_id: Entity identifier.
        n_years: Number of years computed.
        metric_names: Ordered tuple of computed metric names.
        values: Metric values, shape (n_years, n_metrics).
    """

    entity_id: str
    n_years: int
    metric_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]


DEFAULT_TEMPORAL_FEATURE_CONFIG: TemporalFeatureConfig = TemporalFeatureConfig(
    n_fourier_harmonics=5,
    hot_cutoff_percentile=95.0,
    cold_cutoff_percentile=5.0,
    season="warm",
    season_months=(6, 7, 8),
    compute_ar1=True,
)


def require_temporal_feature_config(
    data: dict[str, JSONValue],
    key: str,
) -> TemporalFeatureConfig:
    """Validate and extract TemporalFeatureConfig from parsed data.

    Args:
        data: Dictionary containing temporal feature configuration.
        key: Key name for error messages.

    Returns:
        Validated TemporalFeatureConfig TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    return TemporalFeatureConfig(
        n_fourier_harmonics=_require_positive_int(
            data.get("n_fourier_harmonics"), f"{key}.n_fourier_harmonics"
        ),
        hot_cutoff_percentile=_require_percentile(
            data.get("hot_cutoff_percentile"), f"{key}.hot_cutoff_percentile"
        ),
        cold_cutoff_percentile=_require_percentile(
            data.get("cold_cutoff_percentile"), f"{key}.cold_cutoff_percentile"
        ),
        season=_require_season(data.get("season"), f"{key}.season"),
        season_months=_require_month_tuple(data.get("season_months"), f"{key}.season_months"),
        compute_ar1=_require_bool_field(data.get("compute_ar1"), f"{key}.compute_ar1"),
    )


def require_seasonal_cycle_coefficients(
    data: dict[str, JSONValue],
    key: str,
    n_locations: int,
) -> SeasonalCycleCoefficients:
    """Validate and extract SeasonalCycleCoefficients from parsed data.

    Args:
        data: Dictionary containing seasonal cycle coefficients.
        key: Key name for error messages.
        n_locations: Expected number of locations for inner dimension.

    Returns:
        Validated SeasonalCycleCoefficients TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    n_harmonics = _require_positive_int(data.get("n_harmonics"), f"{key}.n_harmonics")

    return SeasonalCycleCoefficients(
        n_harmonics=n_harmonics,
        cos_coefficients=_require_nested_float_tuple(
            data.get("cos_coefficients"),
            f"{key}.cos_coefficients",
            n_harmonics,
            n_locations,
        ),
        sin_coefficients=_require_nested_float_tuple(
            data.get("sin_coefficients"),
            f"{key}.sin_coefficients",
            n_harmonics,
            n_locations,
        ),
        mean=_require_float_tuple(data.get("mean"), f"{key}.mean", n_locations),
        n_days_per_year=_require_positive_int(
            data.get("n_days_per_year"), f"{key}.n_days_per_year"
        ),
    )


def require_tail_thresholds(
    data: dict[str, JSONValue],
    key: str,
    n_locations: int,
) -> TailThresholds:
    """Validate and extract TailThresholds from parsed data.

    Args:
        data: Dictionary containing tail threshold values.
        key: Key name for error messages.
        n_locations: Expected number of locations for threshold tuples.

    Returns:
        Validated TailThresholds TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    return TailThresholds(
        hot_threshold=_require_float_tuple(
            data.get("hot_threshold"),
            f"{key}.hot_threshold",
            n_locations,
        ),
        cold_threshold=_require_float_tuple(
            data.get("cold_threshold"),
            f"{key}.cold_threshold",
            n_locations,
        ),
        hot_percentile=_require_numeric(data.get("hot_percentile"), f"{key}.hot_percentile"),
        cold_percentile=_require_numeric(data.get("cold_percentile"), f"{key}.cold_percentile"),
    )


def require_temporal_feature_state(
    data: dict[str, JSONValue],
) -> TemporalFeatureState:
    """Validate and extract TemporalFeatureState from parsed data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated TemporalFeatureState TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    n_locations = _require_positive_int(data.get("n_locations"), "n_locations")

    median_baseline = _require_float_tuple(
        data.get("median_baseline"),
        "median_baseline",
        n_locations,
    )

    return TemporalFeatureState(
        config=require_temporal_feature_config(
            _require_json_dict(data.get("config"), "config"), "config"
        ),
        seasonal_cycle=require_seasonal_cycle_coefficients(
            _require_json_dict(data.get("seasonal_cycle"), "seasonal_cycle"),
            "seasonal_cycle",
            n_locations,
        ),
        thresholds=require_tail_thresholds(
            _require_json_dict(data.get("thresholds"), "thresholds"),
            "thresholds",
            n_locations,
        ),
        median_baseline=median_baseline,
        n_locations=n_locations,
    )


def encode_temporal_feature_state(
    state: TemporalFeatureState,
) -> dict[str, JSONValue]:
    """Encode TemporalFeatureState to JSON-serializable dictionary.

    Args:
        state: Validated state to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    config = state["config"]
    cycle = state["seasonal_cycle"]
    thresholds = state["thresholds"]

    return {
        "config": {
            "n_fourier_harmonics": config["n_fourier_harmonics"],
            "hot_cutoff_percentile": config["hot_cutoff_percentile"],
            "cold_cutoff_percentile": config["cold_cutoff_percentile"],
            "season": config["season"],
            "season_months": list(config["season_months"]),
            "compute_ar1": config["compute_ar1"],
        },
        "seasonal_cycle": {
            "n_harmonics": cycle["n_harmonics"],
            "cos_coefficients": [list(row) for row in cycle["cos_coefficients"]],
            "sin_coefficients": [list(row) for row in cycle["sin_coefficients"]],
            "mean": list(cycle["mean"]),
            "n_days_per_year": cycle["n_days_per_year"],
        },
        "thresholds": {
            "hot_threshold": list(thresholds["hot_threshold"]),
            "cold_threshold": list(thresholds["cold_threshold"]),
            "hot_percentile": thresholds["hot_percentile"],
            "cold_percentile": thresholds["cold_percentile"],
        },
        "median_baseline": list(state["median_baseline"]),
        "n_locations": state["n_locations"],
    }


def require_heat_metric_result(
    data: dict[str, JSONValue],
) -> HeatMetricResult:
    """Validate and extract HeatMetricResult from parsed data.

    Args:
        data: Dictionary from JSON parsing.

    Returns:
        Validated HeatMetricResult TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    n_years = _require_positive_int(data.get("n_years"), "n_years")
    metric_names = _require_str_tuple(data.get("metric_names"), "metric_names")
    n_metrics = len(metric_names)

    return HeatMetricResult(
        entity_id=_require_str_field(data.get("entity_id"), "entity_id"),
        n_years=n_years,
        metric_names=metric_names,
        values=_require_nested_float_tuple(
            data.get("values"),
            "values",
            n_years,
            n_metrics,
        ),
    )


def encode_heat_metric_result(
    result: HeatMetricResult,
) -> dict[str, JSONValue]:
    """Encode HeatMetricResult to JSON-serializable dictionary.

    Args:
        result: Validated result to encode.

    Returns:
        Dictionary safe for JSON serialization.
    """
    return {
        "entity_id": result["entity_id"],
        "n_years": result["n_years"],
        "metric_names": list(result["metric_names"]),
        "values": [list(row) for row in result["values"]],
    }


__all__ = [
    "DEFAULT_TEMPORAL_FEATURE_CONFIG",
    "HEAT_METRIC_NAMES",
    "HEAT_METRIC_NAMES_NO_AR1",
    "HeatMetricName",
    "HeatMetricResult",
    "SeasonalCycleCoefficients",
    "TailThresholds",
    "TemporalFeatureConfig",
    "TemporalFeatureState",
    "encode_heat_metric_result",
    "encode_temporal_feature_state",
    "require_heat_metric_result",
    "require_seasonal_cycle_coefficients",
    "require_tail_thresholds",
    "require_temporal_feature_config",
    "require_temporal_feature_state",
]
