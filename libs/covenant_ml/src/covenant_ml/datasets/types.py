"""Dataset types for the pluggable dataset loading system.

Provides TypedDicts for dataset configuration, loading, and metadata.
All types are immutable (total=True) and strictly typed.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

# File format literals
FileFormat = Literal["csv", "arff", "excel"]

# Loading phase literals for progress reporting
LoadPhase = Literal[
    "reading",  # Reading raw data from file
    "parsing",  # Parsing and converting data types
    "encoding",  # Building categorical encodings
    "aggregating",  # Aggregating time-series data
    "caching",  # Writing to parquet cache
    "loading_cache",  # Loading from parquet cache
]


class LoadProgress(TypedDict, total=True):
    """Progress state during dataset loading.

    Reports current phase, completion percentage, and contextual information.
    Immutable snapshot of loading progress at a point in time.

    Attributes:
        phase: Current loading phase.
        bytes_read: Number of bytes read from source file.
        bytes_total: Total bytes in source file.
        rows_processed: Number of rows processed so far.
        rows_total: Total rows (0 if unknown during streaming).
        percent_complete: Completion percentage (0.0 to 100.0).
        message: Human-readable status message.
    """

    phase: LoadPhase
    bytes_read: int
    bytes_total: int
    rows_processed: int
    rows_total: int
    percent_complete: float
    message: str


# Encoding literals
FileEncoding = Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"]

# Label type literals (how the target column encodes classes)
LabelType = Literal["binary_int", "binary_str", "multiclass_int", "multiclass_str"]

# Aggregation strategy for time-series data
AggregationStrategy = Literal[
    "last",  # Take last observation per entity (most recent)
    "first",  # Take first observation per entity (oldest)
    "mean",  # Compute mean of each feature per entity
    "statistics",  # Compute mean, std, min, max per feature (4x features)
]


class CategoricalEncoding(TypedDict, total=True):
    """Label encoding for a single categorical column.

    Stores the mapping from string values to integer codes.
    Used to reproduce encoding for test data or new samples.

    Attributes:
        column_name: Name of the encoded column.
        mapping: Tuple of (value, code) pairs, sorted alphabetically by value.
            Missing values are encoded as code 0 with value "_MISSING_".
        n_categories: Number of unique categories including missing.
    """

    column_name: str
    mapping: tuple[tuple[str, int], ...]
    n_categories: int


class TargetColumnSpec(TypedDict, total=True):
    """Specification for the target/label column.

    Defines how to identify and encode the target column in a dataset.

    Attributes:
        column_name: Name of the target column in the dataset.
        label_type: How labels are encoded (binary_int, binary_str, etc.).
        positive_values: Values that map to class 1 (bankruptcy/default).
        negative_values: Values that map to class 0 (healthy/non-default).
    """

    column_name: str
    label_type: LabelType
    positive_values: tuple[str | int, ...]
    negative_values: tuple[str | int, ...]


class DatasetConfig(TypedDict, total=True):
    """Configuration for loading a single dataset.

    Provides all information needed to locate, parse, and validate a dataset.

    Attributes:
        name: Unique identifier (e.g., "kaggle_company_bankruptcy").
        display_name: Human-readable name for display.
        folder: Subfolder under data/external/.
        file_name: Primary data file name.
        file_format: File format (csv, arff, excel).
        encoding: File encoding (utf-8, utf-8-sig, etc.).
        target: Target column specification.
        exclude_columns: Columns to drop (IDs, dates, names).
        n_samples_expected: Expected sample count for validation.
        n_features_expected: Expected feature count for validation.
        positive_class_ratio_expected: Expected positive class ratio.
    """

    name: str
    display_name: str
    folder: str
    file_name: str
    file_format: FileFormat
    encoding: FileEncoding
    target: TargetColumnSpec
    exclude_columns: tuple[str, ...]
    n_samples_expected: int
    n_features_expected: int
    positive_class_ratio_expected: float


class TimeSeriesSpec(TypedDict, total=True):
    """Specification for time-series dataset handling.

    Defines how to aggregate time-series data into single observations
    and how to join with separate labels files.

    Attributes:
        entity_column: Column identifying unique entities (e.g., "customer_ID").
        time_column: Column indicating time ordering (e.g., "date", "S_2").
        aggregation: Strategy for aggregating features per entity.
        labels_file: Separate labels file name, or empty string if labels in main file.
        labels_entity_column: Entity column name in labels file (for joining).
        include_rank_features: Whether to compute per-entity percentile rank features.
        include_diff_features: Whether to compute row-to-row diff features.
        include_window_features: Whether to compute window aggregations (last N obs).
        window_sizes: Tuple of window sizes for window features (e.g., (3, 6)).
    """

    entity_column: str
    time_column: str
    aggregation: AggregationStrategy
    labels_file: str
    labels_entity_column: str
    include_rank_features: bool
    include_diff_features: bool
    include_window_features: bool
    window_sizes: tuple[int, ...]


class TimeSeriesDatasetConfig(DatasetConfig, total=True):
    """Configuration for time-series datasets.

    Extends DatasetConfig with time-series specific configuration.
    Used for datasets with multiple observations per entity over time.

    Attributes:
        All attributes from DatasetConfig, plus:
        time_series: Time-series handling specification.
    """

    time_series: TimeSeriesSpec


class DatasetMeta(TypedDict, total=True):
    """Metadata about a loaded dataset.

    Contains summary statistics computed after loading.

    Attributes:
        name: Dataset identifier.
        n_samples: Total number of samples.
        n_features: Number of feature columns.
        n_positive: Number of positive class samples.
        n_negative: Number of negative class samples.
        positive_ratio: Fraction of positive samples.
        feature_names: Ordered tuple of feature column names.
        categorical_encodings: Tuple of encodings for categorical columns.
            Empty tuple if no categorical columns were encoded.
    """

    name: str
    n_samples: int
    n_features: int
    n_positive: int
    n_negative: int
    positive_ratio: float
    feature_names: tuple[str, ...]
    categorical_encodings: tuple[CategoricalEncoding, ...]


class LoadedDataset(TypedDict, total=True):
    """A fully loaded and validated dataset ready for ML.

    Contains the feature matrix, labels, and metadata.

    Attributes:
        meta: Dataset metadata with statistics.
        x: Feature matrix of shape (n_samples, n_features).
        y: Binary labels of shape (n_samples,).
    """

    meta: DatasetMeta
    x: NDArray[np.float64]
    y: NDArray[np.int64]


class RegressionLoadedDataset(TypedDict, total=True):
    """A fully loaded dataset with continuous regression targets.

    Parallel to LoadedDataset (classification). The key difference is
    y: NDArray[np.float64] (continuous targets) instead of NDArray[np.int64]
    (binary labels).

    Attributes:
        meta: Dataset metadata with statistics.
        x: Feature matrix of shape (n_samples, n_features).
        y: Continuous target values of shape (n_samples,).
    """

    meta: DatasetMeta
    x: NDArray[np.float64]
    y: NDArray[np.float64]


class DatasetValidationResult(TypedDict, total=True):
    """Result of validating a dataset against its config.

    Attributes:
        is_valid: Whether validation passed.
        dataset_name: Name of the validated dataset.
        errors: Tuple of error messages (empty if valid).
    """

    is_valid: bool
    dataset_name: str
    errors: tuple[str, ...]


# --- Temporal feature types (McKinnon PNAS 2024 methodology) ---


def _require_positive_int(value: JSONValue, field: str) -> int:
    """Validate and return positive integer value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated positive integer.

    Raises:
        ValueError: If value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be positive integer")
    return value


def _require_percentile(value: JSONValue, field: str) -> float:
    """Validate and return percentile value strictly between 0 and 100.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated percentile as float.

    Raises:
        ValueError: If value is not a number in (0, 100).
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    f = float(value)
    if not (0.0 < f < 100.0):
        raise ValueError(f"{field} must be between 0 and 100 exclusive")
    return f


def _require_numeric(value: JSONValue, field: str) -> float:
    """Validate and return numeric value as float.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated float value.

    Raises:
        ValueError: If value is not numeric.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _require_season(value: JSONValue, field: str) -> SeasonDefinition:
    """Validate and return season definition.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated SeasonDefinition literal.

    Raises:
        ValueError: If value is not a valid season.
    """
    if not isinstance(value, str) or value not in ("warm", "cold", "full_year"):
        raise ValueError(f"{field} must be 'warm', 'cold', or 'full_year'")
    if value == "warm":
        return "warm"
    if value == "cold":
        return "cold"
    return "full_year"


def _require_month_tuple(value: JSONValue, field: str) -> tuple[int, ...]:
    """Validate and return tuple of month numbers (1-12).

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated tuple of month integers.

    Raises:
        ValueError: If value is not a non-empty sequence of valid months.
    """
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{field} must be non-empty tuple of ints")
    result: list[int] = []
    for i, m in enumerate(value):
        if not isinstance(m, int) or isinstance(m, bool) or not (1 <= m <= 12):
            raise ValueError(f"{field}[{i}] must be int in 1..12")
        result.append(m)
    return tuple(result)


def _require_float_tuple(value: JSONValue, field: str, expected_len: int) -> tuple[float, ...]:
    """Validate and return tuple of floats with expected length.

    Args:
        value: Value to validate.
        field: Field name for error message.
        expected_len: Required number of elements.

    Returns:
        Validated tuple of floats.

    Raises:
        ValueError: If value is not a sequence of numerics with correct length.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be tuple of floats")
    if len(value) != expected_len:
        raise ValueError(f"{field} length {len(value)} != expected {expected_len}")
    result: list[float] = []
    for i, v in enumerate(value):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{field}[{i}] must be numeric")
        result.append(float(v))
    return tuple(result)


def _require_str_field(value: JSONValue, field: str) -> str:
    """Validate and return string value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated string.

    Raises:
        ValueError: If value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be string")
    return value


def _require_str_tuple(value: JSONValue, field: str) -> tuple[str, ...]:
    """Validate and return tuple of strings.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated tuple of strings.

    Raises:
        ValueError: If value is not a sequence of strings.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be tuple of strings")
    result: list[str] = []
    for i, v in enumerate(value):
        if not isinstance(v, str):
            raise ValueError(f"{field}[{i}] must be string")
        result.append(v)
    return tuple(result)


def _require_nested_float_tuple(
    value: JSONValue,
    field: str,
    expected_outer: int,
    expected_inner: int,
) -> tuple[tuple[float, ...], ...]:
    """Validate and return nested tuple of floats with expected dimensions.

    Args:
        value: Value to validate.
        field: Field name for error message.
        expected_outer: Required outer length (e.g. n_harmonics).
        expected_inner: Required inner length (e.g. n_locations).

    Returns:
        Validated nested tuple of floats.

    Raises:
        ValueError: If value dimensions or element types are wrong.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be nested tuple of floats")
    if len(value) != expected_outer:
        raise ValueError(f"{field} outer length {len(value)} != expected {expected_outer}")
    result: list[tuple[float, ...]] = []
    for i, row in enumerate(value):
        result.append(_require_float_tuple(row, f"{field}[{i}]", expected_inner))
    return tuple(result)


def _require_bool_field(value: JSONValue, field: str) -> bool:
    """Validate and return bool value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated bool.

    Raises:
        ValueError: If value is not a bool.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be bool")
    return value


def _require_json_dict(value: JSONValue, field: str) -> dict[str, JSONValue]:
    """Validate and return JSON dictionary.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated dictionary.

    Raises:
        ValueError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be dictionary")
    return value


# Season definition for temporal analysis
SeasonDefinition = Literal["warm", "cold", "full_year"]

# Heat metric names in canonical order
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

# Canonical ordered tuple of all heat metric names
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

# Metric names excluding AR1 (when compute_ar1=False)
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


# Default temporal feature configuration following McKinnon (PNAS 2024)
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


# --- Rank-trend hypothesis testing types (McKinnon PNAS 2024, steps 4-7) ---

# Metrics where negation before ranking makes rank 1 = most extreme.
# Hot metrics: higher value = more extreme heat.
# ndays_excess_cold is included because higher count = more extreme cold event days.
# ar1 is included because higher autocorrelation = more persistence.
HOT_RANKED_METRICS: tuple[str, ...] = (
    "seasonal_max",
    "cum_excess_hot",
    "avg_excess_hot",
    "ndays_excess_hot",
    "ndays_excess_cold",
    "ar1",
)

# Metrics where direct ranking makes rank 1 = most extreme (smallest value).
# Cold metrics: lower (more negative) value = more extreme cold.
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


def _require_non_negative_int(value: JSONValue, field: str) -> int:
    """Validate and return non-negative integer value.

    Args:
        value: Value to validate.
        field: Field name for error message.

    Returns:
        Validated non-negative integer.

    Raises:
        ValueError: If value is not a non-negative integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be non-negative integer")
    return value


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
    "DEFAULT_TEMPORAL_FEATURE_CONFIG",
    "HEAT_METRIC_NAMES",
    "HEAT_METRIC_NAMES_NO_AR1",
    "HOT_RANKED_METRICS",
    "AggregationStrategy",
    "CategoricalEncoding",
    "DatasetConfig",
    "DatasetMeta",
    "DatasetValidationResult",
    "FileEncoding",
    "FileFormat",
    "HeatMetricName",
    "HeatMetricResult",
    "LabelType",
    "LoadPhase",
    "LoadProgress",
    "LoadedDataset",
    "MetricTrendResult",
    "RankTrendConfig",
    "RankTrendResult",
    "RegressionLoadedDataset",
    "SeasonDefinition",
    "SeasonalCycleCoefficients",
    "TailThresholds",
    "TargetColumnSpec",
    "TemporalFeatureConfig",
    "TemporalFeatureState",
    "TimeSeriesDatasetConfig",
    "TimeSeriesSpec",
    "encode_heat_metric_result",
    "encode_metric_trend_result",
    "encode_rank_trend_result",
    "encode_temporal_feature_state",
    "make_metric_trend_result",
    "make_rank_trend_config",
    "make_rank_trend_result",
    "require_heat_metric_result",
    "require_metric_trend_result",
    "require_rank_trend_config",
    "require_rank_trend_result",
    "require_seasonal_cycle_coefficients",
    "require_tail_thresholds",
    "require_temporal_feature_config",
    "require_temporal_feature_state",
]
