"""Dataset types for the pluggable dataset loading system.

Provides TypedDicts for dataset configuration, loading, and metadata.
All types are immutable (total=True) and strictly typed.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

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
    """

    entity_column: str
    time_column: str
    aggregation: AggregationStrategy
    labels_file: str
    labels_entity_column: str
    include_rank_features: bool
    include_diff_features: bool


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


__all__ = [
    "AggregationStrategy",
    "CategoricalEncoding",
    "DatasetConfig",
    "DatasetMeta",
    "DatasetValidationResult",
    "FileEncoding",
    "FileFormat",
    "LabelType",
    "LoadPhase",
    "LoadProgress",
    "LoadedDataset",
    "TargetColumnSpec",
    "TimeSeriesDatasetConfig",
    "TimeSeriesSpec",
]
