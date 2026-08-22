"""Time-series CSV dataset loader using Polars-native operations.

Loads time-series CSV datasets into LoadedDataset format using Polars
for memory-efficient processing. Uses Polars groupby aggregations to avoid
materializing all rows in Python, enabling large dataset loading.
Integrates parquet caching for fast repeated loads.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._parquet_cache_io import _CacheLock
from covenant_ml.datasets.loaders._parsing import encode_label, find_column_index
from covenant_ml.datasets.loaders._polars_aggregation import (
    aggregate_timeseries,
    build_statistics_feature_names,
)
from covenant_ml.datasets.loaders._polars_encoding import (
    apply_encodings,
    build_categorical_encodings,
    convert_to_numeric,
    detect_categorical_columns,
)
from covenant_ml.datasets.loaders._polars_ranking import (
    compute_diff_features,
    compute_entity_rank_features,
)
from covenant_ml.datasets.loaders._polars_utils import (
    PolarsDataFrameProtocol,
    PolarsReadCSVProtocol,
    convert_encoding,
    report_progress,
    sanitize_array_inplace,
)
from covenant_ml.datasets.loaders._polars_window import (
    compute_multi_window_features,
)
from covenant_ml.datasets.loaders.parquet_cache import (
    _compute_config_hash,
    check_cache,
    get_cache_dir,
    load_from_cache,
    save_to_cache,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import (
    DatasetMeta,
    FileEncoding,
    LoadedDataset,
    LoadProgress,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


class TimeSeriesCSVLoader:
    """Loads time-series CSV datasets using Polars-native operations.

    Uses Polars groupby aggregations to process data without materializing
    all rows in Python memory. Enables loading of large datasets (millions
    of rows) that would otherwise cause memory exhaustion.

    Handles:
    - Multiple observations per entity over time
    - Aggregation strategies (last, first, mean, statistics)
    - Separate labels files (common in Kaggle competitions)
    - Categorical encoding with consistent mapping
    - Missing value handling
    - Parquet caching for fast repeated loads
    """

    def load(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load time-series CSV dataset with optional progress reporting.

        First checks for valid parquet cache. If found, loads from cache.
        Otherwise, loads from CSV using Polars-native operations and
        saves to cache for future loads.

        Args:
            config: Time-series dataset configuration.
            external_dir: Root directory for datasets.
            progress_callback: Optional callback for progress updates.

        Returns:
            LoadedDataset with aggregated features ready for ML.

        Raises:
            FileNotFoundError: If dataset file or labels file doesn't exist.
            ValueError: If columns missing, data invalid, or parsing fails.
        """
        ts_spec = config["time_series"]

        if config.get("group_column") is not None:
            # Aggregation already collapses each entity to one row, so a
            # group column has nothing to group; refuse rather than ignore.
            raise ValueError("group_column is only supported by the CSV loader")

        if not ts_spec["labels_file"]:
            raise ValueError(
                "Time-series datasets must have labels_file specified in time_series spec"
            )

        file_path = external_dir / config["folder"] / config["file_name"]

        # Check cache
        config_hash = _compute_config_hash(self._build_config_string(config))
        cache_dir = get_cache_dir(external_dir, config["folder"], config_hash)

        with _CacheLock(cache_dir):
            cache_info = check_cache(file_path, cache_dir)
            if cache_info["is_valid"]:
                return load_from_cache(cache_dir, progress_callback)

        # Load from CSV
        dataset = self._load_from_csv(config, external_dir, progress_callback)

        # Save to cache
        save_to_cache(dataset, cache_dir, progress_callback)

        return dataset

    def _build_config_string(self, config: TimeSeriesDatasetConfig) -> str:
        """Build config string for cache hash.

        Args:
            config: Dataset configuration.

        Returns:
            String representation for hashing.
        """
        ts_spec = config["time_series"]
        parts = [
            config["name"],
            config["file_name"],
            config["encoding"],
            str(config["target"]),
            str(config["exclude_columns"]),
            ts_spec["entity_column"],
            ts_spec["time_column"],
            ts_spec["aggregation"],
            ts_spec["labels_file"],
            str(ts_spec["include_rank_features"]),
            str(ts_spec["include_diff_features"]),
            str(ts_spec["include_window_features"]),
            str(ts_spec["window_sizes"]),
        ]
        return "|".join(parts)

    def _load_from_csv(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None,
    ) -> LoadedDataset:
        """Load dataset from CSV using Polars-native operations.

        Args:
            config: Time-series dataset configuration.
            external_dir: Root directory for datasets.
            progress_callback: Optional callback for progress updates.

        Returns:
            LoadedDataset with aggregated features.

        Raises:
            FileNotFoundError: If files don't exist.
            ValueError: If data is invalid.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        encoding: FileEncoding = config["encoding"]
        ts_spec = config["time_series"]

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read CSV
        df = self._read_csv(file_path, encoding, progress_callback)

        # Validate columns
        headers = [h.strip() for h in df.columns]
        entity_col = ts_spec["entity_column"]
        time_col = ts_spec["time_column"]
        find_column_index(headers, entity_col)
        find_column_index(headers, time_col)

        # Build feature column list
        feature_columns = self._get_feature_columns(headers, config)

        # Encode and convert
        report_progress(
            progress_callback,
            LoadProgress(
                phase="encoding",
                bytes_read=0,
                bytes_total=0,
                rows_processed=0,
                rows_total=df.height,
                percent_complete=0.0,
                message="Detecting categorical columns...",
            ),
        )

        categorical_columns = detect_categorical_columns(df, feature_columns)
        encodings = build_categorical_encodings(df, feature_columns, categorical_columns)

        report_progress(
            progress_callback,
            LoadProgress(
                phase="encoding",
                bytes_read=0,
                bytes_total=0,
                rows_processed=df.height,
                rows_total=df.height,
                percent_complete=100.0,
                message=f"Found {len(categorical_columns)} categorical columns",
            ),
        )

        df_encoded = apply_encodings(df, feature_columns, encodings, categorical_columns)
        df_numeric = convert_to_numeric(df_encoded, feature_columns, categorical_columns)

        # Load labels
        entity_labels = self._load_labels(config, external_dir, ts_spec)

        # Aggregate
        aggregation = ts_spec["aggregation"]
        n_base_features = len(feature_columns)

        if aggregation == "statistics":
            n_output_features = n_base_features * 4
            output_feature_names = build_statistics_feature_names(feature_columns)
        else:
            n_output_features = n_base_features
            output_feature_names = feature_columns

        report_progress(
            progress_callback,
            LoadProgress(
                phase="aggregating",
                bytes_read=0,
                bytes_total=0,
                rows_processed=0,
                rows_total=0,
                percent_complete=0.0,
                message="Aggregating entities...",
            ),
        )

        x_array, entity_ids = aggregate_timeseries(
            df_numeric, entity_col, time_col, feature_columns, aggregation
        )

        n_entities = len(entity_ids)

        # Compute optional rank features
        if ts_spec["include_rank_features"]:
            rank_result = compute_entity_rank_features(df_numeric, entity_col, feature_columns)
            rank_array: NDArray[np.float64] = rank_result["features"]
            rank_names = rank_result["feature_names"]
            combined: NDArray[np.float64] = np.hstack((x_array, rank_array))
            x_array = combined
            output_feature_names = output_feature_names + rank_names
            n_output_features += len(rank_names)

        # Compute optional diff features
        if ts_spec["include_diff_features"]:
            diff_result = compute_diff_features(df_numeric, entity_col, time_col, feature_columns)
            diff_array: NDArray[np.float64] = diff_result["features"]
            diff_names = diff_result["feature_names"]
            combined_diff: NDArray[np.float64] = np.hstack((x_array, diff_array))
            x_array = combined_diff
            output_feature_names = output_feature_names + diff_names
            n_output_features += len(diff_names)

        # Compute optional window features (last N observations)
        if ts_spec["include_window_features"] and ts_spec["window_sizes"]:
            window_result = compute_multi_window_features(
                df_numeric, entity_col, time_col, feature_columns, ts_spec["window_sizes"]
            )
            window_array: NDArray[np.float64] = window_result["features"]
            window_names = window_result["feature_names"]
            combined_window: NDArray[np.float64] = np.hstack((x_array, window_array))
            x_array = combined_window
            output_feature_names = output_feature_names + window_names
            n_output_features += len(window_names)

        report_progress(
            progress_callback,
            LoadProgress(
                phase="aggregating",
                bytes_read=0,
                bytes_total=0,
                rows_processed=n_entities,
                rows_total=n_entities,
                percent_complete=100.0,
                message=f"Aggregated {n_entities:,} entities with {n_output_features} features",
            ),
        )

        # Build labels array
        y_array = np.zeros(n_entities, dtype=np.int64)
        for idx, entity_id in enumerate(entity_ids):
            if entity_id not in entity_labels:
                raise ValueError(f"Missing labels for 1 entities. First few: ['{entity_id}']")
            y_array[idx] = entity_labels[entity_id]

        sanitize_array_inplace(x_array)

        # Build metadata
        n_positive = int(np.sum(y_array))
        n_negative = n_entities - n_positive
        positive_ratio = n_positive / n_entities if n_entities > 0 else 0.0

        meta = DatasetMeta(
            name=config["name"],
            n_samples=n_entities,
            n_features=n_output_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=positive_ratio,
            feature_names=tuple(output_feature_names),
            categorical_encodings=tuple(encodings),
        )

        return LoadedDataset(meta=meta, x=x_array, y=y_array, groups=None)

    def _read_csv(
        self,
        file_path: Path,
        encoding: FileEncoding,
        progress_callback: ProgressCallbackProtocol | None,
    ) -> PolarsDataFrameProtocol:
        """Read CSV file using Polars.

        Args:
            file_path: Path to CSV file.
            encoding: File encoding.
            progress_callback: Optional progress callback.

        Returns:
            Polars DataFrame.

        Raises:
            ValueError: If no data rows found.
        """
        file_size = file_path.stat().st_size
        polars_encoding = convert_encoding(encoding)

        report_progress(
            progress_callback,
            LoadProgress(
                phase="reading",
                bytes_read=0,
                bytes_total=file_size,
                rows_processed=0,
                rows_total=0,
                percent_complete=0.0,
                message=f"Reading {file_path.name}...",
            ),
        )

        polars_mod = __import__("polars")
        read_csv_fn: PolarsReadCSVProtocol = polars_mod.read_csv
        df: PolarsDataFrameProtocol = read_csv_fn(
            file_path,
            encoding=polars_encoding,
            infer_schema_length=0,
        )

        if df.height == 0:
            raise ValueError(f"No data rows found in {file_path}")

        report_progress(
            progress_callback,
            LoadProgress(
                phase="reading",
                bytes_read=file_size,
                bytes_total=file_size,
                rows_processed=df.height,
                rows_total=df.height,
                percent_complete=100.0,
                message=f"Read {df.height:,} rows from {file_path.name}",
            ),
        )

        return df

    def _get_feature_columns(
        self,
        headers: list[str],
        config: TimeSeriesDatasetConfig,
    ) -> list[str]:
        """Get list of feature column names.

        Args:
            headers: All column headers.
            config: Dataset configuration.

        Returns:
            List of feature column names.
        """
        ts_spec = config["time_series"]
        target_spec = config["target"]

        exclude_set = set(config["exclude_columns"])
        exclude_set.add(ts_spec["entity_column"])
        exclude_set.add(ts_spec["time_column"])
        exclude_set.add(target_spec["column_name"])

        return [h for h in headers if h not in exclude_set]

    def _load_labels(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        ts_spec: TimeSeriesSpec,
    ) -> dict[str, int]:
        """Load labels from separate labels file.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.
            ts_spec: Time-series specification.

        Returns:
            Dictionary mapping entity ID to label (0 or 1).

        Raises:
            FileNotFoundError: If labels file doesn't exist.
            ValueError: If label parsing fails or no data rows.
        """
        target_spec = config["target"]
        labels_path = external_dir / config["folder"] / ts_spec["labels_file"]

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        labels: dict[str, int] = {}

        with open(labels_path, encoding=config["encoding"], newline="") as f:
            reader = csv.reader(f)
            headers: list[str] = []

            for line_values in reader:
                if not headers:
                    headers = [h.strip() for h in line_values]
                    continue

                entity_col_idx = find_column_index(headers, ts_spec["labels_entity_column"])
                target_idx = find_column_index(headers, target_spec["column_name"])

                entity_id = line_values[entity_col_idx] if entity_col_idx < len(line_values) else ""
                target_value = line_values[target_idx] if target_idx < len(line_values) else ""

                labels[entity_id] = encode_label(
                    target_value, target_spec, len(labels), labels_path
                )

        if not labels:
            raise ValueError(f"No data rows found in {labels_path}")

        return labels


def create_timeseries_csv_loader() -> TimeSeriesCSVLoader:
    """Factory function for creating time-series CSV loader.

    Returns:
        New TimeSeriesCSVLoader instance.
    """
    return TimeSeriesCSVLoader()


__all__ = [
    "TimeSeriesCSVLoader",
    "create_timeseries_csv_loader",
]
