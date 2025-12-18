"""Time-series CSV dataset loader.

Loads time-series CSV datasets into LoadedDataset format.
Handles datasets with multiple observations per entity over time,
aggregating them into single feature vectors for ML.

Example use cases:
- Credit card transaction history (AMEX default)
- Stock price time series
- Customer behavior over time
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._parsing import (
    MISSING_VALUES,
    build_categorical_encodings,
    build_encoding_lookup,
    detect_categorical_columns,
    encode_categorical_value,
    encode_label,
    find_column_index,
    parse_numeric_value,
)
from covenant_ml.datasets.types import (
    AggregationStrategy,
    DatasetMeta,
    LoadedDataset,
    TimeSeriesDatasetConfig,
    TimeSeriesSpec,
)


class TimeSeriesCSVLoader:
    """Loads time-series CSV datasets into LoadedDataset format.

    Handles:
    - Multiple observations per entity over time
    - Time-based ordering within each entity
    - Aggregation strategies (last, first, mean, statistics)
    - Separate labels files (common in Kaggle competitions)
    - Categorical encoding with consistent mapping across entities
    - Missing value handling

    The loader aggregates multiple time observations per entity into
    a single feature vector suitable for ML models. Different aggregation
    strategies allow flexibility in how temporal information is compressed.
    """

    def load(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load time-series CSV dataset.

        Args:
            config: Time-series dataset configuration including aggregation strategy.
            external_dir: Root directory for datasets.

        Returns:
            LoadedDataset with aggregated features ready for ML.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If columns missing, data invalid, or parsing fails.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        ts_spec = config["time_series"]

        # Read features data
        headers, rows = self._read_csv(file_path, config["encoding"])

        # Find entity and time column indices
        entity_idx = find_column_index(headers, ts_spec["entity_column"])
        time_idx = find_column_index(headers, ts_spec["time_column"])

        # Build feature column indices (exclude entity, time, target, and exclude_columns)
        target_spec = config["target"]
        exclude_set = set(config["exclude_columns"])
        exclude_set.add(ts_spec["entity_column"])
        exclude_set.add(ts_spec["time_column"])
        exclude_set.add(target_spec["column_name"])

        feature_indices: list[int] = []
        feature_names: list[str] = []
        for i, header in enumerate(headers):
            if header not in exclude_set:
                feature_indices.append(i)
                feature_names.append(header)

        # Detect categorical columns and build encodings from all data
        categorical_columns = detect_categorical_columns(rows, feature_indices)
        encodings = build_categorical_encodings(
            rows, feature_indices, feature_names, categorical_columns
        )
        encoding_lookup = build_encoding_lookup(encodings, feature_names)

        # Group rows by entity
        entity_rows: dict[str, list[list[str]]] = {}
        for row in rows:
            entity_id = row[entity_idx] if entity_idx < len(row) else ""
            if entity_id not in entity_rows:
                entity_rows[entity_id] = []
            entity_rows[entity_id].append(row)

        # Sort each entity's rows by time
        for entity_id in entity_rows:
            entity_rows[entity_id] = self._sort_by_time(
                entity_rows[entity_id], time_idx
            )

        # Load labels (from separate file or main file)
        entity_labels = self._load_labels(
            config, external_dir, ts_spec, list(entity_rows.keys())
        )

        # Compute output dimensions
        n_entities = len(entity_rows)
        n_base_features = len(feature_indices)
        aggregation = ts_spec["aggregation"]

        if aggregation == "statistics":
            # 4 stats per feature: mean, std, min, max
            n_output_features = n_base_features * 4
            output_feature_names = self._build_statistics_feature_names(feature_names)
        else:
            n_output_features = n_base_features
            output_feature_names = feature_names

        # Aggregate features for each entity
        x_array = np.zeros((n_entities, n_output_features), dtype=np.float64)
        y_array = np.zeros(n_entities, dtype=np.int64)

        entity_ids = list(entity_rows.keys())
        for entity_num, entity_id in enumerate(entity_ids):
            entity_data = entity_rows[entity_id]

            # Aggregate features based on strategy
            x_array[entity_num] = self._aggregate_entity(
                entity_data,
                feature_indices,
                encoding_lookup,
                categorical_columns,
                aggregation,
            )

            # Get label for this entity
            y_array[entity_num] = entity_labels[entity_id]

        # Replace any remaining NaN/inf with 0.0
        x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute metadata
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

        return LoadedDataset(meta=meta, x=x_array, y=y_array)

    def _read_csv(
        self,
        file_path: Path,
        encoding: str,
    ) -> tuple[list[str], list[list[str]]]:
        """Read CSV file and return headers and rows.

        Args:
            file_path: Path to CSV file.
            encoding: File encoding to use.

        Returns:
            Tuple of (headers, rows).

        Raises:
            ValueError: If no data rows found.
        """
        rows: list[list[str]] = []
        headers: list[str] = []

        with open(file_path, encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            for line_values in reader:
                if not headers:
                    headers = [h.strip() for h in line_values]
                    continue
                rows.append(line_values)

        if not rows:
            raise ValueError(f"No data rows found in {file_path}")

        return headers, rows

    def _sort_by_time(
        self,
        rows: list[list[str]],
        time_idx: int,
    ) -> list[list[str]]:
        """Sort rows by time column value.

        Handles both numeric and string time values.
        Numeric values are sorted numerically, strings lexicographically.

        Args:
            rows: List of data rows.
            time_idx: Index of time column.

        Returns:
            Rows sorted by time in ascending order.
        """

        def sort_key(row: list[str]) -> tuple[int, float, str]:
            """Generate sort key: (is_numeric, numeric_val, string_val)."""
            value = row[time_idx] if time_idx < len(row) else ""
            stripped = value.strip()

            # Check if numeric
            if stripped in MISSING_VALUES:
                return (0, 0.0, "")

            cleaned = stripped.replace(",", "")
            if cleaned.lstrip("-").replace(".", "").replace("e", "").isdigit():
                return (0, float(cleaned), "")
            return (1, 0.0, stripped)

        return sorted(rows, key=sort_key)

    def _load_labels(
        self,
        config: TimeSeriesDatasetConfig,
        external_dir: Path,
        ts_spec: TimeSeriesSpec,
        entity_ids: list[str],
    ) -> dict[str, int]:
        """Load labels for each entity.

        Labels may be in a separate file or in the main data file.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.
            ts_spec: Time-series specification.
            entity_ids: List of unique entity IDs.

        Returns:
            Dictionary mapping entity ID to label (0 or 1).

        Raises:
            FileNotFoundError: If labels file doesn't exist.
            ValueError: If entity missing label.
        """
        target_spec = config["target"]

        if ts_spec["labels_file"]:
            # Load from separate labels file
            labels_path = external_dir / config["folder"] / ts_spec["labels_file"]
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")

            headers, rows = self._read_csv(labels_path, config["encoding"])
            entity_col_idx = find_column_index(
                headers, ts_spec["labels_entity_column"]
            )
            target_idx = find_column_index(headers, target_spec["column_name"])

            labels: dict[str, int] = {}
            for row_idx, row in enumerate(rows):
                entity_id = row[entity_col_idx] if entity_col_idx < len(row) else ""
                target_value = row[target_idx] if target_idx < len(row) else ""
                labels[entity_id] = encode_label(
                    target_value, target_spec, row_idx, labels_path
                )

            # Verify all entities have labels
            missing = set(entity_ids) - set(labels.keys())
            if missing:
                raise ValueError(
                    f"Missing labels for {len(missing)} entities. "
                    f"First few: {list(missing)[:5]}"
                )

            return labels

        # Labels in main file - take label from first (or only) row per entity
        # This assumes all rows for an entity have the same label
        raise ValueError(
            "Time-series datasets must have labels_file specified in time_series spec"
        )

    def _aggregate_entity(
        self,
        rows: list[list[str]],
        feature_indices: list[int],
        encoding_lookup: dict[int, dict[str, int]],
        categorical_columns: set[int],
        aggregation: AggregationStrategy,
    ) -> NDArray[np.float64]:
        """Aggregate multiple rows for a single entity into one feature vector.

        Args:
            rows: All rows for this entity (sorted by time).
            feature_indices: Indices of feature columns in raw data.
            encoding_lookup: Categorical encoding mappings.
            categorical_columns: Set of categorical feature indices.
            aggregation: Aggregation strategy to use.

        Returns:
            Aggregated feature vector.
        """
        n_features = len(feature_indices)

        if aggregation == "last":
            # Take the last (most recent) observation
            row = rows[-1]
            return self._extract_features(
                row, feature_indices, encoding_lookup, categorical_columns
            )

        if aggregation == "first":
            # Take the first (oldest) observation
            row = rows[0]
            return self._extract_features(
                row, feature_indices, encoding_lookup, categorical_columns
            )

        # For mean and statistics, we need to collect all values
        feature_values: list[list[float]] = [[] for _ in range(n_features)]

        for row in rows:
            for feat_idx, col_idx in enumerate(feature_indices):
                value = row[col_idx] if col_idx < len(row) else ""
                stripped = value.strip()

                if feat_idx in encoding_lookup:
                    # Categorical - encode
                    parsed = encode_categorical_value(value, encoding_lookup[feat_idx])
                elif stripped not in MISSING_VALUES:
                    # Numeric - parse (skip missing for aggregation)
                    parsed = parse_numeric_value(value)
                    feature_values[feat_idx].append(parsed)
                    continue
                else:
                    continue

                feature_values[feat_idx].append(parsed)

        if aggregation == "mean":
            result = np.zeros(n_features, dtype=np.float64)
            for feat_idx in range(n_features):
                vals = feature_values[feat_idx]
                if vals:
                    vals_arr: NDArray[np.float64] = np.array(vals, dtype=np.float64)
                    mean_val: np.float64 = vals_arr.mean()
                    result[feat_idx] = float(mean_val)
            return result

        # aggregation == "statistics"
        # 4 stats per feature: mean, std, min, max
        result = np.zeros(n_features * 4, dtype=np.float64)
        for feat_idx in range(n_features):
            vals = feature_values[feat_idx]
            offset = feat_idx * 4
            if vals:
                arr: NDArray[np.float64] = np.array(vals, dtype=np.float64)
                mean_stat: np.float64 = arr.mean()
                std_stat: np.float64 = arr.std()
                min_stat: np.float64 = arr.min()
                max_stat: np.float64 = arr.max()
                result[offset] = float(mean_stat)
                result[offset + 1] = float(std_stat)
                result[offset + 2] = float(min_stat)
                result[offset + 3] = float(max_stat)
        return result

    def _extract_features(
        self,
        row: list[str],
        feature_indices: list[int],
        encoding_lookup: dict[int, dict[str, int]],
        categorical_columns: set[int],
    ) -> NDArray[np.float64]:
        """Extract features from a single row.

        Args:
            row: Single data row.
            feature_indices: Indices of feature columns.
            encoding_lookup: Categorical encoding mappings.
            categorical_columns: Set of categorical feature indices.

        Returns:
            Feature vector for this row.
        """
        n_features = len(feature_indices)
        result = np.zeros(n_features, dtype=np.float64)

        for feat_idx, col_idx in enumerate(feature_indices):
            value = row[col_idx] if col_idx < len(row) else ""

            if feat_idx in encoding_lookup:
                result[feat_idx] = encode_categorical_value(
                    value, encoding_lookup[feat_idx]
                )
            else:
                result[feat_idx] = parse_numeric_value(value)

        return result

    def _build_statistics_feature_names(
        self,
        base_names: list[str],
    ) -> list[str]:
        """Build feature names for statistics aggregation.

        Creates 4 names per base feature: _mean, _std, _min, _max.

        Args:
            base_names: Original feature names.

        Returns:
            Expanded feature names with statistical suffixes.
        """
        result: list[str] = []
        for name in base_names:
            result.append(f"{name}_mean")
            result.append(f"{name}_std")
            result.append(f"{name}_min")
            result.append(f"{name}_max")
        return result


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
