"""CSV dataset loader.

Loads CSV datasets into LoadedDataset format with strict parsing.
Handles multiple encodings, categorical encoding, and target column encoding.
Uses Polars-based chunked reading with progress reporting for large files.
Integrates parquet caching for fast repeated loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._parsing import (
    build_categorical_encodings,
    build_encoding_lookup,
    detect_categorical_columns,
    encode_categorical_value,
    encode_label,
    find_column_index,
    is_numeric_value,
    is_simple_numeric,
    parse_numeric_value,
)
from covenant_ml.datasets.loaders.chunked_csv_reader import read_csv_with_progress
from covenant_ml.datasets.loaders.parquet_cache import (
    _CacheLock,
    _compute_config_hash,
    check_cache,
    get_cache_dir,
    load_from_cache,
    save_to_cache,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    FileEncoding,
    LoadedDataset,
    LoadProgress,
    TargetColumnSpec,
)


class CSVLoader:
    """Loads CSV datasets into LoadedDataset format.

    Handles:
    - Parquet caching for fast repeated loads
    - Multiple encodings (utf-8, utf-8-sig, latin-1, cp1252)
    - Target column detection and label encoding
    - Column exclusion
    - Automatic categorical column detection and label encoding
    - Missing value handling (replaced with 0.0 for numeric, special code for categorical)
    - Numeric conversion with NaN/inf handling

    Cache is stored in .cache/<config_hash>/ under the dataset folder.
    Cache is invalidated when source file is modified.
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedDataset:
        """Load CSV dataset with caching and optional progress reporting.

        First checks for valid parquet cache. If found, loads from cache.
        Otherwise, loads from CSV and saves to cache for future loads.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.
            progress_callback: Optional callback for progress updates.

        Returns:
            LoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If columns missing, data invalid, or parsing fails.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        encoding: FileEncoding = config["encoding"]

        # Compute config hash for cache key
        config_parts = [
            config["name"],
            config["file_name"],
            config["encoding"],
            str(config["target"]),
            str(config["exclude_columns"]),
            str(config.get("group_column")),
        ]
        config_str = "|".join(config_parts)
        config_hash = _compute_config_hash(config_str)
        cache_dir = get_cache_dir(external_dir, config["folder"], config_hash)

        # Check if valid cache exists under a cache lock to prevent races
        # with concurrent invalidation/removal in other workers.
        with _CacheLock(cache_dir):
            cache_info = check_cache(file_path, cache_dir)
            if cache_info["is_valid"]:
                return load_from_cache(cache_dir, progress_callback)

        # No valid cache - load from CSV
        headers, rows = read_csv_with_progress(file_path, encoding, progress_callback)

        # Find target column index
        target_spec = config["target"]
        target_idx = find_column_index(headers, target_spec["column_name"])

        # Find columns to exclude
        exclude_set = set(config["exclude_columns"])
        exclude_set.add(target_spec["column_name"])  # Target is not a feature

        # The group column identifies which rows are one correlated entity;
        # it is never a feature (a model reading it would memorize entities)
        # and its values are factorized to integer codes for the splitter.
        group_column = config.get("group_column")
        group_idx: int | None = None
        if group_column is not None:
            group_idx = find_column_index(headers, group_column)
            exclude_set.add(group_column)

        # Build feature column indices
        feature_indices: list[int] = []
        feature_names: list[str] = []
        for i, header in enumerate(headers):
            if header not in exclude_set:
                feature_indices.append(i)
                feature_names.append(header)

        # Detect categorical columns and build encodings
        categorical_columns = detect_categorical_columns(rows, feature_indices)
        encodings = build_categorical_encodings(
            rows, feature_indices, feature_names, categorical_columns
        )

        # Build lookup dict for fast encoding access
        encoding_lookup = build_encoding_lookup(encodings, feature_names)

        # Convert to arrays
        n_samples = len(rows)
        n_features = len(feature_indices)

        # Report encoding phase start
        if progress_callback is not None:
            progress_callback(
                LoadProgress(
                    phase="encoding",
                    bytes_read=0,
                    bytes_total=0,
                    rows_processed=0,
                    rows_total=n_samples,
                    percent_complete=0.0,
                    message=f"Encoding {n_samples:,} rows...",
                )
            )

        x_array, y_array, groups_array = _encode_rows(
            rows=rows,
            feature_indices=feature_indices,
            encoding_lookup=encoding_lookup,
            target_idx=target_idx,
            target_spec=target_spec,
            file_path=file_path,
            group_idx=group_idx,
        )

        # Report encoding phase complete
        if progress_callback is not None:
            progress_callback(
                LoadProgress(
                    phase="encoding",
                    bytes_read=0,
                    bytes_total=0,
                    rows_processed=n_samples,
                    rows_total=n_samples,
                    percent_complete=100.0,
                    message=f"Encoded {n_samples:,} rows with {n_features} features",
                )
            )

        # Replace any remaining NaN/inf with 0.0
        x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute metadata
        n_positive = int(np.sum(y_array))
        n_negative = n_samples - n_positive
        positive_ratio = n_positive / n_samples if n_samples > 0 else 0.0

        meta = DatasetMeta(
            name=config["name"],
            n_samples=n_samples,
            n_features=n_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=positive_ratio,
            feature_names=tuple(feature_names),
            categorical_encodings=tuple(encodings),
        )

        dataset = LoadedDataset(meta=meta, x=x_array, y=y_array, groups=groups_array)

        # Save to cache for future loads
        save_to_cache(dataset, cache_dir, progress_callback)

        return dataset

    # Expose shared utilities as instance methods for backward compatibility with tests
    def _is_numeric_value(self, value: str) -> bool:
        """Check if a string value can be parsed as a float.

        Args:
            value: Stripped string value to check.

        Returns:
            True if value is numeric, False if categorical.
        """
        return is_numeric_value(value)

    def _is_simple_numeric(self, value: str) -> bool:
        """Check if a string is a simple numeric value (integer or decimal).

        Args:
            value: String to check (no sign prefix, no scientific notation).

        Returns:
            True if value is a simple numeric format.
        """
        return is_simple_numeric(value)


def _encode_rows(
    *,
    rows: list[list[str]],
    feature_indices: list[int],
    encoding_lookup: dict[int, dict[str, int]],
    target_idx: int,
    target_spec: TargetColumnSpec,
    file_path: Path,
    group_idx: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64] | None]:
    """Convert parsed CSV rows into the feature, label and group arrays.

    Args:
        rows: Raw CSV rows, headers already stripped.
        feature_indices: Column indices that are features, in order.
        encoding_lookup: Categorical label encodings by feature position.
        target_idx: Column index of the target.
        target_spec: How target values encode to labels.
        file_path: Source file, for error messages.
        group_idx: Column index of the group column, or None when the
            dataset is row-independent.

    Returns:
        Feature matrix, labels, and group codes (None without a group
        column). Group values factorize in first-appearance order, so codes
        are deterministic for a given file.
    """
    n_samples = len(rows)
    x_array = np.zeros((n_samples, len(feature_indices)), dtype=np.float64)
    y_array = np.zeros(n_samples, dtype=np.int64)
    groups_array: NDArray[np.int64] | None = None
    group_codes: dict[str, int] = {}
    if group_idx is not None:
        groups_array = np.zeros(n_samples, dtype=np.int64)

    for row_idx, row in enumerate(rows):
        # Extract features
        for feat_idx, col_idx in enumerate(feature_indices):
            value = row[col_idx] if col_idx < len(row) else ""

            if feat_idx in encoding_lookup:
                # Categorical column - apply label encoding
                x_array[row_idx, feat_idx] = encode_categorical_value(
                    value, encoding_lookup[feat_idx]
                )
            else:
                # Numeric column - parse as float
                x_array[row_idx, feat_idx] = parse_numeric_value(value)

        # Extract and encode label
        target_value = row[target_idx] if target_idx < len(row) else ""
        y_array[row_idx] = encode_label(target_value, target_spec, row_idx, file_path)

        if groups_array is not None and group_idx is not None:
            group_value = row[group_idx] if group_idx < len(row) else ""
            if group_value not in group_codes:
                group_codes[group_value] = len(group_codes)
            groups_array[row_idx] = group_codes[group_value]

    return x_array, y_array, groups_array


def create_csv_loader() -> CSVLoader:
    """Factory function for creating CSV loader.

    Returns:
        New CSVLoader instance.
    """
    return CSVLoader()


__all__ = [
    "CSVLoader",
    "create_csv_loader",
]
