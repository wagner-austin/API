"""Regression CSV dataset loader.

Loads CSV datasets into RegressionLoadedDataset format with strict parsing.
Parallel to CSVLoader (classification) but produces continuous float64 targets
instead of integer labels. No label encoding — target is parsed as float directly.

Reuses: chunked_csv_reader, _parsing utilities, parquet cache.
"""

from __future__ import annotations

import math as _math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._parsing import (
    build_categorical_encodings,
    build_encoding_lookup,
    detect_categorical_columns,
    encode_categorical_value,
    find_column_index,
    parse_numeric_value,
)
from covenant_ml.datasets.loaders.chunked_csv_reader import read_csv_with_progress
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import (
    FileEncoding,
    LoadProgress,
    RegressionDatasetConfig,
    RegressionDatasetMeta,
    RegressionLoadedDataset,
)


class RegressionCSVLoader:
    """Loads CSV datasets into RegressionLoadedDataset format.

    Parallel to CSVLoader (classification). Key differences:
    - Target column parsed as float directly (no label encoding)
    - Produces RegressionDatasetMeta with target distribution stats
    - y array is float64 (continuous targets), not int64 (labels)

    Handles:
    - Multiple encodings (utf-8, utf-8-sig, latin-1, cp1252)
    - Target column detection and float parsing
    - Column exclusion
    - Automatic categorical column detection and label encoding
    - Missing value handling (0.0 for numeric, special code for categorical)
    - Numeric conversion with NaN/inf handling
    """

    def load(
        self,
        config: RegressionDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> RegressionLoadedDataset:
        """Load regression CSV dataset with optional progress reporting.

        Args:
            config: Regression dataset configuration.
            external_dir: Root directory for datasets.
            progress_callback: Optional callback for progress updates.

        Returns:
            RegressionLoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If columns missing, data invalid, or parsing fails.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        encoding: FileEncoding = config["encoding"]

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load from CSV
        headers, rows = read_csv_with_progress(file_path, encoding, progress_callback)

        # Find target column index
        target_spec = config["target"]
        target_idx = find_column_index(headers, target_spec["column_name"])

        # Find columns to exclude
        exclude_set = set(config["exclude_columns"])
        exclude_set.add(target_spec["column_name"])  # Target is not a feature

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

        x_array: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
        y_array: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

        for row_idx, row in enumerate(rows):
            # Extract features
            for feat_idx, col_idx in enumerate(feature_indices):
                value = row[col_idx] if col_idx < len(row) else ""

                if feat_idx in encoding_lookup:
                    # Categorical column — apply label encoding
                    x_array[row_idx, feat_idx] = encode_categorical_value(
                        value, encoding_lookup[feat_idx]
                    )
                else:
                    # Numeric column — parse as float
                    x_array[row_idx, feat_idx] = parse_numeric_value(value)

            # Extract target as continuous float (no label encoding)
            target_value = row[target_idx] if target_idx < len(row) else ""
            y_array[row_idx] = parse_numeric_value(target_value)

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

        # Compute target distribution statistics using explicit sum/len to avoid Any
        y_sum = float(np.sum(y_array))
        target_mean = y_sum / n_samples if n_samples > 0 else 0.0
        y_sq_diff_sum = float(np.sum((y_array - target_mean) ** 2))
        target_std = _math.sqrt(y_sq_diff_sum / n_samples) if n_samples > 0 else 0.0
        target_min = float(np.min(y_array))
        target_max = float(np.max(y_array))

        meta = RegressionDatasetMeta(
            name=config["name"],
            n_samples=n_samples,
            n_features=n_features,
            target_mean=target_mean,
            target_std=target_std,
            target_min=target_min,
            target_max=target_max,
            feature_names=tuple(feature_names),
            categorical_encodings=tuple(encodings),
        )

        return RegressionLoadedDataset(meta=meta, x=x_array, y=y_array)


def create_regression_csv_loader() -> RegressionCSVLoader:
    """Factory function for creating regression CSV loader.

    Returns:
        New RegressionCSVLoader instance.
    """
    return RegressionCSVLoader()


__all__ = [
    "RegressionCSVLoader",
    "create_regression_csv_loader",
]
