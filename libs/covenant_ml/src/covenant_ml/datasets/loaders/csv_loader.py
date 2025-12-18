"""CSV dataset loader.

Loads CSV datasets into LoadedDataset format with strict parsing.
Handles multiple encodings, categorical encoding, and target column encoding.
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
    is_numeric_value,
    is_simple_numeric,
    parse_numeric_value,
)
from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
)


class CSVLoader:
    """Loads CSV datasets into LoadedDataset format.

    Handles:
    - Multiple encodings (utf-8, utf-8-sig, latin-1, cp1252)
    - Target column detection and label encoding
    - Column exclusion
    - Automatic categorical column detection and label encoding
    - Missing value handling (replaced with 0.0 for numeric, special code for categorical)
    - Numeric conversion with NaN/inf handling
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load CSV dataset.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.

        Returns:
            LoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If columns missing, data invalid, or parsing fails.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read raw data
        headers, rows = self._read_csv(file_path, config["encoding"])

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

        x_array = np.zeros((n_samples, n_features), dtype=np.float64)
        y_array = np.zeros(n_samples, dtype=np.int64)

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
            y_array[row_idx] = encode_label(
                target_value, target_spec, row_idx, file_path
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
