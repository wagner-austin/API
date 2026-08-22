"""Raw dataset loaders (uncut rows, ARFF parsing)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

_log = get_logger(__name__)


def _safe_float(
    value: str,
    default: float = 0.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Safely convert string to float, handling missing values and outliers.

    Args:
        value: String value to convert.
        default: Default value if conversion fails.
        min_val: If set, clip values below this threshold.
        max_val: If set, clip values above this threshold.

    Returns:
        Converted and optionally clipped float value.
    """
    if value in ("", "?", "NA", "NaN", "nan", "None"):
        return default
    try:
        result = float(value)
        # Handle invalid values
        if math.isnan(result) or math.isinf(result):
            return default
        # Clip to bounds if specified
        if min_val is not None and result < min_val:
            result = min_val
        if max_val is not None and result > max_val:
            result = max_val
        return result
    except (ValueError, TypeError):
        _log.debug("Failed to convert value %r to float, using default %r", value, default)
        return default


class RawDataset(TypedDict):
    """Raw dataset with all columns for automatic feature selection."""

    x: NDArray[np.float64]  # Feature matrix (n_samples, n_features)
    y: NDArray[np.int64]  # Labels (n_samples,)
    feature_names: list[str]  # Column names for each feature
    n_samples: int
    n_features: int
    n_bankrupt: int
    n_healthy: int


def load_taiwan_raw(data_path: Path) -> RawDataset:
    """Load Taiwan dataset with ALL columns for automatic feature selection.

    XGBoost will determine which of the 95 features are most important.
    The first column 'Bankrupt?' is the label (0/1).

    Args:
        data_path: Path to data.csv

    Returns:
        RawDataset with feature matrix, labels, and column names
    """
    rows: list[list[str]] = []
    headers: list[str] = []

    with open(data_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_values in reader:
            if not headers:
                # Strip whitespace from headers
                headers = [h.strip() for h in line_values]
                continue
            rows.append(line_values)

    if not rows:
        raise ValueError(f"No data rows found in {data_path}")

    # First column is label, rest are features
    label_col = headers[0]  # "Bankrupt?"
    feature_cols = headers[1:]

    n_samples = len(rows)
    n_features = len(feature_cols)

    _log.info(
        "Loading Taiwan raw data",
        extra={
            "n_samples": n_samples,
            "n_features": n_features,
            "label_col": label_col,
        },
    )

    # Build arrays
    x_array = np.zeros((n_samples, n_features), dtype=np.float64)
    y_array = np.zeros(n_samples, dtype=np.int64)

    for i, row in enumerate(rows):
        # Label is first column
        y_array[i] = int(_safe_float(row[0] if row else "0"))

        # Features are remaining columns
        for j in range(n_features):
            col_idx = j + 1  # Offset by 1 for label column
            value = row[col_idx] if col_idx < len(row) else "0"
            x_array[i, j] = _safe_float(value)

    n_bankrupt = int(np.sum(y_array))
    n_healthy = n_samples - n_bankrupt

    _log.info(
        "Taiwan raw data loaded",
        extra={
            "n_bankrupt": n_bankrupt,
            "n_healthy": n_healthy,
            "bankruptcy_rate": f"{n_bankrupt / n_samples:.2%}",
        },
    )

    return RawDataset(
        x=x_array,
        y=y_array,
        feature_names=feature_cols,
        n_samples=n_samples,
        n_features=n_features,
        n_bankrupt=n_bankrupt,
        n_healthy=n_healthy,
    )


def load_us_raw(data_path: Path) -> RawDataset:
    """Load US bankruptcy dataset with ALL columns for automatic feature selection.

    XGBoost will determine which of the 18 features (X1-X18) are most important.
    The 'status_label' column contains the label ('alive' or 'failed').

    Args:
        data_path: Path to american_bankruptcy.csv

    Returns:
        RawDataset with feature matrix, labels, and column names
    """
    rows: list[list[str]] = []
    headers: list[str] = []

    with open(data_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for line_values in reader:
            if not headers:
                headers = [h.strip() for h in line_values]
                continue
            rows.append(line_values)

    if not rows:
        raise ValueError(f"No data rows found in {data_path}")

    # Find column indices
    # Format: company_name,status_label,year,X1,X2,...,X18
    status_idx = headers.index("status_label")
    feature_cols = [h for h in headers if h.startswith("X")]

    n_samples = len(rows)
    n_features = len(feature_cols)

    _log.info(
        "Loading US raw data",
        extra={
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )

    # Build arrays
    x_array = np.zeros((n_samples, n_features), dtype=np.float64)
    y_array = np.zeros(n_samples, dtype=np.int64)

    # Get feature column indices
    feature_indices = [headers.index(col) for col in feature_cols]

    for i, row in enumerate(rows):
        # Label: 'failed' = 1, 'alive' = 0
        status = row[status_idx] if status_idx < len(row) else "alive"
        y_array[i] = 1 if status == "failed" else 0

        # Features
        for j, col_idx in enumerate(feature_indices):
            value = row[col_idx] if col_idx < len(row) else "0"
            x_array[i, j] = _safe_float(value)

    n_bankrupt = int(np.sum(y_array))
    n_healthy = n_samples - n_bankrupt

    _log.info(
        "US raw data loaded",
        extra={
            "n_bankrupt": n_bankrupt,
            "n_healthy": n_healthy,
            "bankruptcy_rate": f"{n_bankrupt / n_samples:.2%}",
        },
    )

    return RawDataset(
        x=x_array,
        y=y_array,
        feature_names=feature_cols,
        n_samples=n_samples,
        n_features=n_features,
        n_bankrupt=n_bankrupt,
        n_healthy=n_healthy,
    )


class _ArffParseResult(TypedDict):
    """Result of parsing ARFF file."""

    feature_names: list[str]
    data_rows: list[list[str]]


def _parse_arff_file(data_path: Path) -> _ArffParseResult:
    """Parse ARFF file and extract feature names and data rows.

    Args:
        data_path: Path to ARFF file

    Returns:
        ParseResult with feature_names and data_rows

    Raises:
        ValueError: If no data rows found
    """
    feature_names: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            if line.lower() == "@data":
                in_data = True
                continue

            if not in_data:
                # Parse attribute definitions
                if line.lower().startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attr_name = parts[1]
                        if attr_name.lower() != "class":
                            feature_names.append(attr_name)
            else:
                # Parse data row
                values = line.split(",")
                if len(values) >= len(feature_names) + 1:
                    data_rows.append(values)

    if not data_rows:
        raise ValueError(f"No data rows found in {data_path}")

    return _ArffParseResult(feature_names=feature_names, data_rows=data_rows)


def load_polish_raw(data_path: Path) -> RawDataset:
    """Load Polish bankruptcy dataset with ALL columns for automatic feature selection.

    XGBoost will determine which of the 64 attributes are most important.
    The 'class' column is the label (0/1).

    Args:
        data_path: Path to 1year.arff (or similar ARFF file)

    Returns:
        RawDataset with feature matrix, labels, and column names
    """
    parsed = _parse_arff_file(data_path)
    feature_names = parsed["feature_names"]
    data_rows = parsed["data_rows"]

    n_samples = len(data_rows)
    n_features = len(feature_names)

    _log.info(
        "Loading Polish raw data",
        extra={
            "n_samples": n_samples,
            "n_features": n_features,
        },
    )

    # Build arrays
    x_array = np.zeros((n_samples, n_features), dtype=np.float64)
    y_array = np.zeros(n_samples, dtype=np.int64)

    for i, row in enumerate(data_rows):
        # Features are columns 0 to n_features-1
        for j in range(n_features):
            value = row[j] if j < len(row) else "0"
            x_array[i, j] = _safe_float(value)

        # Label is last column
        label_idx = n_features
        label_str = row[label_idx] if label_idx < len(row) else "0"
        y_array[i] = int(_safe_float(label_str))

    n_bankrupt = int(np.sum(y_array))
    n_healthy = n_samples - n_bankrupt

    _log.info(
        "Polish raw data loaded",
        extra={
            "n_bankrupt": n_bankrupt,
            "n_healthy": n_healthy,
            "bankruptcy_rate": f"{n_bankrupt / n_samples:.2%}",
        },
    )

    return RawDataset(
        x=x_array,
        y=y_array,
        feature_names=feature_names,
        n_samples=n_samples,
        n_features=n_features,
        n_bankrupt=n_bankrupt,
        n_healthy=n_healthy,
    )


__all__ = [
    "RawDataset",
    "load_polish_raw",
    "load_taiwan_raw",
    "load_us_raw",
]
