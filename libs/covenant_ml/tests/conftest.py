"""Shared test fixtures for covenant_ml tests."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

# Path to test data directory
DATA_DIR = Path(__file__).parent / "data"


class USBankruptcyDataset(TypedDict):
    """US bankruptcy dataset loaded from CSV."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]
    feature_names: list[str]
    n_samples: int
    n_features: int
    n_bankrupt: int
    n_healthy: int


def _safe_float(value: str, default: float = 0.0) -> float:
    """Safely convert string to float, handling missing values."""
    if value in ("", "?", "NA", "NaN", "nan", "None"):
        return default
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def load_us_bankruptcy_data() -> USBankruptcyDataset:
    """Load full US bankruptcy dataset for testing.

    Returns:
        USBankruptcyDataset with feature matrix, labels, and metadata.

    Raises:
        FileNotFoundError: If dataset file not found.
    """
    data_path = DATA_DIR / "american_bankruptcy.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"US bankruptcy dataset not found at {data_path}")

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

    return {
        "x": x_array,
        "y": y_array,
        "feature_names": feature_cols,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_bankrupt": n_bankrupt,
        "n_healthy": n_healthy,
    }
