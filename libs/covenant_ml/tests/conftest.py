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

# Rows in the bankruptcy sample the integration tests train on, and the seed
# that picks them; see load_us_bankruptcy_sample.
SAMPLE_ROWS = 6000
SAMPLE_SEED = 42


class USBankruptcyDataset(TypedDict):
    """US bankruptcy dataset loaded from CSV."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]
    feature_names: list[str]
    n_samples: int
    n_features: int
    n_bankrupt: int
    n_healthy: int


# The one parsed sample of this test process; see load_us_bankruptcy_sample.
_SAMPLE_CACHE: list[USBankruptcyDataset] = []


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


def load_us_bankruptcy_sample() -> USBankruptcyDataset:
    """Load a fixed, class-stratified sample of the US bankruptcy dataset.

    The backend integration tests train on this. The file holds 78,682
    rows, 5,220 of them failed companies; every test that fit a model on all
    of it took 9 to 31 s on serendipity, the fleet node that runs this
    package's make check, and together they put the check at 545 s against
    its 300 s budget (board task 2e880b79). SAMPLE_ROWS rows with the file's
    class balance keep real accounting data with real signal under every
    backend at a fraction of the fitting time.

    The file is parsed once per test process and kept in _SAMPLE_CACHE; each
    call returns fresh copies of the arrays and the name list, so a test that
    mutates what it was given cannot change what the next test reads.

    Returns:
        USBankruptcyDataset with the sample's feature matrix, labels, and
        metadata.

    Raises:
        FileNotFoundError: If dataset file not found.
    """
    if not _SAMPLE_CACHE:
        _SAMPLE_CACHE.append(_read_us_bankruptcy_sample())
    cached = _SAMPLE_CACHE[0]
    return {
        "x": cached["x"].copy(),
        "y": cached["y"].copy(),
        "feature_names": list(cached["feature_names"]),
        "n_samples": cached["n_samples"],
        "n_features": cached["n_features"],
        "n_bankrupt": cached["n_bankrupt"],
        "n_healthy": cached["n_healthy"],
    }


def _read_us_bankruptcy_sample() -> USBankruptcyDataset:
    """Parse the bankruptcy CSV and draw the stratified sample from it.

    Returns:
        USBankruptcyDataset with the sample's feature matrix, labels, and
        metadata.

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

    sample = _stratified_sample_indices(y_array)
    x_sample = x_array[sample]
    y_sample = y_array[sample]
    n_bankrupt = int(np.sum(y_sample))

    return {
        "x": x_sample,
        "y": y_sample,
        "feature_names": feature_cols,
        "n_samples": SAMPLE_ROWS,
        "n_features": n_features,
        "n_bankrupt": n_bankrupt,
        "n_healthy": SAMPLE_ROWS - n_bankrupt,
    }


def _stratified_sample_indices(y_array: NDArray[np.int64]) -> NDArray[np.int64]:
    """Pick SAMPLE_ROWS row indices that keep the file's class balance.

    Each class contributes its share of SAMPLE_ROWS, drawn without
    replacement by a fixed-seed generator, and the indices come back sorted
    so the sample keeps the file's row order.

    Args:
        y_array: Labels of every row in the file.

    Returns:
        Sorted row indices, SAMPLE_ROWS of them.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    pos_mask: NDArray[np.bool_] = y_array == 1
    neg_mask: NDArray[np.bool_] = y_array == 0
    pos_indices: NDArray[np.intp] = np.flatnonzero(pos_mask)
    neg_indices: NDArray[np.intp] = np.flatnonzero(neg_mask)
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)
    n_positive = round(SAMPLE_ROWS * len(pos_indices) / len(y_array))
    chosen: NDArray[np.intp] = np.concatenate(
        [pos_indices[:n_positive], neg_indices[: SAMPLE_ROWS - n_positive]]
    )
    ordered: NDArray[np.intp] = np.sort(chosen)
    return ordered.astype(np.int64)
