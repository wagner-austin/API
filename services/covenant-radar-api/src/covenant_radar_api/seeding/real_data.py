"""Load real bankruptcy datasets for model training.

Converts external datasets (Taiwan, US, Polish) into the format
expected by the covenant-radar-api ML training pipeline.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

_log = get_logger(__name__)


def _read_csv_rows(file_path: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Read CSV file and return list of row dictionaries with proper typing."""
    rows: list[dict[str, str]] = []
    with open(file_path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        headers: list[str] = []
        for line_values in reader:
            if not headers:
                headers = line_values
                continue
            row: dict[str, str] = {}
            for i, header in enumerate(headers):
                if i < len(line_values):
                    row[header] = line_values[i]
            rows.append(row)
    return rows


class RealDataSample(TypedDict):
    """A single sample from real bankruptcy data."""

    company_id: str
    debt_to_ebitda: float
    interest_coverage: float
    current_ratio: float
    debt_ratio: float
    quick_ratio: float
    roa: float
    net_worth_ratio: float
    is_bankrupt: int  # 1 = bankrupt, 0 = not bankrupt


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


def load_taiwan_data(data_path: Path) -> list[RealDataSample]:
    """Load Taiwan bankruptcy dataset.

    Maps Taiwan dataset columns to covenant-relevant metrics:
    - Bankrupt? -> is_bankrupt
    - Debt ratio % -> debt_ratio
    - Interest Coverage Ratio -> interest_coverage
    - Current Ratio -> current_ratio
    - Quick Ratio -> quick_ratio
    - ROA(A) before interest -> roa
    - Net worth/Assets -> net_worth_ratio

    Args:
        data_path: Path to data.csv

    Returns:
        List of RealDataSample dictionaries.
    """
    samples: list[RealDataSample] = []
    all_rows = _read_csv_rows(data_path, encoding="utf-8")

    for i, raw_row in enumerate(all_rows):
        # Strip whitespace from column names (Taiwan CSV has leading spaces)
        row = {k.strip(): v for k, v in raw_row.items()}
        # Map Taiwan columns to our features
        # Column names from the dataset
        is_bankrupt = int(_safe_float(row.get("Bankrupt?", "0")))
        debt_ratio = _safe_float(row.get("Debt ratio %", "0"))
        interest_coverage = _safe_float(
            row.get("Interest Coverage Ratio (Interest expense to EBIT)", "0")
        )
        current_ratio = _safe_float(row.get("Current Ratio", "0"))
        quick_ratio = _safe_float(row.get("Quick Ratio", "0"))
        roa = _safe_float(row.get("ROA(A) before interest and % after tax", "0"))
        net_worth_ratio = _safe_float(row.get("Net worth/Assets", "0"))

        # Calculate debt_to_ebitda proxy from debt ratio
        # Higher debt ratio = higher leverage
        debt_to_ebitda = debt_ratio * 10 if debt_ratio > 0 else 0.0

        sample = RealDataSample(
            company_id=f"taiwan_{i}",
            debt_to_ebitda=debt_to_ebitda,
            interest_coverage=interest_coverage,
            current_ratio=current_ratio,
            debt_ratio=debt_ratio,
            quick_ratio=quick_ratio,
            roa=roa,
            net_worth_ratio=net_worth_ratio,
            is_bankrupt=is_bankrupt,
        )
        samples.append(sample)

    return samples


def load_us_data(data_path: Path) -> list[RealDataSample]:
    """Load US bankruptcy dataset.

    The US dataset has columns: company_name, status_label, year, X1-X18
    status_label: 'alive' or 'failed'

    Args:
        data_path: Path to american_bankruptcy.csv

    Returns:
        List of RealDataSample dictionaries.
    """
    samples: list[RealDataSample] = []
    all_rows = _read_csv_rows(data_path, encoding="utf-8-sig")  # utf-8-sig handles BOM

    for i, row in enumerate(all_rows):
        company_name = row.get("company_name", f"us_{i}")
        status = row.get("status_label", "alive")
        is_bankrupt = 1 if status == "failed" else 0

        # Map X1-X18 to financial metrics (based on common conventions)
        # These are normalized/scaled financial ratios
        x1 = _safe_float(row.get("X1", "0"))  # Often working capital/assets
        x2 = _safe_float(row.get("X2", "0"))  # Often retained earnings/assets
        x3 = _safe_float(row.get("X3", "0"))  # Often EBIT/assets
        x4 = _safe_float(row.get("X4", "0"))  # Often market value/debt
        x5 = _safe_float(row.get("X5", "0"))  # Often sales/assets

        # Create proxy metrics from available features
        # Scale down large values
        debt_to_ebitda = max(0, min(20, x1 / 100)) if x1 > 0 else 3.0
        interest_coverage = max(0, min(50, x3 / 10)) if x3 > 0 else 2.0
        current_ratio = max(0, min(10, x2 / 100)) if x2 > 0 else 1.5
        debt_ratio = max(0, min(1, 1 - x4 / 1000)) if x4 > 0 else 0.5
        quick_ratio = max(0, min(10, x5 / 100)) if x5 > 0 else 1.0

        sample = RealDataSample(
            company_id=company_name,
            debt_to_ebitda=debt_to_ebitda,
            interest_coverage=interest_coverage,
            current_ratio=current_ratio,
            debt_ratio=debt_ratio,
            quick_ratio=quick_ratio,
            roa=x3 / 100 if x3 else 0.05,
            net_worth_ratio=x4 / 1000 if x4 else 0.5,
            is_bankrupt=is_bankrupt,
        )
        samples.append(sample)

    return samples


def load_polish_arff(data_path: Path) -> list[RealDataSample]:
    """Load Polish bankruptcy dataset from ARFF format.

    The Polish dataset has 64 attributes (Attr1-Attr64) plus class (0/1).

    Key attributes (based on documentation):
    - Attr1: net profit / total assets
    - Attr4: current assets / short-term liabilities (current ratio)
    - Attr5: (cash + securities) / short-term liabilities (quick ratio)
    - Attr10: equity / total assets
    - Attr13: working capital / total assets
    - Attr27: profit on operating activities / financial expenses (interest coverage proxy)

    Args:
        data_path: Path to 1year.arff (or similar)

    Returns:
        List of RealDataSample dictionaries.
    """
    samples: list[RealDataSample] = []

    with open(data_path, encoding="utf-8") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line == "@data":
                in_data = True
                continue
            if not in_data or not line or line.startswith("%"):
                continue

            # Parse CSV data line
            values = line.split(",")
            if len(values) < 65:  # 64 attrs + 1 class
                continue

            # Extract key attributes
            attr1 = _safe_float(values[0])  # net profit / total assets (ROA proxy)
            attr4 = _safe_float(values[3])  # current ratio
            attr5 = _safe_float(values[4])  # quick ratio proxy
            attr10 = _safe_float(values[9])  # equity / total assets
            # attr13 (working capital / total assets) available but not used
            attr27 = _safe_float(values[26])  # interest coverage proxy
            class_val = int(_safe_float(values[64]))  # 0 or 1

            # Calculate debt_to_ebitda from leverage
            debt_ratio = max(0, min(1, 1 - attr10)) if attr10 else 0.5
            debt_to_ebitda = debt_ratio * 8  # Scale to typical range

            sample = RealDataSample(
                company_id=f"polish_{len(samples)}",
                debt_to_ebitda=debt_to_ebitda,
                interest_coverage=max(0, min(50, attr27)),
                current_ratio=max(0, min(10, attr4)),
                debt_ratio=debt_ratio,
                quick_ratio=max(0, min(10, attr5)),
                roa=attr1,
                net_worth_ratio=attr10,
                is_bankrupt=class_val,
            )
            samples.append(sample)

    return samples


def load_all_real_data(external_dir: Path) -> list[RealDataSample]:
    """Load all available real datasets.

    Args:
        external_dir: Path to data/external directory

    Returns:
        Combined list of samples from all datasets.
    """
    all_samples: list[RealDataSample] = []

    # Taiwan data
    taiwan_path = external_dir / "taiwan_data" / "data.csv"
    if taiwan_path.exists():
        taiwan_samples = load_taiwan_data(taiwan_path)
        all_samples.extend(taiwan_samples)

    # US data
    us_path = external_dir / "us_data" / "american_bankruptcy.csv"
    if us_path.exists():
        us_samples = load_us_data(us_path)
        all_samples.extend(us_samples)

    # Polish data (use 1year for most recent predictions)
    polish_path = external_dir / "polish_data" / "1year.arff"
    if polish_path.exists():
        polish_samples = load_polish_arff(polish_path)
        all_samples.extend(polish_samples)

    return all_samples


def samples_to_arrays(
    samples: list[RealDataSample],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Convert samples to numpy arrays for training.

    Returns:
        Tuple of (X features array, y labels array)
    """
    n_samples = len(samples)
    n_features = 8  # Match the model's expected input

    x_array = np.zeros((n_samples, n_features), dtype=np.float64)
    y_array = np.zeros(n_samples, dtype=np.int64)

    for i, sample in enumerate(samples):
        # Features in same order as covenant_domain.features.LoanFeatures
        x_array[i, 0] = sample["debt_to_ebitda"]
        x_array[i, 1] = sample["interest_coverage"]
        x_array[i, 2] = sample["current_ratio"]
        x_array[i, 3] = 0.0  # leverage_change_1p (not available in static data)
        x_array[i, 4] = 0.0  # leverage_change_4p (not available in static data)
        x_array[i, 5] = 0.0  # sector_encoded (use default)
        x_array[i, 6] = 0.0  # region_encoded (use default)
        x_array[i, 7] = 0.0  # near_breach_count_4p (not available)
        y_array[i] = sample["is_bankrupt"]

    return x_array, y_array


def get_dataset_stats(samples: list[RealDataSample]) -> dict[str, int | float]:
    """Get statistics about the dataset.

    Args:
        samples: List of samples

    Returns:
        Dictionary with dataset statistics.
    """
    n_total = len(samples)
    n_bankrupt = sum(1 for s in samples if s["is_bankrupt"] == 1)
    n_healthy = n_total - n_bankrupt

    return {
        "n_total": n_total,
        "n_bankrupt": n_bankrupt,
        "n_healthy": n_healthy,
        "bankruptcy_rate": n_bankrupt / n_total if n_total > 0 else 0.0,
    }


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
    "RealDataSample",
    "get_dataset_stats",
    "load_all_real_data",
    "load_polish_arff",
    "load_polish_raw",
    "load_taiwan_data",
    "load_taiwan_raw",
    "load_us_data",
    "load_us_raw",
    "samples_to_arrays",
]
