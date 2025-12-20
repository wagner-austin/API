"""Shared parsing utilities for dataset loaders.

Provides common functionality for parsing, encoding, and validation
used across all loader implementations (CSV, ARFF, time-series, etc.).

This module is internal (underscore prefix) - loaders import from here,
but external code should not depend on these internals.
"""

from __future__ import annotations

import math
from pathlib import Path

from covenant_ml.datasets.types import (
    CategoricalEncoding,
    TargetColumnSpec,
)

# Known missing value patterns (case-sensitive)
# Unified set covering CSV, ARFF, and common data formats
MISSING_VALUES: frozenset[str] = frozenset(
    {
        "",
        "?",
        "NA",
        "NaN",
        "nan",
        "None",
        "N/A",
        "n/a",
        "null",
        "NULL",
        ".",
    }
)

# Sentinel value for missing categorical values
CATEGORICAL_MISSING: str = "_MISSING_"


def find_column_index(headers: list[str], column_name: str) -> int:
    """Find column index by name (case-insensitive).

    Args:
        headers: List of column/attribute headers.
        column_name: Column name to find.

    Returns:
        Zero-based column index.

    Raises:
        ValueError: If column not found.
    """
    column_lower = column_name.lower()
    for idx, header in enumerate(headers):
        if header.lower() == column_lower:
            return idx
    raise ValueError(f"Column '{column_name}' not found. Available: {headers}")


def parse_numeric_value(value: str) -> float:
    """Parse a string value to float, handling missing values.

    Missing values are replaced with 0.0.
    Infinity and NaN are replaced with 0.0.

    Args:
        value: String value to parse.

    Returns:
        Parsed float value, or 0.0 for missing/invalid values.
    """
    stripped = value.strip()

    # Handle known missing value patterns
    if stripped in MISSING_VALUES:
        return 0.0

    # Remove thousands separators (comma when used as separator)
    cleaned = stripped.replace(",", "")

    # Parse the numeric value
    parsed: float = float(cleaned)

    # Replace inf/nan with 0.0 using math.isfinite for proper typing
    if not math.isfinite(parsed):
        return 0.0

    return parsed


def is_numeric_value(value: str) -> bool:
    """Check if a string value can be parsed as a float.

    Handles: integers, decimals, negative numbers, scientific notation,
    and special float values (inf, -inf).

    Args:
        value: Stripped string value to check.

    Returns:
        True if value is numeric, False if categorical.
    """
    # Remove thousands separators
    cleaned = value.replace(",", "")

    # Handle special float values (inf, -inf, infinity, etc.)
    lower = cleaned.lower()
    if lower in ("inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"):
        return True

    # Handle sign prefix
    test_str = cleaned.lstrip("-").lstrip("+")
    if not test_str:
        return False

    # Handle scientific notation (e.g., "1e-5", "2.5E10")
    lower_test = test_str.lower()
    if "e" in lower_test:
        parts = lower_test.split("e")
        if len(parts) != 2:
            return False
        mantissa, exponent = parts
        # Validate mantissa (can be decimal)
        if not is_simple_numeric(mantissa):
            return False
        # Validate exponent (integer with optional sign)
        exp_str = exponent.lstrip("-").lstrip("+")
        return bool(exp_str and exp_str.isdigit())

    return is_simple_numeric(test_str)


def is_simple_numeric(value: str) -> bool:
    """Check if a string is a simple numeric value (integer or decimal).

    Args:
        value: String to check (no sign prefix, no scientific notation).

    Returns:
        True if value is a simple numeric format.
    """
    if not value:
        return False

    parts = value.split(".")
    if len(parts) > 2:
        return False

    for part in parts:
        if part and not part.isdigit():
            return False

    # At least one part must have digits
    return any(part.isdigit() for part in parts if part)


def _is_parseable_int(value: str) -> bool:
    """Check if string value can be parsed as an integer."""
    return value.lstrip("-").replace(".", "").isdigit()


def _matches_label_value(stripped: str, label_val: str | int) -> bool:
    """Check if stripped value matches a label value.

    Args:
        stripped: Stripped string value from data.
        label_val: Expected label value (int or string).

    Returns:
        True if values match.
    """
    if isinstance(label_val, int):
        return _is_parseable_int(stripped) and int(float(stripped)) == label_val
    return stripped.lower() == str(label_val).lower()


def encode_label(
    value: str,
    spec: TargetColumnSpec,
    row_idx: int,
    file_path: Path,
) -> int:
    """Encode label value to 0/1.

    Args:
        value: Raw label value from dataset.
        spec: Target column specification with positive/negative values.
        row_idx: Row index for error messages.
        file_path: File path for error messages.

    Returns:
        0 for negative class, 1 for positive class.

    Raises:
        ValueError: If value doesn't match any known label.
    """
    stripped = value.strip()

    for pos_val in spec["positive_values"]:
        if _matches_label_value(stripped, pos_val):
            return 1

    for neg_val in spec["negative_values"]:
        if _matches_label_value(stripped, neg_val):
            return 0

    raise ValueError(
        f"Unknown label value '{stripped}' at row {row_idx} in {file_path}. "
        f"Expected positive={spec['positive_values']} or "
        f"negative={spec['negative_values']}"
    )


def detect_categorical_columns(
    rows: list[list[str]],
    feature_indices: list[int],
) -> set[int]:
    """Detect which feature columns contain categorical (non-numeric) data.

    Scans all rows to determine if a column has any non-numeric values.
    A column is categorical if ANY non-missing value cannot be parsed as float.

    Args:
        rows: All data rows.
        feature_indices: Indices of columns that are features.

    Returns:
        Set of feature indices (position in feature_indices) that are categorical.
    """
    categorical: set[int] = set()

    for feat_idx, col_idx in enumerate(feature_indices):
        for row in rows:
            value = row[col_idx] if col_idx < len(row) else ""
            stripped = value.strip()

            # Skip missing values
            if stripped in MISSING_VALUES:
                continue

            # Check if value is numeric
            if not is_numeric_value(stripped):
                categorical.add(feat_idx)
                break  # No need to check more rows for this column

    return categorical


def build_categorical_encodings(
    rows: list[list[str]],
    feature_indices: list[int],
    feature_names: list[str],
    categorical_columns: set[int],
) -> list[CategoricalEncoding]:
    """Build label encodings for categorical columns.

    Creates alphabetically sorted mappings from string values to integers.
    Missing values are mapped to code 0 with special sentinel value.

    Args:
        rows: All data rows.
        feature_indices: Indices of columns that are features.
        feature_names: Names of feature columns.
        categorical_columns: Set of feature indices that are categorical.

    Returns:
        List of CategoricalEncoding for each categorical column.
    """
    encodings: list[CategoricalEncoding] = []

    for feat_idx in sorted(categorical_columns):
        col_idx = feature_indices[feat_idx]
        column_name = feature_names[feat_idx]

        # Collect unique values
        unique_values: set[str] = set()
        has_missing = False

        for row in rows:
            value = row[col_idx] if col_idx < len(row) else ""
            stripped = value.strip()

            if stripped in MISSING_VALUES:
                has_missing = True
            else:
                unique_values.add(stripped)

        # Sort values alphabetically and assign codes
        sorted_values = sorted(unique_values)

        # Build mapping: missing gets code 0 if present, otherwise start from 0
        mapping_list: list[tuple[str, int]] = []
        code_offset = 0

        if has_missing:
            mapping_list.append((CATEGORICAL_MISSING, 0))
            code_offset = 1

        for idx, val in enumerate(sorted_values):
            mapping_list.append((val, idx + code_offset))

        n_categories = len(mapping_list)

        encodings.append(
            CategoricalEncoding(
                column_name=column_name,
                mapping=tuple(mapping_list),
                n_categories=n_categories,
            )
        )

    return encodings


def encode_categorical_value(
    value: str,
    mapping: dict[str, int],
) -> float:
    """Encode a categorical value using the provided mapping.

    Args:
        value: Raw string value.
        mapping: Dictionary mapping string values to integer codes.

    Returns:
        Float representation of the encoded integer.
    """
    stripped = value.strip()

    if stripped in MISSING_VALUES:
        return float(mapping.get(CATEGORICAL_MISSING, 0))

    return float(mapping[stripped])


def build_encoding_lookup(
    encodings: list[CategoricalEncoding],
    feature_names: list[str],
) -> dict[int, dict[str, int]]:
    """Build lookup dict for fast categorical encoding access.

    Args:
        encodings: List of categorical encodings.
        feature_names: List of feature column names.

    Returns:
        Dictionary mapping feature index to encoding mapping.
    """
    encoding_lookup: dict[int, dict[str, int]] = {}
    for enc in encodings:
        col_idx = feature_names.index(enc["column_name"])
        encoding_lookup[col_idx] = dict(enc["mapping"])
    return encoding_lookup


__all__ = [
    "CATEGORICAL_MISSING",
    "MISSING_VALUES",
    "build_categorical_encodings",
    "build_encoding_lookup",
    "detect_categorical_columns",
    "encode_categorical_value",
    "encode_label",
    "find_column_index",
    "is_numeric_value",
    "is_simple_numeric",
    "parse_numeric_value",
]
