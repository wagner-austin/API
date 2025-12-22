"""Target column and value detection.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import re
from typing import Literal

from scripts.discover_datasets.types import TargetColumnCandidate

# Known target column name patterns (case-insensitive)
# Note: Short patterns (1-2 chars) require exact match, others use word-boundary
TARGET_CANDIDATES: tuple[str, ...] = (
    # Generic target names
    "target",
    "class",
    "label",
    "y",
    "outcome",
    "status",
    "result",
    # Binary classification
    "is_fraud",
    "fraud",
    "fraudulent",
    "is_default",
    "default",
    "default payment next month",
    "is_churn",
    "churn",
    "churned",
    "attrition",
    "attrition_flag",
    # Bankruptcy/distress datasets
    "bankrupt?",
    "bankrupt",
    "bankruptcy",
    "distress",
    "financial distress",
    # Loan/credit datasets
    "loan_status",
    "loan",
    "credit_risk",
    "credit_score",
    "score",
    "risk",
    "risk_flag",
    "bad_loan",
    "good_bad",
    # Delinquency datasets (Give Me Credit, FICO)
    "seriousdlqin2yrs",
    # Health/medical
    "diagnosis",
    "disease",
    "positive",
    # Status variants
    "status_label",
    "class_label",
    "survived",
    "approved",
    "rejected",
)

# Patterns that indicate columns to exclude (case-insensitive)
EXCLUDE_PATTERNS: tuple[str, ...] = (
    "id",
    "customer_id",
    "customer id",
    "company_id",
    "company id",
    "name",
    "company_name",
    "company name",
    "date",
    "index",
    "unnamed",
)

# Known positive class values (lowercase) - the "bad" outcome we want to predict
_POSITIVE_VALUES: frozenset[str] = frozenset(
    {
        "1",
        "1.0",  # Numeric positive
        "2",  # German credit uses 2 for bad
        "yes",
        "y",
        "true",
        "positive",
        "failed",
        "bad",
        "fail",
        "rejected",
        "attrited customer",  # Bank churners
        "default",
        "defaulted",
        "fraud",
        "fraudulent",
        "churn",
        "churned",
        "bankrupt",
        "bankruptcy",
    }
)

# Known negative class values (lowercase) - the "good" outcome
_NEGATIVE_VALUES: frozenset[str] = frozenset(
    {
        "0",
        "0.0",  # Numeric negative
        "no",
        "n",
        "false",
        "negative",
        "alive",
        "good",
        "pass",
        "approved",
        "existing customer",  # Bank churners
        "no default",
        "not fraud",
        "not churn",
        "not bankrupt",
    }
)


def is_binary_column(values: tuple[str, ...]) -> bool:
    """Check if column values represent binary classification.

    Args:
        values: Unique values in the column.

    Returns:
        True if column appears to be binary.
    """
    if len(values) != 2:
        return False

    # Check for common binary patterns
    binary_pairs = (
        # Numeric
        ("0", "1"),
        ("1", "0"),
        ("0.0", "1.0"),  # xlrd returns floats as strings
        ("1.0", "0.0"),
        ("1", "2"),  # german credit uses 1/2
        ("2", "1"),
        # Yes/No variants
        ("yes", "no"),
        ("no", "yes"),
        ("y", "n"),
        ("n", "y"),
        # True/False
        ("true", "false"),
        ("false", "true"),
        # Positive/Negative
        ("positive", "negative"),
        ("negative", "positive"),
        # Status labels
        ("alive", "failed"),
        ("failed", "alive"),
        ("good", "bad"),
        ("bad", "good"),
        ("pass", "fail"),
        ("fail", "pass"),
        ("approved", "rejected"),
        ("rejected", "approved"),
        # Attrition
        ("existing customer", "attrited customer"),
        ("attrited customer", "existing customer"),
    )

    lower_values = tuple(v.lower().strip() for v in values)

    # Check known patterns first
    if any(set(lower_values) == set(pair) for pair in binary_pairs):
        return True

    # Any column with exactly 2 unique values is potentially binary
    return True


def normalize_column_name(col: str) -> str:
    """Normalize column name to lowercase with underscore separators.

    Handles camelCase, PascalCase, spaces, hyphens, and dots by converting to
    snake_case for consistent pattern matching.

    Args:
        col: Original column name.

    Returns:
        Normalized lowercase name with underscore separators.
    """
    # Insert underscore before uppercase letters (handles camelCase/PascalCase)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    # Replace spaces, hyphens, and dots with underscores
    normalized = re.sub(r"[\s\-\.]+", "_", normalized)
    # Lowercase everything
    normalized = normalized.lower()
    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def pattern_matches_column(pattern: str, col_lower: str) -> bool:
    """Check if a target pattern matches a column name.

    Short patterns (1-2 chars) require exact match.
    Longer patterns use word-boundary matching that handles:
    - Exact match
    - Snake_case: separated by underscores
    - CamelCase: converted to snake_case for matching
    - Spaces/hyphens: converted to underscores

    Args:
        pattern: Target pattern to match (already lowercase).
        col_lower: Lowercase column name (may be normalized to snake_case).

    Returns:
        True if pattern matches column name.
    """
    # Short patterns require exact match
    if len(pattern) <= 2:
        return col_lower == pattern

    # Exact match
    if pattern == col_lower:
        return True

    # Check if pattern appears as a complete word/token
    # Pattern should be at start, end, or surrounded by underscores
    escaped = re.escape(pattern)
    word_pattern = rf"(?:^|_){escaped}(?:_|$)"
    return bool(re.search(word_pattern, col_lower))


def find_target_candidates(
    columns: tuple[str, ...],
    sample_rows: tuple[tuple[str, ...], ...],
) -> tuple[TargetColumnCandidate, ...]:
    """Find potential target columns in the dataset.

    Args:
        columns: Column names.
        sample_rows: Sample data rows.

    Returns:
        Tuple of target column candidates.
    """
    candidates: list[TargetColumnCandidate] = []

    for idx, col in enumerate(columns):
        # Normalize column name (handles camelCase, spaces, hyphens)
        col_normalized = normalize_column_name(col)
        # Also keep raw lowercase for patterns that shouldn't be split
        col_lower = col.lower().strip()

        # Check if column name matches known patterns
        is_candidate = False
        for target_name in TARGET_CANDIDATES:
            # Try both normalized (snake_case) and raw lowercase forms
            if pattern_matches_column(target_name, col_normalized):
                is_candidate = True
                break
            # Also check raw lowercase for patterns like "seriousdlqin2yrs"
            if target_name == col_lower:
                is_candidate = True
                break

        if not is_candidate:
            continue

        # Collect unique values from sample
        unique_values: set[str] = set()
        for row in sample_rows:
            if idx < len(row):
                unique_values.add(row[idx].strip())

        unique_tuple = tuple(sorted(unique_values))[:10]  # Limit to 10 values
        n_unique = len(unique_values)
        is_binary = is_binary_column(unique_tuple)

        candidates.append(
            TargetColumnCandidate(
                column_name=col,
                unique_values=unique_tuple,
                n_unique=n_unique,
                is_binary=is_binary,
            )
        )

    return tuple(candidates)


def find_exclude_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Find columns that should be excluded (IDs, names, etc.).

    Args:
        columns: Column names.

    Returns:
        Tuple of columns to exclude.
    """
    excludes: list[str] = []

    for col in columns:
        col_lower = col.lower().strip()

        for pattern in EXCLUDE_PATTERNS:
            if pattern in col_lower:
                excludes.append(col)
                break

    return tuple(excludes)


def recommend_target(candidates: tuple[TargetColumnCandidate, ...]) -> str:
    """Recommend the best target column from candidates.

    Args:
        candidates: Detected target column candidates.

    Returns:
        Recommended column name, or empty string if none suitable.
    """
    if len(candidates) == 0:
        return ""

    # Prefer binary columns
    for candidate in candidates:
        if candidate["is_binary"]:
            return candidate["column_name"]

    # Prefer columns with few unique values
    for candidate in candidates:
        if candidate["n_unique"] <= 5:
            return candidate["column_name"]

    # Return first candidate as fallback
    return candidates[0]["column_name"]


def detect_positive_negative_values(
    unique_values: tuple[str, ...],
) -> tuple[str, str, Literal["binary_int", "binary_str"]]:
    """Detect which value is positive (bad) and which is negative (good).

    Args:
        unique_values: The two unique values from a binary column.

    Returns:
        Tuple of (positive_value, negative_value, label_type).
        Returns empty strings if detection fails.
    """
    if len(unique_values) != 2:
        return "", "", "binary_int"

    val_a, val_b = unique_values[0], unique_values[1]
    lower_a, lower_b = val_a.lower().strip(), val_b.lower().strip()

    # Check if values are numeric
    is_numeric = lower_a in ("0", "1", "0.0", "1.0", "2") and lower_b in (
        "0",
        "1",
        "0.0",
        "1.0",
        "2",
    )
    label_type: Literal["binary_int", "binary_str"] = "binary_int" if is_numeric else "binary_str"

    # Check if a is positive
    if lower_a in _POSITIVE_VALUES:
        return val_a, val_b, label_type

    # Check if b is positive
    if lower_b in _POSITIVE_VALUES:
        return val_b, val_a, label_type

    # Check if a is negative (then b is positive)
    if lower_a in _NEGATIVE_VALUES:
        return val_b, val_a, label_type

    # Check if b is negative (then a is positive)
    if lower_b in _NEGATIVE_VALUES:
        return val_a, val_b, label_type

    # Default: assume first value is positive (alphabetically first)
    # For unknown string values, this is a reasonable default
    return val_a, val_b, label_type


def calculate_positive_ratio(
    sample_rows: tuple[tuple[str, ...], ...],
    columns: tuple[str, ...],
    target_column: str,
    positive_value: str,
) -> float:
    """Calculate the ratio of positive class in sample data.

    Args:
        sample_rows: Sample data rows.
        columns: Column names.
        target_column: Name of the target column.
        positive_value: Value representing positive class.

    Returns:
        Ratio of positive class (0.0 to 1.0), or 0.0 if cannot calculate.
    """
    if not target_column or not positive_value or len(sample_rows) == 0:
        return 0.0

    # Find target column index
    if target_column not in columns:
        return 0.0
    target_idx = columns.index(target_column)

    # Count positive instances
    positive_lower = positive_value.lower().strip()
    n_positive = 0
    n_total = 0

    for row in sample_rows:
        if target_idx < len(row):
            val = row[target_idx].lower().strip()
            n_total += 1
            if val == positive_lower:
                n_positive += 1

    if n_total == 0:
        return 0.0

    return round(n_positive / n_total, 3)


__all__ = [
    "EXCLUDE_PATTERNS",
    "TARGET_CANDIDATES",
    "calculate_positive_ratio",
    "detect_positive_negative_values",
    "find_exclude_columns",
    "find_target_candidates",
    "is_binary_column",
    "normalize_column_name",
    "pattern_matches_column",
    "recommend_target",
]
