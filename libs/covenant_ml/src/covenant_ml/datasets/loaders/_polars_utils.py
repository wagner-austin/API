"""Polars utilities for dataset loaders.

Shared protocols and helper functions for Polars operations.
Internal module - used by loaders, not exported publicly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._parsing import MISSING_VALUES
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import FileEncoding, LoadProgress


class PolarsExprProtocol(Protocol):
    """Protocol for Polars expression operations."""

    def alias(self, name: str) -> PolarsExprProtocol:
        """Rename the expression output."""
        ...

    def mean(self) -> PolarsExprProtocol:
        """Compute mean."""
        ...

    def std(self, ddof: int = 1) -> PolarsExprProtocol:
        """Compute standard deviation.

        Args:
            ddof: Delta degrees of freedom. 0 for population std, 1 for sample std.
        """
        ...

    def min(self) -> PolarsExprProtocol:
        """Compute minimum."""
        ...

    def max(self) -> PolarsExprProtocol:
        """Compute maximum."""
        ...

    def sum(self) -> PolarsExprProtocol:
        """Compute sum."""
        ...

    def last(self) -> PolarsExprProtocol:
        """Get last value."""
        ...

    def diff(self) -> PolarsExprProtocol:
        """Compute row-to-row difference."""
        ...

    def rank(self, method: str = "ordinal", descending: bool = False) -> PolarsExprProtocol:
        """Compute rank.

        Args:
            method: Ranking method (ordinal, min, max, average, dense).
            descending: Whether to rank in descending order.
        """
        ...

    def over(self, expr: str) -> PolarsExprProtocol:
        """Apply expression over groups (window function).

        Args:
            expr: Column name to partition by.
        """
        ...

    def len(self) -> PolarsExprProtocol:
        """Compute length/count."""
        ...

    def unique(self) -> PolarsExprProtocol:
        """Get unique values."""
        ...

    def is_null(self) -> PolarsExprProtocol:
        """Check for null values."""
        ...

    def is_in(self, values: list[str]) -> PolarsExprProtocol:
        """Check if values are in list."""
        ...

    def cast(self, dtype: PolarsDataTypeProtocol) -> PolarsExprProtocol:
        """Cast to dtype."""
        ...

    def fill_null(self, value: float) -> PolarsExprProtocol:
        """Fill null values."""
        ...

    def when(self, condition: PolarsExprProtocol) -> PolarsExprProtocol:
        """Add when clause."""
        ...

    def then(self, value: PolarsExprProtocol) -> PolarsExprProtocol:
        """Add then clause."""
        ...

    def otherwise(self, value: PolarsExprProtocol) -> PolarsExprProtocol:
        """Add otherwise clause."""
        ...

    def eq(self, other: PolarsExprProtocol) -> PolarsExprProtocol:
        """Compare expressions for equality."""
        ...

    def __or__(self, other: PolarsExprProtocol) -> PolarsExprProtocol:
        """Or expressions."""
        ...

    def __truediv__(self, other: PolarsExprProtocol) -> PolarsExprProtocol:
        """Divide expressions."""
        ...


class PolarsDataTypeProtocol(Protocol):
    """Protocol for Polars data types."""

    ...


class PolarsSeriesProtocol(Protocol):
    """Protocol for Polars Series."""

    def to_list(self) -> list[str | None]:
        """Convert to Python list."""
        ...


class PolarsGroupByProtocol(Protocol):
    """Protocol for Polars GroupBy."""

    def first(self) -> PolarsDataFrameProtocol:
        """Get first row per group."""
        ...

    def last(self) -> PolarsDataFrameProtocol:
        """Get last row per group."""
        ...

    def agg(self, exprs: list[PolarsExprProtocol]) -> PolarsDataFrameProtocol:
        """Aggregate groups."""
        ...


class PolarsDataFrameProtocol(Protocol):
    """Protocol for Polars DataFrame operations."""

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        ...

    @property
    def height(self) -> int:
        """Return number of rows."""
        ...

    @property
    def width(self) -> int:
        """Return number of columns."""
        ...

    def sort(self, by: list[str] | str) -> PolarsDataFrameProtocol:
        """Sort DataFrame."""
        ...

    def select(
        self, exprs: list[PolarsExprProtocol] | PolarsExprProtocol
    ) -> PolarsDataFrameProtocol:
        """Select columns."""
        ...

    def group_by(self, by: str) -> PolarsGroupByProtocol:
        """Group by column."""
        ...

    def sample(self, n: int, seed: int) -> PolarsDataFrameProtocol:
        """Sample rows."""
        ...

    def iter_rows(self) -> list[tuple[str | None, ...]]:
        """Iterate rows."""
        ...

    def to_series(self) -> PolarsSeriesProtocol:
        """Convert single-column to Series."""
        ...

    def to_numpy(self) -> NDArray[np.float64]:
        """Convert to numpy array."""
        ...

    def with_columns(self, expr: PolarsExprProtocol) -> PolarsDataFrameProtocol:
        """Add/replace columns."""
        ...


class PolarsColFnProtocol(Protocol):
    """Protocol for Polars col() function."""

    def __call__(self, name: str) -> PolarsExprProtocol:
        """Create column expression."""
        ...


class PolarsLitFnProtocol(Protocol):
    """Protocol for Polars lit() function."""

    def __call__(self, value: float | str | None) -> PolarsExprProtocol:
        """Create literal expression."""
        ...


class PolarsWhenFnProtocol(Protocol):
    """Protocol for Polars when() function."""

    def __call__(self, condition: PolarsExprProtocol) -> PolarsExprProtocol:
        """Create when expression."""
        ...


class PolarsReadCSVProtocol(Protocol):
    """Protocol for Polars read_csv function."""

    def __call__(
        self,
        source: str | Path,
        encoding: str,
        infer_schema_length: int,
    ) -> PolarsDataFrameProtocol:
        """Read CSV file into DataFrame."""
        ...


def convert_encoding(encoding: FileEncoding) -> str:
    """Convert FileEncoding literal to Polars encoding string.

    Args:
        encoding: FileEncoding literal value.

    Returns:
        Encoding string compatible with Polars.
    """
    encoding_map: dict[str, str] = {
        "utf-8": "utf8",
        "utf-8-sig": "utf8",
        "latin-1": "utf8-lossy",
        "cp1252": "utf8-lossy",
    }
    return encoding_map.get(encoding, "utf8")


def report_progress(
    callback: ProgressCallbackProtocol | None,
    progress: LoadProgress,
) -> None:
    """Report progress if callback is provided.

    Args:
        callback: Optional progress callback.
        progress: Progress state to report.
    """
    if callback is not None:
        callback(progress)


def is_numeric_string(value: str) -> bool:
    """Check if a string value can be parsed as a float.

    Args:
        value: String value to check.

    Returns:
        True if value is numeric, False otherwise.
    """
    if value in MISSING_VALUES:
        return True

    cleaned = value.replace(",", "")
    lower = cleaned.lower()

    if lower in ("inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"):
        return True

    test_str = cleaned.lstrip("-").lstrip("+")
    if not test_str:
        return False

    if "e" in test_str.lower():
        parts = test_str.lower().split("e")
        if len(parts) != 2:
            return False
        mantissa, exponent = parts
        if not _is_simple_numeric(mantissa):
            return False
        exp_str = exponent.lstrip("-").lstrip("+")
        return bool(exp_str and exp_str.isdigit())

    return _is_simple_numeric(test_str)


def _is_simple_numeric(value: str) -> bool:
    """Check if a string is a simple numeric value.

    Args:
        value: String to check.

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

    return any(part.isdigit() for part in parts if part)


def extract_entity_ids(
    df: PolarsDataFrameProtocol,
    entity_col: str,
) -> list[str]:
    """Extract entity IDs from DataFrame as string list.

    Args:
        df: Polars DataFrame.
        entity_col: Entity column name.

    Returns:
        List of entity IDs as strings.
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col
    entity_df = df.select(col_fn(entity_col))
    entity_series = entity_df.to_series()
    return [str(e) for e in entity_series.to_list()]


def extract_feature_array(
    df: PolarsDataFrameProtocol,
    feature_columns: list[str],
) -> NDArray[np.float64]:
    """Extract feature matrix from DataFrame.

    Args:
        df: Polars DataFrame.
        feature_columns: Feature column names.

    Returns:
        Feature matrix as numpy array.
    """
    polars_mod = __import__("polars")
    col_fn: PolarsColFnProtocol = polars_mod.col
    feature_df = df.select([col_fn(c) for c in feature_columns])
    return feature_df.to_numpy().astype(np.float64)


def sanitize_array_inplace(x_array: NDArray[np.float64]) -> None:
    """Replace non-finite values with 0.0 in-place.

    Uses column-by-column approach to minimize memory allocation.

    Args:
        x_array: Feature matrix to sanitize in place.
    """
    n_cols = int(x_array.shape[1])
    for col_idx in range(n_cols):
        col_view: NDArray[np.float64] = x_array[:, col_idx]
        np.nan_to_num(col_view, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


__all__ = [
    "PolarsColFnProtocol",
    "PolarsDataFrameProtocol",
    "PolarsDataTypeProtocol",
    "PolarsExprProtocol",
    "PolarsGroupByProtocol",
    "PolarsLitFnProtocol",
    "PolarsReadCSVProtocol",
    "PolarsSeriesProtocol",
    "PolarsWhenFnProtocol",
    "convert_encoding",
    "extract_entity_ids",
    "extract_feature_array",
    "is_numeric_string",
    "report_progress",
    "sanitize_array_inplace",
]
