"""Polars DataFrame protocol for strict type checking.

Provides Protocol-based typing for polars DataFrames without importing
polars directly, enabling strict mypy type checking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class PolarsDataFrameProtocol(Protocol):
    """Protocol for polars DataFrame with typed properties.

    This protocol defines the minimal interface needed for type-safe
    DataFrame operations without importing polars directly.
    """

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        ...

    @property
    def height(self) -> int:
        """Return number of rows."""
        ...

    def write_json(self) -> str:
        """Serialize to JSON string (row-oriented format)."""
        ...


class _PolarsExcelReaderProtocol(Protocol):
    """Protocol for polars read_excel function."""

    def __call__(
        self,
        source: str | Path,
        sheet_name: str | None = None,
        engine: str | None = None,
    ) -> PolarsDataFrameProtocol:
        """Read Excel file into DataFrame."""
        ...


def _get_polars_read_excel() -> _PolarsExcelReaderProtocol:
    """Get polars read_excel function with typing.

    Returns:
        Typed read_excel function.
    """
    polars_mod = __import__("polars")
    read_fn: _PolarsExcelReaderProtocol = polars_mod.read_excel
    return read_fn


__all__ = [
    "PolarsDataFrameProtocol",
    "_get_polars_read_excel",
]
