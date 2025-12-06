"""Protocol definitions for openpyxl library.

Provides type-safe interfaces to openpyxl Workbook, Worksheet, and Cell
classes without importing openpyxl directly.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Protocol


class AlignmentProtocol(Protocol):
    """Protocol for openpyxl Alignment."""

    horizontal: str | None
    vertical: str | None


class FontProtocol(Protocol):
    """Protocol for openpyxl Font."""

    bold: bool
    size: float
    color: str | None


class CellProtocol(Protocol):
    """Protocol for openpyxl Cell."""

    alignment: AlignmentProtocol
    font: FontProtocol
    value: str | int | float | bool | None


class ColumnDimensionProtocol(Protocol):
    """Protocol for openpyxl ColumnDimension."""

    width: float


class _ExcelTable(Protocol):
    """Protocol for openpyxl Table objects."""

    name: str
    ref: str


class _ExcelTableStyle(Protocol):
    """Opaque type for openpyxl TableStyleInfo objects."""


class _GetColumnLetterFn(Protocol):
    """Protocol for openpyxl.utils.get_column_letter function."""

    def __call__(self, col_idx: int) -> str: ...


class WorksheetProtocol(Protocol):
    """Protocol for openpyxl Worksheet."""

    def cell(
        self, row: int, column: int, value: str | int | float | bool | None = None
    ) -> CellProtocol:
        """Get or create cell at (row, column)."""
        ...

    def add_table(self, table: _ExcelTable) -> None:
        """Add a table to the worksheet."""
        ...

    @property
    def column_dimensions(self) -> MutableMapping[str, ColumnDimensionProtocol]:
        """Return column dimensions mapping."""
        ...

    @property
    def title(self) -> str:
        """Return worksheet title."""
        ...

    @title.setter
    def title(self, value: str) -> None:
        """Set worksheet title."""
        ...


class WorkbookProtocol(Protocol):
    """Protocol for openpyxl Workbook."""

    @property
    def sheetnames(self) -> list[str]:
        """Return list of sheet names."""
        ...

    def __getitem__(self, name: str) -> WorksheetProtocol:
        """Get worksheet by name."""
        ...

    def create_sheet(self, title: str) -> WorksheetProtocol:
        """Create a new worksheet."""
        ...

    def save(self, filename: str | Path) -> None:
        """Save workbook to file."""
        ...

    def close(self) -> None:
        """Close workbook."""
        ...

    @property
    def active(self) -> WorksheetProtocol:
        """Return active worksheet."""
        ...

    def remove(self, ws: WorksheetProtocol) -> None:
        """Remove a worksheet."""
        ...


class _LoadWorkbookFn(Protocol):
    """Protocol for openpyxl load_workbook function."""

    def __call__(
        self, path: Path, read_only: bool = False, data_only: bool = False
    ) -> WorkbookProtocol: ...


class _WorkbookCtor(Protocol):
    """Protocol for openpyxl.Workbook constructor."""

    def __call__(self) -> WorkbookProtocol: ...


def _load_workbook(
    path: Path, read_only: bool = False, data_only: bool = False
) -> WorkbookProtocol:
    """Load workbook with proper typing via Protocol.

    Args:
        path: Path to Excel file.
        read_only: Open in read-only mode.
        data_only: Read cell values only, not formulas.

    Returns:
        WorkbookProtocol for the loaded workbook.
    """
    openpyxl_mod = __import__("openpyxl")
    load_fn: _LoadWorkbookFn = openpyxl_mod.load_workbook
    return load_fn(path, read_only=read_only, data_only=data_only)


def _create_workbook() -> WorkbookProtocol:
    """Create a new openpyxl Workbook with strict typing.

    Returns:
        WorkbookProtocol for the new workbook.
    """
    openpyxl_mod = __import__("openpyxl")
    ctor: _WorkbookCtor = openpyxl_mod.Workbook
    return ctor()


def _get_column_letter(col_idx: int) -> str:
    """Get Excel column letter via typed Protocol.

    Args:
        col_idx: 1-based column index.

    Returns:
        Column letter (e.g., "A", "B", "AA").
    """
    utils_mod = __import__("openpyxl.utils", fromlist=["get_column_letter"])
    fn: _GetColumnLetterFn = utils_mod.get_column_letter
    return fn(col_idx)


def _create_styled_table(display_name: str, ref: str) -> _ExcelTable:
    """Create openpyxl Table with default style.

    Args:
        display_name: Table name (must be unique in workbook).
        ref: Cell range reference (e.g., "A1:D10").

    Returns:
        _ExcelTable with TableStyleMedium2 styling.
    """
    table_mod = __import__("openpyxl.worksheet.table", fromlist=["Table", "TableStyleInfo"])
    style: _ExcelTableStyle = table_mod.TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    table: _ExcelTable = table_mod.Table(
        displayName=display_name,
        ref=ref,
        tableStyleInfo=style,
        autoFilter=None,
    )
    return table


def _create_alignment(
    *,
    horizontal: str = "general",
    vertical: str = "bottom",
) -> AlignmentProtocol:
    """Create openpyxl Alignment.

    Args:
        horizontal: Horizontal alignment ("left", "center", "right", "general").
        vertical: Vertical alignment ("top", "center", "bottom").

    Returns:
        AlignmentProtocol instance.
    """
    styles_mod = __import__("openpyxl.styles", fromlist=["Alignment"])
    alignment: AlignmentProtocol = styles_mod.Alignment(
        horizontal=horizontal,
        vertical=vertical,
    )
    return alignment


def _create_font(
    *,
    bold: bool = False,
    size: float = 11.0,
    color: str | None = None,
) -> FontProtocol:
    """Create openpyxl Font.

    Args:
        bold: Whether font is bold.
        size: Font size in points.
        color: Font color as hex string (e.g., "FFFFFFFF").

    Returns:
        FontProtocol instance.
    """
    styles_mod = __import__("openpyxl.styles", fromlist=["Font"])
    font: FontProtocol = styles_mod.Font(bold=bold, size=size, color=color)
    return font


__all__ = [
    "AlignmentProtocol",
    "CellProtocol",
    "ColumnDimensionProtocol",
    "FontProtocol",
    "WorkbookProtocol",
    "WorksheetProtocol",
    "_create_alignment",
    "_create_font",
    "_create_styled_table",
    "_create_workbook",
    "_get_column_letter",
    "_load_workbook",
]
