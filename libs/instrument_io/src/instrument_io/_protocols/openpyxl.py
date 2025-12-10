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


class ColorProtocol(Protocol):
    """Protocol for openpyxl Color."""

    rgb: str


class PatternFillProtocol(Protocol):
    """Protocol for openpyxl PatternFill."""

    start_color: ColorProtocol
    end_color: ColorProtocol
    fill_type: str


class CellProtocol(Protocol):
    """Protocol for openpyxl Cell."""

    alignment: AlignmentProtocol
    font: FontProtocol
    fill: PatternFillProtocol
    value: str | int | float | bool | None
    column_letter: str


class ColumnDimensionProtocol(Protocol):
    """Protocol for openpyxl ColumnDimension."""

    width: float


class _TableStyleInfoProtocol(Protocol):
    """Protocol for openpyxl TableStyleInfo - internal use only."""

    name: str


class _TableProtocol(Protocol):
    """Protocol for openpyxl Table - internal use only."""

    name: str
    ref: str


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

    def add_table(self, table: _TableProtocol) -> None:
        """Add a table to the worksheet."""
        ...

    def __getitem__(self, key: int) -> tuple[CellProtocol, ...]:
        """Get row by 1-based index."""
        ...

    @property
    def columns(self) -> tuple[tuple[CellProtocol, ...], ...]:
        """Return all columns as tuples of cells."""
        ...

    @property
    def column_dimensions(self) -> MutableMapping[str, ColumnDimensionProtocol]:
        """Return column dimensions mapping."""
        ...

    @property
    def max_row(self) -> int:
        """Return maximum row number with data."""
        ...

    @property
    def tables(self) -> dict[str, _TableProtocol]:
        """Return tables mapping."""
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


def _create_table(
    display_name: str,
    ref: str,
    style_name: str = "TableStyleMedium2",
    show_row_stripes: bool = True,
) -> _TableProtocol:
    """Create openpyxl Table with configurable style.

    Args:
        display_name: Table name (must be unique in workbook).
        ref: Cell range reference (e.g., "A1:D10").
        style_name: Excel table style name.
        show_row_stripes: Whether to show alternating row colors.

    Returns:
        _TableProtocol with specified styling.
    """
    table_mod = __import__("openpyxl.worksheet.table", fromlist=["Table", "TableStyleInfo"])
    style: _TableStyleInfoProtocol = table_mod.TableStyleInfo(
        name=style_name,
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=show_row_stripes,
        showColumnStripes=False,
    )
    table: _TableProtocol = table_mod.Table(
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
        color: Font color as hex string (e.g., "006100" for dark green).

    Returns:
        FontProtocol instance.
    """
    styles_mod = __import__("openpyxl.styles", fromlist=["Font"])
    font: FontProtocol = styles_mod.Font(bold=bold, size=size, color=color)
    return font


def _create_pattern_fill(
    *,
    start_color: str,
    end_color: str | None = None,
    fill_type: str = "solid",
) -> PatternFillProtocol:
    """Create openpyxl PatternFill.

    Args:
        start_color: Fill color as hex string (e.g., "C6EFCE" for light green).
        end_color: End color for gradient fills. Defaults to start_color.
        fill_type: Fill pattern type ("solid", "darkDown", etc.).

    Returns:
        PatternFillProtocol instance.
    """
    styles_mod = __import__("openpyxl.styles", fromlist=["PatternFill"])
    actual_end_color = end_color if end_color is not None else start_color
    fill: PatternFillProtocol = styles_mod.PatternFill(
        start_color=start_color,
        end_color=actual_end_color,
        fill_type=fill_type,
    )
    return fill


def _extract_header_strings(row: tuple[CellProtocol, ...]) -> list[str]:
    """Extract header strings from a row of cells.

    Converts cell values to strings. Non-string values are converted via str().
    None values become empty strings.

    Args:
        row: Tuple of cells from worksheet row.

    Returns:
        List of header strings.
    """
    headers: list[str] = []
    for cell in row:
        val = cell.value
        if val is None:
            headers.append("")
        elif isinstance(val, str):
            headers.append(val)
        else:
            headers.append(str(val))
    return headers


def _auto_adjust_column_widths(
    ws: WorksheetProtocol,
    max_width: int = 60,
    padding: int = 2,
) -> None:
    """Auto-adjust column widths based on content.

    Args:
        ws: Worksheet to adjust.
        max_width: Maximum column width.
        padding: Extra padding to add to calculated width.
    """
    for column_cells in ws.columns:
        column_tuple = tuple(column_cells)
        first_cell = column_tuple[0]
        column_letter = first_cell.column_letter
        max_length = 0
        for cell in column_tuple:
            cell_value = cell.value
            if cell_value is not None:
                cell_len = len(str(cell_value))
                if cell_len > max_length:
                    max_length = cell_len
        adjusted_width = min(max_length + padding, max_width)
        ws.column_dimensions[column_letter].width = float(adjusted_width)


__all__ = [
    "AlignmentProtocol",
    "CellProtocol",
    "ColorProtocol",
    "ColumnDimensionProtocol",
    "FontProtocol",
    "PatternFillProtocol",
    "WorkbookProtocol",
    "WorksheetProtocol",
    "_auto_adjust_column_widths",
    "_create_alignment",
    "_create_font",
    "_create_pattern_fill",
    "_create_table",
    "_create_workbook",
    "_extract_header_strings",
    "_get_column_letter",
    "_load_workbook",
]
