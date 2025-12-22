"""Excel file parsing (.xlsx and .xls).

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol

# Maximum rows to sample for analysis
MAX_SAMPLE_ROWS = 1000

# Type alias for Excel cell values
ExcelCellValue = str | int | float | bool | None

# Type alias for a row of Excel cell values
ExcelRow = tuple[ExcelCellValue, ...]


class WorksheetProtocol(Protocol):
    """Protocol for openpyxl Worksheet in read_only mode.

    Uses iter_rows for efficient streaming instead of cell-by-cell access.
    """

    def iter_rows(
        self,
        *,
        values_only: bool = False,
    ) -> Generator[ExcelRow, None, None]:
        """Iterate over rows in the worksheet.

        Args:
            values_only: If True, yield tuples of cell values instead of Cell objects.

        Yields:
            Tuples of cell values when values_only=True.
        """
        ...


class WorkbookProtocol(Protocol):
    """Protocol for openpyxl Workbook to enable testing."""

    @property
    def sheetnames(self) -> list[str]:
        """Return list of sheet names."""
        ...

    def __getitem__(self, name: str) -> WorksheetProtocol:
        """Get worksheet by name."""
        ...

    def close(self) -> None:
        """Close the workbook."""
        ...


class LoadWorkbookProtocol(Protocol):
    """Protocol for openpyxl load_workbook function."""

    def __call__(
        self,
        filename: Path,
        read_only: bool = False,
        data_only: bool = False,
    ) -> WorkbookProtocol:
        """Load a workbook from file."""
        ...


def get_load_workbook() -> LoadWorkbookProtocol:
    """Get openpyxl load_workbook function with typing.

    Returns:
        Typed load_workbook function.
    """
    openpyxl_mod = __import__("openpyxl")
    load_fn: LoadWorkbookProtocol = openpyxl_mod.load_workbook
    return load_fn


def read_excel_header_and_sample(
    path: Path,
) -> tuple[tuple[str, ...], int, tuple[tuple[str, ...], ...]]:
    """Read Excel header and sample rows.

    Prefers sheets named 'Data', 'Sheet1', etc. over description sheets.
    Uses openpyxl iter_rows for efficient read_only streaming.

    Args:
        path: Path to Excel file.

    Returns:
        Tuple of (column names, total row count, sample rows).
    """
    load_workbook = get_load_workbook()
    wb = load_workbook(path, read_only=True, data_only=True)

    # Prefer sheets with data-like names over description sheets
    preferred = ("data", "sheet1", "train", "dataset")
    sheet_name = wb.sheetnames[0]
    for name in wb.sheetnames:
        if name.lower() in preferred:
            sheet_name = name
            break
    ws = wb[sheet_name]

    # Use iter_rows for efficient streaming in read_only mode
    columns: tuple[str, ...] = ()
    sample_rows: list[tuple[str, ...]] = []
    n_rows = 0

    for row in ws.iter_rows(values_only=True):
        if n_rows == 0:
            # First row is header
            columns = tuple(str(val) if val is not None else "" for val in row)
        else:
            # Data rows
            if len(sample_rows) < MAX_SAMPLE_ROWS:
                sample_rows.append(tuple(str(val) if val is not None else "" for val in row))
        n_rows += 1

    wb.close()
    n_data_rows = max(0, n_rows - 1)
    return columns, n_data_rows, tuple(sample_rows)


class XlrdSheetProtocol(Protocol):
    """Protocol for xlrd Sheet."""

    @property
    def nrows(self) -> int:
        """Return number of rows."""
        ...

    @property
    def ncols(self) -> int:
        """Return number of columns."""
        ...

    def cell_value(self, rowx: int, colx: int) -> str | int | float | bool:
        """Get cell value at row/column."""
        ...


class XlrdBookProtocol(Protocol):
    """Protocol for xlrd Book (workbook)."""

    def sheet_names(self) -> list[str]:
        """Return list of sheet names."""
        ...

    def sheet_by_index(self, sheetx: int) -> XlrdSheetProtocol:
        """Get sheet by index."""
        ...


class OpenWorkbookProtocol(Protocol):
    """Protocol for xlrd open_workbook function."""

    def __call__(self, filename: Path) -> XlrdBookProtocol:
        """Open a workbook from file."""
        ...


def get_xlrd_open_workbook() -> OpenWorkbookProtocol:
    """Get xlrd open_workbook function with typing.

    Returns:
        Typed open_workbook function.
    """
    xlrd_mod = __import__("xlrd")
    open_fn: OpenWorkbookProtocol = xlrd_mod.open_workbook
    return open_fn


def read_xls_header_and_sample(
    path: Path,
) -> tuple[tuple[str, ...], int, tuple[tuple[str, ...], ...]]:
    """Read legacy Excel (.xls) header and sample rows using xlrd.

    Args:
        path: Path to .xls file.

    Returns:
        Tuple of (column names, total row count, sample rows).
    """
    open_workbook = get_xlrd_open_workbook()
    wb = open_workbook(path)

    # xlrd always has at least one sheet in a valid .xls file
    ws = wb.sheet_by_index(0)
    n_rows = ws.nrows
    n_cols = ws.ncols

    if n_rows == 0 or n_cols == 0:
        return (), 0, ()

    # Read header row (row 0 in xlrd, 0-indexed)
    columns: list[str] = []
    for col_idx in range(n_cols):
        cell_val = ws.cell_value(0, col_idx)
        columns.append(str(cell_val) if cell_val != "" else "")

    # Read sample data rows (skip header)
    n_data_rows = n_rows - 1
    sample_limit = min(n_data_rows, MAX_SAMPLE_ROWS)
    sample_rows: list[tuple[str, ...]] = []

    for row_idx in range(1, sample_limit + 1):
        row_values: list[str] = []
        for col_idx in range(n_cols):
            cell_val = ws.cell_value(row_idx, col_idx)
            row_values.append(str(cell_val) if cell_val != "" else "")
        sample_rows.append(tuple(row_values))

    return tuple(columns), n_data_rows, tuple(sample_rows)


__all__ = [
    "ExcelCellValue",
    "ExcelRow",
    "LoadWorkbookProtocol",
    "OpenWorkbookProtocol",
    "WorkbookProtocol",
    "WorksheetProtocol",
    "XlrdBookProtocol",
    "XlrdSheetProtocol",
    "get_load_workbook",
    "get_xlrd_open_workbook",
    "read_excel_header_and_sample",
    "read_xls_header_and_sample",
]
