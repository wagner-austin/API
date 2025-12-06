"""Excel file reader implementation.

Provides typed reading of Excel files via polars (preferred) or openpyxl.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from instrument_io._decoders.excel import (
    _decode_rows,
)
from instrument_io._exceptions import ExcelReadError
from instrument_io._json_bridge import CellValue, JSONValue, _load_json_str
from instrument_io._protocols.openpyxl import WorksheetProtocol


def _is_excel_file(path: Path) -> bool:
    """Check if path is an Excel file."""
    return path.is_file() and path.suffix.lower() in (".xlsx", ".xls", ".xlsm")


class _PolarsDataFrameProtocol(Protocol):
    """Protocol for polars DataFrame."""

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
    ) -> _PolarsDataFrameProtocol:
        """Read Excel file into DataFrame."""
        ...


class _PolarsExcelSheetNamesProtocol(Protocol):
    """Protocol for polars excel sheet names function."""

    def __call__(self, source: str | Path) -> list[str]:
        """Get list of sheet names."""
        ...


def _get_polars_read_excel() -> _PolarsExcelReaderProtocol:
    """Get polars read_excel function with typing.

    Returns:
        Typed read_excel function.
    """
    polars_mod = __import__("polars")
    read_fn: _PolarsExcelReaderProtocol = polars_mod.read_excel
    return read_fn


def _get_polars_excel_sheet_names() -> _PolarsExcelSheetNamesProtocol:
    """Get a typed function to list Excel sheet names.

    Implementation note:
    - We intentionally use openpyxl to enumerate sheet names rather than polars.
      polars focuses on reading data and does not expose a stable, typed API
      to list sheet names without attempting a read. Using openpyxl keeps the
      interface simple and strictly typed for sheet discovery.

    Limitations:
    - openpyxl supports .xlsx/.xlsm. It does not support legacy .xls files.
      Callers should gate .xls accordingly at higher levels.
    """
    from instrument_io._protocols.openpyxl import _load_workbook

    def get_sheet_names(source: str | Path) -> list[str]:
        wb = _load_workbook(Path(source), read_only=True, data_only=True)
        names: list[str] = wb.sheetnames
        wb.close()
        return names

    return get_sheet_names


def _parse_polars_json_to_rows(
    json_str: str,
    columns: list[str],
) -> list[dict[str, CellValue]]:
    """Parse polars JSON output to typed rows.

    Args:
        json_str: JSON string from DataFrame.write_json().
        columns: Column names for row dicts.

    Returns:
        List of typed row dictionaries.
    """
    value: JSONValue = _load_json_str(json_str)

    if not isinstance(value, list):
        return []

    rows: list[dict[str, JSONValue]] = []
    for row in value:
        if isinstance(row, dict):
            rows.append(row)

    return _decode_rows(rows, columns)


def _convert_cell_value(value: bool | int | float | str | None) -> CellValue:
    """Convert openpyxl cell value to CellValue type.

    Args:
        value: Raw cell value from openpyxl.

    Returns:
        Typed CellValue (same value, type-narrowed).
    """
    # Type signature guarantees value is one of the allowed types
    return value


def _read_worksheet_row(
    ws: WorksheetProtocol,
    row_idx: int,
    max_cols: int = 1000,
) -> tuple[list[CellValue], bool]:
    """Read a single row from worksheet.

    Args:
        ws: Worksheet to read from.
        row_idx: 1-based row index.
        max_cols: Maximum columns to read (safety limit).

    Returns:
        Tuple of (row values, has_data flag).
    """

    row_values: list[CellValue] = []
    has_data = False

    for col_idx in range(1, max_cols):
        cell = ws.cell(row=row_idx, column=col_idx)
        raw_value = cell.value if hasattr(cell, "value") else None

        cell_value = _convert_cell_value(raw_value)
        row_values.append(cell_value)

        if cell_value is not None:
            has_data = True

        # Check if we've hit empty columns
        if col_idx > 10 and not any(row_values[-10:]):
            row_values = row_values[:-10]
            break

    return row_values, has_data


def _extract_headers(headers_raw: list[CellValue]) -> list[str]:
    """Extract string headers from raw cell values.

    Args:
        headers_raw: Raw cell values from header row.

    Returns:
        List of string headers.
    """
    headers: list[str] = []
    for h in headers_raw:
        if h is None:
            headers.append("")
        elif isinstance(h, str):
            headers.append(h)
        else:
            headers.append(str(h))
    return headers


def _build_row_dicts(
    all_rows: list[list[CellValue]],
    headers: list[str],
    start_row: int,
) -> list[dict[str, CellValue]]:
    """Build row dictionaries from raw rows.

    Args:
        all_rows: All raw row data.
        headers: Column headers.
        start_row: Starting row index (0-based) for data.

    Returns:
        List of row dictionaries.
    """
    result: list[dict[str, CellValue]] = []
    for row in all_rows[start_row:]:
        row_dict: dict[str, CellValue] = {}
        for i, header in enumerate(headers):
            if header and i < len(row):
                row_dict[header] = row[i]
        if row_dict:  # Skip empty rows
            result.append(row_dict)
    return result


class ExcelReader:
    """Reader for Excel files via polars.

    Provides typed access to Excel worksheet data. Uses polars for
    efficient reading with JSON bridge for type safety.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is an Excel file.

        Args:
            path: Path to check.

        Returns:
            True if path is an Excel file (.xlsx, .xls, .xlsm).
        """
        return _is_excel_file(path)

    def list_sheets(self, path: Path) -> list[str]:
        """List all sheet names in workbook.

        Args:
            path: Path to Excel file.

        Returns:
            List of sheet names.

        Raises:
            ExcelReadError: If reading fails.
        """
        if not path.exists():
            raise ExcelReadError(str(path), "File does not exist")

        if not _is_excel_file(path):
            raise ExcelReadError(str(path), "Not an Excel file")

        # openpyxl cannot read legacy .xls files; provide an explicit error
        if path.suffix.lower() == ".xls":
            raise ExcelReadError(
                str(path),
                "Listing sheet names for .xls is not supported; "
                "convert to .xlsx/.xlsm or read data via read_sheet",
            )

        get_names = _get_polars_excel_sheet_names()
        return get_names(path)

    def read_sheet(
        self,
        path: Path,
        sheet_name: str,
    ) -> list[dict[str, CellValue]]:
        """Read a single sheet as list of row dictionaries.

        Args:
            path: Path to Excel file.
            sheet_name: Name of sheet to read.

        Returns:
            List of row dictionaries with typed cell values.
            First row is used as column headers.

        Raises:
            ExcelReadError: If reading fails.
        """
        if not path.exists():
            raise ExcelReadError(str(path), "File does not exist")

        if not _is_excel_file(path):
            raise ExcelReadError(str(path), "Not an Excel file")

        read_excel = _get_polars_read_excel()
        df: _PolarsDataFrameProtocol = read_excel(path, sheet_name=sheet_name)

        columns: list[str] = df.columns
        json_str: str = df.write_json()

        return _parse_polars_json_to_rows(json_str, columns)

    def read_sheets(
        self,
        path: Path,
    ) -> dict[str, list[dict[str, CellValue]]]:
        """Read all sheets from workbook.

        Args:
            path: Path to Excel file.

        Returns:
            Dictionary mapping sheet names to row lists.

        Raises:
            ExcelReadError: If reading fails.
        """
        sheet_names = self.list_sheets(path)
        result: dict[str, list[dict[str, CellValue]]] = {}

        for name in sheet_names:
            result[name] = self.read_sheet(path, name)

        return result

    def read_sheet_with_header_row(
        self,
        path: Path,
        sheet_name: str,
        header_row: int,
        max_row: int = 10000,
    ) -> list[dict[str, CellValue]]:
        """Read sheet with custom header row.

        Args:
            path: Path to Excel file.
            sheet_name: Name of sheet to read.
            header_row: 0-based row index for headers.
            max_row: Maximum rows to read (safety limit, default 10000).

        Returns:
            List of row dictionaries using specified header row.

        Raises:
            ExcelReadError: If reading fails or header_row invalid.
        """
        # For custom header rows, we use openpyxl directly
        from instrument_io._protocols.openpyxl import _load_workbook

        if not path.exists():
            raise ExcelReadError(str(path), "File does not exist")

        wb = _load_workbook(path, read_only=True, data_only=True)

        if sheet_name not in wb.sheetnames:
            wb.close()
            raise ExcelReadError(str(path), f"Sheet '{sheet_name}' not found")

        ws = wb[sheet_name]

        # Read all rows using helper function
        all_rows: list[list[CellValue]] = []
        row_idx = 1

        while row_idx <= max_row:
            row_values, has_data = _read_worksheet_row(ws, row_idx)
            if not has_data:
                break
            all_rows.append(row_values)
            row_idx += 1

        wb.close()

        if header_row >= len(all_rows):
            raise ExcelReadError(
                str(path),
                f"Header row {header_row} exceeds data rows ({len(all_rows)})",
            )

        # Extract headers and build result
        headers = _extract_headers(all_rows[header_row])
        return _build_row_dicts(all_rows, headers, header_row + 1)


__all__ = [
    "ExcelReader",
]
