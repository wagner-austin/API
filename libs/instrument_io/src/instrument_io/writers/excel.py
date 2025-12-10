"""Excel file writer implementation.

Provides typed writing of Excel files via openpyxl.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from instrument_io._exceptions import WriterError
from instrument_io._protocols.openpyxl import (
    WorkbookProtocol,
    WorksheetProtocol,
    _create_table,
    _create_workbook,
    _get_column_letter,
)
from instrument_io.types.common import CellValue


def _collect_columns(rows: list[dict[str, CellValue]]) -> list[str]:
    """Collect all unique column names preserving order.

    Args:
        rows: List of row dictionaries.

    Returns:
        List of unique column names in order of first appearance.
    """
    seen: set[str] = set()
    columns: list[str] = []

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    return columns


def _write_header_row(
    ws: WorksheetProtocol,
    columns: list[str],
    row_num: int,
) -> None:
    """Write header row to worksheet.

    Args:
        ws: Worksheet to write to.
        columns: Column names for headers.
        row_num: Row number (1-based).
    """
    for col_idx, col_name in enumerate(columns, start=1):
        ws.cell(row=row_num, column=col_idx, value=col_name)


def _write_data_row(
    ws: WorksheetProtocol,
    row_data: dict[str, CellValue],
    columns: list[str],
    row_num: int,
) -> None:
    """Write a data row to worksheet.

    Args:
        ws: Worksheet to write to.
        row_data: Row dictionary with typed values.
        columns: Column names defining order.
        row_num: Row number (1-based).
    """
    for col_idx, col_name in enumerate(columns, start=1):
        value = row_data.get(col_name)
        ws.cell(row=row_num, column=col_idx, value=value)


def _set_column_widths(
    ws: WorksheetProtocol,
    columns: list[str],
    rows: list[dict[str, CellValue]],
) -> None:
    """Auto-size column widths based on content.

    Args:
        ws: Worksheet to modify.
        columns: Column names.
        rows: Data rows for calculating widths.
    """
    for col_idx, col_name in enumerate(columns, start=1):
        # Start with header width
        max_width = len(col_name)

        # Check data widths
        for row in rows:
            value = row.get(col_name)
            if value is not None:
                value_len = len(str(value))
                if value_len > max_width:
                    max_width = value_len

        # Apply width with padding (min 8, max 50)
        width = min(max(max_width + 2, 8), 50)
        col_letter = _get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = float(width)


def _add_table_to_sheet(
    ws: WorksheetProtocol,
    columns: list[str],
    row_count: int,
    table_name: str,
) -> None:
    """Add Excel table formatting to worksheet.

    Args:
        ws: Worksheet to modify.
        columns: Column names.
        row_count: Number of data rows (not including header).
        table_name: Name for the table.
    """
    if not columns or row_count == 0:
        return

    # Build table range (header row + data rows)
    end_col = _get_column_letter(len(columns))
    end_row = row_count + 1  # +1 for header
    ref = f"A1:{end_col}{end_row}"

    # Create and add table
    table = _create_table(table_name, ref)
    ws.add_table(table)


def _sanitize_table_name(sheet_name: str, index: int) -> str:
    """Create a valid Excel table name from sheet name.

    Args:
        sheet_name: Original sheet name.
        index: Index for uniqueness.

    Returns:
        Valid table name (alphanumeric, underscore, starts with letter).
    """
    # Remove invalid characters
    clean = "".join(c if c.isalnum() or c == "_" else "_" for c in sheet_name)

    # Ensure starts with letter
    if not clean or not clean[0].isalpha():
        clean = f"Table_{clean}"

    # Add index for uniqueness
    return f"{clean}_{index}"


class ExcelWriter:
    """Writer for Excel files via openpyxl.

    Provides typed writing of data to Excel workbooks with
    automatic table formatting and column sizing.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def __init__(self, auto_table: bool = True, auto_width: bool = True) -> None:
        """Initialize Excel writer.

        Args:
            auto_table: Whether to add Excel table formatting.
            auto_width: Whether to auto-size columns.
        """
        self._auto_table = auto_table
        self._auto_width = auto_width

    def write_sheet(
        self,
        rows: list[dict[str, CellValue]],
        out_path: Path,
        sheet_name: str = "Sheet1",
    ) -> None:
        """Write rows to a single Excel sheet.

        Args:
            rows: List of row dictionaries to write.
            out_path: Output file path.
            sheet_name: Name for the worksheet.

        Raises:
            WriterError: If writing fails.
        """
        self.write_sheets({sheet_name: rows}, out_path)

    def write_sheets(
        self,
        sheets: Mapping[str, list[dict[str, CellValue]]],
        out_path: Path,
    ) -> None:
        """Write multiple sheets to an Excel file.

        Args:
            sheets: Mapping of sheet names to row lists.
            out_path: Output file path.

        Raises:
            WriterError: If writing fails or sheets is empty.
        """
        if not sheets:
            raise WriterError(str(out_path), "No sheets provided")

        # Create workbook
        wb: WorkbookProtocol = _create_workbook()

        # Remove default sheet (we'll create our own)
        default_sheet = wb.active

        for idx, (name, rows) in enumerate(sheets.items()):
            # Create or get sheet
            if idx == 0:
                # Rename default sheet for first
                ws: WorksheetProtocol = default_sheet
                ws.title = name
            else:
                ws = wb.create_sheet(title=name)

            # Get columns
            columns = _collect_columns(rows)

            if not columns:
                # Empty sheet - just set title
                continue

            # Write header row
            _write_header_row(ws, columns, 1)

            # Write data rows
            for row_idx, row_data in enumerate(rows, start=2):
                _write_data_row(ws, row_data, columns, row_idx)

            # Auto-size columns
            if self._auto_width:
                _set_column_widths(ws, columns, rows)

            # Add table formatting
            if self._auto_table and columns:
                table_name = _sanitize_table_name(name, idx)
                _add_table_to_sheet(ws, columns, len(rows), table_name)

        # Ensure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save workbook
        wb.save(out_path)
        wb.close()

    def write_rows_to_sheet(
        self,
        wb: WorkbookProtocol,
        sheet_name: str,
        rows: list[dict[str, CellValue]],
        table_index: int = 0,
    ) -> None:
        """Write rows to a sheet in an existing workbook.

        Used for building up workbooks incrementally.

        Args:
            wb: Existing workbook to write to.
            sheet_name: Name for the worksheet.
            rows: List of row dictionaries.
            table_index: Index for table naming uniqueness.
        """
        # Create sheet
        ws: WorksheetProtocol = wb.create_sheet(title=sheet_name)

        # Get columns
        columns = _collect_columns(rows)

        if not columns:
            return

        # Write header row
        _write_header_row(ws, columns, 1)

        # Write data rows
        for row_idx, row_data in enumerate(rows, start=2):
            _write_data_row(ws, row_data, columns, row_idx)

        # Auto-size columns
        if self._auto_width:
            _set_column_widths(ws, columns, rows)

        # Add table formatting
        if self._auto_table and columns:
            table_name = _sanitize_table_name(sheet_name, table_index)
            _add_table_to_sheet(ws, columns, len(rows), table_name)


__all__ = [
    "ExcelWriter",
]
