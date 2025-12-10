"""Update inventory Excel file with Source and Date columns.

This script adds or updates the Source and Date metadata columns in the
2025 chemical inventory Excel file.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger, setup_logging

from instrument_io._protocols.openpyxl import (
    _extract_header_strings,
    _get_column_letter,
    _load_workbook,
)

logger = get_logger(__name__)

DEFAULT_EXCEL_PATH = Path(
    r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab"
    r"\Notebooks\Emily Truong Notebook\Chemical_Inventory_List_2025.xlsx"
)


def update_inventory_metadata(excel_path: Path | None = None) -> int:
    """Update inventory Excel with Source and Date columns.

    Args:
        excel_path: Path to Excel file (uses default if None)

    Returns:
        Exit code (0 for success)

    Raises:
        FileNotFoundError: If Excel file does not exist
        PermissionError: If file is open or cannot be written
    """
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH

    logger.info("Updating Excel file: %s", excel_path)

    wb = _load_workbook(excel_path)
    ws = wb.active

    header_row = ws[1]
    headers: list[str] = _extract_header_strings(header_row)
    max_row = ws.max_row

    # Update Source Column
    source_col_idx: int
    if "Source" in headers:
        logger.info("Source column already exists. Updating values...")
        source_col_idx = headers.index("Source") + 1
    else:
        logger.info("Adding Source column...")
        source_col_idx = len(headers) + 1
        ws.cell(row=1, column=source_col_idx, value="Source")
        headers.append("Source")

    for row in range(2, max_row + 1):
        ws.cell(row=row, column=source_col_idx, value="UCI Chemical Inventory (Risk & Safety)")

    # Update Date Column
    date_col_idx: int
    if "Date" in headers:
        logger.info("Date column already exists. Updating values...")
        date_col_idx = headers.index("Date") + 1
    else:
        logger.info("Adding Date column...")
        date_col_idx = len(headers) + 1
        ws.cell(row=1, column=date_col_idx, value="Date")
        headers.append("Date")

    for row in range(2, max_row + 1):
        ws.cell(row=row, column=date_col_idx, value="2025-12-05")

    # Update Table Range
    tables = ws.tables
    if tables:
        table_names = list(tables.keys())
        table_name = table_names[0]
        tab = tables[table_name]

        last_col_idx = len(headers)
        last_col_letter = _get_column_letter(last_col_idx)
        new_ref = f"A1:{last_col_letter}{max_row}"

        tab.ref = new_ref
        logger.info("Updated Table '%s' range to %s", table_name, new_ref)

    # Auto-adjust column widths
    ws.column_dimensions[_get_column_letter(source_col_idx)].width = 40
    ws.column_dimensions[_get_column_letter(date_col_idx)].width = 15

    # Save
    logger.info("Saving updated Excel file...")
    wb.save(excel_path)
    logger.info("Success! Inventory file updated with Source and Date.")

    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="update-inventory-source",
        instance_id=None,
        extra_fields=None,
    )
    return update_inventory_metadata()


if __name__ == "__main__":
    raise SystemExit(main())
