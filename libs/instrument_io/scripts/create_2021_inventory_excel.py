"""Create consolidated 2021 chemical inventory Excel file.

This script processes the 2021 chemical inventory from multiple sheets
(CiBR-Trac and Room 428) into a unified, formatted Excel report.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from platform_core.logging import get_logger, setup_logging

from instrument_io._json_bridge import CellValue, _df_json_to_row_dicts, _json_value_to_cell
from instrument_io._protocols.openpyxl import (
    _auto_adjust_column_widths,
    _create_table,
    _create_workbook,
    _get_column_letter,
)

logger = get_logger(__name__)


def _read_cibr_trac_sheet(input_path: Path) -> pl.DataFrame | None:
    """Read and normalize CiBR-Trac sheet.

    Args:
        input_path: Path to input Excel file

    Returns:
        Normalized DataFrame or None if sheet cannot be read
    """
    logger.info("Reading 'CiBR-Trac' sheet...")
    try:
        df_cibr = pl.read_excel(source=input_path, sheet_name="CiBR-Trac", engine="openpyxl")
    except ValueError:
        logger.warning("Sheet 'CiBR-Trac' not found in file")
        return None

    # Select columns and cast types
    df_cibr = df_cibr.with_columns(
        pl.col("Container_Size").cast(pl.Float64, strict=False),
        pl.col("Container_Number").cast(pl.Int64, strict=False),
    )

    # Select needed columns and add literals
    select_cols: list[str] = [
        "Chemical_Name",
        "CAS",
        "Chemical_Physical_State",
        "Container_Size",
        "Units",
        "Container_Number",
    ]
    df_cibr = df_cibr.select(select_cols)

    # Rename columns
    df_cibr = df_cibr.rename(
        {
            "Chemical_Name": "Chemical Name",
            "Chemical_Physical_State": "Physical State",
            "Container_Size": "Amount",
            "Container_Number": "Containers",
        }
    )

    # Add literal columns
    df_cibr = df_cibr.with_columns(
        pl.lit("CiBR-Trac").alias("Location"),
        pl.lit("2021 Inventory (CiBR-Trac)").alias("Source"),
    )

    logger.info("  Loaded %d rows from CiBR-Trac", len(df_cibr))
    return df_cibr


def _read_428_sheet(input_path: Path) -> pl.DataFrame | None:
    """Read and normalize Room 428 sheet.

    Args:
        input_path: Path to input Excel file

    Returns:
        Normalized DataFrame or None if sheet cannot be read
    """
    logger.info("Reading '428' sheet...")
    try:
        df_428 = pl.read_excel(source=input_path, sheet_name="428", engine="openpyxl")
    except ValueError:
        logger.warning("Sheet '428' not found in file")
        return None

    # Construct Location from Room Number and Sublocation, and cast Amount
    df_428 = df_428.with_columns(
        pl.concat_str(
            pl.lit("Room "),
            pl.col("Room Number").cast(pl.Utf8),
            pl.lit(" - "),
            pl.col("Sublocation"),
        ).alias("Location"),
        pl.col("Amount").cast(pl.Float64, strict=False),
    )

    # Select needed columns
    select_cols: list[str] = [
        "Chemical Name",
        "CAS",
        "Physical State",
        "Amount",
        "Units",
        "Location",
    ]
    df_428 = df_428.select(select_cols)

    # Add literal columns
    df_428 = df_428.with_columns(
        pl.lit(1).cast(pl.Int64).alias("Containers"),
        pl.lit("2021 Inventory (Room 428)").alias("Source"),
    )

    # Reorder columns to match CiBR-Trac order for concat compatibility
    final_cols: list[str] = [
        "Chemical Name",
        "CAS",
        "Physical State",
        "Amount",
        "Units",
        "Containers",
        "Location",
        "Source",
    ]
    df_428 = df_428.select(final_cols)

    logger.info("  Loaded %d rows from 428", len(df_428))
    return df_428


def _write_excel_with_formatting(df: pl.DataFrame, output_path: Path) -> None:
    """Write DataFrame to Excel with table formatting.

    Args:
        df: DataFrame to write
        output_path: Output file path

    Raises:
        PermissionError: If file is open or cannot be written
    """
    logger.info("Writing to: %s", output_path)

    wb = _create_workbook()
    ws = wb.active
    ws.title = "Chemical_List_2021"

    # Write Headers
    headers: list[str] = df.columns
    for col_idx, header in enumerate(headers, 1):
        ws.cell(row=1, column=col_idx, value=header)

    # Write Data via JSON serialization to avoid Any types
    rows = _df_json_to_row_dicts(df.write_json())
    for row_idx, row_data in enumerate(rows, 2):
        for col_idx, col_name in enumerate(headers, 1):
            cell_value: CellValue = _json_value_to_cell(row_data.get(col_name))
            ws.cell(row=row_idx, column=col_idx, value=cell_value)

    # Create Excel Table
    last_row = len(df) + 1
    last_col_letter = _get_column_letter(len(headers))
    table_ref = f"A1:{last_col_letter}{last_row}"

    tab = _create_table(
        display_name="Inventory2021",
        ref=table_ref,
        style_name="TableStyleMedium9",
        show_row_stripes=True,
    )
    ws.add_table(tab)

    # Auto-adjust column widths
    _auto_adjust_column_widths(ws, max_width=60, padding=2)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info("Successfully created: %s", output_path)


DEFAULT_BASE_PATH = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab")


def create_2021_inventory_report(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    """Create the 2021 inventory report.

    Args:
        input_path: Path to input Excel file (uses default if None)
        output_path: Path to output Excel file (uses default if None)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if input_path is None:
        input_path = (
            DEFAULT_BASE_PATH / "Important Docs/Chemical Inventory/02252021-Chemical Inventory.xlsx"
        )
    if output_path is None:
        output_path = (
            DEFAULT_BASE_PATH / "Notebooks/Emily Truong Notebook/Chemical_Inventory_List_2021.xlsx"
        )

    logger.info("--- Processing 2021 Inventory ---")
    logger.info("Input: %s", input_path)

    combined_data: list[pl.DataFrame] = []

    # Process CiBR-Trac Sheet
    df_cibr = _read_cibr_trac_sheet(input_path)
    if df_cibr is not None:
        combined_data.append(df_cibr)

    # Process Room 428 Sheet
    df_428 = _read_428_sheet(input_path)
    if df_428 is not None:
        combined_data.append(df_428)

    if not combined_data:
        logger.error("No data loaded. Exiting.")
        return 1

    # Combine DataFrames
    final_df = pl.concat(combined_data)

    # Cast types for consistency
    final_df = final_df.with_columns(
        pl.col("Chemical Name").cast(pl.Utf8),
        pl.col("CAS").cast(pl.Utf8),
        pl.col("Physical State").cast(pl.Utf8),
        pl.col("Amount").cast(pl.Float64, strict=False),
        pl.col("Units").cast(pl.Utf8),
        pl.col("Containers").cast(pl.Int64, strict=False),
        pl.col("Location").cast(pl.Utf8),
        pl.col("Source").cast(pl.Utf8),
    )

    logger.info("Total Combined Rows: %d", len(final_df))

    _write_excel_with_formatting(final_df, output_path)
    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="create-2021-inventory",
        instance_id=None,
        extra_fields=None,
    )
    return create_2021_inventory_report()


if __name__ == "__main__":
    raise SystemExit(main())
