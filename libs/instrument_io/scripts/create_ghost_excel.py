"""Create ghost inventory report Excel file.

This script identifies chemicals that were in the 2021 inventory but are
missing from the 2025 inventory (ghost chemicals).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from platform_core.logging import get_logger, setup_logging

from instrument_io._json_bridge import (
    _df_json_to_row_dicts,
    _get_json_str_value,
    _json_col_to_str_list,
)
from instrument_io._protocols.openpyxl import (
    _auto_adjust_column_widths,
    _create_table,
    _create_workbook,
)

logger = get_logger(__name__)


def _read_2021_inventory(inv_path: Path) -> pl.DataFrame:
    """Read and normalize 2021 inventory from multiple sheets.

    Args:
        inv_path: Path to 2021 inventory Excel file

    Returns:
        Combined and normalized DataFrame

    Raises:
        FileNotFoundError: If file does not exist
        RuntimeError: If no sheets can be read
    """
    known_sheets = ["CiBR-Trac", "428"]
    df_list: list[pl.DataFrame] = []

    for sheet in known_sheets:
        try:
            df_sheet = pl.read_excel(source=inv_path, sheet_name=sheet, engine="openpyxl")
        except ValueError:
            logger.warning("Sheet '%s' not found in file", sheet)
            continue
        df_sheet = df_sheet.with_columns(pl.lit(sheet).alias("Original Sheet"))
        df_list.append(df_sheet)
        logger.info("Successfully read sheet '%s'", sheet)

    if not df_list:
        raise RuntimeError(f"No sheets could be read from {inv_path}")

    # Normalize column names across sheets
    normalized_dfs: list[pl.DataFrame] = []
    for df in df_list:
        cols = df.columns

        # Normalize Chemical Name column
        if "Chemical_Name" not in cols and "Chemical Name" in cols:
            df = df.rename({"Chemical Name": "Chemical_Name"})

        # Handle CAS
        if "CAS" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("CAS"))

        # Handle Physical State
        if "Chemical_Physical_State" not in df.columns:
            if "Physical State" in df.columns:
                df = df.rename({"Physical State": "Chemical_Physical_State"})
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("Chemical_Physical_State"))

        # Cast column types
        df = df.with_columns(
            pl.col("Chemical_Name").cast(pl.Utf8),
            pl.col("CAS").cast(pl.Utf8),
            pl.col("Chemical_Physical_State").cast(pl.Utf8),
            pl.col("Original Sheet").cast(pl.Utf8),
        )
        # Select only key columns
        select_cols: list[str] = [
            "Chemical_Name",
            "CAS",
            "Chemical_Physical_State",
            "Original Sheet",
        ]
        selected_df = df.select(select_cols)
        normalized_dfs.append(selected_df)

    combined = pl.concat(normalized_dfs, how="vertical").unique(subset=["Chemical_Name", "CAS"])
    logger.info("Loaded 2021 Inventory (combined): %d unique entries", len(combined))
    return combined


def _read_2025_inventory(inv_path: Path) -> pl.DataFrame:
    """Read 2025 inventory.

    Args:
        inv_path: Path to 2025 inventory Excel file

    Returns:
        DataFrame with 2025 inventory

    Raises:
        FileNotFoundError: If file does not exist
    """
    return pl.read_excel(source=inv_path, engine="openpyxl")


def _normalize_name_column(df: pl.DataFrame, col_name: str, new_col_name: str) -> pl.DataFrame:
    """Add normalized name column to DataFrame.

    Args:
        df: Source DataFrame
        col_name: Name of column to normalize
        new_col_name: Name for normalized column

    Returns:
        DataFrame with added normalized column
    """
    if col_name not in df.columns:
        return df
    return df.with_columns(
        pl.col(col_name).cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias(new_col_name)
    )


def _write_ghost_excel(ghost_df: pl.DataFrame, output_path: Path) -> None:
    """Write ghost inventory to formatted Excel file.

    Args:
        ghost_df: DataFrame with ghost inventory
        output_path: Output file path

    Raises:
        PermissionError: If file is open or cannot be written
    """
    # Add source column
    ghost_df = ghost_df.with_columns(pl.lit("UCI Chemical Inventory (2021)").alias("Source (2021)"))

    wb = _create_workbook()
    ws = wb.active
    ws.title = "Ghost Inventory Report"

    # Headers
    excel_headers = [
        "Chemical Name (2021)",
        "CAS (2021)",
        "Chemical Physical State (2021)",
        "Original Sheet (2021)",
        "Source (2021)",
    ]

    for col_idx, header in enumerate(excel_headers, 1):
        ws.cell(row=1, column=col_idx, value=header)

    # Write Data via JSON serialization to avoid Any types
    last_row = 1
    rows = _df_json_to_row_dicts(ghost_df.write_json())
    for row_idx, row_data in enumerate(rows, 2):
        last_row = row_idx
        chem_name = _get_json_str_value(row_data, "Chemical_Name")
        ws.cell(row=row_idx, column=1, value=chem_name if chem_name else "")
        cas_val = _get_json_str_value(row_data, "CAS")
        ws.cell(row=row_idx, column=2, value=cas_val if cas_val else "")
        phys_state = _get_json_str_value(row_data, "Chemical_Physical_State")
        ws.cell(row=row_idx, column=3, value=phys_state if phys_state else "")
        orig_sheet = _get_json_str_value(row_data, "Original Sheet")
        ws.cell(row=row_idx, column=4, value=orig_sheet if orig_sheet else "")
        source_val = _get_json_str_value(row_data, "Source (2021)")
        ws.cell(row=row_idx, column=5, value=source_val if source_val else "")

    # Create Excel Table
    if last_row > 1:
        last_col_letter = "E"
        ref = f"A1:{last_col_letter}{last_row}"
        tab = _create_table(
            display_name="GhostInventory",
            ref=ref,
            style_name="TableStyleMedium9",
            show_row_stripes=True,
        )
        ws.add_table(tab)

    # Auto-adjust column widths
    _auto_adjust_column_widths(ws, max_width=60, padding=2)

    wb.save(output_path)
    logger.info("Formatted Ghost Inventory Report saved to: %s", output_path)


DEFAULT_BASE_PATH = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab")


def create_formatted_ghost_excel(
    inv_2021_path: Path | None = None,
    inv_2025_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    """Create the ghost inventory report.

    Args:
        inv_2021_path: Path to 2021 inventory Excel file (uses default if None)
        inv_2025_path: Path to 2025 inventory Excel file (uses default if None)
        output_path: Path to output Excel file (uses default if None)

    Returns:
        Exit code (0 for success)
    """
    if inv_2021_path is None:
        inv_2021_path = (
            DEFAULT_BASE_PATH / "Important Docs/Chemical Inventory/02252021-Chemical Inventory.xlsx"
        )
    if inv_2025_path is None:
        inv_2025_path = (
            DEFAULT_BASE_PATH / "Notebooks/Emily Truong Notebook/Chemical_Inventory_List_2025.xlsx"
        )
    if output_path is None:
        output_path = (
            DEFAULT_BASE_PATH
            / "Notebooks/Emily Truong Notebook/Chemical_Ghost_Inventory_Report_2025.xlsx"
        )

    logger.info("--- Generating Formatted Ghost Inventory Report ---")

    # Load inventories
    df_2021 = _read_2021_inventory(inv_2021_path)
    df_2025 = _read_2025_inventory(inv_2025_path)

    # Normalize names
    df_2021 = _normalize_name_column(df_2021, "Chemical_Name", "Chemical_Name_normalized")
    df_2025 = _normalize_name_column(df_2025, "Chemical Name", "Chemical Name_normalized")

    # Filter valid entries
    df_2021_valid = df_2021.filter(
        pl.col("Chemical_Name_normalized").is_not_null()
        & (pl.col("Chemical_Name_normalized") != "")
    )
    df_2025_valid = df_2025.filter(
        pl.col("Chemical Name_normalized").is_not_null()
        & (pl.col("Chemical Name_normalized") != "")
    )

    names_2021: set[str] = set(
        _json_col_to_str_list(
            df_2021_valid.select("Chemical_Name_normalized").write_json(),
            "Chemical_Name_normalized",
        )
    )
    names_2025: set[str] = set(
        _json_col_to_str_list(
            df_2025_valid.select("Chemical Name_normalized").write_json(),
            "Chemical Name_normalized",
        )
    )

    # Find ghost inventory (in 2021 but NOT in 2025)
    ghost_chemicals_normalized = names_2021 - names_2025

    if not ghost_chemicals_normalized:
        logger.info("Good news! No 'ghost' chemicals found from 2021 inventory.")
        return 0

    logger.info("Found %d ghost chemicals", len(ghost_chemicals_normalized))

    ghost_list: list[str] = sorted(ghost_chemicals_normalized)
    ghost_data_df = df_2021_valid.filter(pl.col("Chemical_Name_normalized").is_in(ghost_list))

    _write_ghost_excel(ghost_data_df, output_path)
    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="create-ghost-excel",
        instance_id=None,
        extra_fields=None,
    )
    return create_formatted_ghost_excel()


if __name__ == "__main__":
    raise SystemExit(main())
