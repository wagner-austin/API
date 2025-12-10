"""Create chemical standards audit Excel report.

This script compares inventory and standards lists to produce an audit report
with status indicators for each chemical.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import polars as pl
from platform_core.logging import get_logger, setup_logging

from instrument_io._json_bridge import _df_json_to_row_dicts, _get_json_str_value
from instrument_io._protocols.openpyxl import (
    _auto_adjust_column_widths,
    _create_font,
    _create_pattern_fill,
    _create_table,
    _create_workbook,
)

logger = get_logger(__name__)


class InventoryRowDict(TypedDict, total=False):
    """Row from inventory DataFrame."""

    chemical_name: str
    cas: str
    physical_state: str
    date: str


class StandardsRowDict(TypedDict, total=False):
    """Row from standards DataFrame."""

    chemical_name: str
    source: str
    date: str
    details: str


class AuditEntry(TypedDict):
    """Entry in the audit report."""

    chemical_name: str | None
    status: str
    cas_inv: str | None
    physical_state: str | None
    source: str
    date: str
    details_std: str | None


def _normalize(name: str | None) -> str:
    """Normalize chemical name for comparison."""
    if not name or not isinstance(name, str):
        return ""
    return name.strip().lower()


def _load_data(
    inventory_path: Path,
    standards_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load inventory and standards Excel files.

    Args:
        inventory_path: Path to inventory Excel file
        standards_path: Path to standards Excel file

    Returns:
        Tuple of (inventory_df, standards_df)

    Raises:
        FileNotFoundError: If files do not exist
        pl.exceptions.ComputeError: If files cannot be parsed
    """
    df_inventory = pl.read_excel(source=inventory_path, engine="openpyxl")
    df_standards = pl.read_excel(source=standards_path, engine="openpyxl")
    return df_inventory, df_standards


def _build_lookup_maps(
    df_inventory: pl.DataFrame,
    df_standards: pl.DataFrame,
) -> tuple[dict[str, dict[str, str | None]], dict[str, dict[str, str | None]]]:
    """Build lookup maps from DataFrames.

    Args:
        df_inventory: Inventory DataFrame
        df_standards: Standards DataFrame

    Returns:
        Tuple of (inv_map, std_map) dictionaries
    """
    inv_map: dict[str, dict[str, str | None]] = {}
    for row in _df_json_to_row_dicts(df_inventory.write_json()):
        chem_name = _get_json_str_value(row, "Chemical Name")
        if chem_name:
            norm_name = _normalize(chem_name)
            inv_map[norm_name] = {
                "Chemical Name": chem_name,
                "CAS": _get_json_str_value(row, "CAS"),
                "Physical State": _get_json_str_value(row, "Physical State"),
                "Date": _get_json_str_value(row, "Date"),
            }

    std_map: dict[str, dict[str, str | None]] = {}
    for row in _df_json_to_row_dicts(df_standards.write_json()):
        chem_name = _get_json_str_value(row, "Chemical Name")
        if chem_name:
            norm_name = _normalize(chem_name)
            std_map[norm_name] = {
                "Chemical Name": chem_name,
                "Source": _get_json_str_value(row, "Source"),
                "Date": _get_json_str_value(row, "Date"),
                "Details": _get_json_str_value(row, "Details"),
            }

    return inv_map, std_map


def _build_audit_data(
    inv_map: dict[str, dict[str, str | None]],
    std_map: dict[str, dict[str, str | None]],
) -> list[AuditEntry]:
    """Build audit entries from lookup maps.

    Args:
        inv_map: Inventory lookup map
        std_map: Standards lookup map

    Returns:
        List of audit entries
    """
    all_names = set(inv_map.keys()).union(set(std_map.keys()))
    audit_data: list[AuditEntry] = []

    for name_norm in sorted(all_names):
        inv_data = inv_map.get(name_norm, {})
        std_data = std_map.get(name_norm, {})

        has_inv = bool(inv_data)
        has_std = bool(std_data)

        if has_inv and has_std:
            status = "OK: Complete"
            name_display = inv_data.get("Chemical Name")
        elif has_inv and not has_std:
            status = "Warning: No Standard"
            name_display = inv_data.get("Chemical Name")
        else:
            # not has_inv and has_std (name came from std_map only)
            status = "Critical: Missing Inventory"
            name_display = std_data.get("Chemical Name")

        # Merge Sources
        sources: list[str] = []
        if has_inv:
            sources.append("UCI Chemical Inventory (Risk & Safety)")
        if has_std:
            std_source = std_data.get("Source")
            if std_source:
                sources.append(std_source)
        combined_source = "; ".join(sources)

        # Merge Dates
        dates: set[str] = set()
        if has_inv:
            inv_date = inv_data.get("Date")
            if inv_date:
                dates.add(inv_date)
        if has_std:
            std_date = std_data.get("Date")
            if std_date:
                dates.add(std_date)
        combined_date = "; ".join(sorted(dates))

        entry: AuditEntry = {
            "chemical_name": name_display,
            "status": status,
            "cas_inv": inv_data.get("CAS"),
            "physical_state": inv_data.get("Physical State"),
            "source": combined_source,
            "date": combined_date,
            "details_std": std_data.get("Details"),
        }
        audit_data.append(entry)

    return audit_data


def _write_audit_excel(audit_data: list[AuditEntry], output_path: Path) -> None:
    """Write audit data to formatted Excel file.

    Args:
        audit_data: List of audit entries
        output_path: Output file path

    Raises:
        PermissionError: If file is open or cannot be written
    """
    wb = _create_workbook()
    ws = wb.active
    ws.title = "Audit Report"

    # Headers
    headers = [
        "Chemical Name",
        "Status",
        "CAS (Inv)",
        "Physical State",
        "Source",
        "Date",
        "Details (Std)",
    ]
    for col_idx, header in enumerate(headers, 1):
        ws.cell(row=1, column=col_idx, value=header)

    # Status cell colors
    green_fill = _create_pattern_fill(start_color="C6EFCE")
    green_font = _create_font(color="006100")
    yellow_fill = _create_pattern_fill(start_color="FFEB9C")
    yellow_font = _create_font(color="9C5700")
    red_fill = _create_pattern_fill(start_color="FFC7CE")
    red_font = _create_font(color="9C0006")

    # Write Data
    last_row = 1
    for row_idx, entry in enumerate(audit_data, 2):
        last_row = row_idx
        ws.cell(row=row_idx, column=1, value=entry["chemical_name"])

        status_cell = ws.cell(row=row_idx, column=2, value=entry["status"])
        if "OK" in entry["status"]:
            status_cell.fill = green_fill
            status_cell.font = green_font
        elif "Warning" in entry["status"]:
            status_cell.fill = yellow_fill
            status_cell.font = yellow_font
        else:
            # Critical status
            status_cell.fill = red_fill
            status_cell.font = red_font

        ws.cell(row=row_idx, column=3, value=entry["cas_inv"])
        ws.cell(row=row_idx, column=4, value=entry["physical_state"])
        ws.cell(row=row_idx, column=5, value=entry["source"])
        ws.cell(row=row_idx, column=6, value=entry["date"])
        ws.cell(row=row_idx, column=7, value=entry["details_std"])

    # Create Table
    ref = f"A1:G{last_row}"
    tab = _create_table(
        display_name="AuditTable",
        ref=ref,
        style_name="TableStyleMedium2",
        show_row_stripes=True,
    )
    ws.add_table(tab)

    # Auto-width columns
    _auto_adjust_column_widths(ws, max_width=50, padding=2)

    logger.info("Saving Audit Excel to: %s", output_path)
    wb.save(output_path)
    logger.info("Success! Audit file created.")


DEFAULT_BASE_PATH = Path(
    r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab"
    r"\Notebooks\Emily Truong Notebook"
)


def create_audit_excel(
    inventory_path: Path | None = None,
    standards_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    """Create the audit Excel report.

    Args:
        inventory_path: Path to inventory Excel file (uses default if None)
        standards_path: Path to standards Excel file (uses default if None)
        output_path: Path to output Excel file (uses default if None)

    Returns:
        Exit code (0 for success)
    """
    if inventory_path is None:
        inventory_path = DEFAULT_BASE_PATH / "Chemical_Inventory_List_2025.xlsx"
    if standards_path is None:
        standards_path = DEFAULT_BASE_PATH / "Chemical_Standards_List_2025.xlsx"
    if output_path is None:
        output_path = DEFAULT_BASE_PATH / "Chemical_Standards_Audit_2025.xlsx"

    # Load Data
    df_inventory, df_standards = _load_data(inventory_path, standards_path)

    # Build lookup maps
    inv_map, std_map = _build_lookup_maps(df_inventory, df_standards)

    # Build audit data
    audit_data = _build_audit_data(inv_map, std_map)

    # Write Excel
    _write_audit_excel(audit_data, output_path)

    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="create-audit-excel",
        instance_id=None,
        extra_fields=None,
    )
    return create_audit_excel()


if __name__ == "__main__":
    raise SystemExit(main())
