"""Compare chemical inventory against standards list.

This script cross-references the 2025 chemical inventory with the standards list
to identify gaps and opportunities.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import polars as pl
from platform_core.logging import get_logger, setup_logging

from instrument_io._json_bridge import _json_col_to_opt_str_list

logger = get_logger(__name__)


class InventoryRow(TypedDict):
    """Row from inventory DataFrame."""

    chemical_name: str
    cas: str


class StandardRow(TypedDict):
    """Row from standards DataFrame."""

    chemical_name: str
    source: str


def _normalize(name: str | None) -> str:
    """Normalize chemical name for comparison."""
    if not name or not isinstance(name, str):
        return ""
    return name.strip().lower()


def _load_inventory(path: Path) -> pl.DataFrame:
    """Load inventory Excel file.

    Args:
        path: Path to inventory Excel file

    Returns:
        Polars DataFrame with inventory data

    Raises:
        FileNotFoundError: If file does not exist
        pl.exceptions.ComputeError: If file cannot be parsed
    """
    return pl.read_excel(source=path, engine="openpyxl")


def _load_standards(path: Path) -> pl.DataFrame:
    """Load standards Excel file.

    Args:
        path: Path to standards Excel file

    Returns:
        Polars DataFrame with standards data

    Raises:
        FileNotFoundError: If file does not exist
        pl.exceptions.ComputeError: If file cannot be parsed
    """
    return pl.read_excel(source=path, engine="openpyxl")


def _extract_chemical_names(
    df: pl.DataFrame,
    column_name: str,
) -> dict[str, str]:
    """Extract chemical names from DataFrame.

    Args:
        df: Source DataFrame
        column_name: Name of column containing chemical names

    Returns:
        Dict mapping normalized names to original names
    """
    if column_name not in df.columns:
        return {}
    raw_names = _json_col_to_opt_str_list(df.select(column_name).write_json(), column_name)
    return {_normalize(name): name for name in raw_names if name}


DEFAULT_BASE_PATH = Path(
    r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab"
    r"\Notebooks\Emily Truong Notebook"
)


def compare_inventory_and_standards(
    inventory_path: Path | None = None,
    standards_path: Path | None = None,
) -> int:
    """Compare inventory against standards and report findings.

    Args:
        inventory_path: Path to inventory Excel file (uses default if None)
        standards_path: Path to standards Excel file (uses default if None)

    Returns:
        Exit code (0 for success)
    """
    if inventory_path is None:
        inventory_path = DEFAULT_BASE_PATH / "Chemical_Inventory_List_2025.xlsx"
    if standards_path is None:
        standards_path = DEFAULT_BASE_PATH / "Chemical_Standards_List_2025.xlsx"

    # Load inventory
    df_inventory = _load_inventory(inventory_path)
    logger.info("Loaded inventory from: %s", inventory_path)

    # Load standards
    df_standards = _load_standards(standards_path)
    logger.info("Loaded standards from: %s", standards_path)

    # Extract and normalize chemical names
    inv_map = _extract_chemical_names(df_inventory, "Chemical Name")
    std_map = _extract_chemical_names(df_standards, "Chemical Name")

    inv_names = set(inv_map.keys())
    std_names = set(std_map.keys())

    # Analysis
    common = inv_names.intersection(std_names)
    missing_in_inventory = std_names - inv_names
    no_standard = inv_names - std_names

    logger.info("Cross-Reference Report")
    logger.info("Total Inventory Items: %d", len(inv_names))
    logger.info("Total Standards: %d", len(std_names))
    logger.info("Matches found: %d", len(common))

    # Report: Standards missing from inventory (Critical)
    if missing_in_inventory:
        logger.warning("CRITICAL: %d standards missing from inventory", len(missing_in_inventory))
        for name_norm in sorted(missing_in_inventory):
            original_name = std_map.get(name_norm, name_norm)
            logger.warning("  Missing: %s", original_name)
    else:
        logger.info("All standards have corresponding inventory entries")

    # Report: Inventory items without standards (Top 15)
    if no_standard:
        logger.info("Inventory items without standards: %d total", len(no_standard))
        for i, name_norm in enumerate(sorted(no_standard)):
            if i >= 15:
                logger.info("  ... and %d more", len(no_standard) - 15)
                break
            original_name = inv_map.get(name_norm, name_norm)
            logger.info("  No standard: %s", original_name)

    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="compare-inventory-standards",
        instance_id=None,
        extra_fields=None,
    )
    return compare_inventory_and_standards()


if __name__ == "__main__":
    raise SystemExit(main())
