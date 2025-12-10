"""Scan for potentially unprocessed chemical data files.

This script identifies Excel files that may contain chemical data but haven't
been explicitly processed by the other scripts.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_BASE_PATH = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab")


def _get_processed_paths(base_path: Path) -> set[str]:
    """Get set of paths that have already been processed.

    Args:
        base_path: Base path for the lab folder

    Returns:
        Set of lowercase normalized paths
    """
    processed_files = {
        # Standards (extract_standards.py)
        "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx",
        "Current Projects/Soil VOC quantitation.xlsx",
        "Notebooks/Avisa Lab Notebook/Standard Calculations (1).xlsx",
        "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/8mix_calc.xlsx",
        "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/std_tidy.xlsx",
        # Inventories
        "Important Docs/Chemical Inventory/02252021-Chemical Inventory.xlsx",
        "Notebooks/Emily Truong Notebook/Chem_Inv.xlsx",
        # Generated Outputs
        "Notebooks/Emily Truong Notebook/Chemical_Inventory_List_2025.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Inventory_List_2021.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Ghost_Inventory_Report_2025.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Standards_List_2025.xlsx",
        "Notebooks/Emily Truong Notebook/Chemical_Standards_Audit_2025.xlsx",
    }

    processed_paths: set[str] = set()
    for p in processed_files:
        full = (base_path / p).resolve()
        processed_paths.add(str(full).lower())

    return processed_paths


def scan_for_missing_data(base_path: Path | None = None) -> int:
    """Scan for potentially unprocessed chemical data files.

    Args:
        base_path: Base path for lab folder (uses default if None)

    Returns:
        Exit code (0 for success)
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    processed_paths = _get_processed_paths(base_path)

    # Find all Excel files
    logger.info("Scanning directory: %s", base_path)
    all_excel = list(base_path.rglob("*.xlsx")) + list(base_path.rglob("*.xls"))

    # Keywords that suggest a file might contain chemical data
    keywords = [
        "inventory",
        "chem",
        "std",
        "standard",
        "stock",
        "mix",
        "analyte",
        "compound",
        "calibration",
    ]

    candidates: list[Path] = []

    for f in all_excel:
        # Skip temporary files (~)
        if f.name.startswith("~"):
            continue

        f_str = str(f.resolve()).lower()

        # If it's already processed, skip
        if f_str in processed_paths:
            continue

        # Check for keywords in filename
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in keywords):
            candidates.append(f)

    # Report
    if not candidates:
        logger.info("No new potential inventory/standard files found!")
        return 0

    logger.info("Potential Unprocessed Data Files (%d)", len(candidates))
    for c in candidates:
        rel_path = c.relative_to(base_path) if c.is_relative_to(base_path) else c
        logger.info("  %s -> %s", c.name, rel_path)

    logger.warning(
        "Review this list. These files have names matching keywords like "
        "'inventory' or 'standard' but haven't been explicitly processed yet."
    )

    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="scan-missing-data",
        instance_id=None,
        extra_fields=None,
    )
    return scan_for_missing_data()


if __name__ == "__main__":
    raise SystemExit(main())
