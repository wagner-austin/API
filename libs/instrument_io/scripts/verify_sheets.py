"""Verify Excel sheets in source files.

This script scans source Excel files and reports all sheets found,
helping identify potentially hidden or missed sheets.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger, setup_logging

from instrument_io._protocols.openpyxl import _load_workbook

logger = get_logger(__name__)

DEFAULT_BASE_PATH = Path(r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab")


def get_files_to_check(base_path: Path) -> list[Path]:
    """Get list of files to check for sheets.

    Args:
        base_path: Base path for lab folder

    Returns:
        List of file paths to check
    """
    return [
        base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx",
        base_path / "Current Projects/Soil VOC quantitation.xlsx",
        base_path / "Notebooks/Avisa Lab Notebook/Standard Calculations (1).xlsx",
        base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/8mix_calc.xlsx",
        base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/std_tidy.xlsx",
        base_path / "Important Docs/Chemical Inventory/02252021-Chemical Inventory.xlsx",
    ]


def verify_all_sheets(base_path: Path | None = None) -> int:
    """Verify sheets in all source Excel files.

    Args:
        base_path: Base path for lab folder (uses default if None)

    Returns:
        Exit code (0 for success)
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    files_to_check = get_files_to_check(base_path)

    logger.info("=== Sheet Verification Audit ===")
    logger.info("Scanning files for hidden sheets...")

    for file_path in files_to_check:
        wb = _load_workbook(file_path, read_only=True)
        sheet_names = wb.sheetnames
        wb.close()

        logger.info(
            "%s: %d sheets -> %s",
            file_path.name,
            len(sheet_names),
            ", ".join(sheet_names),
        )

    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="verify-sheets",
        instance_id=None,
        extra_fields=None,
    )
    return verify_all_sheets()


if __name__ == "__main__":
    raise SystemExit(main())
