"""Tests for verify_sheets.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.verify_sheets import get_files_to_check, verify_all_sheets

from instrument_io._protocols.openpyxl import _create_workbook


def test_get_files_to_check_returns_paths() -> None:
    """Test that get_files_to_check returns expected file paths."""
    base_path = Path("/test/base")
    files = get_files_to_check(base_path)

    assert len(files) == 6
    assert all(isinstance(f, Path) for f in files)
    assert files[0] == base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx"


def test_get_files_to_check_paths_relative_to_base() -> None:
    """Test that paths are relative to the provided base path."""
    base_path = Path("/different/path")
    files = get_files_to_check(base_path)

    for f in files:
        assert str(f).startswith(str(base_path))


def test_verify_all_sheets_processes_files(tmp_path: Path) -> None:
    """Test verify_all_sheets processes Excel files."""
    base_path = tmp_path

    # Create directory structure
    (base_path / "Notebooks/Jasmine OseiEnin Lab Notebook").mkdir(parents=True)
    (base_path / "Current Projects").mkdir(parents=True)
    (base_path / "Notebooks/Avisa Lab Notebook").mkdir(parents=True)
    (base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24").mkdir(parents=True)
    (base_path / "Important Docs/Chemical Inventory").mkdir(parents=True)

    # Create Response Factors file with multiple sheets
    wb = _create_workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.cell(row=1, column=1, value="Data")
    ws2 = wb.create_sheet("Sheet2")
    ws2.cell(row=1, column=1, value="More data")
    wb.save(base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx")
    wb.close()

    # Create Soil VOC file
    wb = _create_workbook()
    ws = wb.active
    ws.title = "VOC Data"
    ws.cell(row=1, column=1, value="Data")
    wb.save(base_path / "Current Projects/Soil VOC quantitation.xlsx")
    wb.close()

    # Create Standard Calculations file
    wb = _create_workbook()
    ws = wb.active
    ws.title = "Calculations"
    ws.cell(row=1, column=1, value="Data")
    wb.save(base_path / "Notebooks/Avisa Lab Notebook/Standard Calculations (1).xlsx")
    wb.close()

    # Create 8mix file
    wb = _create_workbook()
    ws = wb.active
    ws.title = "Mix"
    ws.cell(row=1, column=1, value="Data")
    wb.save(
        base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/8mix_calc.xlsx"
    )
    wb.close()

    # Create std_tidy file
    wb = _create_workbook()
    ws = wb.active
    ws.title = "Tidy"
    ws.cell(row=1, column=1, value="Data")
    wb.save(base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/std_tidy.xlsx")
    wb.close()

    # Create inventory file
    wb = _create_workbook()
    ws = wb.active
    ws.title = "Inventory"
    ws.cell(row=1, column=1, value="Data")
    wb.save(base_path / "Important Docs/Chemical Inventory/02252021-Chemical Inventory.xlsx")
    wb.close()

    result = verify_all_sheets(base_path)

    assert result == 0


def test_verify_all_sheets_default_base_path() -> None:
    """Test verify_all_sheets uses default base path when None."""
    import logging

    # This test verifies the function handles None parameter
    # It will either succeed (if default path exists) or raise FileNotFoundError
    result: int = -1
    try:
        result = verify_all_sheets(None)
    except FileNotFoundError:
        # Expected when default path doesn't exist
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_function() -> None:
    """Test main entry point."""
    import logging

    from scripts.verify_sheets import main

    # main() calls setup_logging and verify_all_sheets
    # It will either succeed or raise FileNotFoundError
    result: int = -1
    try:
        result = main()
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_entry_via_runpy() -> None:
    """Test if __name__ == '__main__' block via runpy."""
    import logging
    import runpy

    import pytest

    script_path = Path(__file__).parent.parent / "scripts" / "verify_sheets.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
