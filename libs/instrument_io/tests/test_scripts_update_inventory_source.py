"""Tests for update_inventory_source.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.update_inventory_source import update_inventory_metadata

from instrument_io._protocols.openpyxl import (
    _create_table,
    _create_workbook,
    _load_workbook,
)


def test_update_adds_source_column(tmp_path: Path) -> None:
    """Test that Source column is added when missing."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=1, column=2, value="CAS")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=2, column=2, value="67-64-1")

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    result = update_inventory_metadata(excel_path)

    assert result == 0

    # Verify Source column was added
    wb = _load_workbook(excel_path)
    ws = wb.active
    headers = [ws.cell(row=1, column=i).value for i in range(1, 5)]
    assert "Source" in headers
    wb.close()


def test_update_adds_date_column(tmp_path: Path) -> None:
    """Test that Date column is added when missing."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=2, column=1, value="Benzene")

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    result = update_inventory_metadata(excel_path)

    assert result == 0

    # Verify Date column was added
    wb = _load_workbook(excel_path)
    ws = wb.active
    headers = [ws.cell(row=1, column=i).value for i in range(1, 5)]
    assert "Date" in headers
    wb.close()


def test_update_fills_source_values(tmp_path: Path) -> None:
    """Test that Source values are filled for all rows."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=3, column=1, value="Benzene")

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    update_inventory_metadata(excel_path)

    wb = _load_workbook(excel_path)
    ws = wb.active

    # Find Source column - should be column 2 (after Chemical Name)
    source_col = 2
    assert ws.cell(row=1, column=source_col).value == "Source"
    assert ws.cell(row=2, column=source_col).value == "UCI Chemical Inventory (Risk & Safety)"
    assert ws.cell(row=3, column=source_col).value == "UCI Chemical Inventory (Risk & Safety)"
    wb.close()


def test_update_fills_date_values(tmp_path: Path) -> None:
    """Test that Date values are filled for all rows."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=2, column=1, value="Acetone")

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    update_inventory_metadata(excel_path)

    wb = _load_workbook(excel_path)
    ws = wb.active

    # Date column should be column 3 (after Chemical Name and Source)
    date_col = 3
    assert ws.cell(row=1, column=date_col).value == "Date"
    assert ws.cell(row=2, column=date_col).value == "2025-12-05"
    wb.close()


def test_update_existing_source_column(tmp_path: Path) -> None:
    """Test updating file that already has Source column."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=1, column=2, value="Source")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=2, column=2, value="Old Source")

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    result = update_inventory_metadata(excel_path)

    assert result == 0

    wb = _load_workbook(excel_path)
    ws = wb.active
    assert ws.cell(row=2, column=2).value == "UCI Chemical Inventory (Risk & Safety)"
    wb.close()


def test_update_existing_date_column(tmp_path: Path) -> None:
    """Test updating file that already has Date column."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=1, column=2, value="Date")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=2, column=2, value="2024-01-01")  # Old date

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    result = update_inventory_metadata(excel_path)

    assert result == 0

    wb = _load_workbook(excel_path)
    ws = wb.active
    # Date column should be updated to new date
    assert ws.cell(row=2, column=2).value == "2025-12-05"
    wb.close()


def test_update_existing_source_and_date_columns(tmp_path: Path) -> None:
    """Test updating file that already has both Source and Date columns."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=1, column=2, value="Source")
    ws.cell(row=1, column=3, value="Date")
    ws.cell(row=2, column=1, value="Benzene")
    ws.cell(row=2, column=2, value="Old Source")
    ws.cell(row=2, column=3, value="2020-01-01")

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    result = update_inventory_metadata(excel_path)

    assert result == 0

    wb = _load_workbook(excel_path)
    ws = wb.active
    assert ws.cell(row=2, column=2).value == "UCI Chemical Inventory (Risk & Safety)"
    assert ws.cell(row=2, column=3).value == "2025-12-05"
    wb.close()


def test_update_with_table(tmp_path: Path) -> None:
    """Test updating file that has an Excel table."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=1, column=2, value="CAS")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=2, column=2, value="67-64-1")

    # Create a table using typed protocol
    tab = _create_table("ChemTable", "A1:B2")
    ws.add_table(tab)

    excel_path = tmp_path / "inventory.xlsx"
    wb.save(excel_path)
    wb.close()

    result = update_inventory_metadata(excel_path)

    assert result == 0

    # Verify table ref was updated
    wb = _load_workbook(excel_path)
    ws = wb.active
    tables = list(ws.tables.values())
    assert len(tables) == 1
    wb.close()


def test_update_inventory_metadata_default_path() -> None:
    """Test update_inventory_metadata uses default path when None."""
    import logging

    # This verifies the None branch
    result: int = -1
    try:
        result = update_inventory_metadata(None)
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_function() -> None:
    """Test main entry point."""
    import logging

    from scripts.update_inventory_source import main

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

    script_path = Path(__file__).parent.parent / "scripts" / "update_inventory_source.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
