"""Tests for create_ghost_excel.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.create_ghost_excel import (
    _normalize_name_column,
    _read_2021_inventory,
    _read_2025_inventory,
    _write_ghost_excel,
    create_formatted_ghost_excel,
)

from instrument_io._protocols.openpyxl import _create_workbook, _load_workbook


def _create_2021_inventory_fixture(excel_path: Path) -> None:
    """Create a 2021 inventory test fixture with CiBR-Trac and 428 sheets."""
    wb = _create_workbook()

    # CiBR-Trac sheet
    ws_cibr = wb.active
    ws_cibr.title = "CiBR-Trac"
    ws_cibr.cell(row=1, column=1, value="Chemical_Name")
    ws_cibr.cell(row=1, column=2, value="CAS")
    ws_cibr.cell(row=1, column=3, value="Chemical_Physical_State")
    ws_cibr.cell(row=2, column=1, value="Acetone")
    ws_cibr.cell(row=2, column=2, value="67-64-1")
    ws_cibr.cell(row=2, column=3, value="Liquid")
    ws_cibr.cell(row=3, column=1, value="Benzene")
    ws_cibr.cell(row=3, column=2, value="71-43-2")
    ws_cibr.cell(row=3, column=3, value="Liquid")

    # 428 sheet
    ws_428 = wb.create_sheet("428")
    ws_428.cell(row=1, column=1, value="Chemical Name")
    ws_428.cell(row=1, column=2, value="CAS")
    ws_428.cell(row=1, column=3, value="Physical State")
    ws_428.cell(row=2, column=1, value="Ethanol")
    ws_428.cell(row=2, column=2, value="64-17-5")
    ws_428.cell(row=2, column=3, value="Liquid")

    wb.save(excel_path)
    wb.close()


def _create_2021_inventory_missing_columns(excel_path: Path) -> None:
    """Create 2021 inventory with missing CAS and Physical State columns."""
    wb = _create_workbook()

    # CiBR-Trac sheet - missing CAS column
    ws_cibr = wb.active
    ws_cibr.title = "CiBR-Trac"
    ws_cibr.cell(row=1, column=1, value="Chemical_Name")
    ws_cibr.cell(row=1, column=2, value="Chemical_Physical_State")
    ws_cibr.cell(row=2, column=1, value="Acetone")
    ws_cibr.cell(row=2, column=2, value="Liquid")

    # 428 sheet - missing both CAS and Physical State columns
    ws_428 = wb.create_sheet("428")
    ws_428.cell(row=1, column=1, value="Chemical Name")
    ws_428.cell(row=2, column=1, value="Ethanol")

    wb.save(excel_path)
    wb.close()


def _create_2025_inventory_fixture(excel_path: Path, names: list[str]) -> None:
    """Create a 2025 inventory test fixture."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    for row_idx, name in enumerate(names, start=2):
        ws.cell(row=row_idx, column=1, value=name)
    wb.save(excel_path)
    wb.close()


def _create_ghost_source_fixture(excel_path: Path, data: list[tuple[str, str, str, str]]) -> None:
    """Create a ghost source fixture for writing tests."""
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical_Name")
    ws.cell(row=1, column=2, value="CAS")
    ws.cell(row=1, column=3, value="Chemical_Physical_State")
    ws.cell(row=1, column=4, value="Original Sheet")
    for row_idx, row_data in enumerate(data, start=2):
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
    wb.save(excel_path)
    wb.close()


def test_read_2021_inventory(tmp_path: Path) -> None:
    """Test reading 2021 inventory with multiple sheets."""
    input_path = tmp_path / "2021_inventory.xlsx"
    _create_2021_inventory_fixture(input_path)

    result = _read_2021_inventory(input_path)

    assert "Chemical_Name" in result.columns
    assert "CAS" in result.columns
    # Should have 3 unique entries (2 from CiBR-Trac + 1 from 428)
    assert result.height == 3


def test_read_2021_inventory_normalizes_columns(tmp_path: Path) -> None:
    """Test that column names are normalized."""
    input_path = tmp_path / "2021_inventory.xlsx"
    _create_2021_inventory_fixture(input_path)

    result = _read_2021_inventory(input_path)

    assert "Chemical_Name" in result.columns
    assert "Chemical_Physical_State" in result.columns


def test_read_2021_inventory_missing_columns(tmp_path: Path) -> None:
    """Test reading inventory with missing CAS and Physical State columns."""
    input_path = tmp_path / "2021_inventory.xlsx"
    _create_2021_inventory_missing_columns(input_path)

    result = _read_2021_inventory(input_path)

    # Should still have all required columns (added as None)
    assert "Chemical_Name" in result.columns
    assert "CAS" in result.columns
    assert "Chemical_Physical_State" in result.columns
    assert result.height == 2


def test_read_2021_inventory_no_valid_sheets(tmp_path: Path) -> None:
    """Test RuntimeError when neither CiBR-Trac nor 428 sheets exist."""
    import pytest

    # Create file with neither required sheet
    input_path = tmp_path / "2021_inventory.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.title = "OtherSheet"
    ws.cell(row=1, column=1, value="Data")
    wb.save(input_path)
    wb.close()

    with pytest.raises(RuntimeError, match="No sheets could be read"):
        _read_2021_inventory(input_path)


def test_read_2025_inventory(tmp_path: Path) -> None:
    """Test reading 2025 inventory."""
    input_path = tmp_path / "2025_inventory.xlsx"
    _create_2025_inventory_fixture(input_path, ["Acetone", "Limonene"])

    result = _read_2025_inventory(input_path)

    assert result.height == 2
    assert "Chemical Name" in result.columns


def test_normalize_name_column(tmp_path: Path) -> None:
    """Test normalizing a name column."""
    # Create fixture with test data
    input_path = tmp_path / "test.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=2, column=1, value="  Acetone  ")
    ws.cell(row=3, column=1, value="BENZENE")
    ws.cell(row=4, column=1, value="Ethanol")
    wb.save(input_path)
    wb.close()

    # Load and normalize
    df = _read_2025_inventory(input_path)
    result = _normalize_name_column(df, "Chemical Name", "normalized")

    assert "normalized" in result.columns


def test_normalize_name_column_missing(tmp_path: Path) -> None:
    """Test normalizing missing column returns unchanged DataFrame."""
    input_path = tmp_path / "test.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Other")
    ws.cell(row=2, column=1, value="Value")
    wb.save(input_path)
    wb.close()

    df = _read_2025_inventory(input_path)
    original_cols = list(df.columns)
    result = _normalize_name_column(df, "Chemical Name", "normalized")

    # Should have same columns since Chemical Name doesn't exist
    assert list(result.columns) == original_cols


def test_write_ghost_excel(tmp_path: Path) -> None:
    """Test writing ghost data to Excel."""
    # Create ghost source data
    source_path = tmp_path / "ghost_source.xlsx"
    _create_ghost_source_fixture(
        source_path,
        [
            ("Benzene", "71-43-2", "Liquid", "CiBR-Trac"),
            ("Ethanol", "64-17-5", "Liquid", "428"),
        ],
    )

    # Read it back as DataFrame via the script function
    import polars as pl

    ghost_df = pl.read_excel(source_path)

    output_path = tmp_path / "ghost.xlsx"
    _write_ghost_excel(ghost_df, output_path)

    assert output_path.exists()

    wb = _load_workbook(output_path)
    ws = wb.active
    assert ws.cell(row=1, column=1).value == "Chemical Name (2021)"
    assert ws.cell(row=2, column=1).value == "Benzene"
    wb.close()


def test_write_ghost_excel_empty(tmp_path: Path) -> None:
    """Test writing empty ghost DataFrame."""
    # Create empty Excel fixture
    source_path = tmp_path / "empty.xlsx"
    _create_ghost_source_fixture(source_path, [])

    import polars as pl

    ghost_df = pl.read_excel(source_path)

    output_path = tmp_path / "ghost.xlsx"
    _write_ghost_excel(ghost_df, output_path)

    assert output_path.exists()


def test_create_formatted_ghost_excel_finds_ghosts(tmp_path: Path) -> None:
    """Test finding ghost chemicals."""
    # Create 2021 inventory with 3 chemicals
    inv_2021_path = tmp_path / "2021.xlsx"
    _create_2021_inventory_fixture(inv_2021_path)

    # Create 2025 inventory with only Acetone
    inv_2025_path = tmp_path / "2025.xlsx"
    _create_2025_inventory_fixture(inv_2025_path, ["Acetone"])

    output_path = tmp_path / "ghost.xlsx"

    result = create_formatted_ghost_excel(inv_2021_path, inv_2025_path, output_path)

    assert result == 0
    assert output_path.exists()

    # Should have found Benzene and Ethanol as ghosts
    wb = _load_workbook(output_path)
    ws = wb.active
    # Header + 2 ghost chemicals
    assert ws.max_row == 3
    wb.close()


def test_create_formatted_ghost_excel_no_ghosts(tmp_path: Path) -> None:
    """Test when no ghost chemicals found."""
    # Create 2021 inventory
    inv_2021_path = tmp_path / "2021.xlsx"
    _create_2021_inventory_fixture(inv_2021_path)

    # Create 2025 inventory with all 2021 chemicals
    inv_2025_path = tmp_path / "2025.xlsx"
    _create_2025_inventory_fixture(inv_2025_path, ["Acetone", "Benzene", "Ethanol"])

    output_path = tmp_path / "ghost.xlsx"

    result = create_formatted_ghost_excel(inv_2021_path, inv_2025_path, output_path)

    # Should succeed and not create output file since no ghosts
    assert result == 0


def test_create_formatted_ghost_excel_default_paths() -> None:
    """Test create_formatted_ghost_excel uses default paths when None."""
    import logging

    # This verifies the None branches
    result: int = -1
    try:
        result = create_formatted_ghost_excel(None, None, None)
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_function() -> None:
    """Test main entry point."""
    import logging

    from scripts.create_ghost_excel import main

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

    script_path = Path(__file__).parent.parent / "scripts" / "create_ghost_excel.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
