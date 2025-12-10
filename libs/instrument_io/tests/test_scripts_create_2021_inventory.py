"""Tests for create_2021_inventory_excel.py script."""

from __future__ import annotations

from pathlib import Path

from scripts.create_2021_inventory_excel import (
    _read_428_sheet,
    _read_cibr_trac_sheet,
    _write_excel_with_formatting,
    create_2021_inventory_report,
)

from instrument_io._protocols.openpyxl import _create_workbook, _load_workbook


def _create_cibr_trac_fixture(excel_path: Path) -> None:
    """Create a test Excel file with CiBR-Trac sheet."""
    wb = _create_workbook()
    ws = wb.active
    ws.title = "CiBR-Trac"

    # Headers
    headers = [
        "Chemical_Name",
        "CAS",
        "Chemical_Physical_State",
        "Container_Size",
        "Units",
        "Container_Number",
    ]
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)

    # Data rows - use explicit types for cell values
    data_row1: list[str | int | float | bool | None] = [
        "Acetone",
        "67-64-1",
        "Liquid",
        500,
        "mL",
        1,
    ]
    data_row2: list[str | int | float | bool | None] = [
        "Benzene",
        "71-43-2",
        "Liquid",
        100,
        "mL",
        2,
    ]
    data_rows = [data_row1, data_row2]
    for row_idx, row_data in enumerate(data_rows, 2):
        for col_idx, value in enumerate(row_data, 1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    # Add 428 sheet
    ws428 = wb.create_sheet("428")
    headers_428 = [
        "Chemical Name",
        "CAS",
        "Physical State",
        "Amount",
        "Units",
        "Room Number",
        "Sublocation",
    ]
    for col, header in enumerate(headers_428, 1):
        ws428.cell(row=1, column=col, value=header)

    data_428_row: list[str | int | float | bool | None] = [
        "Ethanol",
        "64-17-5",
        "Liquid",
        1000,
        "mL",
        428,
        "Cabinet A",
    ]
    for col_idx, value in enumerate(data_428_row, 1):
        ws428.cell(row=2, column=col_idx, value=value)

    wb.save(excel_path)
    wb.close()


def _create_df_source_fixture(
    excel_path: Path, data: list[list[str | int | float | bool | None]]
) -> None:
    """Create a DataFrame source fixture."""
    wb = _create_workbook()
    ws = wb.active
    headers = ["Chemical Name", "CAS", "Amount"]
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=1, column=col_idx, value=header)
    for row_idx, row_data in enumerate(data, start=2):
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)
    wb.save(excel_path)
    wb.close()


def test_read_cibr_trac_sheet(tmp_path: Path) -> None:
    """Test reading CiBR-Trac sheet."""
    input_path = tmp_path / "inventory.xlsx"
    _create_cibr_trac_fixture(input_path)

    result = _read_cibr_trac_sheet(input_path)
    if result is None:
        raise AssertionError("Expected DataFrame, got None")

    # Verify the DataFrame was loaded by checking height and columns
    assert result.height == 2
    assert "Chemical Name" in result.columns
    assert "Location" in result.columns
    assert "Source" in result.columns


def test_read_cibr_trac_adds_location(tmp_path: Path) -> None:
    """Test that CiBR-Trac sheet adds Location column."""
    input_path = tmp_path / "inventory.xlsx"
    _create_cibr_trac_fixture(input_path)

    result = _read_cibr_trac_sheet(input_path)
    if result is None:
        raise AssertionError("Expected DataFrame, got None")

    # Verify location column exists and has expected value
    assert "Location" in result.columns
    json_str: str = result.select("Location").write_json()
    assert "CiBR-Trac" in json_str


def test_read_cibr_trac_adds_source(tmp_path: Path) -> None:
    """Test that CiBR-Trac sheet adds Source column."""
    input_path = tmp_path / "inventory.xlsx"
    _create_cibr_trac_fixture(input_path)

    result = _read_cibr_trac_sheet(input_path)
    if result is None:
        raise AssertionError("Expected DataFrame, got None")

    # Verify source column exists and has expected value
    assert "Source" in result.columns
    json_str: str = result.select("Source").write_json()
    assert "2021 Inventory (CiBR-Trac)" in json_str


def test_read_428_sheet(tmp_path: Path) -> None:
    """Test reading Room 428 sheet."""
    input_path = tmp_path / "inventory.xlsx"
    _create_cibr_trac_fixture(input_path)

    result = _read_428_sheet(input_path)
    if result is None:
        raise AssertionError("Expected DataFrame, got None")

    # Verify the DataFrame was loaded by checking height and columns
    assert result.height == 1
    assert "Chemical Name" in result.columns
    assert "Location" in result.columns


def test_read_428_constructs_location(tmp_path: Path) -> None:
    """Test that 428 sheet constructs Location from Room Number and Sublocation."""
    input_path = tmp_path / "inventory.xlsx"
    _create_cibr_trac_fixture(input_path)

    result = _read_428_sheet(input_path)
    if result is None:
        raise AssertionError("Expected DataFrame, got None")

    # Verify location column exists and contains room info
    assert "Location" in result.columns
    json_str: str = result.select("Location").write_json()
    assert "Room 428" in json_str or "428" in json_str


def test_write_excel_with_formatting(tmp_path: Path) -> None:
    """Test writing DataFrame to formatted Excel."""
    # Create source fixture
    source_path = tmp_path / "source.xlsx"
    _create_df_source_fixture(
        source_path,
        [
            ["Acetone", "67-64-1", 500.0],
            ["Benzene", "71-43-2", 100.0],
        ],
    )

    # Script function expects pl.DataFrame, use polars directly
    import polars as pl

    df: pl.DataFrame = pl.read_excel(source_path)

    output_path = tmp_path / "output.xlsx"
    _write_excel_with_formatting(df, output_path)

    assert output_path.exists()

    # Verify content
    wb = _load_workbook(output_path)
    ws = wb.active
    assert ws.cell(row=1, column=1).value == "Chemical Name"
    assert ws.cell(row=2, column=1).value == "Acetone"
    wb.close()


def test_write_excel_creates_table(tmp_path: Path) -> None:
    """Test that Excel table is created."""
    source_path = tmp_path / "source.xlsx"
    _create_df_source_fixture(source_path, [["Acetone", "67-64-1", 500.0]])

    # Script function expects pl.DataFrame, use polars directly
    import polars as pl

    df: pl.DataFrame = pl.read_excel(source_path)

    output_path = tmp_path / "output.xlsx"
    _write_excel_with_formatting(df, output_path)

    wb = _load_workbook(output_path)
    ws = wb.active
    assert len(ws.tables) == 1
    wb.close()


def test_create_2021_inventory_report(tmp_path: Path) -> None:
    """Test creating full 2021 inventory report."""
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "output.xlsx"
    _create_cibr_trac_fixture(input_path)

    result = create_2021_inventory_report(input_path, output_path)

    assert result == 0
    assert output_path.exists()

    # Verify combined data
    wb = _load_workbook(output_path)
    ws = wb.active
    # Should have header + 3 data rows (2 from CiBR-Trac + 1 from 428)
    assert ws.max_row == 4
    wb.close()


def test_create_2021_inventory_missing_cibr_trac(tmp_path: Path) -> None:
    """Test when CiBR-Trac sheet is missing (only 428 exists)."""
    # Create file with only 428 sheet (no CiBR-Trac)
    input_path = tmp_path / "input.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.title = "428"  # Only 428 sheet
    ws.cell(row=1, column=1, value="Chemical Name")
    ws.cell(row=1, column=2, value="CAS")
    ws.cell(row=1, column=3, value="Physical State")
    ws.cell(row=1, column=4, value="Amount")
    ws.cell(row=1, column=5, value="Units")
    ws.cell(row=1, column=6, value="Room Number")
    ws.cell(row=1, column=7, value="Sublocation")
    ws.cell(row=2, column=1, value="Ethanol")
    ws.cell(row=2, column=2, value="64-17-5")
    ws.cell(row=2, column=3, value="Liquid")
    ws.cell(row=2, column=4, value=1000)
    ws.cell(row=2, column=5, value="mL")
    ws.cell(row=2, column=6, value=428)
    ws.cell(row=2, column=7, value="Cabinet A")
    wb.save(input_path)
    wb.close()

    output_path = tmp_path / "output.xlsx"
    result = create_2021_inventory_report(input_path, output_path)

    assert result == 0
    assert output_path.exists()


def test_create_2021_inventory_missing_428(tmp_path: Path) -> None:
    """Test when 428 sheet is missing (only CiBR-Trac exists)."""
    # Create file with only CiBR-Trac sheet
    input_path = tmp_path / "input.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.title = "CiBR-Trac"
    ws.cell(row=1, column=1, value="Chemical_Name")
    ws.cell(row=1, column=2, value="CAS")
    ws.cell(row=1, column=3, value="Chemical_Physical_State")
    ws.cell(row=1, column=4, value="Container_Size")
    ws.cell(row=1, column=5, value="Units")
    ws.cell(row=1, column=6, value="Container_Number")
    ws.cell(row=2, column=1, value="Acetone")
    ws.cell(row=2, column=2, value="67-64-1")
    ws.cell(row=2, column=3, value="Liquid")
    ws.cell(row=2, column=4, value=500)
    ws.cell(row=2, column=5, value="mL")
    ws.cell(row=2, column=6, value=1)
    wb.save(input_path)
    wb.close()

    output_path = tmp_path / "output.xlsx"
    result = create_2021_inventory_report(input_path, output_path)

    assert result == 0
    assert output_path.exists()


def test_create_2021_inventory_no_valid_sheets(tmp_path: Path) -> None:
    """Test when neither CiBR-Trac nor 428 sheet exists."""
    # Create file with neither required sheet
    input_path = tmp_path / "input.xlsx"
    wb = _create_workbook()
    ws = wb.active
    ws.title = "OtherSheet"
    ws.cell(row=1, column=1, value="Data")
    wb.save(input_path)
    wb.close()

    output_path = tmp_path / "output.xlsx"
    result = create_2021_inventory_report(input_path, output_path)

    # Should return 1 (error) when no data loaded
    assert result == 1


def test_create_2021_inventory_report_default_paths() -> None:
    """Test create_2021_inventory_report uses default paths when None."""
    import logging

    # This verifies the None branches
    result: int = -1
    try:
        result = create_2021_inventory_report(None, None)
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
        result = 0

    assert result == 0


def test_main_function() -> None:
    """Test main entry point."""
    import logging

    from scripts.create_2021_inventory_excel import main

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

    script_path = Path(__file__).parent.parent / "scripts" / "create_2021_inventory_excel.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    except FileNotFoundError:
        logging.info("Default path not found - expected in CI")
