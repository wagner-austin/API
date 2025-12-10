"""Tests for _protocols.polars module."""

from __future__ import annotations

from pathlib import Path

from instrument_io._protocols.openpyxl import _create_workbook
from instrument_io._protocols.polars import _get_polars_read_excel


def test_get_polars_read_excel_returns_callable() -> None:
    """Test that _get_polars_read_excel returns a callable."""
    read_excel = _get_polars_read_excel()
    assert callable(read_excel)


def test_get_polars_read_excel_can_read_file(tmp_path: Path) -> None:
    """Test that the returned function can read an Excel file."""
    # Create a simple Excel file
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Name")
    ws.cell(row=1, column=2, value="Value")
    ws.cell(row=2, column=1, value="Test")
    ws.cell(row=2, column=2, value=123)

    excel_path = tmp_path / "test.xlsx"
    wb.save(excel_path)
    wb.close()

    # Use the protocol function to read
    read_excel = _get_polars_read_excel()
    df = read_excel(source=excel_path, engine="openpyxl")

    assert df.height == 1
    assert "Name" in df.columns
    assert "Value" in df.columns
