"""Tests for types.excel module."""

from __future__ import annotations

from instrument_io.types.common import CellValue
from instrument_io.types.excel import ExcelRow, ExcelRows, ExcelSheets


def test_excel_row_alias() -> None:
    row: ExcelRow = {"Name": "Item", "Value": 100, "Active": True}
    assert row["Name"] == "Item"
    assert row["Value"] == 100
    assert row["Active"] is True


def test_excel_row_with_none() -> None:
    row: ExcelRow = {"Name": "Item", "Missing": None}
    value: CellValue = row["Missing"]
    assert value is None


def test_excel_rows_alias() -> None:
    rows: ExcelRows = [
        {"A": 1, "B": 2},
        {"A": 3, "B": 4},
    ]
    assert len(rows) == 2
    assert rows[0]["A"] == 1
    assert rows[1]["B"] == 4


def test_excel_rows_empty() -> None:
    rows: ExcelRows = []
    assert len(rows) == 0


def test_excel_sheets_alias() -> None:
    sheets: ExcelSheets = {
        "Sheet1": [{"A": 1}],
        "Sheet2": [{"B": 2}, {"B": 3}],
    }
    assert len(sheets) == 2
    # Test actual values, not just existence
    assert sheets["Sheet1"][0]["A"] == 1
    assert sheets["Sheet2"][0]["B"] == 2
    assert len(sheets["Sheet2"]) == 2


def test_excel_sheets_empty() -> None:
    sheets: ExcelSheets = {}
    assert len(sheets) == 0
