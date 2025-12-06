"""Tests for writers.excel module."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import WriterError
from instrument_io._protocols.openpyxl import WorksheetProtocol, _create_workbook
from instrument_io.readers.excel import ExcelReader
from instrument_io.types.common import CellValue
from instrument_io.writers.excel import (
    ExcelWriter,
    _add_table_to_sheet,
    _collect_columns,
    _sanitize_table_name,
)

# Type alias for row dictionaries
RowDict = dict[str, CellValue]


def test_collect_columns_basic() -> None:
    rows: list[RowDict] = [
        {"a": 1, "b": 2},
        {"a": 3, "c": 4},
    ]
    result = _collect_columns(rows)
    assert result == ["a", "b", "c"]


def test_collect_columns_preserves_order() -> None:
    rows: list[RowDict] = [
        {"z": 1, "a": 2, "m": 3},
    ]
    result = _collect_columns(rows)
    assert result == ["z", "a", "m"]


def test_collect_columns_empty() -> None:
    result = _collect_columns([])
    assert result == []


def test_sanitize_table_name_basic() -> None:
    result = _sanitize_table_name("Sheet1", 0)
    assert result == "Sheet1_0"


def test_sanitize_table_name_with_spaces() -> None:
    result = _sanitize_table_name("My Sheet", 1)
    assert result == "My_Sheet_1"


def test_sanitize_table_name_starts_with_number() -> None:
    result = _sanitize_table_name("123Sheet", 0)
    assert result.startswith("Table_")


def test_sanitize_table_name_special_chars() -> None:
    result = _sanitize_table_name("Sheet@#$%", 0)
    # Should only contain alphanumeric and underscore
    assert all(c.isalnum() or c == "_" for c in result)


def test_add_table_to_sheet_no_columns() -> None:
    # Line 124: early return when columns is empty
    wb = _create_workbook()
    ws: WorksheetProtocol = wb.active
    # Should return early without adding table
    _add_table_to_sheet(ws, [], 5, "Table1")
    # No exception means success - table wasn't added
    wb.close()


def test_add_table_to_sheet_zero_rows() -> None:
    # Line 124: early return when row_count is 0
    wb = _create_workbook()
    ws: WorksheetProtocol = wb.active
    # Should return early without adding table
    _add_table_to_sheet(ws, ["Col1", "Col2"], 0, "Table1")
    # No exception means success - table wasn't added
    wb.close()


class TestExcelWriter:
    """Tests for ExcelWriter class."""

    def test_write_sheet_creates_file(self, tmp_path: Path) -> None:
        writer = ExcelWriter()
        rows: list[RowDict] = [
            {"Name": "Item1", "Value": 100},
            {"Name": "Item2", "Value": 200},
        ]
        out_path = tmp_path / "output.xlsx"
        writer.write_sheet(rows, out_path)
        assert out_path.exists()

    def test_write_sheet_can_be_read_back(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False)
        rows: list[RowDict] = [
            {"Name": "Item1", "Value": 100},
        ]
        out_path = tmp_path / "output.xlsx"
        writer.write_sheet(rows, out_path, sheet_name="Data")

        reader = ExcelReader()
        result = reader.read_sheet(out_path, "Data")
        assert len(result) == 1
        assert result[0]["Name"] == "Item1"

    def test_write_sheets_empty_raises(self, tmp_path: Path) -> None:
        writer = ExcelWriter()
        out_path = tmp_path / "output.xlsx"
        with pytest.raises(WriterError):
            writer.write_sheets({}, out_path)

    def test_write_sheets_multiple(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False)
        sheets: dict[str, list[RowDict]] = {
            "Sheet1": [{"A": 1}],
            "Sheet2": [{"B": 2}],
        }
        out_path = tmp_path / "output.xlsx"
        writer.write_sheets(sheets, out_path)

        reader = ExcelReader()
        sheet_names = reader.list_sheets(out_path)
        assert "Sheet1" in sheet_names
        assert "Sheet2" in sheet_names

    def test_write_sheet_with_table(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=True)
        rows: list[RowDict] = [
            {"Name": "Item1", "Value": 100},
        ]
        out_path = tmp_path / "output.xlsx"
        writer.write_sheet(rows, out_path)
        # File should be created without errors
        assert out_path.exists()

    def test_write_sheet_auto_width(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_width=True, auto_table=False)
        rows: list[RowDict] = [
            {"ShortName": "A", "VeryLongColumnNameHere": "B"},
        ]
        out_path = tmp_path / "output.xlsx"
        writer.write_sheet(rows, out_path)
        assert out_path.exists()

    def test_write_sheet_no_auto_features(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False, auto_width=False)
        rows: list[RowDict] = [
            {"Name": "Item1"},
        ]
        out_path = tmp_path / "output.xlsx"
        writer.write_sheet(rows, out_path)
        assert out_path.exists()

    def test_write_sheet_empty_rows(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False)
        rows: list[RowDict] = []
        out_path = tmp_path / "output.xlsx"
        writer.write_sheets({"Empty": rows}, out_path)
        assert out_path.exists()

    def test_write_sheet_creates_parent_dirs(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False)
        rows: list[RowDict] = [{"A": 1}]
        out_path = tmp_path / "subdir" / "nested" / "output.xlsx"
        writer.write_sheet(rows, out_path)
        assert out_path.exists()

    def test_write_rows_to_sheet(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False, auto_width=False)
        wb = _create_workbook()

        rows: list[RowDict] = [{"A": 1, "B": 2}]
        writer.write_rows_to_sheet(wb, "TestSheet", rows, 0)

        out_path = tmp_path / "output.xlsx"
        wb.save(out_path)
        wb.close()

        reader = ExcelReader()
        result = reader.read_sheet(out_path, "TestSheet")
        assert result[0]["A"] == 1

    def test_write_rows_to_sheet_empty(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False, auto_width=False)
        wb = _create_workbook()

        empty_rows: list[RowDict] = []
        writer.write_rows_to_sheet(wb, "Empty", empty_rows, 0)

        out_path = tmp_path / "output.xlsx"
        # Remove default sheet first to avoid extra sheets
        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])
        wb.save(out_path)
        wb.close()

        # Should have created the sheet without errors
        assert out_path.exists()

    def test_write_rows_to_sheet_with_auto_features(self, tmp_path: Path) -> None:
        # Lines 290-296: write_rows_to_sheet with auto_width and auto_table enabled
        writer = ExcelWriter(auto_table=True, auto_width=True)
        wb = _create_workbook()

        rows: list[RowDict] = [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
        writer.write_rows_to_sheet(wb, "AutoSheet", rows, 0)

        out_path = tmp_path / "output.xlsx"
        # Remove default sheet
        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])
        wb.save(out_path)
        wb.close()

        # Verify file created and data readable
        reader = ExcelReader()
        result = reader.read_sheet(out_path, "AutoSheet")
        assert len(result) == 2
        assert result[0]["A"] == 1

    def test_write_various_types(self, tmp_path: Path) -> None:
        writer = ExcelWriter(auto_table=False)
        rows: list[RowDict] = [
            {
                "String": "hello",
                "Int": 42,
                "Float": 3.14,
                "Bool": True,
                "None": None,
            },
        ]
        out_path = tmp_path / "types.xlsx"
        writer.write_sheet(rows, out_path)

        reader = ExcelReader()
        result = reader.read_sheet(out_path, "Sheet1")
        assert result[0]["String"] == "hello"
        assert result[0]["Int"] == 42
