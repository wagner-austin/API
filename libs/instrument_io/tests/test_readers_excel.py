"""Tests for readers.excel module."""

from __future__ import annotations

from pathlib import Path

from instrument_io.readers.excel import (
    ExcelReader,
    _build_row_dicts,
    _convert_cell_value,
    _extract_headers,
    _is_excel_file,
    _parse_polars_json_to_rows,
    _read_worksheet_row,
)


def test_is_excel_file_xlsx(tmp_path: Path) -> None:
    file = tmp_path / "test.xlsx"
    file.touch()
    assert _is_excel_file(file) is True


def test_is_excel_file_xls(tmp_path: Path) -> None:
    file = tmp_path / "test.xls"
    file.touch()
    assert _is_excel_file(file) is True


def test_is_excel_file_xlsm(tmp_path: Path) -> None:
    file = tmp_path / "test.xlsm"
    file.touch()
    assert _is_excel_file(file) is True


def test_is_excel_file_wrong_extension(tmp_path: Path) -> None:
    file = tmp_path / "test.csv"
    file.touch()
    assert _is_excel_file(file) is False


def test_is_excel_file_directory(tmp_path: Path) -> None:
    directory = tmp_path / "test.xlsx"
    directory.mkdir()
    assert _is_excel_file(directory) is False


def test_is_excel_file_not_exists(tmp_path: Path) -> None:
    file = tmp_path / "nonexistent.xlsx"
    assert _is_excel_file(file) is False


def test_convert_cell_value_string() -> None:
    assert _convert_cell_value("hello") == "hello"


def test_convert_cell_value_int() -> None:
    assert _convert_cell_value(42) == 42


def test_convert_cell_value_float() -> None:
    assert _convert_cell_value(3.14) == 3.14


def test_convert_cell_value_bool() -> None:
    assert _convert_cell_value(True) is True


def test_convert_cell_value_none() -> None:
    assert _convert_cell_value(None) is None


def test_parse_polars_json_to_rows_valid() -> None:
    json_str = '[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]'
    result = _parse_polars_json_to_rows(json_str, ["a", "b"])
    assert len(result) == 2
    assert result[0]["a"] == 1
    assert result[1]["b"] == "y"


def test_parse_polars_json_to_rows_empty_array() -> None:
    result = _parse_polars_json_to_rows("[]", ["a", "b"])
    assert result == []


def test_parse_polars_json_to_rows_non_array() -> None:
    result = _parse_polars_json_to_rows('{"a": 1}', ["a"])
    assert result == []


def test_parse_polars_json_to_rows_skips_non_dict() -> None:
    json_str = '[{"a": 1}, "not a dict", {"a": 2}]'
    result = _parse_polars_json_to_rows(json_str, ["a"])
    assert len(result) == 2


def test_extract_headers_strings() -> None:
    result = _extract_headers(["Name", "Value", "Count"])
    assert result == ["Name", "Value", "Count"]


def test_extract_headers_with_none() -> None:
    result = _extract_headers(["Name", None, "Count"])
    assert result == ["Name", "", "Count"]


def test_extract_headers_with_numbers() -> None:
    result = _extract_headers([1, 2.5, "Name"])
    assert result == ["1", "2.5", "Name"]


class TestExcelReader:
    """Tests for ExcelReader class."""

    def test_supports_format_xlsx(self, tmp_path: Path) -> None:
        file = tmp_path / "test.xlsx"
        file.touch()
        reader = ExcelReader()
        assert reader.supports_format(file) is True

    def test_supports_format_csv(self, tmp_path: Path) -> None:
        file = tmp_path / "test.csv"
        file.touch()
        reader = ExcelReader()
        assert reader.supports_format(file) is False

    def test_list_sheets_file_not_exists(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import ExcelReadError

        file = tmp_path / "nonexistent.xlsx"
        reader = ExcelReader()
        with pytest.raises(ExcelReadError) as exc_info:
            reader.list_sheets(file)
        assert "does not exist" in str(exc_info.value)

    def test_list_sheets_not_excel_file(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import ExcelReadError

        file = tmp_path / "test.csv"
        file.touch()
        reader = ExcelReader()
        with pytest.raises(ExcelReadError) as exc_info:
            reader.list_sheets(file)
        assert "Not an Excel file" in str(exc_info.value)

    def test_list_sheets_xls_not_supported(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import ExcelReadError

        file = tmp_path / "test.xls"
        file.touch()
        reader = ExcelReader()
        with pytest.raises(ExcelReadError) as exc_info:
            reader.list_sheets(file)
        assert ".xls is not supported" in str(exc_info.value)

    def test_read_sheet_file_not_exists(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import ExcelReadError

        file = tmp_path / "nonexistent.xlsx"
        reader = ExcelReader()
        with pytest.raises(ExcelReadError) as exc_info:
            reader.read_sheet(file, "Sheet1")
        assert "does not exist" in str(exc_info.value)

    def test_read_sheet_not_excel_file(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import ExcelReadError

        file = tmp_path / "test.csv"
        file.touch()
        reader = ExcelReader()
        with pytest.raises(ExcelReadError) as exc_info:
            reader.read_sheet(file, "Sheet1")
        assert "Not an Excel file" in str(exc_info.value)

    def test_read_sheet_with_header_row_file_not_exists(self, tmp_path: Path) -> None:
        import pytest

        from instrument_io._exceptions import ExcelReadError

        file = tmp_path / "nonexistent.xlsx"
        reader = ExcelReader()
        with pytest.raises(ExcelReadError) as exc_info:
            reader.read_sheet_with_header_row(file, "Sheet1", header_row=0)
        assert "does not exist" in str(exc_info.value)

    def test_read_sheet_with_header_row_max_row_limit(self) -> None:
        """Test reading with max_row limit smaller than data rows.

        Uses real fixture with 15 data rows but sets max_row=3.
        Covers branch 363->370 (while loop exits via condition, not break).
        """
        fixtures = Path(__file__).parent / "fixtures"
        file = fixtures / "excel_edge_cases.xlsx"
        reader = ExcelReader()
        # Read with max_row=3 (only reads 3 rows total: header + 2 data rows)
        result = reader.read_sheet_with_header_row(file, "TestSheet", header_row=0, max_row=3)
        # Should have limited rows due to max_row
        assert len(result) <= 3

    def test_read_sheet_with_header_row_empty_headers_skipped(self) -> None:
        """Test that empty headers are skipped when building row dicts.

        Uses real fixture with some empty header columns.
        Covers branch 213->212 (header is falsy, skip column).
        """
        fixtures = Path(__file__).parent / "fixtures"
        file = fixtures / "excel_edge_cases.xlsx"
        reader = ExcelReader()
        result = reader.read_sheet_with_header_row(file, "TestSheet", header_row=0)
        # Row should only have 'Name' and 'Value' keys, not empty header columns
        # Fixture has 15 data rows
        assert len(result) >= 10
        first_row = result[0]
        assert "Name" in first_row
        assert "Value" in first_row
        # Empty string headers should not appear as keys
        assert "" not in first_row

    def test_read_sheet_with_header_row_short_row(self) -> None:
        """Test reading rows shorter than header count.

        Uses real fixture where some rows have fewer columns than headers.
        Covers branch 213->212 (i >= len(row), skip column).
        """
        fixtures = Path(__file__).parent / "fixtures"
        file = fixtures / "excel_edge_cases.xlsx"
        reader = ExcelReader()
        result = reader.read_sheet_with_header_row(file, "TestSheet", header_row=0)
        # Should handle short rows without error - fixture has 15 data rows
        assert len(result) >= 10


class TestReadWorksheetRow:
    """Tests for _read_worksheet_row internal function."""

    def test_read_row_loop_completes_without_break(self) -> None:
        """Test row reading when loop completes without hitting break.

        Uses real fixture with wide data, calls with small max_cols.
        Covers branch 156->171 (for loop completes normally).
        """
        from instrument_io._protocols.openpyxl import _load_workbook

        fixtures = Path(__file__).parent / "fixtures"
        file = fixtures / "excel_wide_data.xlsx"

        wb = _load_workbook(file, read_only=True, data_only=True)
        ws = wb["WideData"]

        # Call with small max_cols so loop completes without break condition
        # (break requires col_idx > 10, so max_cols=5 will never break)
        row_values, has_data = _read_worksheet_row(ws, row_idx=2, max_cols=5)

        wb.close()

        # Should have read 4 columns (range(1, 5) = 1,2,3,4)
        assert len(row_values) == 4
        assert has_data is True
        assert row_values[0] == 10  # First data column value


class TestBuildRowDicts:
    """Tests for _build_row_dicts internal function."""

    def test_empty_row_dict_skipped(self) -> None:
        """Test that rows resulting in empty dict are skipped.

        When all headers are empty strings, row_dict stays empty.
        Covers branch 215->210 (row_dict is empty, skip append).
        """
        from instrument_io._json_bridge import CellValue

        # Construct test data directly: all-empty headers with data rows
        all_rows: list[list[CellValue]] = [
            ["", "", ""],  # Header row with all empty strings
            ["data1", "data2", "data3"],  # Data row 1
            ["data4", "data5", "data6"],  # Data row 2
        ]

        # Extract headers (all empty strings)
        headers = _extract_headers(all_rows[0])
        assert headers == ["", "", ""]

        # Build row dicts - all headers are empty/falsy, so row_dict stays empty
        result = _build_row_dicts(all_rows, headers, start_row=1)

        # With all-empty headers, no columns match, row_dict is empty, rows skipped
        assert result == []
