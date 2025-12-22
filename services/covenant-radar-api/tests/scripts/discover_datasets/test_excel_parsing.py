"""Tests for Excel parsing functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.discover_datasets.parsers.excel import (
    read_excel_header_and_sample,
    read_xls_header_and_sample,
)

from .conftest import get_workbook_ctor, get_xlwt_workbook_ctor


class TestReadExcelHeaderAndSample:
    """Tests for read_excel_header_and_sample function."""

    def _create_excel_file(
        self,
        path: Path,
        headers: list[str],
        rows: list[list[str | int | float | None]],
    ) -> None:
        """Helper to create Excel file using openpyxl."""
        ctor = get_workbook_ctor()
        wb = ctor()
        ws = wb.active
        for col_idx, header in enumerate(headers, start=1):
            ws.cell(row=1, column=col_idx, value=header)
        for row_idx, row_data in enumerate(rows, start=2):
            for col_idx, value in enumerate(row_data, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        wb.save(path)
        wb.close()

    def test_basic_excel(self) -> None:
        """Test reading basic Excel file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xlsx"
            self._create_excel_file(
                path,
                ["a", "b", "target"],
                [[1, 2, 0], [3, 4, 1]],
            )

            columns, n_rows, sample = read_excel_header_and_sample(path)

            assert columns == ("a", "b", "target")
            assert n_rows == 2
            assert sample == (("1", "2", "0"), ("3", "4", "1"))

    def test_empty_excel(self) -> None:
        """Test reading Excel file with only headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xlsx"
            self._create_excel_file(path, ["a", "b"], [])

            columns, n_rows, sample = read_excel_header_and_sample(path)

            assert columns == ("a", "b")
            assert n_rows == 0
            assert sample == ()

    def test_excel_with_none_values(self) -> None:
        """Test reading Excel file with None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xlsx"
            self._create_excel_file(
                path,
                ["a", "b"],
                [[1, None], [None, 2]],
            )

            columns, n_rows, sample = read_excel_header_and_sample(path)

            assert columns == ("a", "b")
            assert n_rows == 2
            assert sample == (("1", ""), ("", "2"))

    def test_excel_with_numeric_header(self) -> None:
        """Test reading Excel file with numeric header values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xlsx"
            ctor = get_workbook_ctor()
            wb = ctor()
            ws = wb.active
            ws.cell(row=1, column=1, value=1)
            ws.cell(row=1, column=2, value=2)
            ws.cell(row=2, column=1, value="a")
            ws.cell(row=2, column=2, value="b")
            wb.save(path)
            wb.close()

            columns, n_rows, sample = read_excel_header_and_sample(path)

            assert columns == ("1", "2")
            assert n_rows == 1
            assert sample == (("a", "b"),)

    def test_excel_with_none_header(self) -> None:
        """Test reading Excel file with None values in header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xlsx"
            ctor = get_workbook_ctor()
            wb = ctor()
            ws = wb.active
            ws.cell(row=1, column=1, value="a")
            ws.cell(row=1, column=2, value=None)
            ws.cell(row=1, column=3, value="c")
            ws.cell(row=2, column=1, value=1)
            ws.cell(row=2, column=2, value=2)
            ws.cell(row=2, column=3, value=3)
            wb.save(path)
            wb.close()

            columns, n_rows, sample = read_excel_header_and_sample(path)

            assert columns == ("a", "", "c")
            assert n_rows == 1
            assert sample == (("1", "2", "3"),)

    def test_excel_prefers_data_sheet_over_description(self) -> None:
        """Test that 'Data' sheet is selected over a 'Description' sheet.

        When an Excel file has multiple sheets, the scanner should prefer
        sheets named 'Data', 'Sheet1', 'Train', or 'Dataset' over other sheets
        like 'Description' or 'Info'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi_sheet.xlsx"
            ctor = get_workbook_ctor()
            wb = ctor()

            # First sheet is "Description" with metadata (NOT the data)
            ws_desc = wb.active
            ws_desc.title = "Description"
            ws_desc.cell(row=1, column=1, value="Dataset Info")
            ws_desc.cell(row=2, column=1, value="This is a description")

            # Second sheet is "Data" with actual data (should be selected)
            ws_data = wb.create_sheet("Data")
            ws_data.cell(row=1, column=1, value="feature1")
            ws_data.cell(row=1, column=2, value="target")
            ws_data.cell(row=2, column=1, value=1.0)
            ws_data.cell(row=2, column=2, value=0)
            ws_data.cell(row=3, column=1, value=2.0)
            ws_data.cell(row=3, column=2, value=1)

            wb.save(path)
            wb.close()

            columns, n_rows, sample = read_excel_header_and_sample(path)

            # Should read from "Data" sheet, not "Description"
            assert columns == ("feature1", "target")
            assert n_rows == 2
            # Values come back as integers or floats depending on openpyxl version
            assert sample[0][1] == "0"
            assert sample[1][1] == "1"

    def test_large_excel_limits_sample_size(self) -> None:
        """Test that large Excel files are sampled to MAX_SAMPLE_ROWS.

        When an Excel file has more than MAX_SAMPLE_ROWS (1000) data rows,
        only the first 1000 rows are sampled but total count is accurate.
        """
        n_rows = 1100  # More than MAX_SAMPLE_ROWS

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.xlsx"
            ctor = get_workbook_ctor()
            wb = ctor()
            ws = wb.active
            # Write header
            ws.cell(row=1, column=1, value="a")
            ws.cell(row=1, column=2, value="b")
            # Write data rows
            for i in range(n_rows):
                ws.cell(row=i + 2, column=1, value=i)
                ws.cell(row=i + 2, column=2, value="value")
            wb.save(path)
            wb.close()

            columns, total_rows, sample = read_excel_header_and_sample(path)

            assert columns == ("a", "b")
            assert total_rows == n_rows
            assert len(sample) == 1000  # MAX_SAMPLE_ROWS
            # First row should be index 0
            assert sample[0][0] == "0"


class TestReadXlsHeaderAndSample:
    """Tests for read_xls_header_and_sample function."""

    def _create_xls_file(
        self,
        path: Path,
        headers: list[str],
        rows: list[list[str | int | float]],
    ) -> None:
        """Helper to create legacy Excel .xls file using xlwt."""
        ctor = get_xlwt_workbook_ctor()
        wb = ctor()
        ws = wb.add_sheet("Sheet1")
        for col_idx, header in enumerate(headers):
            ws.write(0, col_idx, header)
        for row_idx, row_data in enumerate(rows, start=1):
            for col_idx, value in enumerate(row_data):
                ws.write(row_idx, col_idx, value)
        wb.save(str(path))

    def test_basic_xls(self) -> None:
        """Test reading basic .xls file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xls"
            self._create_xls_file(
                path,
                ["a", "b", "target"],
                [[1, 2, 0], [3, 4, 1]],
            )

            columns, n_rows, sample = read_xls_header_and_sample(path)

            assert columns == ("a", "b", "target")
            assert n_rows == 2
            # xlrd returns numeric values as floats
            assert sample == (("1.0", "2.0", "0.0"), ("3.0", "4.0", "1.0"))

    def test_empty_xls(self) -> None:
        """Test reading .xls file with only headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xls"
            self._create_xls_file(path, ["a", "b"], [])

            columns, n_rows, sample = read_xls_header_and_sample(path)

            assert columns == ("a", "b")
            assert n_rows == 0
            assert sample == ()

    def test_xls_with_string_values(self) -> None:
        """Test reading .xls file with string values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xls"
            self._create_xls_file(
                path,
                ["name", "status"],
                [["Alice", "active"], ["Bob", "inactive"]],
            )

            columns, n_rows, sample = read_xls_header_and_sample(path)

            assert columns == ("name", "status")
            assert n_rows == 2
            assert sample == (("Alice", "active"), ("Bob", "inactive"))

    def test_xls_completely_empty_sheet(self) -> None:
        """Test reading .xls file with an empty sheet (no rows/columns)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.xls"
            ctor = get_xlwt_workbook_ctor()
            wb = ctor()
            wb.add_sheet("Empty")
            wb.save(str(path))

            columns, n_rows, sample = read_xls_header_and_sample(path)

            assert columns == ()
            assert n_rows == 0
            assert sample == ()
