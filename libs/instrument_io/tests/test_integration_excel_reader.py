"""Integration tests for Excel reader.

Tests use real Excel fixture files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io.readers.excel import ExcelReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHROMATOGRAM_XLSX = FIXTURES_DIR / "chromatogram.xlsx"


def _get_xlsx_file() -> Path:
    """Get path to chromatogram.xlsx test file, skip if not found."""
    if not CHROMATOGRAM_XLSX.exists():
        pytest.skip("chromatogram.xlsx test fixture not found")
    return CHROMATOGRAM_XLSX


class TestExcelReaderIntegration:
    """Integration tests using real Excel files."""

    def test_supports_format_real_xlsx(self) -> None:
        """Test that reader recognizes real Excel file."""
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        assert reader.supports_format(xlsx_file) is True

    def test_list_sheets_real_xlsx(self) -> None:
        """Test listing sheets in real Excel file."""
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        sheets = reader.list_sheets(xlsx_file)

        assert len(sheets) == 2
        assert "Chromatogram" in sheets
        assert "Metadata" in sheets

    def test_read_sheet_chromatogram(self) -> None:
        """Test reading Chromatogram sheet from real Excel file."""
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        rows = reader.read_sheet(xlsx_file, sheet_name="Chromatogram")

        # 21 data rows + 1 header = 22 rows total, but read_sheet skips header
        assert len(rows) == 21

        # Verify first row data
        first_row = rows[0]
        assert first_row["Retention Time (min)"] == 0.0
        assert first_row["Intensity"] == 1000
        assert first_row["Sample Name"] == "Sample_001"

        # Verify last row data
        last_row = rows[-1]
        assert last_row["Retention Time (min)"] == 10.0
        assert last_row["Intensity"] == 1000

    def test_read_sheet_metadata(self) -> None:
        """Test reading Metadata sheet from real Excel file."""
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        rows = reader.read_sheet(xlsx_file, sheet_name="Metadata")

        # Metadata sheet has 2 data rows (Instrument, Method)
        assert len(rows) == 2

        # Check instrument row
        assert rows[0]["Property"] == "Instrument"
        assert rows[0]["Value"] == "Agilent 6890"

        # Check method row
        assert rows[1]["Property"] == "Method"
        assert rows[1]["Value"] == "GC-MS Standard"

    def test_read_sheet_with_header_row(self) -> None:
        """Test reading with explicit header row.

        When header_row=1, the first data row becomes the header,
        so we get 20 data rows (rows 2-21 in Excel terms).
        """
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        rows = reader.read_sheet_with_header_row(xlsx_file, sheet_name="Chromatogram", header_row=1)

        # Row 1 becomes header, rows 2-21 become data = 20 rows
        assert len(rows) == 20

    def test_read_chromatogram_data_values(self) -> None:
        """Test that chromatogram data values are correct."""
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        rows = reader.read_sheet(xlsx_file, sheet_name="Chromatogram")

        # Extract retention times and intensities (as floats)
        times: list[float] = []
        intensities: list[float] = []
        for row in rows:
            rt = row["Retention Time (min)"]
            inten = row["Intensity"]
            if isinstance(rt, (int, float)) and isinstance(inten, (int, float)):
                times.append(float(rt))
                intensities.append(float(inten))

        # Verify we got all 21 data points
        assert len(times) == 21
        assert len(intensities) == 21

        # Verify data range
        assert min(times) == 0.0
        assert max(times) == 10.0
        assert max(intensities) == 15000.0  # Peak at 6.5 min

        # Verify peak location
        peak_idx = intensities.index(15000.0)
        assert times[peak_idx] == 6.5

    def test_read_sheets_all(self) -> None:
        """Test reading all sheets from Excel file."""
        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()
        all_sheets = reader.read_sheets(xlsx_file)

        # Should have 2 sheets
        assert len(all_sheets) == 2
        assert "Chromatogram" in all_sheets
        assert "Metadata" in all_sheets

        # Verify Chromatogram sheet data
        assert len(all_sheets["Chromatogram"]) == 21

        # Verify Metadata sheet data
        assert len(all_sheets["Metadata"]) == 2

    def test_read_sheet_with_header_row_sheet_not_found(self) -> None:
        """Test error when sheet doesn't exist in read_sheet_with_header_row."""
        from instrument_io._exceptions import ExcelReadError

        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()

        with pytest.raises(ExcelReadError) as exc_info:
            reader.read_sheet_with_header_row(xlsx_file, "NonExistent", header_row=0)

        assert "not found" in str(exc_info.value)

    def test_read_sheet_with_header_row_exceeds_data(self) -> None:
        """Test error when header_row is beyond available data rows."""
        from instrument_io._exceptions import ExcelReadError

        xlsx_file = _get_xlsx_file()
        reader = ExcelReader()

        # Chromatogram sheet has 22 rows (1 header + 21 data)
        # Setting header_row=100 should fail
        with pytest.raises(ExcelReadError) as exc_info:
            reader.read_sheet_with_header_row(xlsx_file, "Chromatogram", header_row=100)

        assert "exceeds" in str(exc_info.value)
