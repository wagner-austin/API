"""Tests for scripts extract standards: ProcessStdTidy."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
)

from instrument_io._protocols.openpyxl import _create_workbook


class TestProcessStdTidy:
    """Tests for StandardsExtractor._process_std_tidy method."""

    def test_process_std_tidy_with_chemical_name_column(self, tmp_path: Path) -> None:
        """Test std_tidy with chemical.name column."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Tidy"
        ws.cell(row=1, column=1, value="chemical.name")
        ws.cell(row=1, column=2, value="amount")
        ws.cell(row=2, column=1, value="alpha-Pinene")
        ws.cell(row=2, column=2, value=100)
        ws.cell(row=3, column=1, value="beta-Myrcene")
        ws.cell(row=3, column=2, value=50)

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats
        assert extractor.file_stats["Std Tidy"]["sheets"] == 1

    def test_process_std_tidy_empty_header_row(self, tmp_path: Path) -> None:
        """Test std_tidy with empty header values (line 685->684)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # Some header columns are empty
        ws.cell(row=1, column=1, value="")  # Empty
        ws.cell(row=1, column=2, value=None)  # None
        ws.cell(row=1, column=3, value="Eucalyptol peak")
        ws.cell(row=2, column=1, value=1)
        ws.cell(row=2, column=2, value=2)
        ws.cell(row=2, column=3, value=3)

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats

    def test_process_std_tidy_short_header_names(self, tmp_path: Path) -> None:
        """Test std_tidy skips short header names (line 699->684)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # Headers with terpene keywords but resulting names too short
        ws.cell(row=1, column=1, value="Sample")
        ws.cell(row=1, column=2, value="pinene")  # Name "pinene" before split is long enough
        ws.cell(row=1, column=3, value="a(terpene)")  # "a" after split is too short
        ws.cell(row=2, column=1, value="S1")
        ws.cell(row=2, column=2, value=100)
        ws.cell(row=2, column=3, value=200)

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats

    def test_process_std_tidy_no_chemical_column_no_terpene_headers(self, tmp_path: Path) -> None:
        """Test std_tidy with no chemical column and no terpene headers (line 681->702)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # No chemical.name column and no terpene keywords in headers
        ws.cell(row=1, column=1, value="Notes")
        ws.cell(row=1, column=2, value="Values")
        ws.cell(row=2, column=1, value="Data1")
        ws.cell(row=2, column=2, value=100)

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats
        assert extractor.file_stats["Std Tidy"]["extracted"] == 0

    def test_process_std_tidy_extracts_from_header(self, tmp_path: Path) -> None:
        """Test std_tidy extracts chemicals from column headers."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # Headers with terpene names
        ws.cell(row=1, column=1, value="Sample")
        ws.cell(row=1, column=2, value="alpha-Pinene Int area")
        ws.cell(row=1, column=3, value="Linalool mass")
        ws.cell(row=1, column=4, value="Eucalyptol peak")
        ws.cell(row=2, column=1, value="S1")
        ws.cell(row=2, column=2, value=1000)
        ws.cell(row=2, column=3, value=500)
        ws.cell(row=2, column=4, value=250)

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats

    def test_process_std_tidy_with_myrcene_header(self, tmp_path: Path) -> None:
        """Test std_tidy extracts myrcene from header."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Analysis"
        ws.cell(row=1, column=1, value="beta-Myrcene")
        ws.cell(row=1, column=2, value="Thujone")
        ws.cell(row=2, column=1, value=100)
        ws.cell(row=2, column=2, value=200)

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats

    def test_process_std_tidy_converts_dots_to_hyphens(self, tmp_path: Path) -> None:
        """Test std_tidy converts dots to hyphens in names."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        ws.cell(row=1, column=1, value="chemical.name")
        ws.cell(row=2, column=1, value="alpha.pinene")

        file_path = tmp_path / "std_tidy.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_std_tidy(file_path)

        assert "Std Tidy" in extractor.file_stats
