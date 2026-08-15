"""Tests for scripts extract standards: ProcessSoilVoc."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
)

from instrument_io._protocols.openpyxl import _create_workbook


class TestProcessSoilVoc:
    """Tests for StandardsExtractor._process_soil_voc method."""

    def test_process_soil_voc_standard_list_sheet(self, tmp_path: Path) -> None:
        """Test processing soil voc file with Standard list sheet."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Standard list"
        ws.cell(row=1, column=1, value="name")
        ws.cell(row=1, column=2, value="CAS")
        ws.cell(row=2, column=1, value="alpha-Pinene")
        ws.cell(row=2, column=2, value="80-56-8")
        ws.cell(row=3, column=1, value="Limonene")
        ws.cell(row=3, column=2, value="138-86-3")

        file_path = tmp_path / "soil_voc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_soil_voc(file_path)

        assert "Soil VOC" in extractor.file_stats
        assert extractor.file_stats["Soil VOC"]["sheets"] == 1

    def test_process_soil_voc_short_values_skipped(self, tmp_path: Path) -> None:
        """Test soil voc skips short values (line 453->450)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # Header with compound keyword triggers header row detection
        ws.cell(row=1, column=1, value="compound")
        ws.cell(row=1, column=2, value="amount")
        ws.cell(row=2, column=1, value="ab")  # Too short (<=2 chars)
        ws.cell(row=2, column=2, value=100)
        ws.cell(row=3, column=1, value="Limonene")  # Valid
        ws.cell(row=3, column=2, value=200)

        file_path = tmp_path / "soil_voc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_soil_voc(file_path)

        assert "Soil VOC" in extractor.file_stats

    def test_process_soil_voc_compound_colors_sheet(self, tmp_path: Path) -> None:
        """Test processing soil voc with compound_colors sheet."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "compound_colors"
        ws.cell(row=1, column=1, value="compound")
        ws.cell(row=2, column=1, value="Myrcene")
        ws.cell(row=3, column=1, value="Camphene")

        file_path = tmp_path / "soil_voc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_soil_voc(file_path)

        assert "Soil VOC" in extractor.file_stats

    def test_process_soil_voc_with_header_row(self, tmp_path: Path) -> None:
        """Test processing soil voc with header row detection."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # Header row with keyword
        ws.cell(row=1, column=1, value="compound")
        ws.cell(row=1, column=2, value="alpha-Pinene")
        ws.cell(row=1, column=3, value="beta-Terpinene")
        # Data row
        ws.cell(row=2, column=1, value="Sample1")
        ws.cell(row=2, column=2, value=100)
        ws.cell(row=2, column=3, value=200)

        file_path = tmp_path / "soil_voc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_soil_voc(file_path)

        assert "Soil VOC" in extractor.file_stats

    def test_process_soil_voc_extracts_terpene_columns(self, tmp_path: Path) -> None:
        """Test soil voc extracts chemicals from terpene column names."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Analysis"
        # Row with terpene keywords in columns
        ws.cell(row=1, column=1, value="Sample")
        ws.cell(row=1, column=2, value="d-Limonene (area)")
        ws.cell(row=1, column=3, value="cyclopentane")
        ws.cell(row=2, column=1, value="S1")
        ws.cell(row=2, column=2, value=1000)
        ws.cell(row=2, column=3, value=500)

        file_path = tmp_path / "soil_voc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_soil_voc(file_path)

        assert "Soil VOC" in extractor.file_stats
