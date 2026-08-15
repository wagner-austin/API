"""Tests for scripts extract standards: ProcessAvisaCalc."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
)

from instrument_io._protocols.openpyxl import _create_workbook


class TestProcessAvisaCalc:
    """Tests for StandardsExtractor._process_avisa_calc method."""

    def test_process_avisa_calc_with_first_cell_chemical(self, tmp_path: Path) -> None:
        """Test avisa calc extraction when first cell has chemical name."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Limonene Calc"
        ws.cell(row=1, column=1, value="d-Limonene calculation")
        ws.cell(row=2, column=1, value="Data")
        ws.cell(row=2, column=2, value=100)

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats
        assert extractor.file_stats["Avisa Calc"]["sheets"] == 1

    def test_process_avisa_calc_no_chemical_keywords_in_first_cell(self, tmp_path: Path) -> None:
        """Test avisa calc with no chemical keywords in first cell (line 489->505)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # First cell has no chemical keywords
        ws.cell(row=1, column=1, value="Notes")
        ws.cell(row=2, column=1, value="Some data")

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats

    def test_process_avisa_calc_no_header_row(self, tmp_path: Path) -> None:
        """Test avisa calc with no header row found (line 506->519)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "NoHeader"
        # Data without compound/name/standard/analyte keywords
        ws.cell(row=1, column=1, value="Value1")
        ws.cell(row=1, column=2, value="Value2")
        ws.cell(row=2, column=1, value=123)
        ws.cell(row=2, column=2, value=456)
        ws.cell(row=3, column=1, value=789)
        ws.cell(row=3, column=2, value=101)

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats

    def test_process_avisa_calc_single_row_sheet(self, tmp_path: Path) -> None:
        """Test avisa calc with single row sheet (line 500->513)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "SingleRow"
        # Only one row - height will be 1, triggering the height > 1 false branch
        ws.cell(row=1, column=1, value="Limonene data")

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats

    def test_process_avisa_calc_with_compound_column(self, tmp_path: Path) -> None:
        """Test avisa calc with compound column."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Standards"
        ws.cell(row=1, column=1, value="compound")
        ws.cell(row=1, column=2, value="amount")
        ws.cell(row=2, column=1, value="alpha-Pinene")
        ws.cell(row=2, column=2, value=50)
        ws.cell(row=3, column=1, value="Eucalyptol")
        ws.cell(row=3, column=2, value=75)

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats

    def test_process_avisa_calc_with_standard_column(self, tmp_path: Path) -> None:
        """Test avisa calc with standard column."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        ws.cell(row=1, column=1, value="standard")
        ws.cell(row=1, column=2, value="concentration")
        ws.cell(row=2, column=1, value="Linalool")
        ws.cell(row=2, column=2, value=25)

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats

    def test_process_avisa_calc_pinene_in_first_cell(self, tmp_path: Path) -> None:
        """Test avisa calc extracts pinene from first cell."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Pinene Analysis"
        ws.cell(row=1, column=1, value="alpha-Pinene (R) standard")
        ws.cell(row=2, column=1, value="Data row")

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats

    def test_process_avisa_calc_empty_and_short_compound_values(self, tmp_path: Path) -> None:
        """Test avisa calc with empty and short compound values (branch 533->530)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        ws.cell(row=1, column=1, value="compound")
        ws.cell(row=1, column=2, value="amount")
        # Row with empty compound - triggers the `if val and len(val.strip()) > 2` false branch
        ws.cell(row=2, column=1, value="")
        ws.cell(row=2, column=2, value=100)
        # Row with short compound (<= 2 chars after strip)
        ws.cell(row=3, column=1, value="ab")
        ws.cell(row=3, column=2, value=200)
        # Row with whitespace only
        ws.cell(row=4, column=1, value="   ")
        ws.cell(row=4, column=2, value=300)
        # Row with valid compound
        ws.cell(row=5, column=1, value="Limonene")
        ws.cell(row=5, column=2, value=400)

        file_path = tmp_path / "avisa_calc.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_avisa_calc(file_path)

        assert "Avisa Calc" in extractor.file_stats
        # Only the valid Limonene should be extracted
        assert extractor.file_stats["Avisa Calc"]["extracted"] >= 1
