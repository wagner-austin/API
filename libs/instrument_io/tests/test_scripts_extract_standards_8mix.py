"""Tests for scripts extract standards: Process8mix."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
)

from instrument_io._protocols.openpyxl import _create_workbook


class TestProcess8mix:
    """Tests for StandardsExtractor._process_8mix method."""

    def test_process_8mix_with_concentration_header(self, tmp_path: Path) -> None:
        """Test 8mix with concentration header row (line 601)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Mix1"
        # Header row with concentration keyword
        ws.cell(row=1, column=1, value="concentration")
        ws.cell(row=1, column=2, value="alpha-Pinene")
        ws.cell(row=1, column=3, value="Limonene")
        ws.cell(row=1, column=4, value="Myrcene")
        # Data
        ws.cell(row=2, column=1, value=100)
        ws.cell(row=2, column=2, value=50)
        ws.cell(row=2, column=3, value=75)
        ws.cell(row=2, column=4, value=25)

        file_path = tmp_path / "8mix.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_8mix(file_path)

        assert "8mix" in extractor.file_stats
        assert extractor.file_stats["8mix"]["sheets"] == 1
        # Verify that chemicals were extracted (line 601 executed)
        assert extractor.file_stats["8mix"]["extracted"] >= 1

    def test_process_8mix_no_concentration_no_chemicals(self, tmp_path: Path) -> None:
        """Test 8mix with no concentration and no chemical keywords (line 577->586)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        # No concentration header and no chemical keywords
        ws.cell(row=1, column=1, value="Notes")
        ws.cell(row=1, column=2, value="Values")
        ws.cell(row=2, column=1, value="Data1")
        ws.cell(row=2, column=2, value=100)

        file_path = tmp_path / "8mix.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_8mix(file_path)

        assert "8mix" in extractor.file_stats
        assert extractor.file_stats["8mix"]["extracted"] == 0

    def test_process_8mix_column_ending_with_1(self, tmp_path: Path) -> None:
        """Test 8mix skips columns ending with _1 (line 610 branch)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Mix1"
        ws.cell(row=1, column=1, value="concentration")
        ws.cell(row=1, column=2, value="Limonene")
        ws.cell(row=1, column=3, value="Limonene_1")  # Duplicate, should skip
        ws.cell(row=2, column=1, value=100)
        ws.cell(row=2, column=2, value=50)
        ws.cell(row=2, column=3, value=50)

        file_path = tmp_path / "8mix.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_8mix(file_path)

        assert "8mix" in extractor.file_stats

    def test_process_8mix_without_concentration_scans_for_chemicals(self, tmp_path: Path) -> None:
        """Test 8mix scans for known chemicals when no concentration header."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Mix2"
        # No concentration header, but has chemical names
        ws.cell(row=1, column=1, value="Sample")
        ws.cell(row=1, column=2, value="Value")
        ws.cell(row=2, column=1, value="alpha-Pinene standard")
        ws.cell(row=2, column=2, value=100)
        ws.cell(row=3, column=1, value="Limonene test")
        ws.cell(row=3, column=2, value=200)

        file_path = tmp_path / "8mix.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_8mix(file_path)

        assert "8mix" in extractor.file_stats

    def test_process_8mix_with_terpene_in_values(self, tmp_path: Path) -> None:
        """Test 8mix finds terpene chemicals in cell values."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Analysis"
        ws.cell(row=1, column=1, value="gamma-Terpinene analysis")
        ws.cell(row=2, column=1, value="Myrcene measurement")
        ws.cell(row=3, column=1, value="Thujone calibration")

        file_path = tmp_path / "8mix.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_8mix(file_path)

        assert "8mix" in extractor.file_stats

    def test_process_8mix_skips_invalid_columns(self, tmp_path: Path) -> None:
        """Test 8mix skips slope, rt, calc mass columns."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Mix3"
        ws.cell(row=1, column=1, value="concentration")
        ws.cell(row=1, column=2, value="slope")
        ws.cell(row=1, column=3, value="rt")
        ws.cell(row=1, column=4, value="Camphene")
        ws.cell(row=2, column=1, value=100)
        ws.cell(row=2, column=2, value=1.5)
        ws.cell(row=2, column=3, value=5.2)
        ws.cell(row=2, column=4, value=50)

        file_path = tmp_path / "8mix.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        extractor._process_8mix(file_path)

        assert "8mix" in extractor.file_stats
