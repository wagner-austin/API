"""Tests for scripts extract standards: StandardsExtractorProcessMethods."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
)
from scripts.extract_standards_sources import (
    process_chiral_standards,
    process_claire_std,
    process_jasmine_2024,
    process_old_compiled,
    process_response_factors,
    process_standards_and_cals,
    process_universal_list,
)

from instrument_io._protocols.openpyxl import _create_workbook


class TestStandardsExtractorProcessMethods:
    """Tests for StandardsExtractor process methods."""

    def test_process_response_factors(self, tmp_path: Path) -> None:
        """Test processing response factors file."""
        # Create test file using protocol
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Chemical Name")
        ws.cell(row=1, column=2, value="Density (g/mL)")
        ws.cell(row=2, column=1, value="Limonene")
        ws.cell(row=2, column=2, value="0.84")

        file_path = tmp_path / "response_factors.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_response_factors(extractor, file_path)

        assert "Response Factors" in extractor.file_stats
        assert extractor.file_stats["Response Factors"]["sheets"] == 1

    def test_process_response_factors_name_column(self, tmp_path: Path) -> None:
        """Test response factors with 'name' column (line 352-353)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="name")
        ws.cell(row=1, column=2, value="Density")
        ws.cell(row=2, column=1, value="alpha-Pinene")
        ws.cell(row=2, column=2, value="0.86")

        file_path = tmp_path / "response_factors.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_response_factors(extractor, file_path)

        assert "Response Factors" in extractor.file_stats
        assert extractor.file_stats["Response Factors"]["extracted"] >= 1

    def test_process_response_factors_compound_column(self, tmp_path: Path) -> None:
        """Test response factors with 'compound' column (line 352-353)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="compound")
        ws.cell(row=1, column=2, value="Amount")
        ws.cell(row=2, column=1, value="beta-Myrcene")
        ws.cell(row=2, column=2, value="100")

        file_path = tmp_path / "response_factors.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_response_factors(extractor, file_path)

        assert "Response Factors" in extractor.file_stats
        assert extractor.file_stats["Response Factors"]["extracted"] >= 1

    def test_process_response_factors_no_chem_column(self, tmp_path: Path) -> None:
        """Test response factors with no chemical column (line 355->367)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Data")
        ws.cell(row=1, column=2, value="Value")
        ws.cell(row=2, column=1, value="123")
        ws.cell(row=2, column=2, value="456")

        file_path = tmp_path / "response_factors.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_response_factors(extractor, file_path)

        assert "Response Factors" in extractor.file_stats
        assert extractor.file_stats["Response Factors"]["extracted"] == 0

    def test_process_response_factors_empty_chem_values(self, tmp_path: Path) -> None:
        """Test response factors with empty chemical values (branch 354->352)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Chemical Name")
        ws.cell(row=1, column=2, value="Density (g/mL)")
        # Row with empty chemical name - triggers the `if chem:` false branch
        ws.cell(row=2, column=1, value="")
        ws.cell(row=2, column=2, value="0.85")
        # Row with None chemical name
        ws.cell(row=3, column=1, value=None)
        ws.cell(row=3, column=2, value="0.90")
        # Row with valid chemical
        ws.cell(row=4, column=1, value="Limonene")
        ws.cell(row=4, column=2, value="0.84")

        file_path = tmp_path / "response_factors.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_response_factors(extractor, file_path)

        assert "Response Factors" in extractor.file_stats
        # Only the valid Limonene should be extracted
        assert extractor.file_stats["Response Factors"]["extracted"] >= 1

    def test_process_chiral_standards(self, tmp_path: Path) -> None:
        """Test processing chiral standards file."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Retention Times"
        ws.cell(row=1, column=1, value="Compound")
        ws.cell(row=2, column=1, value="(R)-Limonene")
        ws.cell(row=3, column=1, value="(S)-Limonene")

        file_path = tmp_path / "chiral.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_chiral_standards(extractor, file_path)

        assert "ChiralStandards" in extractor.file_stats

    def test_process_chiral_standards_no_compound_column(self, tmp_path: Path) -> None:
        """Test chiral standards with no Compound column (line 764->775)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Retention Times"
        ws.cell(row=1, column=1, value="Name")  # Not "Compound"
        ws.cell(row=2, column=1, value="Limonene")

        file_path = tmp_path / "chiral.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_chiral_standards(extractor, file_path)

        assert "ChiralStandards" in extractor.file_stats
        assert extractor.file_stats["ChiralStandards"]["extracted"] == 0

    def test_process_chiral_standards_empty_compound_values(self, tmp_path: Path) -> None:
        """Test chiral standards with empty Compound values (branch 754->753)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Retention Times"
        ws.cell(row=1, column=1, value="Compound")
        ws.cell(row=1, column=2, value="RT")
        # Row with empty compound - triggers the `if val:` false branch
        ws.cell(row=2, column=1, value="")
        ws.cell(row=2, column=2, value=1.5)
        # Row with None compound
        ws.cell(row=3, column=1, value=None)
        ws.cell(row=3, column=2, value=2.5)
        # Row with valid compound
        ws.cell(row=4, column=1, value="(R)-Limonene")
        ws.cell(row=4, column=2, value=3.5)

        file_path = tmp_path / "chiral.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_chiral_standards(extractor, file_path)

        assert "ChiralStandards" in extractor.file_stats
        # Only the valid (R)-Limonene should be extracted
        assert extractor.file_stats["ChiralStandards"]["extracted"] >= 1

    def test_process_standards_and_cals(self, tmp_path: Path) -> None:
        """Test processing standards and cals file."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Work list"
        ws.cell(row=1, column=1, value="Mixture Arrangment")
        ws.cell(row=2, column=1, value="Limonene / Pinene / Myrcene")

        file_path = tmp_path / "standards.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_standards_and_cals(extractor, file_path)

        assert "StandardsAndCals" in extractor.file_stats

    def test_process_standards_and_cals_no_mix_column(self, tmp_path: Path) -> None:
        """Test standards and cals with no mixture/arrangment column (line 727->732, 732->exit)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Work list"
        ws.cell(row=1, column=1, value="Data")
        ws.cell(row=1, column=2, value="Value")
        ws.cell(row=2, column=1, value="123")
        ws.cell(row=2, column=2, value="456")

        file_path = tmp_path / "standards.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_standards_and_cals(extractor, file_path)

        assert "StandardsAndCals" in extractor.file_stats
        assert extractor.file_stats["StandardsAndCals"]["extracted"] == 0

    def test_process_standards_and_cals_empty_mixture_values(self, tmp_path: Path) -> None:
        """Test standards and cals with empty mixture values (branch 723->722)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Work list"
        ws.cell(row=1, column=1, value="Mixture Arrangment")
        ws.cell(row=1, column=2, value="Notes")
        # Row with empty mixture - triggers the `if val:` false branch
        ws.cell(row=2, column=1, value="")
        ws.cell(row=2, column=2, value="empty entry")
        # Row with None mixture
        ws.cell(row=3, column=1, value=None)
        ws.cell(row=3, column=2, value="null entry")
        # Row with valid mixture
        ws.cell(row=4, column=1, value="Limonene / Pinene")
        ws.cell(row=4, column=2, value="valid entry")

        file_path = tmp_path / "standards.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_standards_and_cals(extractor, file_path)

        assert "StandardsAndCals" in extractor.file_stats
        # Only the valid mixture should result in extractions
        assert extractor.file_stats["StandardsAndCals"]["extracted"] >= 1

    def test_process_jasmine_2024(self, tmp_path: Path) -> None:
        """Test processing Jasmine 2024 file."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Chemical Name")
        ws.cell(row=2, column=1, value="Limonene")

        file_path = tmp_path / "jasmine.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_jasmine_2024(extractor, file_path)

        assert "Jasmine2024" in extractor.file_stats

    def test_process_jasmine_2024_no_chemical_column(self, tmp_path: Path) -> None:
        """Test Jasmine 2024 with no chemical column (line 838->843)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Data")  # Not a chemical column
        ws.cell(row=2, column=1, value="123")

        file_path = tmp_path / "jasmine.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_jasmine_2024(extractor, file_path)

        assert "Jasmine2024" in extractor.file_stats
        assert extractor.file_stats["Jasmine2024"]["extracted"] == 0

    def test_process_claire_std(self, tmp_path: Path) -> None:
        """Test processing Claire std file."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Compound")
        ws.cell(row=2, column=1, value="Limonene")

        file_path = tmp_path / "claire.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_claire_std(extractor, file_path)

        assert "ClaireStd" in extractor.file_stats

    def test_process_claire_std_no_compound_column(self, tmp_path: Path) -> None:
        """Test Claire std with no Compound column (line 859->864)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Name")  # Not "Compound"
        ws.cell(row=2, column=1, value="Limonene")

        file_path = tmp_path / "claire.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_claire_std(extractor, file_path)

        assert "ClaireStd" in extractor.file_stats
        assert extractor.file_stats["ClaireStd"]["extracted"] == 0

    def test_process_old_compiled(self, tmp_path: Path) -> None:
        """Test processing old compiled file."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Rearrangment"
        ws.cell(row=1, column=1, value="Compiled standard list")
        ws.cell(row=2, column=1, value="Limonene")

        file_path = tmp_path / "old_compiled.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_old_compiled(extractor, file_path)

        assert "OldCompiled" in extractor.file_stats

    def test_process_old_compiled_no_column(self, tmp_path: Path) -> None:
        """Test old compiled with no Compiled standard list column (line 881->892)."""
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Rearrangment"
        ws.cell(row=1, column=1, value="Data")  # Not the expected column
        ws.cell(row=2, column=1, value="123")

        file_path = tmp_path / "old_compiled.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_old_compiled(extractor, file_path)

        assert "OldCompiled" in extractor.file_stats
        assert extractor.file_stats["OldCompiled"]["extracted"] == 0

    def test_process_universal_list(self, tmp_path: Path) -> None:
        """Test processing universal list file."""
        wb = _create_workbook()

        # Standards list sheet
        ws1 = wb.active
        ws1.title = "Standards list"
        ws1.cell(row=1, column=1, value="Chemical Name")
        ws1.cell(row=2, column=1, value="Limonene")

        # RT combined sheet - header row contains compound names, needs data row
        ws2 = wb.create_sheet("RT combined(in progress)")
        ws2.cell(row=1, column=1, value="Pinene")
        ws2.cell(row=1, column=2, value="Myrcene")
        # Add data row so polars doesn't see empty sheet
        ws2.cell(row=2, column=1, value=1.23)
        ws2.cell(row=2, column=2, value=4.56)

        file_path = tmp_path / "universal.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_universal_list(extractor, file_path)

        assert "UniversalList" in extractor.file_stats

    def test_process_universal_list_no_chemical_column(self, tmp_path: Path) -> None:
        """Test universal list with no chemical column (line 793->804)."""
        wb = _create_workbook()

        # Standards list sheet without chemical column
        ws1 = wb.active
        ws1.title = "Standards list"
        ws1.cell(row=1, column=1, value="Data")  # Not a chemical column
        ws1.cell(row=2, column=1, value="123")

        # RT combined sheet
        ws2 = wb.create_sheet("RT combined(in progress)")
        ws2.cell(row=1, column=1, value="Value")
        ws2.cell(row=2, column=1, value=1.0)

        file_path = tmp_path / "universal.xlsx"
        wb.save(file_path)
        wb.close()

        extractor = StandardsExtractor()
        process_universal_list(extractor, file_path)

        assert "UniversalList" in extractor.file_stats
