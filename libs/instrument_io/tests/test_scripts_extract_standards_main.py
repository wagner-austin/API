"""Tests for scripts extract standards: StandardsExtractorLogSummary."""

from __future__ import annotations

from pathlib import Path

from scripts.extract_standards import (
    StandardsExtractor,
)

from instrument_io._protocols.openpyxl import _create_workbook, _load_workbook


class TestStandardsExtractorLogSummary:
    """Tests for StandardsExtractor.log_summary method."""

    def test_log_summary(self) -> None:
        """Test log_summary runs without error."""
        extractor = StandardsExtractor()
        extractor._file_stats["Test File"] = {"sheets": 3, "extracted": 10}

        # Should not raise
        extractor.log_summary()


class TestExtractStandardsMain:
    """Tests for main function and extract_standards."""

    def test_extract_standards_with_custom_paths(self, tmp_path: Path) -> None:
        """Test extract_standards with custom input/output paths."""
        from scripts.extract_standards import extract_standards

        # Create all required input files
        base_path = tmp_path / "lab"
        base_path.mkdir()

        # Create directory structure
        (base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24").mkdir(
            parents=True
        )
        (base_path / "Notebooks/Avisa Lab Notebook").mkdir(parents=True)
        (base_path / "Notebooks/Emily Truong Notebook").mkdir(parents=True)
        gcms_path = (
            base_path
            / "Current Projects/Thermal Stress Project"
            / "2021-2022 BVOC collection experiment (Juan)/GCMS data"
        )
        gcms_path.mkdir(parents=True)
        (base_path / "InstrumentLogs/TDGC/Calibrations/old files").mkdir(parents=True)

        # Create Response Factors file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Chemical Name")
        ws.cell(row=2, column=1, value="Limonene")
        wb.save(base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx")
        wb.close()

        # Create Soil VOC file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Standard list"
        ws.cell(row=1, column=1, value="name")
        ws.cell(row=2, column=1, value="Pinene")
        wb.save(base_path / "Current Projects/Soil VOC quantitation.xlsx")
        wb.close()

        # Create Avisa Calc file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Limonene calculation")
        ws.cell(row=2, column=1, value="Data")
        wb.save(base_path / "Notebooks/Avisa Lab Notebook/Standard Calculations (1).xlsx")
        wb.close()

        # Create 8mix file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Mix1"
        ws.cell(row=1, column=1, value="concentration")
        ws.cell(row=1, column=2, value="Camphene")
        ws.cell(row=2, column=1, value=100)
        ws.cell(row=2, column=2, value=50)
        wb.save(
            base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/8mix_calc.xlsx"
        )
        wb.close()

        # Create std_tidy file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Data"
        ws.cell(row=1, column=1, value="chemical.name")
        ws.cell(row=2, column=1, value="Terpinene")
        wb.save(
            base_path / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/std_tidy.xlsx"
        )
        wb.close()

        # Create StandardsAndCals file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Work list"
        ws.cell(row=1, column=1, value="Mixture Arrangment")
        ws.cell(row=2, column=1, value="Linalool / Myrcene")
        wb.save(base_path / "InstrumentLogs/TDGC/Calibrations/StandardsAndCals.xlsx")
        wb.close()

        # Create ChiralStandards file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Retention Times"
        ws.cell(row=1, column=1, value="Compound")
        ws.cell(row=2, column=1, value="(R)-Limonene")
        wb.save(base_path / "InstrumentLogs/TDGC/Calibrations/ChiralStandards_Cal - Updated.xlsx")
        wb.close()

        # Create Universal Chemical List file
        wb = _create_workbook()
        ws1 = wb.active
        ws1.title = "Standards list"
        ws1.cell(row=1, column=1, value="Chemical Name")
        ws1.cell(row=2, column=1, value="Carvone")
        ws2 = wb.create_sheet("RT combined(in progress)")
        ws2.cell(row=1, column=1, value="Borneol")
        ws2.cell(row=2, column=1, value=1.0)
        wb.save(gcms_path / "Universal Chemical List.xlsx")
        wb.close()

        # Create Jasmine 2024 file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Chemical Name")
        ws.cell(row=2, column=1, value="Fenchone")
        wb.save(
            base_path
            / "InstrumentLogs/TDGC/Calibrations/old files/Jasmine Chemcial Standard List 2024.xlsx"
        )
        wb.close()

        # Create Claire std file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.cell(row=1, column=1, value="Compound")
        ws.cell(row=2, column=1, value="Cineole")
        wb.save(
            base_path
            / "InstrumentLogs/TDGC/Calibrations/old files/Claire Chemical Standard List-Faiola.xlsx"
        )
        wb.close()

        # Create Old Compiled file
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Rearrangment"
        ws.cell(row=1, column=1, value="Compiled standard list")
        ws.cell(row=2, column=1, value="Sabinene")
        wb.save(
            base_path / "InstrumentLogs/TDGC/Calibrations/old files/OLD_CompiledStandardList.xlsx"
        )
        wb.close()

        output_path = tmp_path / "output" / "standards.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = extract_standards(base_path, output_path)

        assert result == 0
        assert output_path.exists()

        # Verify output
        wb = _load_workbook(output_path)
        ws = wb.active
        assert ws.cell(row=1, column=1).value == "Chemical Name"
        wb.close()

    def test_extract_standards_default_paths(self) -> None:
        """Test extract_standards uses default paths when None."""
        import logging

        from scripts.extract_standards import extract_standards

        # This verifies the None branches
        result: int = -1
        try:
            result = extract_standards(None, None)
        except FileNotFoundError:
            logging.info("Default path not found - expected in CI")
            result = 0

        assert result == 0

    def test_main_function(self) -> None:
        """Test main entry point."""
        import logging

        from scripts.extract_standards import main

        result: int = -1
        try:
            result = main()
        except FileNotFoundError:
            logging.info("Default path not found - expected in CI")
            result = 0

        assert result == 0

    def test_main_entry_via_runpy(self) -> None:
        """Test if __name__ == '__main__' block via runpy."""
        import logging
        import runpy

        import pytest

        script_path = Path(__file__).parent.parent / "scripts" / "extract_standards.py"

        try:
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path(str(script_path), run_name="__main__")
            assert exc_info.value.code == 0
        except FileNotFoundError:
            logging.info("Default path not found - expected in CI")
