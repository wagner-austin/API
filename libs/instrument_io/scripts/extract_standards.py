"""Extract chemical standards from multiple Excel sources.

This script scans various Excel files to build a consolidated master list
of chemical standards used in the lab.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from platform_core.logging import get_logger, setup_logging

from instrument_io._protocols.openpyxl import (
    _auto_adjust_column_widths,
    _create_table,
    _create_workbook,
)

logger = get_logger(__name__)


class StandardEntry(TypedDict):
    """Entry for a chemical standard."""

    chemical_name: str
    source: str
    date: str
    type_: str
    details: str


class FileStats(TypedDict):
    """Statistics for a processed file."""

    sheets: int
    extracted: int


def _deduplicate_headers(headers: list[str]) -> list[str]:
    """Ensure column names are unique for Polars.

    Args:
        headers: List of header names

    Returns:
        List of unique header names
    """
    counts: dict[str, int] = {}
    new_headers: list[str] = []
    for h in headers:
        original = str(h)
        if original in counts:
            counts[original] += 1
            new_headers.append(f"{original}_{counts[original]}")
        else:
            counts[original] = 0
            new_headers.append(original)
    return new_headers


class StandardsExtractor:
    """Extracts chemical standards from Excel files."""

    def __init__(self) -> None:
        """Initialize the extractor."""
        self._standards_list: list[StandardEntry] = []
        self._seen_names: set[str] = set()
        self._file_stats: dict[str, FileStats] = {}

    @property
    def standards_list(self) -> list[StandardEntry]:
        """Get the list of extracted standards."""
        return self._standards_list

    @property
    def file_stats(self) -> dict[str, FileStats]:
        """Get file processing statistics."""
        return self._file_stats

    def _is_valid_chemical_name(self, name: str) -> bool:
        """Check if a name is a valid chemical name.

        Args:
            name: Chemical name to validate

        Returns:
            True if valid, False otherwise
        """
        if not name or not isinstance(name, str) or len(name.strip()) < 3:
            return False

        name_lower = name.lower()

        # Skip obvious non-chemical entries
        skip_exact = [
            "null",
            "none",
            "na",
            "total",
            "rt",
            "id",
            "code",
            "date",
            "column",
            "sheet",
            "nan",
            "area",
            "mass",
            "sample",
            "control",
            "chemical name",
            "compound",
            "name",
            "standard",
            "injected volume",
            "response factor",
            "relative to",
            "concentration",
            "cartridge",
            "point",
            "volume",
            "slope",
            "calc mass",
            "int area",
            "peak area",
            "1ul",
            "2ul",
            "other",
            "monoterpene",
            "monoterpenoid",
            "sesquiterpene",
            "alkane",
            "benzoic-acid",
            "notes",
            "achieved",
            "min area",
            "max area",
            "standard ran? (y/n)",
            "compound rf",
            "ana notes",
            "claire notes",
            "reference compound",
            "vial label",
            "dilute",
            "concentrate",
            "mixture",
        ]
        if name_lower in skip_exact:
            return False

        # Skip if starts with these
        skip_startswith = [
            "sample",
            "samplw",
            "tic:",
            "col-",
            "column-",
            "relative to",
            "standard volume",
            "unknown",
            "mt",
            "omt",
            "sqt",
            "osqt",
        ]
        if any(name_lower.startswith(s) for s in skip_startswith):
            return False

        # Skip if contains these substrings
        skip_contains = ["\\data-ms", "-d\\", "injected", "response factor", " and u", "(y/n)"]
        if any(s in name_lower for s in skip_contains):
            return False

        # Skip pure numbers
        if re.match(r"^-?\d+[-\d]*$", name):
            return False

        # Skip formulas/equations
        if re.search(r"\*x\s*\+", name):
            return False

        # Skip very long entries
        return len(name) <= 80

    def _normalize_name(self, name: str) -> str:
        """Normalize a chemical name for deduplication.

        Args:
            name: Chemical name to normalize

        Returns:
            Normalized name
        """
        result = name.lower()
        result = re.sub(r"[\s\-\,\.\[\]\(\)]+", "", result)
        result = re.sub(r"^(alpha|a|α)", "alpha", result)
        result = re.sub(r"^(beta|b|β)", "beta", result)
        result = re.sub(r"^(gamma|g|y|γ)", "gamma", result)
        result = re.sub(r"^r\+", "", result)
        result = re.sub(r"^s\+", "", result)
        result = re.sub(r"^\+", "", result)
        result = re.sub(r"^\-", "", result)
        result = re.sub(r"^\+/\-", "", result)
        result = re.sub(r"^\?", "", result)
        result = re.sub(r"^cis", "", result)
        return re.sub(r"^trans", "", result)

    def _clean_display_name(self, name: str) -> str:
        """Clean up display name with Greek letter prefixes.

        Args:
            name: Chemical name to clean

        Returns:
            Cleaned display name
        """
        display_name = name
        if re.match(r"^alpha[\s\-]", display_name, re.I):
            display_name = "α-" + re.sub(r"^alpha[\s\-]+", "", display_name, flags=re.I)
        elif re.match(r"^a-", display_name, re.I):
            display_name = "α-" + display_name[2:]
        elif re.match(r"^beta[\s\-]", display_name, re.I):
            display_name = "β-" + re.sub(r"^beta[\s\-]+", "", display_name, flags=re.I)
        elif re.match(r"^b-", display_name, re.I):
            display_name = "β-" + display_name[2:]
        elif re.match(r"^gamma[\s\-]", display_name, re.I):
            display_name = "γ-" + re.sub(r"^gamma[\s\-]+", "", display_name, flags=re.I)
        elif re.match(r"^y-", display_name, re.I):
            display_name = "γ-" + display_name[2:]

        # Capitalize first letter after Greek prefix
        if display_name.startswith(("α-", "β-", "γ-")):
            prefix = display_name[:2]
            rest = display_name[2:]
            display_name = prefix + rest[0].upper() + rest[1:] if rest else prefix
        else:
            display_name = (
                display_name[0].upper() + display_name[1:] if display_name else display_name
            )

        return display_name

    def add_standard(
        self,
        name: str | None,
        source: str,
        date: str,
        type_: str,
        details: str,
    ) -> bool:
        """Add a standard to the list if valid.

        Args:
            name: Chemical name
            source: Source file/sheet
            date: Date of file modification
            type_: Type of standard
            details: Additional details

        Returns:
            True if added, False otherwise
        """
        if not name or not isinstance(name, str):
            return False

        name = name.strip()

        # Strip R-style X prefix
        if name.startswith("X") and len(name) > 1 and (name[1].isdigit() or name[1] == "."):
            name = name[1:].lstrip(".")

        # Convert dots and underscores to hyphens
        name = name.replace(".", "-").replace("_", "-")
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")

        if not self._is_valid_chemical_name(name):
            return False

        norm_name = self._normalize_name(name)
        if norm_name in self._seen_names:
            return False

        self._seen_names.add(norm_name)
        display_name = self._clean_display_name(name)

        self._standards_list.append(
            StandardEntry(
                chemical_name=display_name,
                source=source,
                date=date,
                type_=type_,
                details=details,
            )
        )
        return True

    def _get_file_date(self, path: Path) -> str:
        """Get file modification date.

        Args:
            path: Path to file

        Returns:
            Date string in YYYY-MM-DD format
        """
        mod_time = os.path.getmtime(path)
        return datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")

    def write_output(self, output_path: Path) -> None:
        """Write extracted standards to Excel file.

        Args:
            output_path: Output file path

        Raises:
            PermissionError: If file is open or cannot be written
        """
        wb = _create_workbook()
        ws = wb.active
        ws.title = "Chemical Standards"

        headers = ["Chemical Name", "Source", "Date", "Type", "Details"]
        for col_idx, header in enumerate(headers, 1):
            ws.cell(row=1, column=col_idx, value=header)

        for row_idx, entry in enumerate(self._standards_list, 2):
            ws.cell(row=row_idx, column=1, value=entry["chemical_name"])
            ws.cell(row=row_idx, column=2, value=entry["source"])
            ws.cell(row=row_idx, column=3, value=entry["date"])
            ws.cell(row=row_idx, column=4, value=entry["type_"])
            ws.cell(row=row_idx, column=5, value=entry["details"])

        last_row = len(self._standards_list) + 1
        if last_row > 1:
            tab = _create_table(
                display_name="ChemicalStandards2025",
                ref=f"A1:E{last_row}",
                style_name="TableStyleMedium9",
                show_row_stripes=True,
            )
            ws.add_table(tab)

        _auto_adjust_column_widths(ws, max_width=60, padding=2)

        wb.save(output_path)
        logger.info("Saved %d unique standards to: %s", len(self._standards_list), output_path)

    def log_summary(self) -> None:
        """Log extraction summary."""
        logger.info("=== Extraction Summary ===")
        total_sheets = 0
        total_extracted = 0
        for file_name, stats in self._file_stats.items():
            logger.info(
                "  %s: %d sheets, %d extracted", file_name, stats["sheets"], stats["extracted"]
            )
            total_sheets += stats["sheets"]
            total_extracted += stats["extracted"]
        logger.info(
            "TOTAL: %d sheets, %d unique standards", total_sheets, len(self._standards_list)
        )


DEFAULT_BASE_PATH = Path("C:/Users/austi/PROJECTS/UC Irvine/Celia Louise Braun Faiola - FaiolaLab")


def extract_standards(
    base_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    """Extract standards from all configured files.

    Args:
        base_path: Base path for input files (uses default if None)
        output_path: Output file path (uses default if None)

    Returns:
        Exit code (0 for success)
    """
    if base_path is None:
        base_path = DEFAULT_BASE_PATH

    files = {
        "Response Factors": base_path
        / "Notebooks/Jasmine OseiEnin Lab Notebook/Response factors.xlsx",
        "Soil VOC": base_path / "Current Projects/Soil VOC quantitation.xlsx",
        "Avisa Calc": base_path / "Notebooks/Avisa Lab Notebook/Standard Calculations (1).xlsx",
        "8mix": base_path
        / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/8mix_calc.xlsx",
        "Std Tidy": base_path
        / "Notebooks/Jasmine OseiEnin Lab Notebook/2023-2024/Summer 24/std_tidy.xlsx",
        "StandardsAndCals": base_path / "InstrumentLogs/TDGC/Calibrations/StandardsAndCals.xlsx",
        "ChiralStandards": base_path
        / "InstrumentLogs/TDGC/Calibrations/ChiralStandards_Cal - Updated.xlsx",
        "UniversalList": (
            base_path
            / "Current Projects/Thermal Stress Project"
            / "2021-2022 BVOC collection experiment (Juan)/GCMS data/Universal Chemical List.xlsx"
        ),
        "Jasmine2024": base_path
        / "InstrumentLogs/TDGC/Calibrations/old files/Jasmine Chemcial Standard List 2024.xlsx",
        "ClaireStd": base_path
        / "InstrumentLogs/TDGC/Calibrations/old files/Claire Chemical Standard List-Faiola.xlsx",
        "OldCompiled": base_path
        / "InstrumentLogs/TDGC/Calibrations/old files/OLD_CompiledStandardList.xlsx",
    }

    if output_path is None:
        output_path = (
            base_path / "Notebooks/Emily Truong Notebook/Chemical_Standards_List_2025.xlsx"
        )

    logger.info("=== Chemical Standards Extraction ===")

    from scripts.extract_standards_sources import (
        process_8mix,
        process_avisa_calc,
        process_chiral_standards,
        process_claire_std,
        process_jasmine_2024,
        process_old_compiled,
        process_response_factors,
        process_soil_voc,
        process_standards_and_cals,
        process_std_tidy,
        process_universal_list,
    )

    extractor = StandardsExtractor()

    # Process all files
    process_response_factors(extractor, files["Response Factors"])
    process_soil_voc(extractor, files["Soil VOC"])
    process_avisa_calc(extractor, files["Avisa Calc"])
    process_8mix(extractor, files["8mix"])
    process_std_tidy(extractor, files["Std Tidy"])
    process_standards_and_cals(extractor, files["StandardsAndCals"])
    process_chiral_standards(extractor, files["ChiralStandards"])
    process_universal_list(extractor, files["UniversalList"])
    process_jasmine_2024(extractor, files["Jasmine2024"])
    process_claire_std(extractor, files["ClaireStd"])
    process_old_compiled(extractor, files["OldCompiled"])

    extractor.log_summary()
    extractor.write_output(output_path)

    return 0


def main() -> int:
    """Entry point for script."""
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="extract-standards",
        instance_id=None,
        extra_fields=None,
    )
    return extract_standards()


if __name__ == "__main__":
    raise SystemExit(main())
