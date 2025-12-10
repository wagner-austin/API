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

import polars as pl
from platform_core.logging import get_logger, setup_logging

from instrument_io._json_bridge import (
    _df_get_cell_str,
    _df_get_row_values,
    _df_json_to_row_dicts,
    _df_slice_to_rows,
    _get_json_str_value,
    _json_col_to_str_list,
)
from instrument_io._protocols.openpyxl import (
    _auto_adjust_column_widths,
    _create_table,
    _create_workbook,
    _load_workbook,
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

    def _process_response_factors(self, file_path: Path) -> None:
        """Process Response Factors Excel file."""
        logger.info("1. Response Factors: %s", file_path.name)
        self._file_stats["Response Factors"] = FileStats(sheets=0, extracted=0)

        rf_date = self._get_file_date(file_path)
        wb_rf = _load_workbook(file_path, read_only=True)
        rf_sheet_names = wb_rf.sheetnames
        wb_rf.close()
        self._file_stats["Response Factors"]["sheets"] = len(rf_sheet_names)

        for sheet_name in rf_sheet_names:
            count_before = len(self._standards_list)
            df_rf = pl.read_excel(
                source=file_path, sheet_name=sheet_name, engine="openpyxl", has_header=True
            )

            # Find chemical name column
            chem_col: str | None = None
            for col in df_rf.columns:
                col_lower = col.lower()
                if "chemical" in col_lower and "name" in col_lower:
                    chem_col = col
                    break
                if col_lower in ("name", "compound"):
                    chem_col = col

            if chem_col:
                for row in _df_json_to_row_dicts(df_rf.write_json()):
                    chem = _get_json_str_value(row, chem_col)
                    if chem:
                        density = _get_json_str_value(row, "Density (g/mL)") or _get_json_str_value(
                            row, "Density"
                        )
                        details = f"Density: {density}" if density else f"Sheet: {sheet_name}"
                        self.add_standard(
                            chem, "Jasmine - Response Factors", rf_date, "Standard Mix", details
                        )

            count_after = len(self._standards_list)
            extracted = count_after - count_before
            self._file_stats["Response Factors"]["extracted"] += extracted
            logger.info("    %s: %d chemicals", sheet_name, extracted)

    def _process_soil_voc(self, file_path: Path) -> None:
        """Process Soil VOC Quantitation Excel file."""
        logger.info("2. Soil VOC: %s", file_path.name)
        self._file_stats["Soil VOC"] = FileStats(sheets=0, extracted=0)

        soil_date = self._get_file_date(file_path)
        wb_soil = _load_workbook(file_path, read_only=True)
        soil_sheet_names = wb_soil.sheetnames
        wb_soil.close()
        self._file_stats["Soil VOC"]["sheets"] = len(soil_sheet_names)

        for sheet_name in soil_sheet_names:
            count_before = len(self._standards_list)
            extracted_chems: set[str] = set()

            df_raw = pl.read_excel(
                source=file_path,
                sheet_name=sheet_name,
                engine="openpyxl",
                has_header=False,
                infer_schema_length=None,
            )

            # Strategy 1: Known structured sheets
            if sheet_name in ["Standard list", "compound_colors (2)", "compound_colors", "Sheet1"]:
                df_clean = pl.read_excel(
                    source=file_path, sheet_name=sheet_name, engine="openpyxl", has_header=True
                )
                for col in ["name", "compound", "Name", "Compound"]:
                    if col in df_clean.columns:
                        for val in _json_col_to_str_list(df_clean.select(col).write_json(), col):
                            extracted_chems.add(val.strip())

            # Strategy 2: Find header row and extract from columns
            if not extracted_chems:
                header_row_idx = -1
                df_raw_json = df_raw.write_json()
                for r_idx in range(min(df_raw.height, 5)):
                    row_vals = [
                        v.strip().lower() for v in _df_get_row_values(df_raw_json, r_idx) if v
                    ]
                    if any(
                        kw in val
                        for val in row_vals
                        for kw in ["compound", "name", "pinene", "terpene", "alkane"]
                    ):
                        header_row_idx = r_idx
                        break

                if header_row_idx != -1:
                    raw_headers = _df_get_row_values(df_raw_json, header_row_idx)
                    headers = [
                        val.strip() if val.strip() else f"col_{i}"
                        for i, val in enumerate(raw_headers)
                    ]
                    headers = _deduplicate_headers(headers)

                    data_rows = _df_slice_to_rows(df_raw_json, header_row_idx + 1)
                    df_processed = pl.DataFrame(
                        data_rows,
                        schema=headers,
                        orient="row",
                        infer_schema_length=None,
                    )

                    # Extract from column names
                    for col in df_processed.columns:
                        col_lower = col.lower()
                        if any(
                            kw in col_lower
                            for kw in ["pinene", "terpene", "limonene", "alkane", "cyclo"]
                        ):
                            extracted_chems.add(col.split("(")[0].strip())

                    # Extract from name/compound columns
                    for col in df_processed.columns:
                        if col.lower() in ["compound", "name", "chemical name", "analyte"]:
                            for val in _json_col_to_str_list(
                                df_processed.select(col).write_json(), col
                            ):
                                if val and len(val.strip()) > 2:
                                    extracted_chems.add(val.strip())

            for chem in extracted_chems:
                self.add_standard(
                    chem, "Soil VOC Project", soil_date, "Standard", f"Sheet: {sheet_name}"
                )

            count_after = len(self._standards_list)
            extracted = count_after - count_before
            self._file_stats["Soil VOC"]["extracted"] += extracted
            logger.info("    %s: %d chemicals", sheet_name, extracted)

    def _process_avisa_calc(self, file_path: Path) -> None:
        """Process Avisa Standard Calculations Excel file."""
        logger.info("3. Avisa Calc: %s", file_path.name)
        self._file_stats["Avisa Calc"] = FileStats(sheets=0, extracted=0)

        avisa_date = self._get_file_date(file_path)
        wb_avisa = _load_workbook(file_path, read_only=True)
        avisa_sheet_names = wb_avisa.sheetnames
        wb_avisa.close()
        self._file_stats["Avisa Calc"]["sheets"] = len(avisa_sheet_names)

        for sheet_name in avisa_sheet_names:
            count_before = len(self._standards_list)
            df_raw = pl.read_excel(
                source=file_path,
                sheet_name=sheet_name,
                engine="openpyxl",
                has_header=False,
                infer_schema_length=None,
            )
            df_raw_json = df_raw.write_json()

            # Check first cell for chemical name
            first_col = df_raw.columns[0]
            first_cell = _df_get_cell_str(df_raw_json, 0, first_col)
            if first_cell and any(
                kw in first_cell.lower()
                for kw in ["limonene", "pinene", "camphor", "terpene", "linalool", "eucalyptol"]
            ):
                self.add_standard(
                    first_cell.split("(")[0].strip(),
                    "Avisa - Standard Calculations",
                    avisa_date,
                    "Calculated Standard",
                    f"Sheet: {sheet_name}",
                )

            # Find header row
            header_row_idx = -1
            if df_raw.height > 1:
                for r_idx in range(min(df_raw.height, 5)):
                    row_vals = [
                        v.strip().lower() for v in _df_get_row_values(df_raw_json, r_idx) if v
                    ]
                    if any(
                        kw in val
                        for val in row_vals
                        for kw in ["compound", "name", "standard", "analyte"]
                    ):
                        header_row_idx = r_idx
                        break

            if header_row_idx != -1:
                raw_headers = _df_get_row_values(df_raw_json, header_row_idx)
                headers = [
                    val.strip() if val.strip() else f"col_{i}" for i, val in enumerate(raw_headers)
                ]
                headers = _deduplicate_headers(headers)

                data_rows = _df_slice_to_rows(df_raw_json, header_row_idx + 1)
                df_processed = pl.DataFrame(
                    data_rows,
                    schema=headers,
                    orient="row",
                    infer_schema_length=None,
                )

                for col in df_processed.columns:
                    if any(kw in col.lower() for kw in ["compound", "name", "standard", "analyte"]):
                        for val in _json_col_to_str_list(
                            df_processed.select(col).write_json(), col
                        ):
                            if val and len(val.strip()) > 2:
                                self.add_standard(
                                    val.strip(),
                                    "Avisa - Standard Calculations",
                                    avisa_date,
                                    "Calculated Standard",
                                    f"Sheet: {sheet_name}",
                                )

            count_after = len(self._standards_list)
            extracted = count_after - count_before
            self._file_stats["Avisa Calc"]["extracted"] += extracted
            logger.info("    %s: %d chemicals", sheet_name, extracted)

    def _process_8mix(self, file_path: Path) -> None:
        """Process 8mix_calc Excel file."""
        logger.info("4. 8mix: %s", file_path.name)
        self._file_stats["8mix"] = FileStats(sheets=0, extracted=0)

        mix_date = self._get_file_date(file_path)
        wb_8mix = _load_workbook(file_path, read_only=True)
        mix_sheet_names = wb_8mix.sheetnames
        wb_8mix.close()
        self._file_stats["8mix"]["sheets"] = len(mix_sheet_names)

        for sheet_name in mix_sheet_names:
            count_before = len(self._standards_list)
            df_raw = pl.read_excel(
                source=file_path,
                sheet_name=sheet_name,
                engine="openpyxl",
                has_header=False,
                infer_schema_length=None,
            )
            df_raw_json = df_raw.write_json()

            # Find header row with "concentration"
            header_row_idx = -1
            for r_idx in range(min(df_raw.height, 5)):
                row_vals = [v.strip().lower() for v in _df_get_row_values(df_raw_json, r_idx) if v]
                if "concentration" in row_vals:
                    header_row_idx = r_idx
                    break

            if header_row_idx != -1:
                raw_headers = _df_get_row_values(df_raw_json, header_row_idx)
                headers = [
                    val.strip() if val.strip() else f"col_{i}" for i, val in enumerate(raw_headers)
                ]
                headers = _deduplicate_headers(headers)

                skip_cols = [
                    "concentration",
                    "standard",
                    "cartridge",
                    "slope",
                    "rt",
                    "calc mass",
                    "column",
                ]
                for col in headers:
                    col_lower = col.lower()
                    if (
                        not any(skip in col_lower for skip in skip_cols)
                        and not col_lower.endswith("_1")
                        and len(col.strip()) > 2
                    ):
                        self.add_standard(
                            col.split("(")[0].strip(),
                            "Avisa - 8mix",
                            mix_date,
                            "8-Mix Component",
                            f"Sheet: {sheet_name}",
                        )
            else:
                # Scan for known chemical names
                for r_idx in range(min(df_raw.height, 5)):
                    for val in _df_get_row_values(df_raw_json, r_idx):
                        if val and any(
                            kw in val.lower()
                            for kw in [
                                "pinene",
                                "terpene",
                                "limonene",
                                "thujone",
                                "linalool",
                                "eucalyptol",
                                "myrcene",
                            ]
                        ):
                            self.add_standard(
                                val.split("(")[0].strip(),
                                "Avisa - 8mix",
                                mix_date,
                                "8-Mix Component",
                                f"Sheet: {sheet_name}",
                            )

            count_after = len(self._standards_list)
            extracted = count_after - count_before
            self._file_stats["8mix"]["extracted"] += extracted
            logger.info("    %s: %d chemicals", sheet_name, extracted)

    def _process_std_tidy(self, file_path: Path) -> None:
        """Process std_tidy Excel file."""
        logger.info("5. Std Tidy: %s", file_path.name)
        self._file_stats["Std Tidy"] = FileStats(sheets=0, extracted=0)

        tidy_date = self._get_file_date(file_path)
        wb_tidy = _load_workbook(file_path, read_only=True)
        tidy_sheet_names = wb_tidy.sheetnames
        wb_tidy.close()
        self._file_stats["Std Tidy"]["sheets"] = len(tidy_sheet_names)

        for sheet_name in tidy_sheet_names:
            count_before = len(self._standards_list)
            extracted_chems: set[str] = set()

            df_raw = pl.read_excel(
                source=file_path,
                sheet_name=sheet_name,
                engine="openpyxl",
                has_header=False,
                infer_schema_length=None,
            )

            # Check for chemical.name column
            df_with_header = pl.read_excel(
                source=file_path, sheet_name=sheet_name, engine="openpyxl", has_header=True
            )
            for col in df_with_header.columns:
                if "chemical" in col.lower() and "name" in col.lower():
                    for val in _json_col_to_str_list(df_with_header.select(col).write_json(), col):
                        clean_name = val.replace(".", "-").strip()
                        extracted_chems.add(clean_name)

            # Extract from column headers
            df_raw_json = df_raw.write_json()
            header_row = _df_get_row_values(df_raw_json, 0)
            for val in header_row:
                if val:
                    val_lower = val.lower()
                    if any(
                        kw in val_lower
                        for kw in [
                            "pinene",
                            "terpene",
                            "myrcene",
                            "linalool",
                            "eucalyptol",
                            "thujone",
                        ]
                    ):
                        name = val.split("(")[0].split("Int")[0].split("mass")[0].strip()
                        if len(name) > 2:
                            extracted_chems.add(name)

            for chem in extracted_chems:
                self.add_standard(
                    chem,
                    "Avisa - Tidy Standards",
                    tidy_date,
                    "Standard Mix Component",
                    f"Sheet: {sheet_name}",
                )

            count_after = len(self._standards_list)
            extracted = count_after - count_before
            self._file_stats["Std Tidy"]["extracted"] += extracted
            logger.info("    %s: %d chemicals", sheet_name, extracted)

    def _process_standards_and_cals(self, file_path: Path) -> None:
        """Process StandardsAndCals Excel file."""
        logger.info("6. StandardsAndCals: %s", file_path.name)
        self._file_stats["StandardsAndCals"] = FileStats(sheets=0, extracted=0)

        sc_date = self._get_file_date(file_path)
        df_wl = pl.read_excel(
            source=file_path, sheet_name="Work list", engine="openpyxl", has_header=True
        )

        mix_col: str | None = None
        for col in df_wl.columns:
            if "mixture" in col.lower() or "arrangment" in col.lower():
                mix_col = col
                break

        if mix_col:
            count_before = len(self._standards_list)
            for val in _json_col_to_str_list(df_wl.select(mix_col).write_json(), mix_col):
                if val:
                    parts = val.split("/")
                    for part in parts:
                        cleaned = part.strip()
                        val_truncated = val[:30] + "..." if len(val) > 30 else val
                        self.add_standard(
                            cleaned,
                            "StandardsAndCals - Work list",
                            sc_date,
                            "Mix Component",
                            f"From mix: {val_truncated}",
                        )

            extracted = len(self._standards_list) - count_before
            self._file_stats["StandardsAndCals"]["extracted"] += extracted
            self._file_stats["StandardsAndCals"]["sheets"] = 1
            logger.info("    Work list: %d chemicals", extracted)

    def _process_chiral_standards(self, file_path: Path) -> None:
        """Process ChiralStandards Excel file."""
        logger.info("7. ChiralStandards: %s", file_path.name)
        self._file_stats["ChiralStandards"] = FileStats(sheets=0, extracted=0)

        chiral_date = self._get_file_date(file_path)
        df_rt = pl.read_excel(
            source=file_path, sheet_name="Retention Times", engine="openpyxl", has_header=True
        )

        count_before = len(self._standards_list)
        if "Compound" in df_rt.columns:
            for val in _json_col_to_str_list(df_rt.select("Compound").write_json(), "Compound"):
                if val:
                    self.add_standard(
                        val,
                        "ChiralStandards - RT",
                        chiral_date,
                        "Chiral Standard",
                        "Retention Times Sheet",
                    )

        extracted = len(self._standards_list) - count_before
        self._file_stats["ChiralStandards"]["extracted"] += extracted
        self._file_stats["ChiralStandards"]["sheets"] = 1
        logger.info("    Retention Times: %d chemicals", extracted)

    def _process_universal_list(self, file_path: Path) -> None:
        """Process Universal Chemical List Excel file."""
        logger.info("8. UniversalList: %s", file_path.name)
        self._file_stats["UniversalList"] = FileStats(sheets=0, extracted=0)

        univ_date = self._get_file_date(file_path)

        # Standards list sheet
        df_std = pl.read_excel(
            source=file_path, sheet_name="Standards list", engine="openpyxl", has_header=True
        )
        count_before = len(self._standards_list)
        col_name = next((c for c in df_std.columns if "chemical" in c.lower()), None)
        if col_name:
            for val in _json_col_to_str_list(df_std.select(col_name).write_json(), col_name):
                self.add_standard(
                    val,
                    "UniversalList - Standards",
                    univ_date,
                    "Standard",
                    "Standards list sheet",
                )

        extracted = len(self._standards_list) - count_before
        self._file_stats["UniversalList"]["extracted"] += extracted
        logger.info("    Standards list: %d chemicals", extracted)

        # RT combined Sheet
        df_rt = pl.read_excel(
            source=file_path,
            sheet_name="RT combined(in progress)",
            engine="openpyxl",
            has_header=True,
        )
        count_before = len(self._standards_list)
        for col in df_rt.columns:
            self.add_standard(
                col, "UniversalList - RT Combined", univ_date, "Tracked Compound", "Column Header"
            )

        extracted = len(self._standards_list) - count_before
        self._file_stats["UniversalList"]["extracted"] += extracted
        self._file_stats["UniversalList"]["sheets"] = 2
        logger.info("    RT combined: %d chemicals", extracted)

    def _process_jasmine_2024(self, file_path: Path) -> None:
        """Process Jasmine 2024 Chemical Standard List."""
        logger.info("9. Jasmine2024: %s", file_path.name)
        self._file_stats["Jasmine2024"] = FileStats(sheets=0, extracted=0)

        jas_date = self._get_file_date(file_path)
        df_j = pl.read_excel(
            source=file_path, sheet_name="Sheet1", engine="openpyxl", has_header=True
        )

        count_before = len(self._standards_list)
        col_name = next((c for c in df_j.columns if "chemical" in c.lower()), None)
        if col_name:
            for val in _json_col_to_str_list(df_j.select(col_name).write_json(), col_name):
                self.add_standard(val, "Jasmine 2024 List", jas_date, "Standard", "Sheet1")

        extracted = len(self._standards_list) - count_before
        self._file_stats["Jasmine2024"]["extracted"] += extracted
        self._file_stats["Jasmine2024"]["sheets"] = 1
        logger.info("    Sheet1: %d chemicals", extracted)

    def _process_claire_std(self, file_path: Path) -> None:
        """Process Claire Chemical Standard List."""
        logger.info("10. ClaireStd: %s", file_path.name)
        self._file_stats["ClaireStd"] = FileStats(sheets=0, extracted=0)

        claire_date = self._get_file_date(file_path)
        df_c = pl.read_excel(
            source=file_path, sheet_name="Sheet1", engine="openpyxl", has_header=True
        )

        count_before = len(self._standards_list)
        if "Compound" in df_c.columns:
            for val in _json_col_to_str_list(df_c.select("Compound").write_json(), "Compound"):
                self.add_standard(val, "Claire Faiola List", claire_date, "Standard", "Sheet1")

        extracted = len(self._standards_list) - count_before
        self._file_stats["ClaireStd"]["extracted"] += extracted
        self._file_stats["ClaireStd"]["sheets"] = 1
        logger.info("    Sheet1: %d chemicals", extracted)

    def _process_old_compiled(self, file_path: Path) -> None:
        """Process OLD_CompiledStandardList Excel file."""
        logger.info("11. OldCompiled: %s", file_path.name)
        self._file_stats["OldCompiled"] = FileStats(sheets=0, extracted=0)

        old_date = self._get_file_date(file_path)
        df_old = pl.read_excel(
            source=file_path, sheet_name="Rearrangment", engine="openpyxl", has_header=True
        )

        count_before = len(self._standards_list)
        col_name = "Compiled standard list"
        if col_name in df_old.columns:
            for val in _json_col_to_str_list(df_old.select(col_name).write_json(), col_name):
                self.add_standard(
                    val,
                    "Old Compiled List",
                    old_date,
                    "Historical Standard",
                    "Rearrangment Sheet",
                )

        extracted = len(self._standards_list) - count_before
        self._file_stats["OldCompiled"]["extracted"] += extracted
        self._file_stats["OldCompiled"]["sheets"] = 1
        logger.info("    Rearrangment: %d chemicals", extracted)

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

    extractor = StandardsExtractor()

    # Process all files
    extractor._process_response_factors(files["Response Factors"])
    extractor._process_soil_voc(files["Soil VOC"])
    extractor._process_avisa_calc(files["Avisa Calc"])
    extractor._process_8mix(files["8mix"])
    extractor._process_std_tidy(files["Std Tidy"])
    extractor._process_standards_and_cals(files["StandardsAndCals"])
    extractor._process_chiral_standards(files["ChiralStandards"])
    extractor._process_universal_list(files["UniversalList"])
    extractor._process_jasmine_2024(files["Jasmine2024"])
    extractor._process_claire_std(files["ClaireStd"])
    extractor._process_old_compiled(files["OldCompiled"])

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
