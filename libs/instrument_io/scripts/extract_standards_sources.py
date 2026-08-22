"""Per-source parsers for the chemical standards extraction.

Each ``process_*`` function reads one spreadsheet family and feeds
the shared :class:`scripts.extract_standards.StandardsExtractor`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from platform_core.logging import get_logger

from instrument_io._json_bridge import (
    _df_get_cell_str,
    _df_get_row_values,
    _df_json_to_row_dicts,
    _df_slice_to_rows,
    _get_json_str_value,
    _json_col_to_str_list,
)
from instrument_io._protocols.openpyxl import (
    _load_workbook,
)
from scripts.extract_standards import (
    FileStats,
    StandardsExtractor,
    _deduplicate_headers,
)

logger = get_logger(__name__)


def process_response_factors(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process Response Factors Excel file."""
    logger.info("1. Response Factors: %s", file_path.name)
    extractor._file_stats["Response Factors"] = FileStats(sheets=0, extracted=0)

    rf_date = extractor._get_file_date(file_path)
    wb_rf = _load_workbook(file_path, read_only=True)
    rf_sheet_names = wb_rf.sheetnames
    wb_rf.close()
    extractor._file_stats["Response Factors"]["sheets"] = len(rf_sheet_names)

    for sheet_name in rf_sheet_names:
        count_before = len(extractor._standards_list)
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
                    extractor.add_standard(
                        chem, "Jasmine - Response Factors", rf_date, "Standard Mix", details
                    )

        count_after = len(extractor._standards_list)
        extracted = count_after - count_before
        extractor._file_stats["Response Factors"]["extracted"] += extracted
        logger.info("    %s: %d chemicals", sheet_name, extracted)


def process_soil_voc(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process Soil VOC Quantitation Excel file."""
    logger.info("2. Soil VOC: %s", file_path.name)
    extractor._file_stats["Soil VOC"] = FileStats(sheets=0, extracted=0)

    soil_date = extractor._get_file_date(file_path)
    wb_soil = _load_workbook(file_path, read_only=True)
    soil_sheet_names = wb_soil.sheetnames
    wb_soil.close()
    extractor._file_stats["Soil VOC"]["sheets"] = len(soil_sheet_names)

    for sheet_name in soil_sheet_names:
        count_before = len(extractor._standards_list)
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
                row_vals = [v.strip().lower() for v in _df_get_row_values(df_raw_json, r_idx) if v]
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
            extractor.add_standard(
                chem, "Soil VOC Project", soil_date, "Standard", f"Sheet: {sheet_name}"
            )

        count_after = len(extractor._standards_list)
        extracted = count_after - count_before
        extractor._file_stats["Soil VOC"]["extracted"] += extracted
        logger.info("    %s: %d chemicals", sheet_name, extracted)


def process_avisa_calc(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process Avisa Standard Calculations Excel file."""
    logger.info("3. Avisa Calc: %s", file_path.name)
    extractor._file_stats["Avisa Calc"] = FileStats(sheets=0, extracted=0)

    avisa_date = extractor._get_file_date(file_path)
    wb_avisa = _load_workbook(file_path, read_only=True)
    avisa_sheet_names = wb_avisa.sheetnames
    wb_avisa.close()
    extractor._file_stats["Avisa Calc"]["sheets"] = len(avisa_sheet_names)

    for sheet_name in avisa_sheet_names:
        count_before = len(extractor._standards_list)
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
            extractor.add_standard(
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
                row_vals = [v.strip().lower() for v in _df_get_row_values(df_raw_json, r_idx) if v]
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
                    for val in _json_col_to_str_list(df_processed.select(col).write_json(), col):
                        if val and len(val.strip()) > 2:
                            extractor.add_standard(
                                val.strip(),
                                "Avisa - Standard Calculations",
                                avisa_date,
                                "Calculated Standard",
                                f"Sheet: {sheet_name}",
                            )

        count_after = len(extractor._standards_list)
        extracted = count_after - count_before
        extractor._file_stats["Avisa Calc"]["extracted"] += extracted
        logger.info("    %s: %d chemicals", sheet_name, extracted)


def process_8mix(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process 8mix_calc Excel file."""
    logger.info("4. 8mix: %s", file_path.name)
    extractor._file_stats["8mix"] = FileStats(sheets=0, extracted=0)

    mix_date = extractor._get_file_date(file_path)
    wb_8mix = _load_workbook(file_path, read_only=True)
    mix_sheet_names = wb_8mix.sheetnames
    wb_8mix.close()
    extractor._file_stats["8mix"]["sheets"] = len(mix_sheet_names)

    for sheet_name in mix_sheet_names:
        count_before = len(extractor._standards_list)
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
                    extractor.add_standard(
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
                        extractor.add_standard(
                            val.split("(")[0].strip(),
                            "Avisa - 8mix",
                            mix_date,
                            "8-Mix Component",
                            f"Sheet: {sheet_name}",
                        )

        count_after = len(extractor._standards_list)
        extracted = count_after - count_before
        extractor._file_stats["8mix"]["extracted"] += extracted
        logger.info("    %s: %d chemicals", sheet_name, extracted)


def process_std_tidy(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process std_tidy Excel file."""
    logger.info("5. Std Tidy: %s", file_path.name)
    extractor._file_stats["Std Tidy"] = FileStats(sheets=0, extracted=0)

    tidy_date = extractor._get_file_date(file_path)
    wb_tidy = _load_workbook(file_path, read_only=True)
    tidy_sheet_names = wb_tidy.sheetnames
    wb_tidy.close()
    extractor._file_stats["Std Tidy"]["sheets"] = len(tidy_sheet_names)

    for sheet_name in tidy_sheet_names:
        count_before = len(extractor._standards_list)
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
            extractor.add_standard(
                chem,
                "Avisa - Tidy Standards",
                tidy_date,
                "Standard Mix Component",
                f"Sheet: {sheet_name}",
            )

        count_after = len(extractor._standards_list)
        extracted = count_after - count_before
        extractor._file_stats["Std Tidy"]["extracted"] += extracted
        logger.info("    %s: %d chemicals", sheet_name, extracted)


def process_standards_and_cals(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process StandardsAndCals Excel file."""
    logger.info("6. StandardsAndCals: %s", file_path.name)
    extractor._file_stats["StandardsAndCals"] = FileStats(sheets=0, extracted=0)

    sc_date = extractor._get_file_date(file_path)
    df_wl = pl.read_excel(
        source=file_path, sheet_name="Work list", engine="openpyxl", has_header=True
    )

    mix_col: str | None = None
    for col in df_wl.columns:
        if "mixture" in col.lower() or "arrangment" in col.lower():
            mix_col = col
            break

    if mix_col:
        count_before = len(extractor._standards_list)
        for val in _json_col_to_str_list(df_wl.select(mix_col).write_json(), mix_col):
            if val:
                parts = val.split("/")
                for part in parts:
                    cleaned = part.strip()
                    val_truncated = val[:30] + "..." if len(val) > 30 else val
                    extractor.add_standard(
                        cleaned,
                        "StandardsAndCals - Work list",
                        sc_date,
                        "Mix Component",
                        f"From mix: {val_truncated}",
                    )

        extracted = len(extractor._standards_list) - count_before
        extractor._file_stats["StandardsAndCals"]["extracted"] += extracted
        extractor._file_stats["StandardsAndCals"]["sheets"] = 1
        logger.info("    Work list: %d chemicals", extracted)


def process_chiral_standards(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process ChiralStandards Excel file."""
    logger.info("7. ChiralStandards: %s", file_path.name)
    extractor._file_stats["ChiralStandards"] = FileStats(sheets=0, extracted=0)

    chiral_date = extractor._get_file_date(file_path)
    df_rt = pl.read_excel(
        source=file_path, sheet_name="Retention Times", engine="openpyxl", has_header=True
    )

    count_before = len(extractor._standards_list)
    if "Compound" in df_rt.columns:
        for val in _json_col_to_str_list(df_rt.select("Compound").write_json(), "Compound"):
            if val:
                extractor.add_standard(
                    val,
                    "ChiralStandards - RT",
                    chiral_date,
                    "Chiral Standard",
                    "Retention Times Sheet",
                )

    extracted = len(extractor._standards_list) - count_before
    extractor._file_stats["ChiralStandards"]["extracted"] += extracted
    extractor._file_stats["ChiralStandards"]["sheets"] = 1
    logger.info("    Retention Times: %d chemicals", extracted)


def process_universal_list(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process Universal Chemical List Excel file."""
    logger.info("8. UniversalList: %s", file_path.name)
    extractor._file_stats["UniversalList"] = FileStats(sheets=0, extracted=0)

    univ_date = extractor._get_file_date(file_path)

    # Standards list sheet
    df_std = pl.read_excel(
        source=file_path, sheet_name="Standards list", engine="openpyxl", has_header=True
    )
    count_before = len(extractor._standards_list)
    col_name = next((c for c in df_std.columns if "chemical" in c.lower()), None)
    if col_name:
        for val in _json_col_to_str_list(df_std.select(col_name).write_json(), col_name):
            extractor.add_standard(
                val,
                "UniversalList - Standards",
                univ_date,
                "Standard",
                "Standards list sheet",
            )

    extracted = len(extractor._standards_list) - count_before
    extractor._file_stats["UniversalList"]["extracted"] += extracted
    logger.info("    Standards list: %d chemicals", extracted)

    # RT combined Sheet
    df_rt = pl.read_excel(
        source=file_path,
        sheet_name="RT combined(in progress)",
        engine="openpyxl",
        has_header=True,
    )
    count_before = len(extractor._standards_list)
    for col in df_rt.columns:
        extractor.add_standard(
            col, "UniversalList - RT Combined", univ_date, "Tracked Compound", "Column Header"
        )

    extracted = len(extractor._standards_list) - count_before
    extractor._file_stats["UniversalList"]["extracted"] += extracted
    extractor._file_stats["UniversalList"]["sheets"] = 2
    logger.info("    RT combined: %d chemicals", extracted)


def process_jasmine_2024(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process Jasmine 2024 Chemical Standard List."""
    logger.info("9. Jasmine2024: %s", file_path.name)
    extractor._file_stats["Jasmine2024"] = FileStats(sheets=0, extracted=0)

    jas_date = extractor._get_file_date(file_path)
    df_j = pl.read_excel(source=file_path, sheet_name="Sheet1", engine="openpyxl", has_header=True)

    count_before = len(extractor._standards_list)
    col_name = next((c for c in df_j.columns if "chemical" in c.lower()), None)
    if col_name:
        for val in _json_col_to_str_list(df_j.select(col_name).write_json(), col_name):
            extractor.add_standard(val, "Jasmine 2024 List", jas_date, "Standard", "Sheet1")

    extracted = len(extractor._standards_list) - count_before
    extractor._file_stats["Jasmine2024"]["extracted"] += extracted
    extractor._file_stats["Jasmine2024"]["sheets"] = 1
    logger.info("    Sheet1: %d chemicals", extracted)


def process_claire_std(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process Claire Chemical Standard List."""
    logger.info("10. ClaireStd: %s", file_path.name)
    extractor._file_stats["ClaireStd"] = FileStats(sheets=0, extracted=0)

    claire_date = extractor._get_file_date(file_path)
    df_c = pl.read_excel(source=file_path, sheet_name="Sheet1", engine="openpyxl", has_header=True)

    count_before = len(extractor._standards_list)
    if "Compound" in df_c.columns:
        for val in _json_col_to_str_list(df_c.select("Compound").write_json(), "Compound"):
            extractor.add_standard(val, "Claire Faiola List", claire_date, "Standard", "Sheet1")

    extracted = len(extractor._standards_list) - count_before
    extractor._file_stats["ClaireStd"]["extracted"] += extracted
    extractor._file_stats["ClaireStd"]["sheets"] = 1
    logger.info("    Sheet1: %d chemicals", extracted)


def process_old_compiled(extractor: StandardsExtractor, file_path: Path) -> None:
    """Process OLD_CompiledStandardList Excel file."""
    logger.info("11. OldCompiled: %s", file_path.name)
    extractor._file_stats["OldCompiled"] = FileStats(sheets=0, extracted=0)

    old_date = extractor._get_file_date(file_path)
    df_old = pl.read_excel(
        source=file_path, sheet_name="Rearrangment", engine="openpyxl", has_header=True
    )

    count_before = len(extractor._standards_list)
    col_name = "Compiled standard list"
    if col_name in df_old.columns:
        for val in _json_col_to_str_list(df_old.select(col_name).write_json(), col_name):
            extractor.add_standard(
                val,
                "Old Compiled List",
                old_date,
                "Historical Standard",
                "Rearrangment Sheet",
            )

    extracted = len(extractor._standards_list) - count_before
    extractor._file_stats["OldCompiled"]["extracted"] += extracted
    extractor._file_stats["OldCompiled"]["sheets"] = 1
    logger.info("    Rearrangment: %d chemicals", extracted)
