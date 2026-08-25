"""Build the ``metab_blank`` classification corpus from the Emily table.

The metabolomics-dashboard's blank analyses name the task: every
untargeted metabolomics run must separate real biological peaks from
contamination (solvent impurities, plasticizers, carryover) that blank
samples capture. The lab's standard decision rule — keep a peak when its
biological-sample average is at least three times its blank average, or
when it appears in samples only — IS the label. The corpus asks whether
that verdict is predictable from a peak's PHYSICOCHEMICAL properties
alone: m/z, charge, retention time, chromatographic peak width, and the
mass-defect signatures (plain and Kendrick/CH2) that contaminant
homolog series like plasticizers and PEG are known to carry.

Honesty rules, applied by construction:

- The label derives from the intensity columns, so NO intensity-derived
  quantity is a feature — not the sample or blank averages, not the
  workbook's Anova/q/fold-change/abundance/CV columns, nothing
  downstream of the numbers that define the verdict.
- ``Identifications`` (database hits) is excluded too: predictors are
  pre-annotation measurables only, as on every corpus in this family.
- The blank average uses the twelve INDIVIDUAL blank columns; the
  pooled ``250220_ebtruong_combine`` injection is excluded exactly as
  the other pooled columns in this family are. The dashboard's open
  leaf-vs-root blank-assignment dispute is sidestepped by using all
  blanks together (its documented option C), which no tissue
  assignment can change.
- ``Neutral mass (Da)`` is absent on 93% of rows (Progenesis assigns
  it only to deconvolved compounds) and is not a feature; m/z and
  charge carry the mass information for every row.
- Peaks detected in neither the biological samples nor the individual
  blanks are DROPPED and counted, never imputed.
- ``rt_bin`` (0.1-minute retention windows) is the GROUP column, never
  a feature: adducts and in-source fragments of one molecule co-elute,
  and a row-wise split would let the model see a compound's siblings
  across train and test.

Usage:
    poetry run python -m scripts.build_metab_blank_corpus \
        --workbook .../Emily_Data_Pruned_Labeled.xlsx \
        --out ../../services/covenant-radar-api/data/external/metab_blank/data.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import JSONValue, dump_json_str

from covenant_ml.datasets.xlsx_reader import read_xlsx_sheet
from scripts.build_metab_corpus import sha256_file

#: The worksheet the dashboard's blank analyses ran on.
SHEET_NAME = "Normalized"

#: The 23 biological sample columns (leaf and root; drought, ambient,
#: watered), by exact header.
BIOLOGICAL_COLUMNS = (
    "BL - Drought",
    "CL - Drought",
    "EL - Drought",
    "GL - Drought",
    "IL - Ambient",
    "JL - Ambient",
    "LL - Ambient",
    "ML - Ambient",
    "OL - Watered",
    "PL - Watered",
    "RL - Watered",
    "TL - Watered",
    "AR - Drought",
    "DR - Drought",
    "ER - Drought",
    "GR - Drought",
    "HR - Ambient",
    "IR - Ambient",
    "JR - Ambient",
    "MR - Ambient",
    "RR - Watered",
    "SR - Watered",
    "TR - Watered",
)

#: The twelve individual blank columns; the pooled combine is excluded.
BLANK_COLUMNS = (
    "250220_ebtruong_blank1",
    "250220_ebtruong_blank2",
    "250220_ebtruong_blank3",
    "250220_ebtruong_blank4",
    "BL2",
    "BL3",
    "BL4",
    "blank1_root",
    "blank2_root",
    "blank3_root",
    "Blk1",
    "Blk2",
)

#: Metadata headers every row must carry a value for.
REQUIRED_METADATA = ("Compound", "m/z", "Charge", "Retention time (min)")

#: The peak-width header (required present, value may repeat zeros).
PEAK_WIDTH_HEADER = "Chromatographic peak width (min)"

#: The lab's standard keep rule: sample average at least this multiple
#: of the blank average.
BLANK_RATIO = 3.0

#: Retention-time co-elution windows per minute (0.1-minute windows).
RT_BINS_PER_MINUTE = 10

#: Kendrick mass scaling for the CH2 base unit.
KENDRICK_CH2 = 14.0 / 14.01565

GROUP_COLUMN = "rt_bin"
TARGET_COLUMN = "real"

#: Feature columns in output order — physicochemical measurables only.
FEATURE_COLUMNS = (
    "mz",
    "charge",
    "rt_minutes",
    "peak_width_minutes",
    "mz_defect",
    "kendrick_mass_defect",
)

OUTPUT_HEADER = (GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)


class CorpusResult(TypedDict):
    """The assembled corpus, ready to write.

    Args:
        header: Output column names in order.
        rows: One output row per kept peak, values as strings.
        n_rt_bins: Distinct retention-time bins (group count).
        n_real: Rows labelled real (kept by the lab's rule).
        n_blank: Rows labelled blank-dominated.
        n_dropped_undetected: Rows detected in neither biological
            samples nor individual blanks.
    """

    header: tuple[str, ...]
    rows: list[list[str]]
    n_rt_bins: int
    n_real: int
    n_blank: int
    n_dropped_undetected: int


def _write(message: str) -> None:
    """Write a message to stdout.

    Args:
        message: Text to emit.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def _column_mean(row: list[str], indices: tuple[int, ...]) -> float:
    """Mean intensity over the named columns, empty cells as zero.

    An empty cell in this table means the peak was not detected in that
    injection — a genuine zero for the averaging rule, not missing data
    to impute.

    Args:
        row: The sheet row.
        indices: Column indices to average.

    Returns:
        The mean intensity.
    """
    total = 0.0
    for i in indices:
        cell = row[i]
        if cell != "":
            total += float(cell)
    return total / len(indices)


def _label(sample_avg: float, blank_avg: float) -> int | None:
    """Apply the lab's standard blank-filter rule.

    Args:
        sample_avg: Mean intensity over the biological samples.
        blank_avg: Mean intensity over the individual blanks.

    Returns:
        1 for a real peak (samples-only, or at least ``BLANK_RATIO``
        times the blank average), 0 for a blank-dominated peak, None
        for a peak detected in neither (dropped by the caller).
    """
    if sample_avg == 0.0 and blank_avg == 0.0:
        return None
    if blank_avg == 0.0 or sample_avg >= BLANK_RATIO * blank_avg:
        return 1
    return 0


def _kendrick_mass_defect(mz: float) -> float:
    """Kendrick mass defect of an m/z on the CH2 base unit.

    Contaminant homolog series (plasticizers, PEG) differ by CH2 units
    and share a Kendrick mass defect, which is what makes the quantity
    a contamination signature.

    Args:
        mz: The measured m/z.

    Returns:
        ``round(km) - km`` for the Kendrick mass ``km``, in [-0.5, 0.5].
    """
    kendrick_mass = mz * KENDRICK_CH2
    return round(kendrick_mass) - kendrick_mass


class _Columns(TypedDict):
    """Resolved column indices of the Normalized sheet.

    Args:
        compound: The Compound id column.
        mz: The m/z column.
        charge: The Charge column.
        rt: The retention-time column.
        width: The chromatographic peak-width column.
        bio: The biological sample columns, in declared order.
        blank: The individual blank columns, in declared order.
    """

    compound: int
    mz: int
    charge: int
    rt: int
    width: int
    bio: tuple[int, ...]
    blank: tuple[int, ...]


def _resolve_columns(header: list[str]) -> _Columns:
    """Resolve every required header to its column index.

    Args:
        header: The sheet's header row.

    Returns:
        The resolved indices.

    Raises:
        ValueError: If a required header is missing.
    """
    column_index: dict[str, int] = {name: i for i, name in enumerate(header)}
    for name in (*REQUIRED_METADATA, PEAK_WIDTH_HEADER, *BIOLOGICAL_COLUMNS, *BLANK_COLUMNS):
        if name not in column_index:
            raise ValueError(f"sheet '{SHEET_NAME}' is missing required header '{name}'")
    return _Columns(
        compound=column_index["Compound"],
        mz=column_index["m/z"],
        charge=column_index["Charge"],
        rt=column_index["Retention time (min)"],
        width=column_index[PEAK_WIDTH_HEADER],
        bio=tuple(column_index[name] for name in BIOLOGICAL_COLUMNS),
        blank=tuple(column_index[name] for name in BLANK_COLUMNS),
    )


def build_corpus(workbook: Path) -> CorpusResult:
    """Assemble the corpus from the Normalized sheet.

    Args:
        workbook: Path to ``Emily_Data_Pruned_Labeled.xlsx``.

    Returns:
        The assembled corpus and its counts.

    Raises:
        ValueError: If a required header is missing, a required cell is
            empty, or no rows survive.
    """
    rows = read_xlsx_sheet(workbook, SHEET_NAME)
    if not rows:
        raise ValueError(f"sheet '{SHEET_NAME}' is empty")
    columns = _resolve_columns(rows[0])
    compound_idx = columns["compound"]
    mz_idx = columns["mz"]
    charge_idx = columns["charge"]
    rt_idx = columns["rt"]
    width_idx = columns["width"]
    bio_indices = columns["bio"]
    blank_indices = columns["blank"]

    out_rows: list[list[str]] = []
    rt_bins: set[str] = set()
    n_real = 0
    n_blank = 0
    n_undetected = 0
    for row in rows[1:]:
        if row[compound_idx] == "":
            continue
        for name, idx in (
            ("m/z", mz_idx),
            ("Charge", charge_idx),
            ("Retention time (min)", rt_idx),
            (PEAK_WIDTH_HEADER, width_idx),
        ):
            if row[idx] == "":
                raise ValueError(
                    f"peak '{row[compound_idx]}' has no value for required column '{name}'"
                )
        label = _label(_column_mean(row, bio_indices), _column_mean(row, blank_indices))
        if label is None:
            n_undetected += 1
            continue
        if label == 1:
            n_real += 1
        else:
            n_blank += 1
        mz = float(row[mz_idx])
        rt = float(row[rt_idx])
        rt_bin = str(int(rt * RT_BINS_PER_MINUTE))
        rt_bins.add(rt_bin)
        out_rows.append(
            [
                rt_bin,
                row[mz_idx],
                row[charge_idx],
                row[rt_idx],
                row[width_idx],
                f"{mz - math.floor(mz):.6f}",
                f"{_kendrick_mass_defect(mz):.6f}",
                str(label),
            ]
        )

    if not out_rows:
        raise ValueError("no rows survived the drop rules — corpus would be empty")
    return CorpusResult(
        header=OUTPUT_HEADER,
        rows=out_rows,
        n_rt_bins=len(rt_bins),
        n_real=n_real,
        n_blank=n_blank,
        n_dropped_undetected=n_undetected,
    )


def write_outputs(out: Path, result: CorpusResult, workbook: Path) -> None:
    """Write ``data.csv`` and the sha256-pinned ``MANIFEST.json``.

    Args:
        out: Output path for the corpus CSV; the manifest lands beside
            it.
        result: The assembled corpus.
        workbook: The source workbook path.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(result["header"])
        writer.writerows(result["rows"])

    n_rows = len(result["rows"])
    manifest: JSONValue = {
        "sources": {
            "workbook": {"file_name": workbook.name, "sha256": sha256_file(workbook)},
        },
        "corpus": {
            "rows": n_rows,
            "rt_bins": result["n_rt_bins"],
            "real": result["n_real"],
            "blank_dominated": result["n_blank"],
            "dropped_undetected": result["n_dropped_undetected"],
            "positive_ratio": round(result["n_real"] / n_rows, 6),
        },
    }
    manifest_path = out.parent / "MANIFEST.json"
    manifest_path.write_text(dump_json_str(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Build the metab_blank blank-vs-real peak corpus.")
    parser.add_argument(
        "--workbook", type=Path, required=True, help="Emily_Data_Pruned_Labeled.xlsx."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output path for data.csv.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the corpus, write it, and report its shape and drops.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parsed = build_parser().parse_args(argv)
    workbook: Path = parsed.workbook
    out: Path = parsed.out

    result = build_corpus(workbook)
    write_outputs(out, result, workbook)

    n_rows = len(result["rows"])
    _write(
        f"metab_blank: {n_rows} rows across {result['n_rt_bins']} rt bins -> {out}\n"
        f"  real {result['n_real']} / blank-dominated {result['n_blank']} "
        f"(positive ratio {result['n_real'] / n_rows:.4f})\n"
        f"  dropped: {result['n_dropped_undetected']} undetected-in-samples-and-blanks\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
