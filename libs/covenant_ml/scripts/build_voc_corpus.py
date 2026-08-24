"""Build the ``voc_match_quality`` regression corpus from the VOC field sites.

The Faiola Lab's aggregated GC-MS peak table
(``Aggregated_Summarized_Output.xlsx``, one sheet per California reserve
site) records, for every chromatogram peak, the top NIST library match
and its quality. The corpus asks the GC-MS twin of metab_confidence's
question: given only what the instrument measured — where the peak
elutes, what plant was sampled, and how crowded the chromatogram is
around it — how well will the NIST library identify it? One row per
peak; the target is ``Match1.Quality``, verbatim.

Honesty rules, applied by construction:

- Predictors are pre-annotation measurables ONLY: the plant species
  (known at sampling), the retention time, and chromatogram-context
  statistics computed from peak positions alone. Library outputs
  (match names, Match2/Match3 qualities), curation outputs (Compound,
  Class, Comments) and ``MatchScore`` (verified equal to
  ``Match1.Quality`` on every row) never become features. Run identity
  (DataFolderName, CartridgeNum, DateRun) never becomes a feature.
- ``site`` is the GROUP column, never a feature: peaks from one
  reserve share plants, weather and instrument sessions, so the split
  must be by whole site.
- Drops are counted and printed, never imputed: rows with no retention
  time cannot be placed in a chromatogram; rows with no species are the
  provenance-flagged misfiled cartridges; rows with no match quality
  have no target; rows whose quality falls outside NIST's 1-99 scale
  are recorded data defects (two exist: 944 and 994) and are dropped,
  not guessed at.
- Chromatogram context is computed from every peak with a valid
  retention time in the run — a peak with no library hit still crowds
  its neighbours.

Usage:
    poetry run python -m scripts.build_voc_corpus \
        --workbook .../Aggregated_Summarized_Output.xlsx \
        --out ../../services/covenant-radar-api/data/external/voc_match_quality/data.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import JSONValue, dump_json_str

from covenant_ml.datasets.xlsx_reader import read_xlsx_sheet
from scripts.build_metab_corpus import sha256_file

#: The workbook's ten site sheets, in output order.
SITE_SHEETS = (
    "Angelo",
    "BlueOak",
    "EmersonOaks",
    "FortOrd",
    "Lassen",
    "PointReyes",
    "Rancho",
    "Sagehen",
    "StuntRanch",
    "Yosemite",
)

GROUP_COLUMN = "site"
TARGET_COLUMN = "match1_quality"

#: Feature columns in output order — pre-annotation measurables only.
FEATURE_COLUMNS = (
    "species",
    "rt_minutes",
    "run_n_peaks",
    "rt_rank_frac",
    "gap_prev_minutes",
    "peaks_within_0p1min",
    "peaks_within_1min",
    "run_rt_span",
)

OUTPUT_HEADER = (GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)

#: Headers every site sheet must carry.
REQUIRED_HEADERS = ("DataFolderName", "Species", "RetentionTime", "Match1.Quality")

#: NIST match quality bounds; values outside are recorded data defects.
QUALITY_MIN = 1
QUALITY_MAX = 99

#: Co-elution and regional crowding windows, in minutes.
CO_ELUTION_WINDOW = 0.1
REGION_WINDOW = 1.0


class SheetPeak(TypedDict):
    """One peak row read from a site sheet, values verbatim.

    Args:
        run: The ``DataFolderName`` (one chromatogram).
        species: The plant species code, stripped.
        rt_text: The retention time cell, verbatim ("" when absent).
        quality_text: The ``Match1.Quality`` cell, verbatim ("" when
            absent).
    """

    run: str
    species: str
    rt_text: str
    quality_text: str


class DropCounts(TypedDict):
    """Counted drop rules, in precedence order.

    Args:
        no_rt: Rows with no retention time (unplaceable in a
            chromatogram; also invisible to context statistics).
        no_species: Rows with no species (the misfiled cartridges).
        no_quality: Rows with no ``Match1.Quality`` (no target).
        quality_range: Rows whose quality falls outside NIST's 1-99.
    """

    no_rt: int
    no_species: int
    no_quality: int
    quality_range: int


class CorpusResult(TypedDict):
    """The assembled corpus, ready to write.

    Args:
        header: Output column names in order.
        rows: One output row per kept peak, values as strings.
        n_sites: Distinct site count (group count).
        n_runs: Distinct (site, run) chromatogram count.
        drops: The counted drop rules.
        target_mean: Mean of the kept quality targets.
    """

    header: tuple[str, ...]
    rows: list[list[str]]
    n_sites: int
    n_runs: int
    drops: DropCounts
    target_mean: float


def _write(message: str) -> None:
    """Write a message to stdout.

    Args:
        message: Text to emit.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def parse_sheet(workbook: Path, site: str) -> list[SheetPeak]:
    """Read one site sheet's peak rows with verbatim cell values.

    All-empty spreadsheet rows are structural padding and are skipped;
    every data row is returned, drops are decided later where they can
    be counted.

    Args:
        workbook: Path to the aggregated workbook.
        site: The site sheet name.

    Returns:
        The sheet's peak rows, in sheet order.

    Raises:
        ValueError: If the sheet is empty or missing a required header.
    """
    rows = read_xlsx_sheet(workbook, site)
    if not rows:
        raise ValueError(f"sheet '{site}' is empty")
    header = rows[0]
    column_index: dict[str, int] = {name: i for i, name in enumerate(header)}
    for name in REQUIRED_HEADERS:
        if name not in column_index:
            raise ValueError(f"sheet '{site}' is missing required header '{name}'")
    run_idx = column_index["DataFolderName"]
    species_idx = column_index["Species"]
    rt_idx = column_index["RetentionTime"]
    quality_idx = column_index["Match1.Quality"]

    peaks: list[SheetPeak] = []
    for row in rows[1:]:
        if all(cell == "" for cell in row):
            continue
        peaks.append(
            SheetPeak(
                run=row[run_idx],
                species=row[species_idx].strip(),
                rt_text=row[rt_idx],
                quality_text=row[quality_idx],
            )
        )
    return peaks


class _RunContext(TypedDict):
    """Chromatogram-context statistics for one run's placeable peaks.

    Args:
        rts: Retention times of every peak with a valid RT, sorted.
        rank_of: Peak's sorted rank keyed by its index in the run's
            placeable-peak order.
    """

    rts: list[float]
    rank_of: dict[int, int]


def _run_context(rts_in_order: list[float]) -> _RunContext:
    """Rank one run's peaks by retention time, ties broken by file order.

    Args:
        rts_in_order: Valid retention times in sheet order.

    Returns:
        The sorted times and each peak's rank.
    """
    decorated: list[tuple[float, int]] = sorted((rt, i) for i, rt in enumerate(rts_in_order))
    rank_of = {peak_index: rank for rank, (_, peak_index) in enumerate(decorated)}
    return _RunContext(rts=[rt for rt, _ in decorated], rank_of=rank_of)


def _context_row(context: _RunContext, peak_index: int) -> list[str]:
    """Compute one peak's chromatogram-context feature values.

    Args:
        context: The peak's run context.
        peak_index: The peak's index in the run's placeable-peak order.

    Returns:
        Feature strings: run_n_peaks, rt_rank_frac, gap_prev_minutes,
        peaks_within_0p1min, peaks_within_1min, run_rt_span.
    """
    rts = context["rts"]
    n_peaks = len(rts)
    rank = context["rank_of"][peak_index]
    rt = rts[rank]
    # The first peak's "previous" event is the injection at t = 0.
    gap_prev = rt - rts[rank - 1] if rank > 0 else rt
    within_close = sum(1 for other in rts if abs(other - rt) <= CO_ELUTION_WINDOW) - 1
    within_region = sum(1 for other in rts if abs(other - rt) <= REGION_WINDOW) - 1
    return [
        str(n_peaks),
        f"{(rank + 0.5) / n_peaks:.6f}",
        f"{gap_prev:.6f}",
        str(within_close),
        str(within_region),
        f"{rts[-1] - rts[0]:.6f}",
    ]


def build_corpus(workbook: Path) -> CorpusResult:
    """Assemble the corpus across every site sheet.

    Args:
        workbook: Path to the aggregated workbook.

    Returns:
        The assembled corpus and its drop counts.

    Raises:
        ValueError: If a sheet is defective or no rows survive.
    """
    drops = DropCounts(no_rt=0, no_species=0, no_quality=0, quality_range=0)
    rows: list[list[str]] = []
    n_runs = 0
    target_sum = 0.0

    for site in SITE_SHEETS:
        peaks = parse_sheet(workbook, site)

        by_run: dict[str, list[SheetPeak]] = {}
        for peak in peaks:
            by_run.setdefault(peak["run"], []).append(peak)
        n_runs += len(by_run)

        for run_peaks in by_run.values():
            placeable = [p for p in run_peaks if p["rt_text"] != ""]
            drops["no_rt"] += len(run_peaks) - len(placeable)
            context = _run_context([float(p["rt_text"]) for p in placeable])
            for peak_index, peak in enumerate(placeable):
                if peak["species"] == "":
                    drops["no_species"] += 1
                    continue
                if peak["quality_text"] == "":
                    drops["no_quality"] += 1
                    continue
                quality = int(peak["quality_text"])
                if not QUALITY_MIN <= quality <= QUALITY_MAX:
                    drops["quality_range"] += 1
                    continue
                target_sum += quality
                rows.append(
                    [
                        site,
                        peak["species"],
                        peak["rt_text"],
                        *_context_row(context, peak_index),
                        peak["quality_text"],
                    ]
                )

    if not rows:
        raise ValueError("no rows survived the drop rules — corpus would be empty")
    return CorpusResult(
        header=OUTPUT_HEADER,
        rows=rows,
        n_sites=len(SITE_SHEETS),
        n_runs=n_runs,
        drops=drops,
        target_mean=target_sum / len(rows),
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

    drops = result["drops"]
    manifest: JSONValue = {
        "sources": {
            "workbook": {"file_name": workbook.name, "sha256": sha256_file(workbook)},
        },
        "corpus": {
            "rows": len(result["rows"]),
            "sites": result["n_sites"],
            "runs": result["n_runs"],
            "dropped_no_rt": drops["no_rt"],
            "dropped_no_species": drops["no_species"],
            "dropped_no_quality": drops["no_quality"],
            "dropped_quality_out_of_range": drops["quality_range"],
            "target_mean": round(result["target_mean"], 6),
        },
    }
    manifest_path = out.parent / "MANIFEST.json"
    manifest_path.write_text(dump_json_str(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Build the voc_match_quality NIST-match-quality corpus."
    )
    parser.add_argument(
        "--workbook", type=Path, required=True, help="Aggregated_Summarized_Output.xlsx."
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

    drops = result["drops"]
    _write(
        f"voc_match_quality: {len(result['rows'])} rows across {result['n_sites']} sites "
        f"({result['n_runs']} chromatograms) -> {out}\n"
        f"  dropped: {drops['no_rt']} no-rt, {drops['no_species']} no-species, "
        f"{drops['no_quality']} no-quality, {drops['quality_range']} quality-out-of-range\n"
        f"  {TARGET_COLUMN}: mean {result['target_mean']:.4f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
