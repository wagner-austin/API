"""Build the ``metab_confidence`` regression corpus from the artcal campaign.

The corpus asks a deployable question: given only what the instrument
measured — precursor mass, retention time, MS1 feature height, the MS2
spectrum's shape, and how the feature behaves across the biological
samples — how confident will CSI:FingerID be in its best structure call?
One row per mzMine feature that SIRIUS gave a rank-1 structure; the
target is COSMIC's ``ConfidenceScoreExact``.

Sources (all sha256-pinned in the output MANIFEST):

- the mzMine DIA export MGF (one MS1 ``CORRELATED MS`` block and one MS2
  block per feature) — every spectral predictor comes from here;
- the mzMine MetaboAnalyst quant table — per-sample detection and
  intensity statistics;
- SIRIUS ``structure_identifications.tsv`` — the rank-1
  ``ConfidenceScoreExact`` target, and NOTHING else.

Honesty rules, applied by construction:

- Predictors are pre-annotation measurables ONLY. Annotation outputs
  (adduct, formula, ionMass error, any SIRIUS score) never become
  features: they are downstream of the answer.
- ``rt_bin`` (0.1-minute retention windows) is the GROUP column, never a
  feature: adducts and in-source fragments of one molecule co-elute, and
  a row-wise split would let the model see a compound's siblings across
  train and test.
- Rows whose ``ConfidenceScoreExact`` is ``-Infinity`` (SIRIUS: no exact
  confidence computable) are DROPPED and counted, never imputed.
- Features undetected in every biological sample are DROPPED and
  counted (the pooled ``combine.mzML`` injection is not a biological
  sample and is excluded from the statistics).
- Structural defects in the sources (missing or duplicate blocks, peak
  counts that disagree with the declaration, non-positive intensities)
  are REFUSALS naming the feature, not skips.

Usage:
    poetry run python -m scripts.build_metab_corpus \
        --mgf .../artcal_metab_dia_sirius.mgf \
        --quant .../artcal_metab_dia_metaboanalyst.csv \
        --structures .../structure_identifications.tsv \
        --out ../../services/covenant-radar-api/data/external/metab_confidence/data.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import JSONValue, dump_json_str

#: The pooled combined injection in the quant table; not a biological
#: sample, so it never counts toward detection or intensity statistics.
EXCLUDED_SAMPLE_COLUMNS = ("combine.mzML",)

#: Retention-time co-elution window width for the group column.
RT_BIN_SECONDS = 6.0

GROUP_COLUMN = "rt_bin"
TARGET_COLUMN = "confidence_exact"

#: Feature columns in output order — pre-annotation measurables only.
FEATURE_COLUMNS = (
    "precursor_mz",
    "rt_seconds",
    "log10_ms1_height",
    "ms1_n_peaks",
    "ms2_n_peaks",
    "ms2_log10_total_intensity",
    "ms2_log10_max_intensity",
    "ms2_frac_top3",
    "n_samples_detected",
    "log10_mean_detected_intensity",
    "log10_max_detected_intensity",
)

OUTPUT_HEADER = (GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)

#: SIRIUS's spelling for "no exact confidence computable".
INFINITE_CONFIDENCE = "-Infinity"

#: Columns the structures TSV must carry.
REQUIRED_STRUCTURE_COLUMNS = (
    "structurePerIdRank",
    "mappingFeatureId",
    "ConfidenceScoreExact",
)


class Ms1Block(TypedDict):
    """The measurables read from one feature's MS1 ``CORRELATED MS`` block.

    Args:
        precursor_mz: The ``PEPMASS`` value, verbatim.
        rt_seconds: The ``RTINSECONDS`` value, verbatim.
        ms1_height: The ``FEATURE_MS1_HEIGHT`` value.
        n_peaks: Number of correlated MS1 peaks (isotopes/adducts).
    """

    precursor_mz: str
    rt_seconds: str
    ms1_height: float
    n_peaks: int


class Ms2Block(TypedDict):
    """The measurables read from one feature's MS2 block.

    Args:
        n_peaks: Fragment peak count.
        total_intensity: Sum of fragment intensities.
        max_intensity: Largest fragment intensity.
        top3_intensity: Sum of the three largest fragment intensities.
    """

    n_peaks: int
    total_intensity: float
    max_intensity: float
    top3_intensity: float


class QuantStats(TypedDict):
    """Per-feature statistics over the biological sample columns.

    Args:
        n_detected: Samples with a recorded intensity.
        mean_detected: Mean intensity over detected samples (0.0 when
            none are detected; such rows are dropped, never written).
        max_detected: Largest intensity over detected samples (0.0 when
            none are detected).
    """

    n_detected: int
    mean_detected: float
    max_detected: float


class CorpusResult(TypedDict):
    """The assembled corpus, ready to write.

    Args:
        header: Output column names in order.
        rows: One output row per kept feature, values as strings.
        n_rt_bins: Distinct retention-time bins (group count).
        n_dropped_infinite: Rows dropped for ``-Infinity`` confidence.
        n_dropped_undetected: Rows dropped for zero biological detections.
        target_mean: Mean of the kept confidence targets.
    """

    header: tuple[str, ...]
    rows: list[list[str]]
    n_rt_bins: int
    n_dropped_infinite: int
    n_dropped_undetected: int
    target_mean: float


def _write(message: str) -> None:
    """Write a message to stdout.

    Args:
        message: Text to emit.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def _log10_positive(value: float, what: str) -> float:
    """Take log10 of a value that the data promises is positive.

    Args:
        value: The measured value.
        what: Description used in the refusal message.

    Returns:
        ``log10(value)``.

    Raises:
        ValueError: If the value is not strictly positive.
    """
    if value <= 0.0:
        raise ValueError(f"{what} must be positive, got {value}")
    return math.log10(value)


def parse_structures(path: Path) -> dict[str, str]:
    """Read rank-1 ``ConfidenceScoreExact`` targets keyed by feature id.

    The TSV is tab-separated with UNQUOTED fields — compound names may
    contain literal quote characters — so quoting is disabled.

    Args:
        path: Path to ``structure_identifications.tsv``.

    Returns:
        Mapping of ``mappingFeatureId`` to the verbatim confidence value
        (which may be ``-Infinity``).

    Raises:
        ValueError: If a required column is missing or a feature id
            carries more than one rank-1 row.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        column_index: dict[str, int] = {name: i for i, name in enumerate(header)}
        for name in REQUIRED_STRUCTURE_COLUMNS:
            if name not in column_index:
                raise ValueError(f"structures TSV is missing required column '{name}'")
        rank_idx = column_index["structurePerIdRank"]
        fid_idx = column_index["mappingFeatureId"]
        conf_idx = column_index["ConfidenceScoreExact"]

        targets: dict[str, str] = {}
        for row in reader:
            if not row or row[rank_idx] != "1":
                continue
            fid = row[fid_idx]
            if fid in targets:
                raise ValueError(f"feature {fid} has more than one rank-1 structure row")
            targets[fid] = row[conf_idx]
    return targets


def _finish_block(
    fid: str,
    mslevel: str,
    fields: dict[str, str],
    intensities: list[float],
    ms1: dict[str, Ms1Block],
    ms2: dict[str, Ms2Block],
) -> None:
    """Validate one finished MGF block and store its measurables.

    Args:
        fid: The block's ``FEATURE_ID``.
        mslevel: The block's ``MSLEVEL``.
        fields: The block's ``KEY=value`` header fields.
        intensities: Parsed peak intensities, in file order.
        ms1: MS1 blocks by feature id, updated in place.
        ms2: MS2 blocks by feature id, updated in place.

    Raises:
        ValueError: If the block is structurally defective — unknown
            MSLEVEL, duplicate block for the feature, a peak count that
            disagrees with the ``Num peaks`` declaration, or a missing
            required field.
    """
    declared = int(fields["Num peaks"])
    if declared != len(intensities):
        raise ValueError(
            f"feature {fid} MSLEVEL={mslevel} declares {declared} peaks "
            f"but lists {len(intensities)}"
        )
    if mslevel == "1":
        if fid in ms1:
            raise ValueError(f"feature {fid} has more than one MS1 block")
        ms1[fid] = Ms1Block(
            precursor_mz=fields["PEPMASS"],
            rt_seconds=fields["RTINSECONDS"],
            ms1_height=float(fields["FEATURE_MS1_HEIGHT"]),
            n_peaks=len(intensities),
        )
        return
    if mslevel == "2":
        if fid in ms2:
            raise ValueError(f"feature {fid} has more than one MS2 block")
        ordered = sorted(intensities, reverse=True)
        ms2[fid] = Ms2Block(
            n_peaks=len(intensities),
            total_intensity=sum(intensities),
            max_intensity=ordered[0] if ordered else 0.0,
            top3_intensity=sum(ordered[:3]),
        )
        return
    raise ValueError(f"feature {fid} has unexpected MSLEVEL '{mslevel}'")


def parse_mgf(
    path: Path, wanted: frozenset[str]
) -> tuple[dict[str, Ms1Block], dict[str, Ms2Block]]:
    """Stream the DIA MGF and collect block measurables for wanted features.

    Each feature exports two blocks: an MS1 ``CORRELATED MS`` block and
    an MS2 fragment block. Header lines are ``KEY=value``; peak lines
    are ``mz intensity`` pairs.

    Args:
        path: Path to the mzMine DIA export MGF.
        wanted: Feature ids to collect (all others stream past).

    Returns:
        MS1 and MS2 block measurables keyed by feature id.

    Raises:
        ValueError: If a wanted feature's block is structurally
            defective (via :func:`_finish_block`).
    """
    ms1: dict[str, Ms1Block] = {}
    ms2: dict[str, Ms2Block] = {}
    in_block = False
    fields: dict[str, str] = {}
    intensities: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                in_block = True
                fields = {}
                intensities = []
            elif line == "END IONS":
                fid = fields.get("FEATURE_ID", "")
                if fid in wanted:
                    _finish_block(fid, fields.get("MSLEVEL", ""), fields, intensities, ms1, ms2)
                in_block = False
            elif in_block and line:
                if "=" in line:
                    key, _, value = line.partition("=")
                    fields[key] = value
                else:
                    intensities.append(float(line.split()[1]))
    return ms1, ms2


def parse_quant(path: Path, wanted: frozenset[str]) -> dict[str, QuantStats]:
    """Read per-feature biological-sample statistics from the quant table.

    Row keys are ``<feature_id>/<mz>mz/<rt>min``; the integer before the
    first slash is the mzMine feature id. The second header row carries
    sample-group metadata and is skipped.

    Args:
        path: Path to the MetaboAnalyst quant CSV.
        wanted: Feature ids to collect.

    Returns:
        Detection and intensity statistics keyed by feature id.

    Raises:
        ValueError: If no biological sample columns exist, a feature id
            repeats, or a recorded intensity is not positive.
    """
    stats: dict[str, QuantStats] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        sample_indices = [
            i
            for i, name in enumerate(header)
            if name.endswith(".mzML") and name not in EXCLUDED_SAMPLE_COLUMNS
        ]
        if not sample_indices:
            raise ValueError("quant table has no biological sample columns")
        next(reader)  # sample-group metadata row (Organ_Water)
        for row in reader:
            if not row:
                continue
            fid = row[0].split("/", 1)[0]
            if fid not in wanted:
                continue
            if fid in stats:
                raise ValueError(f"feature {fid} appears twice in the quant table")
            detected: list[float] = []
            for i in sample_indices:
                cell = row[i] if i < len(row) else ""
                if cell == "":
                    continue
                value = float(cell)
                if value <= 0.0:
                    raise ValueError(
                        f"feature {fid} has non-positive intensity {value} in '{header[i]}'"
                    )
                detected.append(value)
            n_detected = len(detected)
            stats[fid] = QuantStats(
                n_detected=n_detected,
                mean_detected=sum(detected) / n_detected if n_detected > 0 else 0.0,
                max_detected=max(detected) if n_detected > 0 else 0.0,
            )
    return stats


def _corpus_row(
    fid: str,
    ms1_block: Ms1Block,
    ms2_block: Ms2Block,
    quant: QuantStats,
    confidence: str,
) -> list[str]:
    """Assemble one output row for a kept feature.

    Args:
        fid: The feature id (used in refusal messages).
        ms1_block: The feature's MS1 measurables.
        ms2_block: The feature's MS2 measurables.
        quant: The feature's biological-sample statistics.
        confidence: The verbatim confidence target.

    Returns:
        Values as strings in ``OUTPUT_HEADER`` order.
    """
    rt = float(ms1_block["rt_seconds"])
    total = ms2_block["total_intensity"]
    return [
        str(int(rt // RT_BIN_SECONDS)),
        ms1_block["precursor_mz"],
        ms1_block["rt_seconds"],
        f"{_log10_positive(ms1_block['ms1_height'], f'feature {fid} FEATURE_MS1_HEIGHT'):.6f}",
        str(ms1_block["n_peaks"]),
        str(ms2_block["n_peaks"]),
        f"{_log10_positive(total, f'feature {fid} MS2 total intensity'):.6f}",
        f"{_log10_positive(ms2_block['max_intensity'], f'feature {fid} MS2 max intensity'):.6f}",
        f"{ms2_block['top3_intensity'] / total:.6f}",
        str(quant["n_detected"]),
        f"{_log10_positive(quant['mean_detected'], f'feature {fid} mean intensity'):.6f}",
        f"{_log10_positive(quant['max_detected'], f'feature {fid} max intensity'):.6f}",
        confidence,
    ]


def build_corpus(
    targets: dict[str, str],
    ms1: dict[str, Ms1Block],
    ms2: dict[str, Ms2Block],
    quant: dict[str, QuantStats],
) -> CorpusResult:
    """Join the three sources into the corpus, in feature-id order.

    Args:
        targets: Rank-1 confidence targets by feature id.
        ms1: MS1 block measurables by feature id.
        ms2: MS2 block measurables by feature id.
        quant: Biological-sample statistics by feature id.

    Returns:
        The assembled corpus and its drop counts.

    Raises:
        ValueError: If a target feature is missing from any source, or
            no rows survive the drop rules.
    """
    rows: list[list[str]] = []
    rt_bins: set[str] = set()
    n_infinite = 0
    n_undetected = 0
    target_sum = 0.0
    for fid in sorted(targets, key=int):
        confidence = targets[fid]
        if confidence == INFINITE_CONFIDENCE:
            n_infinite += 1
            continue
        ms1_block = ms1.get(fid)
        if ms1_block is None:
            raise ValueError(f"feature {fid} has a structure call but no MS1 block in the MGF")
        ms2_block = ms2.get(fid)
        if ms2_block is None:
            raise ValueError(f"feature {fid} has a structure call but no MS2 block in the MGF")
        quant_stats = quant.get(fid)
        if quant_stats is None:
            raise ValueError(f"feature {fid} has a structure call but no quant table row")
        if quant_stats["n_detected"] == 0:
            n_undetected += 1
            continue
        row = _corpus_row(fid, ms1_block, ms2_block, quant_stats, confidence)
        rt_bins.add(row[0])
        target_sum += float(confidence)
        rows.append(row)

    if not rows:
        raise ValueError("no rows survived the drop rules — corpus would be empty")
    return CorpusResult(
        header=OUTPUT_HEADER,
        rows=rows,
        n_rt_bins=len(rt_bins),
        n_dropped_infinite=n_infinite,
        n_dropped_undetected=n_undetected,
        target_mean=target_sum / len(rows),
    )


def sha256_file(path: Path) -> str:
    """Compute a file's sha256 hex digest by streaming.

    Args:
        path: File to hash.

    Returns:
        The 64-character hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    out: Path,
    result: CorpusResult,
    sources: tuple[Path, Path, Path],
) -> None:
    """Write ``data.csv`` and the sha256-pinned ``MANIFEST.json``.

    Args:
        out: Output path for the corpus CSV; the manifest lands beside it.
        result: The assembled corpus.
        sources: The MGF, quant and structures source paths, in order.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(result["header"])
        writer.writerows(result["rows"])

    mgf_path, quant_path, structures_path = sources
    manifest: JSONValue = {
        "sources": {
            "mgf": {"file_name": mgf_path.name, "sha256": sha256_file(mgf_path)},
            "quant": {"file_name": quant_path.name, "sha256": sha256_file(quant_path)},
            "structures": {
                "file_name": structures_path.name,
                "sha256": sha256_file(structures_path),
            },
        },
        "corpus": {
            "rows": len(result["rows"]),
            "rt_bins": result["n_rt_bins"],
            "dropped_infinite_confidence": result["n_dropped_infinite"],
            "dropped_undetected": result["n_dropped_undetected"],
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
        description="Build the metab_confidence structure-confidence corpus."
    )
    parser.add_argument("--mgf", type=Path, required=True, help="mzMine DIA export MGF.")
    parser.add_argument("--quant", type=Path, required=True, help="MetaboAnalyst quant CSV.")
    parser.add_argument(
        "--structures", type=Path, required=True, help="SIRIUS structure_identifications.tsv."
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
    mgf_path: Path = parsed.mgf
    quant_path: Path = parsed.quant
    structures_path: Path = parsed.structures
    out: Path = parsed.out

    targets = parse_structures(structures_path)
    wanted = frozenset(targets)
    ms1, ms2 = parse_mgf(mgf_path, wanted)
    quant = parse_quant(quant_path, wanted)
    result = build_corpus(targets, ms1, ms2, quant)
    write_outputs(out, result, (mgf_path, quant_path, structures_path))

    _write(
        f"metab_confidence: {len(result['rows'])} rows across "
        f"{result['n_rt_bins']} rt bins -> {out}\n"
        f"  dropped: {result['n_dropped_infinite']} infinite-confidence, "
        f"{result['n_dropped_undetected']} undetected-in-biological-samples\n"
        f"  {TARGET_COLUMN}: mean {result['target_mean']:.4f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
