"""Every trace in a batch, reduced to the figures its autopsies turn on.

The session-scale findings of 2026-08-02/03 -- the ~30-35k worth ceiling, the
never-above-1.19 ratio on structurally lost seeds, and the expansion race
that decides Very Hard by sample 1,500 with zero combat -- were each read
out of the per-sample traces with a throwaway script. Three throwaways is a
pattern: this is that reading, kept.

One row per match: where our worth peaked and against what, the our/their
ratio at the samples the race and the closer turn on, the extractor count at
the race's finish line, and where the worth first halved after its peak --
the collapse marker. Tab-separated like :mod:`scripts.results` and
:mod:`scripts.ledger`, for the same reason: pipe it anywhere.

Run as ``python -m scripts.autopsy <batch-name>``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

TRACE_ROOT = Path("runs/traces")

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_BAD_USAGE = 2

#: Samples the trajectory is probed at: the race's finish line (the solo-24
#: traces show extractor counts at 1,500 separating every win from every
#: loss), the closer's typical latch window, and the band where every
#: recorded win finished.
PROBES = (1500, 2000, 2500, 3400)

#: Engine frames per planner sample, the lockstep stride every recorded
#: trace was taken at.
FRAMES_PER_SAMPLE = 75

#: Fewest columns a line must split into to be a tick row. Thirteen is the
#: original shape; the income pair (2026-08-05) landed between ``rival`` and
#: ``world`` precisely so that every column this script indexes -- extractors
#: at 4, lost at 5, worth at 10, rival at 11 -- kept its position, which is
#: why one bound reads both the 13-column archive and the 15-column current
#: shape instead of splitting the trace record into eras.
_MIN_COLUMNS = 13


class TracePoint(TypedDict):
    """One sample of a match, the columns autopsies read.

    Attributes:
        sample: The observation index (the trace's frame over the stride).
        extractors: Extractors standing.
        lost: Own units lost so far.
        worth: Our total worth.
        rival: The strongest rival's army value.
    """

    sample: int
    extractors: int
    lost: int
    worth: int
    rival: int


class MatchAutopsy(TypedDict):
    """One match reduced to the figures the verdicts turned on.

    Attributes:
        file: The trace's stem, for tracing a row back.
        samples: Observations recorded.
        peak_worth: Our highest total worth.
        peak_sample: When it was reached.
        rival_at_peak: Their army value at that moment.
        halved_sample: When our worth first fell below half its peak after
            the peak, -1 when it never did -- the collapse marker.
        extractors_at_race: Extractors standing at the first probe, the
            race's finish line.
        ratios: Our worth over their army value at each probe, in
            :data:`PROBES` order, 0.0 where their army was zero.
    """

    file: str
    samples: int
    peak_worth: int
    peak_sample: int
    rival_at_peak: int
    halved_sample: int
    extractors_at_race: int
    ratios: tuple[float, ...]


def decode_trace(text: str) -> tuple[TracePoint, ...]:
    """Read a recorder trace into its autopsy columns.

    The trace is the recorder's fixed-width table: a header naming the columns
    -- thirteen before the income pair, fifteen since -- then one row per
    sample. Anything that is not such a row -- the header, a blank line -- is
    skipped by shape, the same tolerance :func:`scripts.analyze_sweep.drops`
    reads with, and :data:`_MIN_COLUMNS` says why one bound reads both shapes.

    Args:
        text: The trace file's content.

    Returns:
        The points, in recorded order.
    """
    points: list[TracePoint] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < _MIN_COLUMNS or not parts[0].lstrip("-").isdigit():
            continue
        points.append(
            TracePoint(
                sample=int(parts[0]) // FRAMES_PER_SAMPLE,
                extractors=int(parts[4]),
                lost=int(parts[5]),
                worth=int(parts[10]),
                rival=int(parts[11]),
            )
        )
    return tuple(points)


def _at(points: Sequence[TracePoint], target: int) -> TracePoint:
    """Return the recorded point nearest a target sample.

    Args:
        points: The match's points, never empty.
        target: The sample asked about.

    Returns:
        The nearest point.
    """

    def distance(point: TracePoint) -> int:
        return abs(point["sample"] - target)

    return min(points, key=distance)


def autopsy(points: Sequence[TracePoint], stem: str) -> MatchAutopsy | None:
    """Reduce one match's points to the figures its verdict turned on.

    Args:
        points: The match's points.
        stem: The trace's stem, carried into the row.

    Returns:
        The autopsy, or None for an empty trace -- a match that died before
        its first sample, which is a fact for the launch log rather than a
        row here.
    """
    if not points:
        return None

    def worth_of(index: int) -> int:
        return points[index]["worth"]

    peak_index = max(range(len(points)), key=worth_of)
    peak = points[peak_index]
    halved = next(
        (p for p in points[peak_index:] if p["worth"] < peak["worth"] / 2),
        None,
    )
    return MatchAutopsy(
        file=stem,
        samples=points[-1]["sample"],
        peak_worth=peak["worth"],
        peak_sample=peak["sample"],
        rival_at_peak=peak["rival"],
        halved_sample=halved["sample"] if halved is not None else -1,
        extractors_at_race=_at(points, PROBES[0])["extractors"],
        ratios=tuple(
            round(point["worth"] / point["rival"], 2) if point["rival"] > 0 else 0.0
            for probe in PROBES
            for point in (_at(points, probe),)
        ),
    )


def rows(batch: Path) -> tuple[str, ...]:
    """Render one row per trace in the batch, header first.

    Args:
        batch: The batch's trace directory.

    Returns:
        Tab-separated rows. Header only when the directory holds no
        readable trace.
    """
    header = (
        "file",
        "samples",
        "peak_worth",
        "peak_sample",
        "rival_at_peak",
        "halved_sample",
        "extr_at_race",
        *(f"ratio_s{probe}" for probe in PROBES),
    )
    lines = ["\t".join(header)]
    for path in sorted(batch.glob("*.ndjson")):
        record = autopsy(decode_trace(path.read_text(encoding="utf-8")), path.stem)
        if record is None:
            continue
        lines.append(
            "\t".join(
                (
                    record["file"],
                    str(record["samples"]),
                    str(record["peak_worth"]),
                    str(record["peak_sample"]),
                    str(record["rival_at_peak"]),
                    str(record["halved_sample"]),
                    str(record["extractors_at_race"]),
                    *(str(ratio) for ratio in record["ratios"]),
                )
            )
        )
    return tuple(lines)


def main(argv: Sequence[str] | None = None, root: Path = TRACE_ROOT) -> int:
    """Print the autopsy table for the batch named on the command line.

    Args:
        argv: ``<batch-name>``. ``None`` reads the process arguments.
        root: The trace root, a parameter so a test can point it at a
            scratch tree.

    Returns:
        ``EXIT_OK``, ``EXIT_EMPTY`` when the batch has no traces, or
        ``EXIT_BAD_USAGE`` on any other argument shape.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stdout.write("usage: autopsy <batch-name>\n")
        return EXIT_BAD_USAGE
    table = rows(root / args[0])
    if len(table) == 1:
        sys.stdout.write("no traces for that batch\n")
        return EXIT_EMPTY
    for line in table:
        sys.stdout.write(line + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
