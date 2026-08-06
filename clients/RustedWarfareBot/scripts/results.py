"""Every ad-hoc probe ever run, as one tab-separated table.

The longitudinal view for the OTHER half of the record: :mod:`scripts.ledger`
covers the sweep batches under ``runs/sweeps``, but the campaign's arms are
screened one match at a time as ``runs/<name>.out`` files first -- ten of them
in the 2026-08-01 basics batch alone -- and answering "which arm ever moved
the rival's dip" meant grepping a hundred scorecards. One row per probe here:
arm, seed, verdict, survival, the rival dip and peak, and the endpoint
figures, sorted so the arms that moved the needle read first.

``results spends`` prints the long-form spend ledger instead -- one row per
budget channel per match -- because the mechanism behind every verdict this
project has produced was read out of an ``asked/got/spent`` line, and joining
those across matches is how a starved channel is spotted the night it starves.

Nothing is written and nothing appends: the ``.out`` files are the store, and
regenerating the table from them on every run is what keeps it impossible to
drift from the record it summarises.

Run as ``python -m scripts.results [spends]``.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from rw_bot.harness.sweep import LABEL_WIDTH

RUNS_ROOT = Path("runs")

EXIT_OK = 0
EXIT_BAD_USAGE = 2

#: The banner every headless-match launch prints before anything else; a
#: ``.out`` without it is a build log or a probe of another kind, not a match.
PLAY_BANNER = "==> play"


class SpendRow(TypedDict):
    """One budget channel's ledger line from one match.

    Attributes:
        channel: The claim label, e.g. ``produce:c_tank``.
        asked: Ticks the channel claimed.
        got: Claims granted.
        spent: Credits spent in total.
    """

    channel: str
    asked: int
    got: int
    spent: int


class ProbeRecord(TypedDict):
    """One ad-hoc match, reduced to the figures verdicts turn on.

    Attributes:
        file: The scorecard's stem, for tracing a row back to its report.
        arm: The doctrine played.
        seed: The engine seed, -1 when the launch banner does not carry one.
        difficulty: The opponents' difficulty, -1 when the banner does not
            carry one. Carried because the record mixes rungs -- a 5,900 dip
            at Hard beside a 2,450 at Impossible is the exact confusion the
            table would otherwise create.
        map: The skirmish map's file stem, empty when the banner does not
            carry one -- the engine's own hardcoded sandbox.
        verdict: First word of the scorecard verdict, ``incomplete`` when the
            match died before printing one -- a crashed or wedged run is a
            fact worth a row, not a file to skip silently.
        samples: Observations survived, 0 when unreported.
        dip: The rival's worst army-value fall from its peak, 0 when
            unreported -- the lethality figure ([[policy-situation]]).
        peak: The rival's army-value peak, 0 when unreported.
        rival_end: The rival's closing army value, 0 when unreported.
        extractors_end: Extractors standing at the end, 0 when unreported.
        attacks: Attack orders sent, 0 when unreported.
        engaged_gone: Engaged enemies destroyed, 0 when unreported.
        spends: The match's budget ledger, in report order.
    """

    file: str
    arm: str
    seed: int
    difficulty: int
    map: str
    verdict: str
    samples: int
    dip: int
    peak: int
    rival_end: int
    extractors_end: int
    attacks: int
    engaged_gone: int
    spends: tuple[SpendRow, ...]


def scorecard_fields(text: str) -> dict[str, str]:
    """Read a scorecard's label/value pairs by the shape the sweep trusts.

    Args:
        text: The ``.out`` file's content.

    Returns:
        Values by label.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) > LABEL_WIDTH and line[LABEL_WIDTH] != " " and line[:1].islower():
            out[line[:LABEL_WIDTH].strip()] = line[LABEL_WIDTH:].strip()
    return out


def _arrow_end(value: str) -> int:
    """Return the end figure of a ``start -> end`` scorecard value.

    Args:
        value: The raw field text.

    Returns:
        The end integer, zero when the shape is absent.
    """
    m = re.search(r"->\s*(-?\d+)", value)
    return int(m.group(1)) if m else 0


def _leading_int(value: str) -> int:
    """Return the first integer of a scorecard value, zero when absent.

    Args:
        value: The raw field text.

    Returns:
        The leading integer.
    """
    m = re.match(r"(\d+)", value)
    return int(m.group(1)) if m else 0


def parse_probe(text: str, stem: str) -> ProbeRecord | None:
    """Reduce one ``.out`` file to a record, or rule it not a match at all.

    Args:
        text: The file's content.
        stem: The file's stem, carried into the record.

    Returns:
        The record, or None when the file is not a headless-match scorecard
        -- an agent build log, a sweep wrapper, or any other ``.out``.
    """
    if PLAY_BANNER not in text:
        return None
    fields = scorecard_fields(text)
    seed = re.search(r"-Seed (\d+)", text)
    difficulty = re.search(r"-Difficulty (\d+)", text)
    map_name = re.search(r"-Map \"?[^\"\r\n]*?([^\"/\\\r\n]+)\.tmx", text)
    arm = re.search(r"^doctrine: (\S+)", text, re.M)
    rival = fields.get("best rival", "")
    dip = re.search(r"worst dip (\d+)", rival)
    peak = re.search(r"peak (\d+)", rival)
    ledger = r"^spend\s+(\S+)\s+asked\s+(\d+)\s+got\s+(\d+)\s+spent\s+(\d+)"
    spends = tuple(
        SpendRow(
            channel=m.group(1),
            asked=int(m.group(2)),
            got=int(m.group(3)),
            spent=int(m.group(4)),
        )
        for m in re.finditer(ledger, text, re.M)
    )
    return ProbeRecord(
        file=stem,
        arm=arm.group(1) if arm else "?",
        seed=int(seed.group(1)) if seed else -1,
        difficulty=int(difficulty.group(1)) if difficulty else -1,
        map=map_name.group(1) if map_name else "",
        verdict=fields.get("verdict", "incomplete").split(" ")[0],
        samples=_leading_int(fields.get("samples seen", "")),
        dip=int(dip.group(1)) if dip else 0,
        peak=int(peak.group(1)) if peak else 0,
        rival_end=_arrow_end(rival.split("(")[0]) if rival else 0,
        extractors_end=_arrow_end(fields.get("extractors", "")),
        attacks=_leading_int(fields.get("attack orders", "")),
        engaged_gone=_leading_int(fields.get("engaged gone", "")),
        spends=spends,
    )


def collect(root: Path) -> tuple[ProbeRecord, ...]:
    """Parse every match scorecard under the runs directory.

    Args:
        root: The runs directory.

    Returns:
        Records sorted by dip, then survival, best first -- the question the
        table exists for is "what moved the needle", so the needle-movers
        read first.
    """
    records = [
        record
        for path in sorted(root.glob("*.out"))
        for record in (parse_probe(path.read_text(encoding="utf-8", errors="replace"), path.stem),)
        if record is not None
    ]

    def needle(record: ProbeRecord) -> tuple[int, int]:
        return (record["dip"], record["samples"])

    return tuple(sorted(records, key=needle, reverse=True))


def index_rows(records: Sequence[ProbeRecord]) -> tuple[str, ...]:
    """Render the one-row-per-match table.

    Args:
        records: The parsed probes.

    Returns:
        Tab-separated rows, header first.
    """
    header = (
        "file",
        "arm",
        "seed",
        "diff",
        "map",
        "verdict",
        "samples",
        "dip",
        "peak",
        "rival_end",
        "extr_end",
        "attacks",
        "engaged_gone",
    )
    lines = ["\t".join(header)]
    for r in records:
        lines.append(
            "\t".join(
                (
                    r["file"],
                    r["arm"],
                    str(r["seed"]),
                    str(r["difficulty"]),
                    r["map"],
                    r["verdict"],
                    str(r["samples"]),
                    str(r["dip"]),
                    str(r["peak"]),
                    str(r["rival_end"]),
                    str(r["extractors_end"]),
                    str(r["attacks"]),
                    str(r["engaged_gone"]),
                )
            )
        )
    return tuple(lines)


def spend_rows(records: Sequence[ProbeRecord]) -> tuple[str, ...]:
    """Render the long-form spend ledger, one row per channel per match.

    Args:
        records: The parsed probes.

    Returns:
        Tab-separated rows, header first.
    """
    lines = ["\t".join(("file", "arm", "seed", "channel", "asked", "got", "spent"))]
    for r in records:
        for s in r["spends"]:
            lines.append(
                "\t".join(
                    (
                        r["file"],
                        r["arm"],
                        str(r["seed"]),
                        s["channel"],
                        str(s["asked"]),
                        str(s["got"]),
                        str(s["spent"]),
                    )
                )
            )
    return tuple(lines)


def main(argv: Sequence[str] | None = None, root: Path = RUNS_ROOT) -> int:
    """Print the table.

    Args:
        argv: ``[spends]`` for the spend ledger, empty for the match index.
            ``None`` reads the process arguments.
        root: The runs directory, a parameter so a test can point it at a
            scratch tree.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on any other argument.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if args not in ([], ["spends"]):
        sys.stdout.write("usage: results [spends]\n")
        return EXIT_BAD_USAGE
    records = collect(root)
    table = spend_rows(records) if args else index_rows(records)
    for line in table:
        sys.stdout.write(line + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
