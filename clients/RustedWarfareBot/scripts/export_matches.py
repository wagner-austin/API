"""Every match of a batch as training rows, filed where the ML service reads.

The covenant-radar service trains on external CSV datasets by *registration* --
``taiwan`` is a directory holding one ``data.csv``, and ``rw_matches`` is the
same shape for this game. This script writes that file: one row per recorded
sample, joined with the two records a tick alone cannot carry -- the loss
ledger's killer attributions, and the scorecard's verdict, which is the label
the whole dataset exists to predict ([[policy-trace]], [[policy-verdict]]).

Rows carry the match's identity (batch, arm, seed) precisely so the training
side can split by match. Fifteen hundred rows of one match agree with each
other far more than they agree with anything else; split by row, the test set
is the training set wearing a different frame number.

Only traces carrying the income pair (15 columns or more) export: it is the
point of the dataset -- the race law in one column per sample ([[policy-economy]]) -- and a
13-column archive trace has no honest value to put there. Skips are counted
and printed, never silent.

Run as ``python -m scripts.export_matches <batch-name> [<batch-name> ...]``.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from scripts.ledger import scorecard_fields

SWEEP_ROOT = Path("runs/sweeps")
TRACE_ROOT = Path("runs/traces")

#: Where the covenant-radar service's external datasets live, one directory
#: per dataset name. Written relative to this repository because the two are
#: siblings in one monorepo; ``main`` takes it as a parameter like the roots.
EXPORT_ROOT = Path("../../services/covenant-radar-api/data/external/rw_matches")

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_BAD_USAGE = 2

#: Columns of one exported row, in order. ``lost_cum`` is the running sum of
#: the tick's own ``lost`` column; ``killed_cum`` counts only the ledger
#: entries that name a killer -- the distinction the trace records on purpose,
#: a blank killer being a unit that left the roster some other way.
#: ``difficulty`` is the card's own statement of what was played (runner
#: files a ``match`` line since 2026-08-06), blank for cards that predate
#: it -- metadata for slicing, excluded from the model's features exactly
#: like arm and seed.
HEADER = (
    "match",
    "arm",
    "seed",
    "verdict",
    "won",
    "frame",
    "army",
    "credits",
    "enemies",
    "extractors",
    "lost",
    "lost_cum",
    "killed_cum",
    "producers",
    "idle",
    "orders",
    "refused",
    "worth",
    "rival",
    "income",
    "rival_income",
    "difficulty",
)

#: Fewest columns a tick row must split into to export -- the shape with
#: the income pair at 12-13. Later columns (plan at 15, workers at 16) are
#: read when present. The 13-column archive shape is counted, reported and
#: skipped rather than padded: an invented zero income is not a measurement.
_TICK_COLUMNS = 15

#: Column counts of a loss-table row: frame, unit, type, x, y, killer -- the
#: killer column renders ``-`` for blank, but a stripped trailing field can
#: also leave five tokens, so both shapes read.
_LOSS_COLUMNS = (5, 6)

#: The killer token for a loss nothing had damaged: not attributed.
_NO_KILLER = "-"


class ParsedTrace(TypedDict):
    """One trace file, split into the two tables it holds.

    Attributes:
        ticks: The 15-column sample rows, as integers, in recorded order.
        losses: ``(frame, attributed)`` per loss-ledger row, where attributed
            means the row names a killer.
        legacy: Whether the file held tick rows of the pre-income shape.
    """

    ticks: tuple[tuple[int, ...], ...]
    losses: tuple[tuple[int, bool], ...]
    legacy: bool


def parse_trace(text: str) -> ParsedTrace:
    """Read both of a trace file's tables by row shape.

    The same tolerance every other trace reader uses: headers and blank lines
    are skipped because they do not look like data rows, and the two tables
    are told apart by column count alone.

    Args:
        text: The trace file's content.

    Returns:
        The parsed trace.
    """
    ticks: list[tuple[int, ...]] = []
    losses: list[tuple[int, bool]] = []
    legacy = False
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        if len(parts) >= _TICK_COLUMNS:
            ticks.append(tuple(int(p) for p in parts[:_TICK_COLUMNS]))
        elif len(parts) in _LOSS_COLUMNS:
            killer = parts[5] if len(parts) == 6 else _NO_KILLER
            losses.append((int(parts[0]), killer != _NO_KILLER))
        else:
            legacy = True
    return ParsedTrace(ticks=tuple(ticks), losses=tuple(losses), legacy=legacy)


def match_rows(
    match: str, arm: str, seed: str, verdict: str, difficulty: str, parsed: ParsedTrace
) -> tuple[str, ...]:
    """Join one match's ticks with its loss ledger and verdict.

    Args:
        match: The row identity the training side splits on.
        arm: The sweep arm the match ran under.
        seed: The match's seed.
        verdict: The scorecard's grade, first word.
        difficulty: The card's stated difficulty, empty when the card
            predates the ``match`` line.
        parsed: The match's parsed trace.

    Returns:
        One CSV row per tick, in recorded order.
    """
    won = "1" if verdict == "won" else "0"
    kills = sorted(frame for frame, attributed in parsed["losses"] if attributed)
    rows = []
    lost_cum = 0
    killed = 0
    for tick in parsed["ticks"]:
        lost_cum += tick[5]
        while killed < len(kills) and kills[killed] <= tick[0]:
            killed += 1
        rows.append(
            ",".join(
                (
                    match,
                    arm,
                    seed,
                    verdict,
                    won,
                    *(str(value) for value in tick[:5]),
                    str(tick[5]),
                    str(lost_cum),
                    str(killed),
                    *(str(value) for value in tick[6:14]),
                    difficulty,
                )
            )
        )
    return tuple(rows)


def export(batches: Sequence[str], sweeps: Path, traces: Path) -> tuple[list[str], list[str]]:
    """Collect every exportable match of the named batches.

    Args:
        batches: The batch names, in the order given.
        sweeps: The scorecard root.
        traces: The trace root.

    Returns:
        The CSV data rows, and one note per skipped match saying why.
    """
    rows: list[str] = []
    skipped: list[str] = []
    for batch in batches:
        for path in sorted((traces / batch).glob("*.ndjson")):
            parsed = parse_trace(path.read_text(encoding="utf-8"))
            card = sweeps / batch / f"{path.stem}.txt"
            if not parsed["ticks"]:
                shape = "pre-income trace shape" if parsed["legacy"] else "empty trace"
                skipped.append(f"{batch}/{path.stem}: {shape}")
                continue
            if not card.exists():
                skipped.append(f"{batch}/{path.stem}: no scorecard")
                continue
            fields = scorecard_fields(card.read_text(encoding="utf-8"))
            verdict = fields.get("verdict", "?").split(" ")[0]
            found = re.search(r"difficulty (-?\d+)", fields.get("match", ""))
            difficulty = found.group(1) if found else ""
            arm, _, seed = path.stem.rpartition("-s")
            rows.extend(match_rows(f"{batch}/{path.stem}", arm, seed, verdict, difficulty, parsed))
    return rows, skipped


def main(
    argv: Sequence[str] | None = None,
    sweeps: Path = SWEEP_ROOT,
    traces: Path = TRACE_ROOT,
    dest: Path = EXPORT_ROOT,
) -> int:
    """Write the dataset for the batches named on the command line.

    Args:
        argv: ``<batch-name> [<batch-name> ...]``. ``None`` reads the process
            arguments.
        sweeps: The scorecard root, a parameter so a test can point it at a
            scratch tree.
        traces: The trace root, likewise.
        dest: The dataset directory, likewise. Gains ``data.csv``.

    Returns:
        ``EXIT_OK``, ``EXIT_EMPTY`` when nothing exported, or
        ``EXIT_BAD_USAGE`` on an empty argument list.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        sys.stdout.write("usage: export_matches <batch-name> [<batch-name> ...]\n")
        return EXIT_BAD_USAGE
    rows, skipped = export(args, sweeps, traces)
    for note in skipped:
        sys.stdout.write(f"skipped {note}\n")
    if not rows:
        sys.stdout.write("nothing to export\n")
        return EXIT_EMPTY
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "data.csv"
    target.write_text("\n".join([",".join(HEADER), *rows]) + "\n", encoding="utf-8")
    matches = len({row.split(",", 1)[0] for row in rows})
    sys.stdout.write(f"wrote {len(rows)} rows from {matches} matches to {target}\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
