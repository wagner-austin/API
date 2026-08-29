"""Tabulate any A/B sweep: per-match rows, then per-arm aggregates.

The per-batch reading companion to :mod:`scripts.ledger`'s cross-batch table.
Joins each scorecard with its trace so extractor drops -- the figure every
verdict this project has produced turns on -- appear beside the endpoint
numbers ([[policy-trace]]).

Run as ``python -m scripts.analyze_sweep <sweep-name>``.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.provenance import summarize_arm

LABEL_WIDTH = 15

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_BAD_USAGE = 2


def fields(path: Path) -> dict[str, str]:
    """Read a scorecard's label/value pairs by shape.

    Args:
        path: The scorecard file.

    Returns:
        Values by label.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if len(line) > LABEL_WIDTH and line[LABEL_WIDTH] != " " and line[0].islower():
            out[line[:LABEL_WIDTH].strip()] = line[LABEL_WIDTH:].strip()
    return out


def arrow_end(value: str) -> int:
    """Return the end figure of a ``start -> end`` scorecard value.

    Args:
        value: The raw field text.

    Returns:
        The end integer, zero when the shape is absent.
    """
    m = re.search(r"->\s*(-?\d+)", value)
    return int(m.group(1)) if m else 0


def drops(batch: str, stem: str, traces: Path) -> tuple[int, int]:
    """Return the extractor peak and peak-to-end drop from a match's trace.

    Args:
        batch: The sweep the match belongs to.
        stem: The match's file stem.
        traces: The trace root directory.

    Returns:
        Peak extractors, and how many of them were gone by the end.
    """
    trace = traces / batch / f"{stem}.ndjson"
    peak = end = 0
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 12 and parts[0].isdigit():
                value = int(parts[4])
                peak = max(peak, value)
                end = value
    return peak, peak - end


def main(
    argv: Sequence[str] | None = None,
    sweeps: Path = Path("runs/sweeps"),
    traces: Path = Path("runs/traces"),
) -> int:
    """Print the table for the sweep named on the command line.

    Args:
        argv: ``<sweep-name>``. ``None`` reads the process arguments.
        sweeps: The results root, a parameter so a test can point it at a
            scratch tree instead of the real record.
        traces: The trace root, likewise.

    Returns:
        ``EXIT_OK``, or ``EXIT_EMPTY`` when the batch has no results yet.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stdout.write("usage: analyze_sweep <sweep-name>\n")
        return EXIT_BAD_USAGE
    batch = args[0]
    results = sweeps / batch
    rows: list[dict[str, str | int]] = []
    for path in sorted(results.glob("*.txt")):
        arm, _, seed = path.stem.rpartition("-s")
        f = fields(path)
        rival = f.get("best rival", "")
        dip = re.search(r"worst dip (\d+)", rival)
        enemies = f.get("enemies seen", "")
        engage = re.search(r"\((\d+) engageable\)", enemies)
        peak, dropped = drops(batch, path.stem, traces)
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "verdict": f.get("verdict", "?").split(" ")[0],
                "extr_end": arrow_end(f.get("extractors", "0 -> 0")),
                "peak": peak,
                "dropped": dropped,
                "worth_end": arrow_end(f.get("total worth", "0 -> 0")),
                "rival_end": arrow_end(rival.split("(")[0]) if rival else 0,
                "dip": int(dip.group(1)) if dip else 0,
                "targets_end": arrow_end(enemies.split("(")[0]) if enemies else 0,
                "engageable": int(engage.group(1)) if engage else 0,
                "intercepted": int(f.get("intercepted", "0") or 0),
                "income": f.get("income", "?"),
            }
        )
    if not rows:
        sys.stdout.write("no results yet\n")
        return EXIT_EMPTY
    sys.stdout.write(
        f"{'arm':8} {'seed':>8} {'verdict':10} {'extr':>4} {'peak':>4} {'drop':>4} "
        f"{'worth':>7} {'rival':>7} {'dip':>6} {'tgts':>4} {'eng':>3} {'icpt':>5} {'income':>7}\n"
    )
    for r in sorted(rows, key=_row_order):
        sys.stdout.write(
            f"{r['arm']:8} {r['seed']:>8} {r['verdict']:10} {r['extr_end']:>4} {r['peak']:>4} "
            f"{r['dropped']:>4} {r['worth_end']:>7} {r['rival_end']:>7} {r['dip']:>6} "
            f"{r['targets_end']:>4} {r['engageable']:>3} {r['intercepted']:>5} {r['income']:>7}\n"
        )
    sys.stdout.write("\n")
    for arm in sorted({str(r["arm"]) for r in rows}):
        summary = summarize_arm(rows, arm)
        sys.stdout.write(
            f"{arm:8}  won {summary['wins']}/{summary['matches']}  "
            f"lost {summary['losses']}  drops {summary['drops']}  "
            f"median worth {summary['median_worth']}  "
            f"unengageable {summary['unengageable']}  "
            f"intercepts {summary['intercepts']}\n"
        )
    return EXIT_OK


def _row_order(row: dict[str, str | int]) -> tuple[str, int]:
    """Order rows by arm then numeric seed."""
    return (str(row["arm"]), int(str(row["seed"])))


if __name__ == "__main__":
    raise SystemExit(main(None))
