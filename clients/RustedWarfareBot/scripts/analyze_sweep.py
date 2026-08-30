"""Tabulate any A/B sweep: per-match rows, then per-arm aggregates.

The per-batch reading companion to :mod:`scripts.ledger`'s cross-batch table.
Joins each scorecard with its trace so extractor drops -- the figure every
verdict this project has produced turns on -- appear beside the endpoint
numbers ([[policy-trace]]).

Run as ``python -m scripts.analyze_sweep <sweep-name>``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness.records import read_batch_rows
from rw_bot.harness.scorecards import row_order
from rw_bot.provenance import summarize_arm

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_BAD_USAGE = 2


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
    # Read through the one reader a batch has. This script used to carry its
    # own copy of the label/value parser and its own row shape, differing from
    # the harness's by a single character -- two readers of one format, one
    # edit away from a table and an aggregate that disagree about the same
    # batch, with neither of them failing.
    rows = list(read_batch_rows(sweeps / batch, traces, batch))
    if not rows:
        sys.stdout.write("no results yet\n")
        return EXIT_EMPTY
    sys.stdout.write(
        f"{'arm':8} {'seed':>8} {'verdict':10} {'extr':>4} {'peak':>4} {'drop':>4} "
        f"{'worth':>7} {'rival':>7} {'dip':>6} {'tgts':>4} {'eng':>3} {'icpt':>5} {'income':>7}\n"
    )
    for r in sorted(rows, key=row_order):
        sys.stdout.write(
            f"{r['arm']:8} {r['seed']:>8} {r['verdict']:10} {r['extr_end']:>4} {r['peak']:>4} "
            f"{r['dropped']:>4} {r['worth_end']:>7} {r['rival_end']:>7} {r['dip']:>6} "
            f"{r['targets_end']:>4} {r['engageable']:>3} {r['intercepted']:>5} {r['income']:>7}\n"
        )
    sys.stdout.write("\n")
    for arm in sorted({row["arm"] for row in rows}):
        summary = summarize_arm(rows, arm)
        sys.stdout.write(
            f"{arm:8}  won {summary['wins']}/{summary['matches']}  "
            f"lost {summary['losses']}  drops {summary['drops']}  "
            f"median worth {summary['median_worth']}  "
            f"unengageable {summary['unengageable']}  "
            f"intercepts {summary['intercepts']}\n"
        )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
