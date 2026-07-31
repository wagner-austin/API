"""Every match ever played, as one tab-separated table.

The longitudinal view the per-batch scorecards cannot give: one row per
match across every batch in ``runs/sweeps``, with the batch, arm, seed,
verdict and the load-bearing figures as columns. Pipe it anywhere -- pandas,
a spreadsheet, ``awk`` -- the point is that "how has the champion's win rate
moved across five batches" becomes a one-liner instead of an archaeology
session ([[policy-trace]]).

Run as ``python -m scripts.ledger [sweep-name-filter]``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness.sweep import LABEL_WIDTH

SWEEP_ROOT = Path("runs/sweeps")

#: Scorecard labels carried into the table, in column order.
COLUMNS = (
    "verdict",
    "extractors",
    "income",
    "army value",
    "total worth",
    "best rival",
    "intercepted",
    "raids",
    "marches",
    "samples seen",
)

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def scorecard_fields(text: str) -> dict[str, str]:
    """Read a scorecard's label/value pairs by the shape the sweep trusts.

    Args:
        text: The scorecard file's content.

    Returns:
        Values by label.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) > LABEL_WIDTH and line[LABEL_WIDTH] != " " and line[:1].islower():
            out[line[:LABEL_WIDTH].strip()] = line[LABEL_WIDTH:].strip()
    return out


def rows(root: Path, batch_filter: str) -> list[str]:
    """Build one row per result file under the sweep root.

    Args:
        root: The sweeps directory.
        batch_filter: Substring a batch directory must contain, empty for all.

    Returns:
        Tab-separated rows, header first.
    """
    lines = ["\t".join(("batch", "arm", "seed", *COLUMNS))]
    for batch_dir in sorted(root.iterdir()):
        if not batch_dir.is_dir() or batch_filter not in batch_dir.name:
            continue
        for card in sorted(batch_dir.glob("*.txt")):
            stem = card.stem
            arm, _, seed = stem.rpartition("-s")
            fields = scorecard_fields(card.read_text(encoding="utf-8"))
            lines.append(
                "\t".join((batch_dir.name, arm, seed, *(fields.get(c, "") for c in COLUMNS)))
            )
    return lines


def main(argv: Sequence[str] | None = None, root: Path = SWEEP_ROOT) -> int:
    """Print the table.

    Args:
        argv: ``[batch-filter]``. ``None`` reads the process arguments.
        root: The sweeps directory, a parameter so a test can point it at a
            scratch tree.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on extra arguments.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) > 1:
        sys.stdout.write("usage: ledger [batch-filter]\n")
        return EXIT_BAD_USAGE
    batch_filter = args[0] if args else ""
    for line in rows(root, batch_filter):
        sys.stdout.write(line + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
