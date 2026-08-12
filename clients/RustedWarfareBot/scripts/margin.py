"""CLI over the dense margin: per-batch means and paired deltas.

The scoring itself lives in :mod:`rw_bot.harness.margin` so the
doctrine-search driver can read it without importing a script; this
module only walks ``runs/sweeps`` and prints.

Run as ``python -m scripts.margin <batch> [batch...]``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness.margin import batch_margins, report

SWEEP_ROOT = Path("runs/sweeps")

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None, root: Path = SWEEP_ROOT) -> int:
    """Print margin summaries for the named batches.

    Args:
        argv: Batch names under the sweep root. ``None`` reads
            ``sys.argv[1:]``.
        root: The sweeps directory, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` with no batches named.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        sys.stdout.write("usage: margin <batch> [batch...]\n")
        return EXIT_BAD_USAGE
    for batch in args:
        for line in report(batch, batch_margins(root / batch)):
            sys.stdout.write(line + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
