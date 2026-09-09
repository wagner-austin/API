"""CLI over the dense margin: per-batch means, paired deltas, and power.

The scoring itself lives in :mod:`rw_bot.harness.margin` and the power
statements in :mod:`rw_bot.harness.verdict_power`, so the doctrine-search
driver can read both without importing a script; this module only walks
``runs/sweeps`` and prints.

Run as ``python -m scripts.margin <batch> [batch...] [--power <effect>]``.
With ``--power``, every batch's report is followed by the shared power
instruments over the same cards: the paired-continuous minimum detectable
effect against the stated effect of interest, and the win-bit McNemar test
with its falsifiability reach -- the arithmetic every panel verdict used to
hand-roll, now emitted by tracked code (log 2026-09-09, the fleet
research-integrity audit).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness.margin import batch_margins, report
from rw_bot.harness.verdict_power import paired_power_lines, win_power_lines

SWEEP_ROOT = Path("runs/sweeps")

EXIT_OK = 0
EXIT_BAD_USAGE = 2

USAGE = "usage: margin <batch> [batch...] [--power <effect-of-interest>]\n"


def main(argv: Sequence[str] | None = None, root: Path = SWEEP_ROOT) -> int:
    """Print margin summaries, and power statements when asked.

    Args:
        argv: Batch names under the sweep root, optionally followed by
            ``--power <effect>``. ``None`` reads ``sys.argv[1:]``.
        root: The sweeps directory, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` with no batches named or a
        ``--power`` flag missing its effect.

    Raises:
        ValueError: When the ``--power`` effect is not a number -- the same
            loud path every numeric argument in this scripts package takes.
        AppError: Through the power instruments, when a pair is too thin to
            power or the effect of interest is not positive.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    effect: float | None = None
    if "--power" in args:
        at = args.index("--power")
        if at + 1 >= len(args):
            sys.stdout.write(USAGE)
            return EXIT_BAD_USAGE
        effect = float(args[at + 1])
        del args[at : at + 2]
    if not args:
        sys.stdout.write(USAGE)
        return EXIT_BAD_USAGE
    for batch in args:
        margins = batch_margins(root / batch)
        for line in report(batch, margins):
            sys.stdout.write(line + "\n")
        if effect is not None:
            for line in paired_power_lines(margins, effect):
                sys.stdout.write(line + "\n")
            for line in win_power_lines(margins):
                sys.stdout.write(line + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
