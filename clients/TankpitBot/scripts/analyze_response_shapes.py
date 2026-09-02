"""Diff the real server's response shapes against the sim's.

A shape only the real server produces is a MISSING law; a shape only
the sim produces is an INVENTED one ([[capture-differ]]).

Usage: poetry run python -m scripts.analyze_response_shapes [sim_dir ...]

The LIVE side is always the real archive. The argument names the SIM
side, because the useful question is almost always "how faithful is
THIS sim", and a mixed ``runs/sim`` answers it about several past sims
at once: the directory accumulates every baseline ever generated, so
captures from before a fix still carry the pre-fix shapes. Point this
at a freshly generated baseline to read the CURRENT sim; the default
reads whatever ``runs/sim`` holds.

Read the invented side first. The two archives are rarely sampled at
comparable size, and an under-sampled sim makes almost every live shape
look "missing" while saying nothing about fidelity — whereas a shape
the sim DID produce is one it produces regardless of sample size.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import _test_hooks
from tankpit_bot.analysis.response_shapes import (
    analyze_response_shapes,
    format_response_shape_diff,
)

#: The real archive: the bot's own sessions and the user-piloted sniffs.
LIVE_DIRECTORIES: tuple[Path, ...] = (Path("runs") / "bot", Path("runs") / "sniff")

#: Swept as the sim side when the command line names no directory.
DEFAULT_SIM_DIRECTORIES: tuple[Path, ...] = (Path("runs") / "sim",)

#: Divergence rows rendered per verdict. The report says explicitly how
#: many it dropped rather than truncating in silence.
ROW_LIMIT = 20


def _sim_directories() -> list[Path]:
    """Resolve the sim directories to diff against.

    Returns:
        The directories named on the command line, or
        :data:`DEFAULT_SIM_DIRECTORIES` when none were.
    """
    if len(sys.argv) > 1:
        return [Path(argument) for argument in sys.argv[1:]]
    return list(DEFAULT_SIM_DIRECTORIES)


def main() -> None:
    """Diff the archives and print every one-sided response shape.

    Raises:
        SystemExit: If a named directory does not exist. Sweeping on
            regardless would report a fidelity verdict over an archive
            the run never read — and with an empty sim side every live
            shape reads as a missing law, which looks like a result.
    """
    _test_hooks.setup_rich_logging(level="INFO")

    directories = list(LIVE_DIRECTORIES) + _sim_directories()
    for directory in directories:
        if not _test_hooks.path_exists(directory):
            sys.stdout.write(f"No such directory: {directory}\n")
            raise SystemExit(1)

    diff = analyze_response_shapes(list(LIVE_DIRECTORIES), _sim_directories())
    sys.stdout.write(format_response_shape_diff(diff, ROW_LIMIT))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_SIM_DIRECTORIES", "LIVE_DIRECTORIES", "ROW_LIMIT", "main"]
