"""Does the sim survive every command a real client actually sends?

Usage: poetry run python -m scripts.analyze_command_coverage [dir ...]

Exits NONZERO when the archive holds a command byte the sim does not
map, because that is not a report — it is a crash waiting for the
first real client. ``SimServer.queue_command`` refuses any kind
outside ``SUPPORTED_KINDS`` and every client frame reaches it.

Not part of ``make check``: the archive lives under gitignored
``runs/``, so a fresh clone has nothing to audit. It belongs beside
the other archive-driven checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import _test_hooks
from tankpit_bot.analysis.command_coverage import (
    analyze_command_coverage,
    crashing_rows,
    format_command_coverage,
)

#: The real archive: the bot's own sessions and the user-piloted
#: sniffs. The sniffs are what make this audit work at all — they are
#: REAL CLIENT traffic, and the commands our bot never sends are
#: exactly the ones that crash the sim ([[client-commands]]).
DEFAULT_DIRECTORIES: tuple[Path, ...] = (Path("runs") / "bot", Path("runs") / "sniff")


def _directories() -> list[Path]:
    """Resolve the archive directories to audit.

    Returns:
        The directories named on the command line, or
        :data:`DEFAULT_DIRECTORIES` when none were.
    """
    if len(sys.argv) > 1:
        return [Path(argument) for argument in sys.argv[1:]]
    return list(DEFAULT_DIRECTORIES)


def main() -> None:
    """Audit client-command coverage and fail on an unmapped byte.

    Raises:
        SystemExit: Code 1 if a named directory does not exist, or if
            the archive holds a command byte the sim cannot handle.
    """
    _test_hooks.setup_rich_logging(level="INFO")
    directories = _directories()
    for directory in directories:
        if not _test_hooks.path_exists(directory):
            sys.stdout.write(f"No such directory: {directory}\n")
            raise SystemExit(1)

    coverage = analyze_command_coverage(directories)
    sys.stdout.write(format_command_coverage(coverage))
    sys.stdout.write("\n")
    if crashing_rows(coverage):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_DIRECTORIES", "main"]
