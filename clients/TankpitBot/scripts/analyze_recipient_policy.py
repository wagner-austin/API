"""Mine the capture archive for each message family's recipient policy.

Which connections receive a given server message? A single-client sim
cannot tell "broadcast to the room" from "send to this client", so the
archive decides it ([[recipient-policy]]).

Usage: poetry run python -m scripts.analyze_recipient_policy [dir ...]

With no arguments both archive directories are swept.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import _test_hooks
from tankpit_bot.analysis.recipient_policy import (
    analyze_recipient_policy,
    format_recipient_policy,
)

#: Swept when the command line names no directory. Both hold
#: ``*.capture_session.json`` files: ``runs/bot`` the bot's own
#: sessions, ``runs/sniff`` the user-piloted captures.
DEFAULT_DIRECTORIES: tuple[Path, ...] = (Path("runs") / "bot", Path("runs") / "sniff")


def _requested_directories() -> list[Path]:
    """Resolve the directories to sweep from the command line.

    Returns:
        The directories named on the command line, or
        :data:`DEFAULT_DIRECTORIES` when none were.
    """
    if len(sys.argv) > 1:
        return [Path(argument) for argument in sys.argv[1:]]
    return list(DEFAULT_DIRECTORIES)


def main() -> None:
    """Sweep the archive and print each family's evidence and verdict.

    Raises:
        SystemExit: If a named directory does not exist. A silently
            skipped directory would report a verdict over an archive it
            never read, which is the one failure this sweep must not
            have.
    """
    _test_hooks.setup_rich_logging(level="INFO")

    directories = _requested_directories()
    for directory in directories:
        if not _test_hooks.path_exists(directory):
            sys.stdout.write(f"No such directory: {directory}\n")
            raise SystemExit(1)

    result = analyze_recipient_policy(directories)
    sys.stdout.write(format_recipient_policy(result))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_DIRECTORIES", "main"]
