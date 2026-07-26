"""Play a build order against a live game and print a scorecard.

The bot's entry point. Connects to a running agent, executes a plan by
observing the world and deciding from it, and reports what it achieved so one
run can be compared with another.

Run against a game started with ``-javaagent:...=channelPort=27200``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.control.channel import open_channel
from rw_bot.mechanics.catalogue import UnitStats, decode_catalogue
from rw_bot.policy.runner import format_scorecard, run

#: The opening.
#:
#: Every entry must be something a builder can actually construct. That is not
#: derivable from the unit catalogue -- it carries prices and stats, not build
#: lists -- and the engine reports a refusal only in its own log. A laboratory
#: was in this plan until a live run stalled on it with 11,258 credits banked
#: and the engine saying "Unit 'builder' can not queue build:laboratory".
DEFAULT_PLAN: tuple[str, ...] = (
    "landFactory",
    "landFactory",
    "landFactory",
)

DEFAULT_MAX_SAMPLES = 120

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_BAD_USAGE = 2


def load_catalogue(path: Path) -> dict[str, UnitStats]:
    """Read the unit catalogue produced by ``-printunits``.

    Args:
        path: Archived catalogue dump.

    Returns:
        Unit stats by type name.

    Raises:
        OSError: When the file cannot be read.
        CatalogueError: When the dump cannot be decoded.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    return {unit["type_name"]: unit for unit in decode_catalogue(lines)}


def main(argv: Sequence[str] | None = None) -> int:
    """Connect, play the plan, and report.

    Args:
        argv: ``<port> <catalogue-path> [max-samples]``. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when the plan completed, ``EXIT_INCOMPLETE`` when it did
        not, ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (2, 3):
        sys.stdout.write("usage: play <port> <catalogue-path> [max-samples]\n")
        return EXIT_BAD_USAGE
    max_samples = int(args[2]) if len(args) == 3 else DEFAULT_MAX_SAMPLES

    catalogue = load_catalogue(Path(args[1]))
    sys.stdout.write(f"plan: {' -> '.join(DEFAULT_PLAN)}\n")
    for name in DEFAULT_PLAN:
        sys.stdout.write(f"  {name} costs {catalogue[name]['price']}\n")

    channel = open_channel(int(args[0]))
    card = run(channel, DEFAULT_PLAN, catalogue, max_samples)
    channel.close()

    for line in format_scorecard(card):
        sys.stdout.write(f"{line}\n")
    return EXIT_OK if card["outcome"] == "done" else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))
