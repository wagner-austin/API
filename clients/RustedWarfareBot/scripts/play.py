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
from rw_bot.mechanics.placement import TypePlacement, decode_placements
from rw_bot.policy.runner import format_scorecard, run

#: The opening.
#:
#: Every entry must be something a builder can actually construct. That is not
#: derivable from the unit catalogue -- it carries prices and stats, not build
#: lists -- and the engine reports a refusal only in its own log. A laboratory
#: was in this plan until a live run stalled on it with 11,258 credits banked
#: and the engine saying "Unit 'builder' can not queue build:laboratory".
#:
#: Extractors first, because they are the only entry that pays for the rest:
#: each generates credits, and a factory built before them is a factory bought
#: with the starting balance and nothing after it. Where they go is not a choice
#: the plan makes -- the engine allows them on resource pools and nowhere else
#: ([[mechanics-resource-pools]]).
#: The last entry is a unit rather than a structure, and it is what proves the
#: second verb. A Scout is produced by the Command Center the match starts
#: with, so it needs no prerequisite -- and at $700 it is dear enough that a
#: fixed stall window sized for a Builder would have declared it refused.
DEFAULT_PLAN: tuple[str, ...] = (
    "extractorT1",
    "extractorT1",
    "landFactory",
    "extractorT1",
    "landFactory",
    "scout",
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


def load_placements(path: Path) -> dict[str, TypePlacement]:
    """Read the placement rules produced by ``make type-flags``.

    Args:
        path: Archived placement dump.

    Returns:
        Placement rules by type name.

    Raises:
        OSError: When the file cannot be read.
        PlacementError: When the dump cannot be decoded.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    return {place["type_name"]: place for place in decode_placements(lines)}


def main(argv: Sequence[str] | None = None) -> int:
    """Connect, play the plan, and report.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [max-samples]``.
            ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when the plan completed, ``EXIT_INCOMPLETE`` when it did
        not, ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write("usage: play <port> <catalogue-path> <placement-path> [max-samples]\n")
        return EXIT_BAD_USAGE
    max_samples = int(args[3]) if len(args) == 4 else DEFAULT_MAX_SAMPLES

    catalogue = load_catalogue(Path(args[1]))
    placements = load_placements(Path(args[2]))
    sys.stdout.write(f"plan: {' -> '.join(DEFAULT_PLAN)}\n")
    for name in DEFAULT_PLAN:
        site = "on a resource pool" if placements[name]["needs_pool"] else "on the ring"
        sys.stdout.write(f"  {name} costs {catalogue[name]['price']}, goes {site}\n")

    channel = open_channel(int(args[0]))
    card = run(channel, DEFAULT_PLAN, catalogue, placements, max_samples)
    channel.close()

    for line in format_scorecard(card):
        sys.stdout.write(f"{line}\n")
    return EXIT_OK if card["outcome"] == "done" else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))
