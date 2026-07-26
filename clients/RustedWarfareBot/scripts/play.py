"""Play a build order against a live game and print a scorecard.

The bot's entry point. Connects to a running agent, executes a plan by
observing the world and deciding from it, and reports what it achieved so one
run can be compared with another.

Run against a game started with ``-javaagent:...=channelPort=27200``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.control.channel import open_channel
from rw_bot.mechanics.build_tree import decode_build_tree
from rw_bot.mechanics.catalogue import UnitStats, decode_catalogue
from rw_bot.mechanics.placement import TypePlacement, decode_placements
from rw_bot.policy.campaign import fight, format_battle
from rw_bot.policy.expand import expand
from rw_bot.policy.runner import format_scorecard, run

#: What the bot is asked for -- goals, not a build order.
#:
#: The distinction is the point. No factory appears here, and two of these
#: entries need one: a Land Factory is what makes a tank, and the plan the bot
#: actually executes has one inserted before them by
#: :func:`rw_bot.policy.expand.expand`. Writing the prerequisite out by hand is
#: what the build tree exists to stop.
#:
#: Extractors first, because they are the only entry that pays for the rest:
#: each generates credits, and anything built before them is bought with the
#: starting balance and nothing after it. Where they go is not a choice the plan
#: makes -- the engine allows them on resource pools and nowhere else
#: ([[mechanics-resource-pools]]).
DEFAULT_GOALS: tuple[str, ...] = (
    "extractorT1",
    "extractorT1",
    "extractorT1",
    "c_tank",
    "c_tank",
    "c_tank",
    "c_tank",
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


def load_build_tree(path: Path) -> dict[str, frozenset[str]]:
    """Read the build tree produced by ``make type-flags``.

    The same file the placement rules come from, because both are one pass over
    one registry and two files could drift against different game builds.

    Args:
        path: Archived type dump.

    Returns:
        Product type names by producer type name.

    Raises:
        OSError: When the file cannot be read.
        BuildTreeError: When the dump cannot be decoded.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    return decode_build_tree(lines)


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


def reinforcements(
    goals: Sequence[str],
    placements: Mapping[str, TypePlacement],
) -> tuple[str, ...]:
    """Return the goal types worth making again once the plan is finished.

    The plan ends; wanting the units it asked for does not. So reinforcement
    repeats the goals rather than ranking units by some invented notion of
    combat worth -- a number attached to a guess is still a guess.

    Structures are dropped. Rebuilding an extractor means choosing a resource
    pool to put it on, which is the build policy's decision and not something
    a producer queue can express ([[mechanics-resource-pools]]).

    Duplicates are collapsed, keeping first appearance. Asking for four tanks
    does not mean four preferences; it means one, wanted repeatedly.

    Args:
        goals: What the plan was asked for, in order.
        placements: Placement rules by type name, for telling a structure from
            a unit.

    Returns:
        Unit type names to keep making, in preference order.
    """
    wanted: list[str] = []
    for name in goals:
        if placements[name]["needs_pool"] or name in wanted:
            continue
        wanted.append(name)
    return tuple(wanted)


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
    tree = load_build_tree(Path(args[2]))

    # Expansion needs to know what the player already has, so it runs against a
    # real observation rather than an assumed opening roster. One sample is
    # spent on it, which costs nothing: the loop reads its own.
    channel = open_channel(int(args[0]))
    opening = channel.next_sample()
    owned = [e["type_name"] for e in opening["entities"] if e["mine"] and e["complete"]]
    plan = expand(DEFAULT_GOALS, tree, owned, catalogue)

    sys.stdout.write(f"goals: {' -> '.join(DEFAULT_GOALS)}\n")
    sys.stdout.write(f"plan:  {' -> '.join(plan)}\n")
    for name in plan:
        site = "on a resource pool" if placements[name]["needs_pool"] else "on the ring"
        sys.stdout.write(f"  {name} costs {catalogue[name]['price']}, goes {site}\n")

    card = run(channel, plan, catalogue, placements, max_samples)
    for line in format_scorecard(card):
        sys.stdout.write(f"{line}\n")

    # Building is not playing. The fight phase only runs on a plan that
    # finished, because sending a half-built army at five opponents loses the
    # army and proves nothing.
    if card["outcome"] == "done":
        battle = fight(
            channel,
            catalogue,
            max_samples,
            reinforce=reinforcements(DEFAULT_GOALS, placements),
        )
        for line in format_battle(battle):
            sys.stdout.write(f"{line}\n")
    channel.close()

    return EXIT_OK if card["outcome"] == "done" else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))
