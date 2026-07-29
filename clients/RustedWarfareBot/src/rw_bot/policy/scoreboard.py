"""Reading the engine's own scoreboard.

``gameFramework.g.f`` computes income, army value and building value per player
and the agent puts one record per player on the wire, so how the match is going
is read rather than estimated. Everything here is a pure function of one
observation.

The distinction these functions exist to keep straight is *whose* figure is
being read. Our own total says nothing on its own; measured against an opponent
it says whether the match is being lost, and measured against an opponent's own
history it says whether the army has ever cost anybody anything -- which are
three different questions that one number cannot answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.combat import PLACEHOLDER_TYPE, is_mobile
from rw_bot.wire.state import Entity, PlayerStat, Sample


def local_player(sample: Sample) -> PlayerStat | None:
    """Return the local player's scoreboard row, or None when the stream has none.

    Args:
        sample: One observation of the world.

    Returns:
        The row, or None on a stream that predates the player record.
    """
    for player in sample["players"]:
        if player["local"]:
            return player
    return None


def worth_of(player: PlayerStat | None) -> int:
    """Return everything a player holds, mobile and standing alike.

    Army value alone is the wrong score for a bot that can buy defences. A
    turret is a building, so the engine books it under building value -- and
    reading only the army figure would show the most cost-effective thing the
    bot can buy as a pure loss ([[policy-production]]).

    Args:
        player: The scoreboard row, or None when the stream carries none.

    Returns:
        Army value plus building value, or zero when there is no row.
    """
    if player is None:
        return 0
    return player["army_value"] + player["building_value"]


def standing_of(sample: Sample, catalogue: Mapping[str, UnitStats]) -> tuple[Entity, ...]:
    """Return the finished buildings the player owns.

    **Added because a policy could not be told from its absence.** Defence is
    the third expansion question and nothing reported whether it had ever bought
    anything: our own buildings appear in no report line, the trace carries no
    column for them, and the expander keeps the *income* reason when defence
    declines, so the defence reason never reaches a log at all
    ([[policy-holding-ground]]). Asked whether a single turret had ever been
    built in twelve full matches, the honest answer was that the run output
    could not say -- which is a measurement gap rather than a finding.

    Buildings rather than everything owned, because the mobile half already has
    two lines of its own and the question here is what is standing on the map.

    The map editor's placeholder is excluded, the same exclusion every other
    reader of the roster makes and for the same reason: it is owned, finished
    and immobile, so it passes every structural test and is not a building. Its
    first appearance here was in a wiped match reporting ``structures
    editorOrBuilder x1`` -- which read as one surviving building where the truth
    was none ([[policy-loop]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for telling a building from a unit.

    Returns:
        The finished immobile entities the player owns, in roster order.
    """
    return tuple(
        entity
        for entity in sample["entities"]
        if entity["mine"]
        and entity["complete"]
        and entity["type_name"] != PLACEHOLDER_TYPE
        and not is_mobile(entity, catalogue)
    )


def composition_of(entities: Sequence[Entity]) -> tuple[tuple[str, int], ...]:
    """Return how many of each type a set of entities is made of.

    What it counts is decided entirely by what it is handed, and there are two
    callers that matter. Given the army it reports the mix that actually
    fights, builders and structures having already been excluded
    ([[policy-combat]]). Given the visible hostiles it reports what the
    opponents are fielding -- which is the only way to see whether they hold
    types the bot cannot reach, and the question a whole tier of the game turns
    on ([[policy-holding-ground]]).

    Ordered commonest first, and by type name within a tie, so two runs that
    built the same army report it identically and a diff between them is a
    difference in the game rather than in dictionary iteration.

    Args:
        entities: The entities to count.

    Returns:
        Type name and count, commonest first.
    """

    def rank(pair: tuple[str, int]) -> tuple[int, str]:
        return -pair[1], pair[0]

    counted: dict[str, int] = {}
    for unit in entities:
        counted[unit["type_name"]] = counted.get(unit["type_name"], 0) + 1
    return tuple(sorted(counted.items(), key=rank))


def deepest_dip(peaks: dict[int, int], sample: Sample) -> int:
    """Update each opponent's running peak and return the deepest fall from one.

    **Per opponent, because the maximum hides exactly the case that matters.**
    Measuring the dip against :func:`best_rival` asks whether the *leader* was
    ever set back, and grinding a weaker opponent down while a stronger one
    builds leaves that maximum tracking the stronger -- so a match spent
    successfully destroying somebody reports the same zero as a match spent
    achieving nothing. Measured that way nine matches all reported roughly the
    same 700, which should have been suspicious on its own: an identical figure
    across arms that differ is a constant, not a measurement. Per opponent the
    same matches show 1,600 to 8,150 ([[policy-verdict]]).

    Peaks are carried in the caller's dictionary and updated in place, the same
    shape the attack bookkeeping uses, so the loop keeps one copy of the state
    rather than threading a second accumulator through the report.

    Args:
        peaks: Highest worth seen per player index, updated in place.
        sample: One observation of the world.

    Returns:
        The largest fall from any single opponent's own peak at this
        observation. Zero when every opponent is at or above its peak.
    """
    deepest = 0
    for player in sample["players"]:
        if not player["hostile"]:
            continue
        worth = worth_of(player)
        peak = max(peaks.get(player["index"], worth), worth)
        peaks[player["index"]] = peak
        deepest = max(deepest, peak - worth)
    return deepest


def best_rival(sample: Sample) -> int:
    """Return the strongest hostile player's total worth.

    The comparison is the point. Our own figure says nothing on its own;
    measured against the strongest opponent it says whether the match is being
    lost, and unlike the visible-enemy count it cannot be inflated by our own
    scouting ([[policy-verdict]]).

    Args:
        sample: One observation of the world.

    Returns:
        The largest hostile total, or zero when nothing hostile remains.
    """
    return max((worth_of(p) for p in sample["players"] if p["hostile"]), default=0)


__all__ = [
    "best_rival",
    "composition_of",
    "deepest_dip",
    "local_player",
    "standing_of",
    "worth_of",
]
