"""Deciding what to attack, and with what.

Separate from :mod:`rw_bot.policy.build_order` because it answers a different
question. That module decides what to *make*; this one decides what to *do*
with what was made, and the two share nothing but the sample they read. Neither
opens a socket -- dispatch is the runner's concern.

The bot needed this because building well is not playing. Measured over five
minutes past a completed plan, it lost nothing and took no damage while banking
credits from 8,539 to 21,164 and watching visible enemy units go from 54 to 126
([[policy-loop]]). It was not winning; it had not been reached yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.wire.state import Entity, Sample

#: Type name of the map editor's placeholder, which is owned and cannot fight.
#:
#: The same exclusion producer selection needs, for the same reason: it is an
#: owned entity in every sample and not a playable unit ([[policy-loop]]).
PLACEHOLDER_TYPE = "editorOrBuilder"


#: Units each successive wave waits for, in order.
#:
#: The shipped AI's ladder: three for the first attack, five for the next few,
#: seven thereafter. Its groups are created empty with a target size and recruit
#: until full before they move, and the size climbs with the number of groups it
#: has already sent ([[engine-ai-triggers]]).
WAVE_SIZES = (3, 3, 5, 5, 5, 7)


def wave_size(waves_sent: int) -> int:
    """Return how many units the next wave waits for.

    Args:
        waves_sent: Waves already released.

    Returns:
        The size the next one needs, the last rung repeating thereafter.
    """
    return WAVE_SIZES[min(waves_sent, len(WAVE_SIZES) - 1)]


class Muster(TypedDict):
    """Who may attack this sample, and who is still gathering.

    Attributes:
        released: Engine ids cleared to attack.
        gathering: Units waiting to form the next wave.
        wanted: How many the next wave needs.
        waves: Waves released so far, including any released this sample.
        reason: Human-readable justification, for the run log.
    """

    released: frozenset[int]
    gathering: int
    wanted: int
    waves: int
    reason: str


def muster(army: Sequence[Entity], released: frozenset[int], waves: int) -> Muster:
    """Decide which units are cleared to attack, and which keep gathering.

    Fill, then commit. Attacking with whatever exists feeds units in one at a
    time and loses each of them separately; the same units sent together are a
    wave. That is the shipped AI's rule ([[engine-ai-triggers]]).

    **Membership, not a flag.** A boolean "have we started" was the first
    attempt and it was worse than nothing: it latched on the first wave and
    every reinforcement thereafter walked into the fight alone, which is the
    trickle the rule exists to prevent. Measured over 1,500 samples it produced
    45 reinforcements for a net army growth of one. So a unit is either in a
    released wave or in the reserve, and only the reserve gathers.

    Survivors of a released wave keep their clearance. They do not turn round
    to wait for reinforcements — abandoning an attack while still in range of
    it is the worst of both behaviours, and it is why the reserve is counted
    separately rather than the whole army being re-tested against the
    threshold.

    Args:
        army: Units available to fight, as :func:`find_army` reports them.
        released: Engine ids already cleared by an earlier wave.
        waves: Waves released so far.

    Returns:
        The decision, carrying the state the next call needs.
    """
    alive = {unit["unit_id"] for unit in army}
    survivors = alive & released
    reserve = alive - released
    wanted = wave_size(waves)

    if len(reserve) >= wanted:
        return Muster(
            released=frozenset(alive),
            gathering=0,
            wanted=wave_size(waves + 1),
            waves=waves + 1,
            reason=f"wave {waves + 1} of {len(reserve)} released",
        )
    return Muster(
        released=frozenset(survivors),
        gathering=len(reserve),
        wanted=wanted,
        waves=waves,
        reason=f"{len(survivors)} committed, mustering {len(reserve)}/{wanted}",
    )


#: How close counts as arrived at the rally point, in world units.
#:
#: The engine's own rally group drops a member once it is within this of the
#: centre — a squared 3,600 in its tick, so 60 ([[engine-ai-zones]]). Reused
#: rather than guessed because the question is identical: when has a unit
#: finished gathering.
RALLY_RADIUS = 60.0


class Deployment(TypedDict):
    """One unit ordered to a position.

    Attributes:
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.
        reason: Why, for the run log.
    """

    unit_id: int
    x: float
    y: float
    reason: str


def rally(reserve: Sequence[Entity], point: tuple[float, float]) -> tuple[Deployment, ...]:
    """Send the units still gathering to the place they gather.

    The wave gate created a reserve and gave it nowhere to be. Units that are
    not yet cleared to attack sit wherever they rolled out of the factory,
    which spreads the next wave across the map and means it arrives piecemeal
    even after the gate releases it — the trickle again, one step earlier.

    Rallying them at a point solves that and doubles as the only defensive
    posture the bot has: units waiting near the base are units standing between
    an attacker and the base.

    Already-arrived units are not re-ordered. The engine runs a waypoint until
    it is replaced, so re-issuing every sample would reset the walk at the
    sampling rate and nothing would ever arrive — the same failure the attack
    path already learned ([[policy-combat]]).

    Args:
        reserve: Units still gathering, which is the army minus the released
            wave.
        point: Where to gather, as world x and y.

    Returns:
        One deployment per unit not yet within :data:`RALLY_RADIUS`.
    """
    limit = RALLY_RADIUS**2
    return tuple(
        Deployment(
            unit_id=unit["unit_id"],
            x=point[0],
            y=point[1],
            reason=f"{unit['type_name']} rallying",
        )
        for unit in reserve
        if (unit["x"] - point[0]) ** 2 + (unit["y"] - point[1]) ** 2 > limit
    )


class Engagement(TypedDict):
    """One unit ordered onto one target.

    Attributes:
        attacker_id: Engine identity of the unit to order.
        target_id: Engine identity of the unit to attack.
        reason: Why this pairing, for the run log.
    """

    attacker_id: int
    target_id: int
    reason: str


def is_armed(entity: Entity, catalogue: Mapping[str, UnitStats]) -> bool:
    """Report whether an entity has a weapon at all.

    Read from the catalogue rather than guessed from the type name. The engine
    prints an attack range for units that have one and omits it for units that
    do not, so a Builder is unarmed by the same source that says a Tank is not
    ([[mechanics-unit-catalogue]]).

    A type the catalogue does not know counts as unarmed. Sending a unit whose
    weapon is unknown into a fight is the more expensive of the two mistakes.

    Args:
        entity: The entity to test.
        catalogue: Unit stats by type name.

    Returns:
        True when the catalogue gives it a weapon.
    """
    stats = catalogue.get(entity["type_name"])
    return stats is not None and stats["weapon"] is not None


def is_mobile(entity: Entity, catalogue: Mapping[str, UnitStats]) -> bool:
    """Report whether an entity can move to a fight.

    A turret is armed and cannot be sent anywhere, so ordering one to attack a
    distant target produces a command the engine accepts and cannot carry out.

    Args:
        entity: The entity to test.
        catalogue: Unit stats by type name.

    Returns:
        True when the catalogue gives it a non-zero speed.
    """
    stats = catalogue.get(entity["type_name"])
    return stats is not None and stats["speed"] != 0.0


def find_army(sample: Sample, catalogue: Mapping[str, UnitStats]) -> tuple[Entity, ...]:
    """Return the units that can be sent to fight.

    Owned, finished, armed, mobile, and not the editor placeholder. Each
    exclusion is load-bearing: an unfinished unit does not exist yet, an unarmed
    Builder sent at a tank is a Builder thrown away, a turret cannot travel, and
    the placeholder is not a unit at all.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name.

    Returns:
        The army, in roster order.
    """
    return tuple(
        entity
        for entity in sample["entities"]
        if entity["mine"]
        and entity["complete"]
        and entity["type_name"] != PLACEHOLDER_TYPE
        and is_armed(entity, catalogue)
        and is_mobile(entity, catalogue)
    )


def find_targets(sample: Sample) -> tuple[Entity, ...]:
    """Return the hostile entities currently visible.

    Hostility is the engine's own answer, carried per entity, rather than the
    negation of ownership. The two differ: an ally and a neutral map object are
    both not-mine and neither is an enemy, and attacking either is a wasted
    order at best ([[wire-contract-ndjson]]).

    Args:
        sample: One observation of the world.

    Returns:
        Every visible hostile entity, in roster order.
    """
    return tuple(entity for entity in sample["entities"] if entity["hostile"])


def choose_target(
    army: Sequence[Entity],
    targets: Sequence[Entity],
    holding: int | None = None,
) -> Entity | None:
    """Pick the target the army should commit to, keeping the current one.

    **Commitment is the point of the ``holding`` argument.** Choosing afresh
    every sample is what made the bot look busy and achieve little: nearest is
    measured from the army's centre, that centre shifts whenever a unit dies or
    a new one rolls out, and the whole army was re-tasked on a flip that could
    be a few world units wide. One measured run spent 743 attack orders across
    48 units on 24 targets -- about fifteen re-orders each
    ([[policy-combat]]).

    So a target already being attacked is kept while it remains visible and
    hostile, and a new one is chosen only when it is not. The engine's own AI
    reaches the same place from the other direction: it holds a target and
    refreshes on a timer rather than on a change of mind
    ([[ai-opponent-strategy]]).

    Purity is not lost by this. The prior choice is an argument rather than
    hidden state, exactly as the build loop passes its own progress in, so the
    function is still a value in and a value out.

    Nearest is measured to the army's centre rather than per unit, so a split
    force converges instead of each unit wandering to its own closest enemy.
    Concentrating fire is the one tactic that matters at this scale: two tanks
    on one target kill it in half the time and take half the return fire.

    Distance is squared and left squared -- only the ordering is used, and a
    square root would cost precision for nothing.

    Args:
        army: The units available to fight.
        targets: The hostile entities to choose between.
        holding: Engine identity of the target already being attacked, if any.
        fighting: The units cleared to attack. ``None`` means the whole army,
            which is what a caller with no wave discipline wants; a caller that
            musters passes the released wave so reinforcements still gathering
            are not ordered in alone ([[engine-ai-triggers]]).

    Returns:
        The chosen target, or None when either side is empty.
    """
    if not army or not targets:
        return None
    for target in targets:
        if target["unit_id"] == holding:
            return target
    centre_x = sum(unit["x"] for unit in army) / len(army)
    centre_y = sum(unit["y"] for unit in army) / len(army)

    best: Entity | None = None
    best_distance = 0.0
    for target in targets:
        distance = (target["x"] - centre_x) ** 2 + (target["y"] - centre_y) ** 2
        if best is None or distance < best_distance:
            best = target
            best_distance = distance
    return best


def engagements(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    holding: int | None = None,
    fighting: Sequence[Entity] | None = None,
) -> tuple[Engagement, ...]:
    """Decide who attacks what this sample.

    The whole army is sent at one target rather than spread across several, and
    that target persists across samples, both for the reasons given in
    :func:`choose_target`.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name.
        holding: Engine identity of the target already being attacked, if any.

    Returns:
        One engagement per available unit, empty when there is no army or
        nothing hostile in sight.
    """
    army = find_army(sample, catalogue) if fighting is None else tuple(fighting)
    target = choose_target(army, find_targets(sample), holding)
    if target is None:
        return ()
    return tuple(
        Engagement(
            attacker_id=unit["unit_id"],
            target_id=target["unit_id"],
            reason=f"{unit['type_name']} -> {target['type_name']} {target['unit_id']}",
        )
        for unit in army
    )


__all__ = [
    "PLACEHOLDER_TYPE",
    "RALLY_RADIUS",
    "WAVE_SIZES",
    "Deployment",
    "Engagement",
    "Muster",
    "choose_target",
    "engagements",
    "find_army",
    "find_targets",
    "is_armed",
    "is_mobile",
    "muster",
    "rally",
    "wave_size",
]
