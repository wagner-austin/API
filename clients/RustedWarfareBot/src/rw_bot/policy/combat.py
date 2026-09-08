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
from rw_bot.mechanics.combat_profile import CombatProfile, can_engage, is_armed
from rw_bot.wire.state import Entity, Sample

#: Type name of the map editor's placeholder, which is owned and cannot fight.
#:
#: The same exclusion producer selection needs, for the same reason: it is an
#: owned entity in every sample and not a playable unit ([[policy-loop]]).
PLACEHOLDER_TYPE = "editorOrBuilder"


#: The fewest units that count as attacking together.
#:
#: The ladder's own first rung, reused rather than reinvented: it is the number
#: the shipped AI uses for its first attack group, and the number below which
#: this module already calls an attack a trickle ([[engine-ai-triggers]]).
FIRST_WAVE = 3

#: Units each successive wave waits for, in order.
#:
#: The shipped AI's ladder: three for the first attack, five for the next few,
#: seven thereafter. Its groups are created empty with a target size and recruit
#: until full before they move, and the size climbs with the number of groups it
#: has already sent ([[engine-ai-triggers]]).
WAVE_SIZES: tuple[int, ...] = (FIRST_WAVE, FIRST_WAVE, 5, 5, 5, 7)


def ladder_to(mass: int) -> tuple[int, ...]:
    """Return the shipped ladder with its final rung replaced.

    The early rungs are left alone deliberately. They govern the opening, when
    the player has three units and holding them back is the difference between
    a first attack and no attack at all; the final rung governs the other
    twenty-eight minutes, and it is the one worth asking a question about. An
    experiment that moved both would not be able to say which end mattered.

    Args:
        mass: Units the sustained wave waits for. Values at or below the last
            fixed rung leave the ladder unchanged, so the shipped behaviour is
            reachable rather than a special case.

    Returns:
        The ladder to muster against.
    """
    return (*WAVE_SIZES[:-1], max(mass, WAVE_SIZES[-2]))


def wave_size(waves_sent: int, ladder: Sequence[int] = WAVE_SIZES) -> int:
    """Return how many units the next wave waits for.

    Args:
        waves_sent: Waves already released.
        ladder: Sizes in order, the last rung repeating thereafter.

    Returns:
        The size the next one needs, the last rung repeating thereafter.
    """
    return ladder[min(waves_sent, len(ladder) - 1)]


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


def muster(
    army: Sequence[Entity],
    released: frozenset[int],
    waves: int,
    ladder: Sequence[int] = WAVE_SIZES,
    force: bool = False,
) -> Muster:
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

    **A wave reduced below the size that makes a wave is not one any more.**
    Survivors used to keep their clearance permanently, on the reasoning that
    turning round mid-attack is the worst of both behaviours. Measured, that
    reasoning was wrong in the one way that mattered: of 48 units lost in a
    1500-sample match, 46 died more than 2,000 world units from home and not one
    died within 900 ([[policy-combat]]). Nothing was attacking the base. The
    army was walking into defended ground and dying, and the last survivor of
    each wave kept its clearance and walked in after them, alone -- which is
    precisely the trickle this gate exists to prevent, happening on the way out
    instead of the way in.

    So clearance is held only while the wave is still a wave, and the threshold
    is the ladder's own first rung rather than a new number: below
    :data:`FIRST_WAVE` the survivors return to the reserve, rally home, and go
    out again with the next one.

    Args:
        army: Units available to fight, as :func:`find_army` reports them.
        released: Engine ids already cleared by an earlier wave.
        waves: Waves released so far.
        ladder: How many units each successive wave waits for. Defaults to the
            shipped AI's, which is a number copied from an opponent playing a
            different economy: measured, this bot feeds about sixty tanks into
            defended ground across a match and sets the leader back by roughly
            a thousand credits, so how much to mass before committing is a
            question rather than a constant ([[policy-combat]]).
        force: Release the reserve now rather than at the ladder's rung --
            the riposte: the enemy's attack just burned itself on our ground,
            and the window before its next group finishes staging is when a
            stockpile converts ([[policy-combat]], [[ai-opponent-strategy]]).
            The anti-trickle floor still holds: fewer than a first wave is
            not a punch, forced or not.

    Returns:
        The decision, carrying the state the next call needs.
    """
    alive = {unit["unit_id"] for unit in army}
    survivors = alive & released
    if len(survivors) < FIRST_WAVE:
        # Decimated, so no longer a wave. Handing them back to the reserve is
        # what sends them home to re-gather rather than in to die one at a time.
        survivors = set()
    reserve = alive - survivors
    wanted = wave_size(waves, ladder)
    if force and len(reserve) >= FIRST_WAVE:
        wanted = min(wanted, len(reserve))

    if len(reserve) >= wanted:
        return Muster(
            released=frozenset(alive),
            gathering=0,
            wanted=wave_size(waves + 1, ladder),
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


def find_army(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
) -> tuple[Entity, ...]:
    """Return the units that can be sent to fight.

    Owned, finished, armed, mobile, and not the editor placeholder. Each
    exclusion is load-bearing: an unfinished unit does not exist yet, an unarmed
    Builder sent at a tank is a Builder thrown away, a turret cannot travel, and
    the placeholder is not a unit at all.

    Armament comes from the registry rather than the stat catalogue, because the
    catalogue describes 90 of 173 types and an absent entry there is
    indistinguishable from an unarmed unit ([[mechanics-combat-profile]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for mobility.
        profiles: Combat profiles by type name, for armament.

    Returns:
        The army, in roster order.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe an
            owned type.
    """
    return tuple(
        entity
        for entity in sample["entities"]
        if entity["mine"]
        and entity["complete"]
        and entity["type_name"] != PLACEHOLDER_TYPE
        and is_armed(profiles, entity)
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


def engageable(
    profiles: Mapping[str, CombatProfile],
    army: Sequence[Entity],
    targets: Sequence[Entity],
) -> tuple[Entity, ...]:
    """Return the targets at least one of these units can actually shoot.

    **This is the filter whose absence could hang a whole match.** ``c_tank`` --
    the only unit the opening plan builds -- declares ``canAttackFlyingUnits:
    false``. Combat used to select on *having* a weapon and never on the weapon
    reaching the target, so on a water map the army could commit to a
    helicopter, hold it for as long as it stayed visible because commitment
    keeps a visible target, and never fire a shot or pick anything else
    ([[mechanics-combat-profile]]).

    Args:
        profiles: Combat profiles by type name.
        army: The units available to fight.
        targets: The hostile entities to filter.

    Returns:
        The targets some unit in ``army`` can engage, in roster order.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe a
            visible type.
    """
    return tuple(
        target
        for target in targets
        if any(can_engage(profiles, attacker, target) for attacker in army)
    )


__all__ = [
    "FIRST_WAVE",
    "PLACEHOLDER_TYPE",
    "RALLY_RADIUS",
    "WAVE_SIZES",
    "Deployment",
    "Muster",
    "engageable",
    "find_army",
    "find_targets",
    "is_mobile",
    "muster",
    "rally",
    "wave_size",
]
