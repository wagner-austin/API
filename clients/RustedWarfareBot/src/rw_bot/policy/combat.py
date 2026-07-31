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

#: How many kill-groups may fill at once. Unbounded grouping was measured
#: (screen-vh9f): trades improved everywhere and the winning seed's kill was
#: lost -- an army facing many visible targets spreads to the point where
#: no group has punch density beyond bare lethality. Two keeps the
#: no-overkill edge in field fights while the army stays a fist.
MAX_OPEN_GROUPS = 2


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
    48 units on 24 targets -- about fifteen re-orders each ([[policy-combat]]).

    So a target already being attacked is kept while it remains in the candidate
    list, and a new one is chosen only when it is not. The engine's own AI
    reaches the same place from the other direction: it holds a target and
    refreshes on a timer rather than on a change of mind
    ([[ai-opponent-strategy]]).

    Purity is not lost by this. The prior choice is an argument rather than
    hidden state, exactly as the plan passes its own progress in, so the
    function is still a value in and a value out.

    Nearest is measured to the army's centre rather than per unit, so a split
    force converges instead of each unit wandering to its own closest enemy.
    Concentrating fire is the one tactic that matters at this scale: two tanks
    on one target kill it in half the time and take half the return fire.

    **Health breaks ties and nothing more.** Equidistant targets are ordinary on
    a symmetric map, and resolving them by roster order is arbitrary where
    resolving them by what is closest to dying is not. It is deliberately not a
    scoring model: ranking a distant cripple above a near healthy unit would be
    a number invented here rather than measured ([[policy-combat]]).

    Distance is squared and left squared -- only the ordering is used, and a
    square root would cost precision for nothing.

    Args:
        army: The units available to fight.
        targets: The hostile entities to choose between.
        holding: Engine identity of the target already being attacked, if any.

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
    best_key: tuple[float, float] = (0.0, 0.0)
    for target in targets:
        distance = (target["x"] - centre_x) ** 2 + (target["y"] - centre_y) ** 2
        key = (distance, target["hp"])
        if best is None or key < best_key:
            best = target
            best_key = key
    return best


def engagements(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    held: Mapping[int, int] | None = None,
    fighting: Sequence[Entity] | None = None,
) -> tuple[Engagement, ...]:
    """Decide who attacks what this sample: kill-sized groups, held per unit.

    Fire concentrates until one volley kills, and no further. The whole army
    on one target was measured through five screening rounds at Very Hard:
    every match ran even to sample 1000 and was lost on trade quality in the
    window after, with a twenty-five unit wave volleying single tanks while
    the opponent's spread fire killed efficiently ([[policy-combat]], log
    2026-07-31). So a target is assigned attackers until their combined
    volley damage covers its hit points -- the engine's own figures, no
    invented constant -- and the next attacker starts the next-nearest
    target's group. When every visible target's group is already lethal, the
    overflow joins the nearest group rather than standing idle: overkill
    beats an armed unit watching a fight.

    Assignments persist per attacker for :func:`choose_target`'s reason --
    re-choosing every sample re-tasked the whole army on a centre shift a few
    world units wide. An attacker keeps its target while that target remains
    engageable; only freed attackers (their target died or left) are dealt
    into groups afresh.

    Only units that can reach a target's layer join its group. The rest are
    left alone rather than sent: an order a unit cannot carry out is accepted
    by the engine and then does nothing, which is indistinguishable from a
    unit that is simply losing ([[mechanics-combat-profile]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for volley damage.
        profiles: Combat profiles by type name, for armament and reachability.
        held: Target already assigned per attacker, from the previous sample.
        fighting: The units cleared to attack. ``None`` means the whole army,
            which is what a caller with no wave discipline wants; a caller that
            musters passes the released wave so reinforcements still gathering
            are not ordered in alone ([[engine-ai-triggers]]).

    Returns:
        One engagement per unit with a reachable target, empty when there is
        no army or nothing it can touch.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe a
            visible type.
    """
    army = find_army(sample, catalogue, profiles) if fighting is None else tuple(fighting)
    candidates = engageable(profiles, army, find_targets(sample))
    if not army or not candidates:
        return ()
    ordered = _by_convergence(army, candidates)
    by_id = {target["unit_id"]: target for target in ordered}
    committed: dict[int, float] = {target["unit_id"]: 0.0 for target in ordered}
    assigned: dict[int, Entity] = {}
    kept = held or {}
    for unit in army:
        target = by_id.get(kept.get(unit["unit_id"], -1))
        if target is not None and can_engage(profiles, unit, target):
            assigned[unit["unit_id"]] = target
            committed[target["unit_id"]] += _volley(catalogue, unit)
    for unit in army:
        if unit["unit_id"] in assigned:
            continue
        reachable = [t for t in ordered if can_engage(profiles, unit, t)]
        if not reachable:
            continue
        # At most two groups exist at once. Unbounded groups were measured
        # (screen-vh9f): trades improved everywhere -- the hardest seed's
        # rival fell from 114k to 19k -- and the winning seed's kill was
        # lost, because an army facing a whole visible base diluted into
        # many barely-lethal groups and the turret return fire ground it
        # down. Two keeps the no-overkill edge in field fights, where
        # targets arrive a few at a time, and keeps the army a fist against
        # fortifications. Lethal groups still count: a fresh target may open
        # a group only while fewer than two have been started at all.
        started = sum(1 for value in committed.values() if value > 0.0)
        open_groups = [
            t
            for t in reachable
            if committed[t["unit_id"]] < t["hp"]
            and (committed[t["unit_id"]] > 0.0 or started < MAX_OPEN_GROUPS)
        ]
        # Overflow joins the nearest reachable group: overkill beats idling.
        target = open_groups[0] if open_groups else reachable[0]
        assigned[unit["unit_id"]] = target
        committed[target["unit_id"]] += _volley(catalogue, unit)
    return tuple(
        Engagement(
            attacker_id=unit["unit_id"],
            target_id=assigned[unit["unit_id"]]["unit_id"],
            reason=(
                f"{unit['type_name']} -> "
                f"{assigned[unit['unit_id']]['type_name']} "
                f"{assigned[unit['unit_id']]['unit_id']}"
            ),
        )
        for unit in army
        if unit["unit_id"] in assigned
    )


def _by_convergence(army: Sequence[Entity], targets: Sequence[Entity]) -> tuple[Entity, ...]:
    """Order targets by distance from the army's centre, health breaking ties.

    **Distance-first survived a measured challenge, and the challenger is
    recorded.** Ranking visible hostile income structures ahead of distance
    -- the arithmetically appealing "wallet outranks the war" -- doubled the
    extractor losses and strangled two of three screening seeds: the army
    chased extractors past their escorts and ate free damage the whole walk
    (screen-vh9m, log 2026-07-31). Economy kills convert matches, but the
    instrument for them is the raid party and the fights the army wins on
    the way in, not a global preference that ignores what is shooting.

    The key is :func:`choose_target`'s: nearest to the army's centre so a
    split force converges, health breaking ties so the target closest to
    dying fills first.
    """
    centre_x = sum(unit["x"] for unit in army) / len(army)
    centre_y = sum(unit["y"] for unit in army) / len(army)

    def convergence_key(target: Entity) -> tuple[float, float]:
        distance = (target["x"] - centre_x) ** 2 + (target["y"] - centre_y) ** 2
        return (distance, target["hp"])

    return tuple(sorted(targets, key=convergence_key))


def _volley(catalogue: Mapping[str, UnitStats], unit: Entity) -> float:
    """Return one full volley's damage from a unit, the engine's own figure.

    The larger of the direct and splash volleys: a unit contributes whichever
    kind of damage it actually deals, and an unarmed unit contributes nothing
    -- though an unarmed unit never reaches a group, because its profile
    reaches no layer at all ([[mechanics-combat-profile]]).
    """
    weapon = catalogue[unit["type_name"]]["weapon"]
    if weapon is None:
        return 0.0
    return max(weapon["direct_damage_volley"], weapon["area_damage_volley"])


__all__ = [
    "FIRST_WAVE",
    "PLACEHOLDER_TYPE",
    "RALLY_RADIUS",
    "WAVE_SIZES",
    "Deployment",
    "Engagement",
    "Muster",
    "choose_target",
    "engageable",
    "engagements",
    "find_army",
    "find_targets",
    "is_mobile",
    "muster",
    "rally",
    "wave_size",
]
