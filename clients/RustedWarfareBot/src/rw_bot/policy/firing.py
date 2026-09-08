"""Deciding what each cleared unit fires at, and in what order.

Extracted from :mod:`rw_bot.policy.combat` when the tactical genome's
first alleles pushed that module over the size ceiling
([[impossible-tactical-genome]]): discovery (what exists, what is
engageable) stays there; this module owns the FIRING decisions -- which
target the army commits to, how kill-sized groups form, and the allele
constants the doctrine carries for both. The split is by role, not
convenience: everything here consumes what combat discovers and produces
the pairings dispatch sends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile, can_engage
from rw_bot.policy.combat import engageable, find_army, find_targets
from rw_bot.wire.state import Entity, Sample

#: How many kill-groups may fill at once. Unbounded grouping was measured
#: (screen-vh9f): trades improved everywhere and the winning seed's kill was
#: lost -- an army facing many visible targets spreads to the point where
#: no group has punch density beyond bare lethality. Two keeps the
#: no-overkill edge in field fights while the army stays a fist. Since the
#: tactical genome ([[impossible-tactical-genome]]) this is the IDENTITY
#: value of the ``groupcap`` allele, not a law: it was set on three seeds
#: "pending a wins-based reason", and the searcher owns it now.
MAX_OPEN_GROUPS = 2

#: Target-priority alleles ([[impossible-tactical-genome]]). The identity
#: orders by convergence -- nearest the army's centre, health breaking
#: ties -- the ordering that survived the economy-first challenge
#: (screen-vh9m: a global wallet preference chased extractors past their
#: escorts and ate free damage the whole walk). The variants reorder only
#: WITHIN the engageable set, so nothing here re-creates that walk:
#: RANGE ranks the longest-reaching target first (the artillery that tops
#: every Impossible death ledger dies before the tanks in front of it, at
#: the measured risk of pulling the army deeper), and FRAIL ranks the
#: lowest absolute hit points first (the fastest possible removal of
#: firing units).
PRIO_CONVERGENCE = 0
PRIO_RANGE = 1
PRIO_FRAIL = 2


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
    groups: int = MAX_OPEN_GROUPS,
    prio: int = PRIO_CONVERGENCE,
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
        groups: Kill-groups that may fill at once -- the ``groupcap``
            allele, identity :data:`MAX_OPEN_GROUPS`. Zero behaves as one
            rolling group (the pre-arc single focus), because only a
            started group can accept members.
        prio: Target-priority allele, identity :data:`PRIO_CONVERGENCE`.
            See the allele constants for the variants and their measured
            history.

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
    ordered = _prioritized(profiles, _by_convergence(army, candidates), prio)
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
            and (committed[t["unit_id"]] > 0.0 or started < groups)
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


def _prioritized(
    profiles: Mapping[str, CombatProfile], ordered: Sequence[Entity], prio: int
) -> tuple[Entity, ...]:
    """Apply the target-priority allele to the convergence ordering.

    The identity returns the convergence order untouched. The variants
    re-sort STABLY on one engine-derived figure each, so convergence still
    breaks every tie and nothing here re-creates the refuted global
    economy preference (screen-vh9m) -- the candidate set was already
    filtered to what the army can engage.

    Args:
        profiles: Combat profiles by type name, for the range figure.
        ordered: Targets in convergence order.
        prio: :data:`PRIO_CONVERGENCE`, :data:`PRIO_RANGE` or
            :data:`PRIO_FRAIL`.

    Returns:
        The targets in allele order.
    """

    def longest_reach(target: Entity) -> float:
        return -profiles[target["type_name"]]["attack_range"]

    def frailest(target: Entity) -> float:
        return target["hp"]

    if prio == PRIO_RANGE:
        return tuple(sorted(ordered, key=longest_reach))
    if prio == PRIO_FRAIL:
        return tuple(sorted(ordered, key=frailest))
    return tuple(ordered)


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
    "MAX_OPEN_GROUPS",
    "PRIO_CONVERGENCE",
    "PRIO_FRAIL",
    "PRIO_RANGE",
    "Engagement",
    "choose_target",
    "engagements",
]
