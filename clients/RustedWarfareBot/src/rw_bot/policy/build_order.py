"""A build-order policy: decide the next action from one observation.

Deliberately a pure function. ``decide`` takes a world sample and the plan's
progress and returns what to do; it opens no sockets, reads no clock, and
mutates nothing. That is what makes the playing logic testable without a game
running, and it is the same separation the agent holds on its own side —
dispatch there, decision here (wiki: runtime-split-java-agent-python-brain).

The policy is small on purpose. It executes a fixed sequence of structures,
waiting when it cannot afford the next one and stopping when it cannot make
progress at all. It does not fight, expand, or react to an opponent. What it
demonstrates is a loop that observes, chooses, acts, and can be scored.

**Where** a structure would stand is a different job and lives in
:mod:`rw_bot.policy.siting`. This module decides *what* to make next and
whether anything owned can make it; that one answers which pool or which ring
slot is free, and answers it identically for the economy, so the two spenders
cannot collide over one site ([[policy-loop]]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.progress import (
    completed_count,
    next_unsatisfied_index,
    unsatisfied_indices,
)
from rw_bot.policy.siting import (
    find_anchor,
    next_ring_site,
    no_pool_reason,
    survey_pools,
)
from rw_bot.wire.state import BuildOption, Entity, Sample

BUILDER_TYPE = "builder"

#: Type name of the map editor's placeholder unit.
#:
#: An owned entity in every sample, parked off-map at (-1000, -1000) with
#: 170,000 hit points, and not a playable unit. It answers the engine's build
#: queries for nearly every type in the game, so anything that selects a
#: producer by capability has to exclude it by name -- see
#: :func:`find_producer` for what including it costs.
PLACEHOLDER_TYPE = "editorOrBuilder"


class Decision(TypedDict):
    """What the policy wants to happen next.

    Attributes:
        action: ``"build"`` to place the next structure at a chosen position,
            ``"produce"`` to have a building make a unit that rolls out of it,
            ``"wait"`` to do nothing this tick, ``"done"`` when the plan is
            complete, ``"blocked"`` when it cannot proceed at all, and
            ``"stalled"`` when an order was accepted but never took effect.
        reason: Human-readable justification, carried for the run log so a
            session can be read back without re-deriving why it stalled.
        type_name: What to make. Empty unless ``action`` is ``"build"`` or
            ``"produce"``.
        unit_id: The unit the plan intends to order, zero when there is none.
            Set for ``"build"`` and ``"produce"``, and **also for a ``"wait"``
            that already knows which unit it is holding** -- one waiting on
            price or on an action the engine has not offered yet.

            That last case reads like a detail and was worth most of the
            economy. The plan reserves its worker while it waits, and the only
            way it could say so was a boolean, which switched off *every*
            spender rather than removing *one* worker. Instrumented, the
            expander was skipped on 572 of 800 samples while six workers stood
            free ([[policy-economy]]). Zero still means "no unit", so a caller
            filtering on it is safe on the waits that genuinely hold nothing.
        x: Placement world x. Zero unless ``action`` is ``"build"`` -- a
            produced unit appears where the engine puts it, not where the
            planner asks.
        y: Placement world y. Zero unless ``action`` is ``"build"``.
        deficit: Credits still missing for the entry being waited on; zero for
            every other decision. Carried as a number because the tracker
            cannot judge a save it cannot see -- the shortfall was already in
            the wait's reason string, and parsing it back out would be the
            same figure laundered through prose. What the tracker does with
            it: a shortfall that never shrinks is a save that is not
            happening, and the plan is ruled blocked rather than holding its
            worker hostage forever ([[policy-economy]]).
    """

    action: Literal["build", "produce", "wait", "done", "blocked", "stalled"]
    reason: str
    type_name: str
    unit_id: int
    x: float
    y: float
    deficit: int


def find_producer(sample: Sample, target: str, free: Sequence[Entity] = ()) -> BuildOption | None:
    """Return the option by which something the player owns makes ``target``.

    This replaces asking "which of my units is a builder", which was a guess
    dressed as a constant. The engine answers the real question per unit, and
    the answer rides in the sample ([[wire-contract-ndjson]]).

    Entities of :data:`PLACEHOLDER_TYPE` are skipped, and that exclusion is
    load-bearing rather than tidy. The map editor's placeholder is an owned
    entity in every sample, parked off-map, and it offers 108 of the 123 options
    in the archived capture -- a superset of everything the real Builder can
    make, plus 95 more. Without the exclusion almost no plan entry is ever
    unbuildable, so the check this function exists to support would pass on
    types nothing playable can produce, and the resulting order would go to a
    unit at (-1000, -1000) and do nothing at all. That is the same silent
    failure the stall detector was built for, wearing a check that looks like
    protection.

    Unavailable options are returned rather than skipped. An action that exists
    but is not usable yet is a wait; one that does not exist is a dead plan
    entry, and the caller needs to tell those apart.

    **A placed structure is only ever ordered from a free worker.** A produced
    unit is not: a factory's queue is the engine's own, and asking a busy one is
    how the queue gets filled. The distinction is the option's own ``placed``
    flag, so nothing here has to know which types are structures.

    Args:
        sample: One observation of the world.
        target: Type name the plan asks for.
        free: Workers not already carrying out an order. Empty means no worker
            is free, so no placed option qualifies.

    Returns:
        The option, preferring an available one, or None when nothing the
        player owns makes it.
    """
    placeholders = {
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["type_name"] == PLACEHOLDER_TYPE
    }
    idle = {worker["unit_id"] for worker in free}
    fallback: BuildOption | None = None
    for option in sample["options"]:
        if option["produces"] != target or option["unit_id"] in placeholders:
            continue
        if option["placed"] and option["unit_id"] not in idle:
            continue
        if option["available"]:
            return option
        if fallback is None:
            fallback = option
    return fallback


def _any_producer_exists(sample: Sample, target: str) -> bool:
    """Report whether anything the player owns offers ``target`` at all.

    The busy-blind twin of :func:`find_producer`: same option scan, same
    placeholder exclusion, no idle filter. It exists to split "not playable
    from here" (nothing owned makes it -- a dead plan entry) from "every
    capable unit has its hands full" (a wait the world resolves the moment a
    worker frees), which one Hard batch showed are a match apart
    ([[policy-loop]]).

    Args:
        sample: One observation of the world.
        target: Type name the plan asks for.

    Returns:
        True when some owned, non-placeholder entity offers the target.
    """
    owned = {
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and entity["type_name"] != PLACEHOLDER_TYPE
    }
    return any(
        option["produces"] == target and option["unit_id"] in owned for option in sample["options"]
    )


def _undescribed(
    target: str,
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
) -> Decision | None:
    """Return the block for a target the dumps do not describe, or None.

    Both dumps are read from the live engine and cover every registered type,
    so a miss means the plan names something that does not exist in this build
    -- a typo, or a type from a mod that is not loaded. That cannot resolve on
    its own, which is why it blocks rather than waits.

    Args:
        target: Type name the plan asks for.
        catalogue: Unit stats by type name.
        placements: Placement rules by type name.

    Returns:
        The blocking decision, or None when both dumps describe the target.
    """
    if target not in catalogue:
        return _decision(
            "blocked",
            f"{target!r} is not in the unit catalogue, so its price is unknown",
        )
    if target not in placements:
        return _decision(
            "blocked",
            f"{target!r} is not in the placement dump, so where it may stand is unknown",
        )
    return None


def decide(
    sample: Sample,
    plan: Sequence[str],
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
    profiles: Mapping[str, CombatProfile],
    free: Sequence[Entity] = (),
    claimed: Sequence[tuple[float, float]] = (),
) -> Decision:
    """Choose the next action from one observation.

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.
        catalogue: Unit stats by type name, for prices.
        placements: Placement rules by type name, for where a structure may
            stand.
        profiles: Combat profiles by type name, for the threat filter.
        free: Workers not already carrying out an order. A placed structure is
            ordered from one of these; which of them is free is decided by the
            loop, because it is the only thing that can see what each was last
            sent to do ([[policy-loop]]).

    Returns:
        The decision.
    """
    built = completed_count(sample, plan)
    pending = unsatisfied_indices(sample, plan)
    if not pending:
        return _decision("done", f"all {len(plan)} plan entries satisfied")

    # **An entry with nowhere to stand defers to the next one; it does not stop
    # the plan.** The opening is a sequence -- extractors, then the factory,
    # then the army -- and it used to wait on whichever entry it had reached.
    # When the third extractor had no free pool, the wait was permanent: the
    # factory was never built, no army was ever produced, and matches ended
    # with five idle builders, 60,676 credits and an army of nothing, against
    # an opponent left to do as it liked. Two of twelve duels were lost that
    # way, and both had been level until the pools ran out
    # ([[policy-holding-ground]]).
    #
    # Only *placement* defers. Being unaffordable or having no producer is a
    # condition on the whole plan rather than on one entry, so those still stop
    # it -- skipping past them would spend the next entry's credits on the
    # strength of the current one's being short.
    deferred: Decision | None = None
    for index in pending:
        attempt, deferrable = _attempt(
            sample, plan, index, built, catalogue, placements, profiles, free, claimed
        )
        if not deferrable:
            return attempt
        if deferred is None:
            deferred = attempt
    # Every entry left is unplaceable. Reported with the *first* entry's own
    # reason -- a full ring and an occupied pool are different problems and the
    # run log has to say which ([[policy-loop]]).
    return deferred if deferred is not None else _decision("wait", "nothing left to place")


def _attempt(
    sample: Sample,
    plan: Sequence[str],
    index: int,
    built: int,
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
    profiles: Mapping[str, CombatProfile],
    free: Sequence[Entity],
    claimed: Sequence[tuple[float, float]],
) -> tuple[Decision, bool]:
    """Decide one plan entry, and say whether the next entry may be tried.

    Args:
        sample: One observation of the world.
        plan: What to make, in order.
        index: Which entry to try.
        built: How much of the plan the world already shows, for the reason.
        catalogue: Unit stats by type name, for prices.
        placements: Placement rules by type name.
        profiles: Combat profiles by type name, for the threat filter.
        free: Workers not already carrying out an order.

    Returns:
        The decision, and whether it is *deferrable* -- true only when the
        entry has nowhere legal to stand, which is a fact about this entry.
        Being unaffordable or having no producer is a fact about the plan, so
        those are not deferrable and stop it.
    """
    target = plan[index]
    undescribed = _undescribed(target, catalogue, placements)
    if undescribed is not None:
        return undescribed, False
    stats = catalogue[target]
    placement = placements[target]

    # The engine's own answer to "can anything I own make this", asked before
    # an order is spent rather than inferred from one that quietly did nothing.
    # A builder cannot construct a laboratory, and the refusal produces no
    # roster change and no error -- a plan naming one used to run for three
    # hundred samples reporting progress ([[policy-loop]]).
    producer = find_producer(sample, target, free)
    if producer is None:
        # **A busy workforce is a wait, not a block, and conflating them cost
        # every Hard win in a batch.** The moment defence siting stopped being
        # silently refused, a rich match kept all eight workers employed on
        # turrets, the plan's Land Factory never met a free worker, and this
        # branch ruled the plan "not playable from here" -- permanently, for a
        # plan that needed one worker for one order: army 0 -> 0, attack
        # orders 0, wins 1/10 where the same doctrine had won 10/12
        # (log: 2026-07-31). Playable-from-here is a fact about what the
        # player OWNS; whose hands are full right now is not it. The wait
        # carries no unit -- there is no specific worker to hold -- and the
        # campaign answers it by sending the plan the next worker that frees
        # ([[policy-loop]]).
        if _any_producer_exists(sample, target):
            return (
                _waiting_on(f"every unit that can make {target} is busy", 0),
                False,
            )
        return (
            _decision(
                "blocked",
                f"nothing the player owns can make {target}; the plan is not playable from here",
            ),
            False,
        )
    if not producer["available"]:
        # Available is a property of the world, not of the plan. A prerequisite
        # can still be built and tech can still be researched, so this is a
        # wait -- and an unbounded one today: only the price wait carries a
        # deficit for the savings clock to judge, and the raid batch showed a
        # pool wait holding a worker 335 samples (log: 2026-07-29). Open.
        return (
            _waiting_on(
                f"unit {producer['unit_id']} can make {target} but the action is not available yet",
                producer["unit_id"],
            ),
            False,
        )

    # Where it goes is settled before whether it is affordable, so a structure
    # with nowhere legal to stand says so rather than reporting a price. Credits
    # arrive on their own; a taken pool does not become free by waiting for the
    # bank.
    site: tuple[float, float] | None = None
    if producer["placed"]:
        # Indexed rather than searched with a guard: a placed option is only
        # returned for a worker the caller listed as free, so a miss here would
        # mean those two reads disagreed about the same list.
        worker = {entity["unit_id"]: entity for entity in free}[producer["unit_id"]]
        placed = _placed_site(sample, worker, target, placement, catalogue, profiles, claimed)
        if isinstance(placed, dict):
            # Nowhere legal to stand. Handed back as deferrable, carrying its
            # own reason -- a full ring and an occupied pool are different
            # problems and the run log has to say which.
            return placed, True
        site = placed

    if sample["credits"] < stats["price"]:
        # The dominant wait, and the one that used to stop the whole economy.
        # It names its worker so the expander can skip that one and run on the
        # rest, rather than being switched off entirely ([[policy-economy]]).
        return (
            _waiting_on(
                f"{target} costs {stats['price']}, holding {sample['credits']}",
                producer["unit_id"],
                deficit=stats["price"] - sample["credits"],
            ),
            False,
        )

    if site is None:
        # A produced unit rolls out of the building that made it. The engine
        # chooses where, so this decision carries no position -- offering one
        # would be a number the planner invented.
        return (
            Decision(
                action="produce",
                reason=f"producing {target} ({built + 1} of {len(plan)})",
                type_name=target,
                unit_id=producer["unit_id"],
                x=0.0,
                y=0.0,
                deficit=0,
            ),
            False,
        )

    return (
        Decision(
            action="build",
            reason=f"building {target} ({built + 1} of {len(plan)})",
            type_name=target,
            unit_id=producer["unit_id"],
            x=site[0],
            y=site[1],
            deficit=0,
        ),
        False,
    )


def _placed_site(
    sample: Sample,
    builder: Entity,
    target: str,
    placement: TypePlacement,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    claimed: Sequence[tuple[float, float]],
) -> tuple[float, float] | Decision:
    """Choose where a placed structure goes, or explain why nowhere will do.

    Two placement rules, because the engine has two. A structure bound to a
    resource pool has exactly as many legal sites as there are usable pools, and
    the ring is irrelevant to it -- offering a ring position would produce an
    order the engine refuses without saying so. Everything else takes the next
    ring position around the anchor.

    Args:
        sample: One observation of the world.
        builder: The worker that will place it, chosen by the caller. Which
            worker is free is the loop's business, not this module's
            ([[policy-loop]]).
        target: The type being placed, for the failure message.
        placement: The engine's placement rule for it.
        catalogue: Unit stats by type name, for the anchor and pool judgement.
        profiles: Combat profiles by type name, for the threat filter.

    Returns:
        The world point to build at, or the decision that stops this tick.
    """
    # Measure from the most stable reference available. A structure never moves;
    # the builder does, and measuring from it collapses the ring. With no
    # structure owned the builder is the only reference there is, so the
    # collapse is unavoidable rather than chosen -- and a player who has lost
    # every building should still be able to rebuild.
    anchor = find_anchor(sample, catalogue) or builder

    if not placement["needs_pool"]:
        site = next_ring_site(sample, anchor, catalogue)
        if site is None:
            # Every ring position is taken. A wait rather than a block: a
            # structure destroyed frees its slot, so the world can leave this
            # state on its own. Unbounded today -- see the producer wait above.
            return _waiting_on(
                f"{target} needs a ring position and all are taken", builder["unit_id"]
            )
        return site

    survey = survey_pools(sample, anchor, builder, catalogue, profiles, claimed)
    chosen = survey["pool"]
    if chosen is None:
        # Not "blocked". No pool being usable is a state the world can leave on
        # its own -- fog lifts as units move, a destroyed extractor frees its
        # pool, and a killed enemy stops covering the route to one -- so this is
        # a wait. Unbounded today -- see the producer wait above.
        return _waiting_on(no_pool_reason(target, survey), builder["unit_id"])
    return (chosen["x"], chosen["y"])


def _decision(
    action: Literal["build", "produce", "wait", "done", "blocked", "stalled"], reason: str
) -> Decision:
    """Build a decision that carries no order.

    Args:
        action: What to do.
        reason: Why.

    Returns:
        The decision, with the order fields zeroed.
    """
    return Decision(action=action, reason=reason, type_name="", unit_id=0, x=0.0, y=0.0, deficit=0)


def _waiting_on(reason: str, unit_id: int, deficit: int = 0) -> Decision:
    """Build a wait that still names the unit the plan is holding.

    **A wait used to name nothing, and that cost the economy most of the match.**
    The plan reserves its worker while merely waiting to afford something, and
    the only way to express that to the expander was a boolean -- which switched
    off *every* spender rather than removing *one* worker. Measured on an
    instrumented match: the expander was skipped on 572 of 800 samples while six
    workers stood free ([[policy-economy]]).

    Args:
        reason: Why the plan is waiting.
        unit_id: The unit it intends to order once it can.
        deficit: Credits still missing, for the price wait and only it -- the
            site waits carry zero because the world can end them on its own,
            and only a shortfall is judged for convergence.

    Returns:
        The decision, carrying the unit and no position.
    """
    return Decision(
        action="wait", reason=reason, type_name="", unit_id=unit_id, x=0.0, y=0.0, deficit=deficit
    )


__all__ = [
    "BUILDER_TYPE",
    "PLACEHOLDER_TYPE",
    "Decision",
    "completed_count",
    "decide",
    "find_producer",
    "next_unsatisfied_index",
]
