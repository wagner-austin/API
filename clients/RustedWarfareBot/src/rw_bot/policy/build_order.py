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

Placement is the one part that is not simple, because the engine's rules are
not. Most structures go on a ring around the base. Extractors go on resource
pools and nowhere else, and which types those are is read from the engine
rather than assumed ([[mechanics-resource-pools]]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.wire.state import Entity, ResourcePool, Sample

BUILDER_TYPE = "builder"

#: World-unit radius within which an entity is treated as occupying a pool.
#:
#: A pool is one tile, tiles are 20 world units across, and an extractor's own
#: collision radius is 18 — so anything standing within a tile of the centre is
#: on it. Measured from the map rather than guessed: the tile grid is 20x20 and
#: the extractor's ``radius`` is declared in its unit definition.
#:
#: This is deliberately not the engine's own occupancy test, because the engine
#: has none to ask. Its placement check answers "is this tile a pool", not "is
#: this pool taken"; a second extractor on an occupied pool fails later, on the
#: ordinary overlap rule. So occupancy is the planner's judgement, and it is
#: made here where it can be read and changed.
POOL_OCCUPIED_RADIUS = 20.0

#: World-space offsets from the builder at which successive structures are
#: placed. A ring rather than a line: buildings occupy space, and walking the
#: builder steadily away from its base would put later structures out of reach
#: of anything that has to defend them.
PLACEMENT_RING: tuple[tuple[float, float], ...] = (
    (200.0, 120.0),
    (-200.0, 120.0),
    (200.0, -120.0),
    (-200.0, -120.0),
    (280.0, 0.0),
    (-280.0, 0.0),
    (0.0, 240.0),
    (0.0, -240.0),
)


class Decision(TypedDict):
    """What the policy wants to happen next.

    Attributes:
        action: ``"build"`` to place the next structure, ``"wait"`` to do
            nothing this tick, ``"done"`` when the plan is complete,
            ``"blocked"`` when it cannot proceed at all, and ``"stalled"``
            when an order was accepted but never took effect.
        reason: Human-readable justification, carried for the run log so a
            session can be read back without re-deriving why it stalled.
        type_name: Structure to place. Empty unless ``action`` is ``"build"``.
        unit_id: Builder to order. Zero unless ``action`` is ``"build"``.
        x: Placement world x. Zero unless ``action`` is ``"build"``.
        y: Placement world y. Zero unless ``action`` is ``"build"``.
    """

    action: Literal["build", "wait", "done", "blocked", "stalled"]
    reason: str
    type_name: str
    unit_id: int
    x: float
    y: float


def completed_count(sample: Sample, plan: Sequence[str]) -> int:
    """Count how much of the plan the world already shows.

    Progress is read from the roster rather than tracked in a counter, so a
    structure that was destroyed is no longer counted as built and the policy
    will replace it. Counting from observation is also what makes the policy
    resumable: a planner that reconnects mid-match sees the same progress as
    one that watched the whole thing.

    Only owned entities count. The stream carries every visible entity, so
    without the ownership check an opponent building the same structure in
    view would advance this plan.

    Args:
        sample: One observation of the world.
        plan: Structures to build, in order.

    Returns:
        How many plan entries are satisfied by structures currently owned.
    """
    remaining: list[str] = list(plan)
    built = 0
    for entity in sample["entities"]:
        if not entity["mine"]:
            continue
        if entity["type_name"] in remaining:
            remaining.remove(entity["type_name"])
            built += 1
    return built


def next_unsatisfied_index(sample: Sample, plan: Sequence[str]) -> int:
    """Return the index of the first plan entry the world does not already show.

    This is deliberately not the same question as :func:`completed_count`, and
    conflating them is a real defect rather than a hypothetical one. The count
    answers "how many entries are satisfied"; using it as "which entry is next"
    is only correct when the satisfied entries form a prefix of the plan.

    They diverge the moment a plan names something the player already owns.
    Every game starts with a builder, so the plan ``("landFactory", "builder")``
    counts as one-satisfied, jumps straight to index 1, builds a second builder
    and never builds the factory at all. Scanning for the first unsatisfied
    entry fixes the order while keeping the inventory reading that makes
    progress resumable.

    Args:
        sample: One observation of the world.
        plan: Structures to build, in order.

    Returns:
        The index to build next, or ``len(plan)`` when the plan is satisfied.
    """
    owned: list[str] = [e["type_name"] for e in sample["entities"] if e["mine"]]
    for index, wanted in enumerate(plan):
        if wanted in owned:
            owned.remove(wanted)
            continue
        return index
    return len(plan)


def find_anchor(sample: Sample, catalogue: Mapping[str, UnitStats]) -> Entity | None:
    """Return the fixed structure that placement offsets are measured from.

    The offsets in :data:`PLACEMENT_RING` describe a ring around a base, and a
    ring only spreads if its centre holds still. Measuring from the builder does
    not work, because the builder walks to each site it is sent to: by the third
    structure it is standing next to the first one, and the third offset lands
    on top of it. Observed directly — factory one at ``(4450, 2730)`` and the
    third order at ``(4451, 2646)``, 84 world units apart on a grid where a land
    factory is wider than that, which the engine silently refused.

    The anchor is the oldest owned immobile entity: lowest engine id among
    entities the catalogue gives a speed of zero. Lowest id is the oldest
    because ids are assigned once at construction, so the anchor is the Command
    Center at the start of a match and stays the same structure as more are
    built. Immobility is read from the catalogue rather than assumed by type
    name ([[mechanics-unit-catalogue]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the speed field.

    Returns:
        The anchor entity, or None when the player owns no immobile structure,
        in which case the caller measures from the builder instead.
    """
    oldest: Entity | None = None
    for entity in sample["entities"]:
        if not entity["mine"]:
            continue
        stats = catalogue.get(entity["type_name"])
        if stats is None or stats["speed"] != 0.0:
            continue
        if oldest is None or entity["unit_id"] < oldest["unit_id"]:
            oldest = entity
    return oldest


def find_free_pool(
    sample: Sample, anchor: Entity, catalogue: Mapping[str, UnitStats]
) -> ResourcePool | None:
    """Return the nearest visible resource pool with no structure on it.

    Nearest to the anchor rather than to the builder, so the economy grows out
    from the base instead of trailing whichever pool the builder last walked
    past. Distance is squared and left squared: the ordering is all that
    matters and a square root would only cost precision.

    Args:
        sample: One observation of the world.
        anchor: The structure to measure distance from.
        catalogue: Unit stats by type name, for the speed field.

    Returns:
        The nearest free pool, or None when every visible pool is taken — which
        is also the answer when none is visible at all.
    """
    best: ResourcePool | None = None
    best_distance = 0.0
    for pool in sample["pools"]:
        if _is_occupied(sample, pool, catalogue):
            continue
        distance = (pool["x"] - anchor["x"]) ** 2 + (pool["y"] - anchor["y"]) ** 2
        if best is None or distance < best_distance:
            best = pool
            best_distance = distance
    return best


def _is_occupied(sample: Sample, pool: ResourcePool, catalogue: Mapping[str, UnitStats]) -> bool:
    """Report whether a structure is standing on a pool.

    Only immobile entities count. A builder walking across a pool — or parked
    on one, which is exactly where it stands after building there — does not
    stop anything being built on it, and counting it would make the pool the
    bot just used look permanently taken.

    Ownership does not matter. An opponent's extractor holds a pool exactly as
    firmly as ours does, and ordering onto it would spend the credits and
    produce nothing.

    A type the catalogue does not know is treated as not occupying. The two
    errors are not symmetric: guessing "free" wrongly costs one order the
    engine refuses, which the stall detector already catches, while guessing
    "occupied" wrongly hides the pool for the rest of the run.

    Args:
        sample: One observation of the world.
        pool: The pool to test.
        catalogue: Unit stats by type name, for the speed field.

    Returns:
        True when an immobile visible entity is within
        :data:`POOL_OCCUPIED_RADIUS` of the pool's centre.
    """
    limit = POOL_OCCUPIED_RADIUS**2
    for entity in sample["entities"]:
        stats = catalogue.get(entity["type_name"])
        if stats is None or stats["speed"] != 0.0:
            continue
        if (entity["x"] - pool["x"]) ** 2 + (entity["y"] - pool["y"]) ** 2 <= limit:
            return True
    return False


def find_builder(sample: Sample) -> Entity | None:
    """Return a builder from the roster, or None when the player owns none.

    Enemy builders are visible in the stream and are not orderable, so the
    ownership check is what keeps this from returning one.

    Args:
        sample: One observation of the world.

    Returns:
        The first owned builder, or None.
    """
    for entity in sample["entities"]:
        if entity["mine"] and entity["type_name"] == BUILDER_TYPE:
            return entity
    return None


def decide(
    sample: Sample,
    plan: Sequence[str],
    catalogue: Mapping[str, UnitStats],
    placements: Mapping[str, TypePlacement],
) -> Decision:
    """Choose the next action from one observation.

    Args:
        sample: One observation of the world.
        plan: Structures to build, in order.
        catalogue: Unit stats by type name, for prices.
        placements: Placement rules by type name, for where a structure may
            stand.

    Returns:
        The decision.
    """
    built = completed_count(sample, plan)
    index = next_unsatisfied_index(sample, plan)
    if index >= len(plan):
        return _decision("done", f"all {len(plan)} structures built")

    target = plan[index]
    stats = catalogue.get(target)
    if stats is None:
        return _decision(
            "blocked",
            f"{target!r} is not in the unit catalogue, so its price is unknown",
        )
    placement = placements.get(target)
    if placement is None:
        return _decision(
            "blocked",
            f"{target!r} is not in the placement dump, so where it may stand is unknown",
        )

    builder = find_builder(sample)
    if builder is None:
        return _decision("blocked", "the player owns no builder")

    # Measure from the most stable reference available. A structure never
    # moves; the builder does, and measuring from it collapses the ring. With
    # no structure owned the builder is the only reference there is, so the
    # collapse is unavoidable rather than chosen -- and a player who has lost
    # every building should still be able to rebuild.
    anchor = find_anchor(sample, catalogue) or builder

    site = _site_for(sample, index, placement, anchor, catalogue)
    if site is None:
        # Not "blocked". Every pool in sight being taken is a state the world
        # can leave on its own -- fog lifts as units move, and a destroyed
        # extractor frees its pool -- so this is a wait, and the stall detector
        # is what stops it waiting forever.
        return _decision(
            "wait",
            f"{target} needs a resource pool and every one of the "
            f"{len(sample['pools'])} in sight is occupied",
        )

    if sample["credits"] < stats["price"]:
        return _decision(
            "wait",
            f"{target} costs {stats['price']}, holding {sample['credits']}",
        )

    return Decision(
        action="build",
        reason=f"building {target} ({built + 1} of {len(plan)})",
        type_name=target,
        unit_id=builder["unit_id"],
        x=site[0],
        y=site[1],
    )


def _site_for(
    sample: Sample,
    index: int,
    placement: TypePlacement,
    anchor: Entity,
    catalogue: Mapping[str, UnitStats],
) -> tuple[float, float] | None:
    """Choose where to put the next structure.

    Two placement rules, because the engine has two. A structure bound to a
    resource pool has exactly as many legal sites as there are free pools, and
    the ring is irrelevant to it — offering a ring position would produce an
    order the engine refuses without saying so. Everything else takes the next
    ring position around the anchor.

    Args:
        sample: One observation of the world.
        index: The structure's position in the plan, which selects the ring
            offset.
        placement: The engine's placement rule for it.
        anchor: The structure ring offsets are measured from.
        catalogue: Unit stats by type name, for judging pool occupancy.

    Returns:
        The world point to build at, or None when the structure needs a pool
        and no free one is visible.
    """
    if placement["needs_pool"]:
        pool = find_free_pool(sample, anchor, catalogue)
        if pool is None:
            return None
        return pool["x"], pool["y"]

    offset = PLACEMENT_RING[index % len(PLACEMENT_RING)]
    return anchor["x"] + offset[0], anchor["y"] + offset[1]


def _decision(
    action: Literal["build", "wait", "done", "blocked", "stalled"], reason: str
) -> Decision:
    """Build a decision that carries no order.

    Args:
        action: What to do.
        reason: Why.

    Returns:
        The decision, with the order fields zeroed.
    """
    return Decision(action=action, reason=reason, type_name="", unit_id=0, x=0.0, y=0.0)


__all__ = [
    "BUILDER_TYPE",
    "PLACEMENT_RING",
    "POOL_OCCUPIED_RADIUS",
    "Decision",
    "completed_count",
    "decide",
    "find_anchor",
    "find_builder",
    "find_free_pool",
    "next_unsatisfied_index",
]
