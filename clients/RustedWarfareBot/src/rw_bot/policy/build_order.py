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
rather than assumed ([[mechanics-resource-pools]]). Which pool is worth having
is a separate question from which pool is legal, and it is answered by who can
shoot the way there ([[policy-threat]]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.threat import route_is_exposed
from rw_bot.wire.state import BuildOption, Entity, ResourcePool, Sample

BUILDER_TYPE = "builder"

#: Type name of the map editor's placeholder unit.
#:
#: An owned entity in every sample, parked off-map at (-1000, -1000) with
#: 170,000 hit points, and not a playable unit. It answers the engine's build
#: queries for nearly every type in the game, so anything that selects a
#: producer by capability has to exclude it by name -- see
#: :func:`find_producer` for what including it costs.
PLACEHOLDER_TYPE = "editorOrBuilder"

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
        action: ``"build"`` to place the next structure at a chosen position,
            ``"produce"`` to have a building make a unit that rolls out of it,
            ``"wait"`` to do nothing this tick, ``"done"`` when the plan is
            complete, ``"blocked"`` when it cannot proceed at all, and
            ``"stalled"`` when an order was accepted but never took effect.
        reason: Human-readable justification, carried for the run log so a
            session can be read back without re-deriving why it stalled.
        type_name: What to make. Empty unless ``action`` is ``"build"`` or
            ``"produce"``.
        unit_id: Unit to order. Zero unless ``action`` is ``"build"`` or
            ``"produce"``.
        x: Placement world x. Zero unless ``action`` is ``"build"`` -- a
            produced unit appears where the engine puts it, not where the
            planner asks.
        y: Placement world y. Zero unless ``action`` is ``"build"``.
    """

    action: Literal["build", "produce", "wait", "done", "blocked", "stalled"]
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

    Only **finished** ones count, which is a distinction the roster alone
    cannot make. A building joins the roster the moment construction starts,
    so presence is not completion: counting on presence reported a plan done
    while a factory was still a shell, and an unfinished factory produces
    nothing, so the next entry could be ordered against a building that could
    not accept it.

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.

    Returns:
        How many plan entries are satisfied by finished structures the player
        owns.
    """
    remaining: list[str] = list(plan)
    built = 0
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"]:
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

    Unfinished structures do not satisfy an entry, for the same reason they do
    not advance :func:`completed_count`. That keeps the two answers consistent
    -- a half-built factory leaves the plan pointing at the same entry, and the
    caller's once-per-position rule is what stops it being ordered twice while
    it goes up.

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.

    Returns:
        The index to build next, or ``len(plan)`` when the plan is satisfied.
    """
    owned: list[str] = [e["type_name"] for e in sample["entities"] if e["mine"] and e["complete"]]
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


class PoolSurvey(TypedDict):
    """Which pool to build on, and what happened to the ones passed over.

    The counts exist so a refusal can say something true. "Every pool in sight
    is occupied" was the only explanation the policy could give, and once pools
    can also be ruled out for sitting under enemy guns that sentence becomes a
    lie on exactly the runs where the reason matters most.

    Attributes:
        pool: The pool to build on, or None when none qualifies.
        visible: How many pools the sample carries.
        occupied: How many already have a structure standing on them.
        exposed: How many were reachable only through hostile fire.
    """

    pool: ResourcePool | None
    visible: int
    occupied: int
    exposed: int


def survey_pools(
    sample: Sample, anchor: Entity, builder: Entity, catalogue: Mapping[str, UnitStats]
) -> PoolSurvey:
    """Choose a resource pool to build on, and account for the rejects.

    Two filters and then a ranking. A pool with something standing on it is out.
    A pool the builder can only reach by walking through hostile fire is out —
    the route is what matters, not the destination, because the builder this
    rule was written for died in transit rather than on arrival
    ([[policy-threat]]). What survives is ranked by distance.

    **The two measurements start from different places, deliberately.** Exposure
    is measured along the builder's walk, because the builder is what gets shot
    and it starts from wherever it happens to be standing. Distance is measured
    from the anchor, because the economy should grow outward from the base
    rather than trail whichever pool the builder last walked past. Using one
    origin for both would get one of the two questions wrong.

    Distance stays squared. The ordering is all that is wanted from it, and a
    square root would only cost precision.

    Args:
        sample: One observation of the world.
        anchor: The structure to measure distance from.
        builder: The unit that will walk to the pool.
        catalogue: Unit stats by type name, for speeds and attack ranges.

    Returns:
        The chosen pool with the counts behind the choice. ``pool`` is None when
        every visible pool is occupied or exposed, and when none is visible at
        all.
    """
    best: ResourcePool | None = None
    best_distance = 0.0
    occupied = 0
    exposed = 0
    origin = (builder["x"], builder["y"])
    for pool in sample["pools"]:
        if _is_occupied(sample, pool, catalogue):
            occupied += 1
            continue
        if route_is_exposed(sample, catalogue, origin, (pool["x"], pool["y"])):
            exposed += 1
            continue
        distance = (pool["x"] - anchor["x"]) ** 2 + (pool["y"] - anchor["y"]) ** 2
        if best is None or distance < best_distance:
            best = pool
            best_distance = distance
    return PoolSurvey(pool=best, visible=len(sample["pools"]), occupied=occupied, exposed=exposed)


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


def find_producer(sample: Sample, target: str) -> BuildOption | None:
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

    Args:
        sample: One observation of the world.
        target: Type name the plan asks for.

    Returns:
        The option, preferring an available one, or None when nothing the
        player owns makes it.
    """
    placeholders = {
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["type_name"] == PLACEHOLDER_TYPE
    }
    fallback: BuildOption | None = None
    for option in sample["options"]:
        if option["produces"] != target or option["unit_id"] in placeholders:
            continue
        if option["available"]:
            return option
        if fallback is None:
            fallback = option
    return fallback


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
) -> Decision:
    """Choose the next action from one observation.

    Args:
        sample: One observation of the world.
        plan: What to make, in order. Entries may be structures or units.
        catalogue: Unit stats by type name, for prices.
        placements: Placement rules by type name, for where a structure may
            stand.

    Returns:
        The decision.
    """
    built = completed_count(sample, plan)
    index = next_unsatisfied_index(sample, plan)
    if index >= len(plan):
        return _decision("done", f"all {len(plan)} plan entries satisfied")

    target = plan[index]
    undescribed = _undescribed(target, catalogue, placements)
    if undescribed is not None:
        return undescribed
    stats = catalogue[target]
    placement = placements[target]

    # The engine's own answer to "can anything I own make this", asked before
    # an order is spent rather than inferred from one that quietly did nothing.
    # A builder cannot construct a laboratory, and the refusal produces no
    # roster change and no error -- a plan naming one used to run for three
    # hundred samples reporting progress ([[policy-loop]]).
    producer = find_producer(sample, target)
    if producer is None:
        return _decision(
            "blocked",
            f"nothing the player owns can make {target}; the plan is not playable from here",
        )
    if not producer["available"]:
        # Available is a property of the world, not of the plan. A prerequisite
        # can still be built and tech can still be researched, so this is a
        # wait and the stall detector bounds it.
        return _decision(
            "wait",
            f"unit {producer['unit_id']} can make {target} but the action is not available yet",
        )

    # Where it goes is settled before whether it is affordable, so a structure
    # with nowhere legal to stand says so rather than reporting a price. Credits
    # arrive on their own; a taken pool does not become free by waiting for the
    # bank.
    site: tuple[float, float] | None = None
    if producer["placed"]:
        placed = _placed_site(sample, index, target, placement, catalogue)
        if isinstance(placed, dict):
            return placed
        site = placed

    if sample["credits"] < stats["price"]:
        return _decision(
            "wait",
            f"{target} costs {stats['price']}, holding {sample['credits']}",
        )

    if site is None:
        # A produced unit rolls out of the building that made it. The engine
        # chooses where, so this decision carries no position -- offering one
        # would be a number the planner invented.
        return Decision(
            action="produce",
            reason=f"producing {target} ({built + 1} of {len(plan)})",
            type_name=target,
            unit_id=producer["unit_id"],
            x=0.0,
            y=0.0,
        )

    return Decision(
        action="build",
        reason=f"building {target} ({built + 1} of {len(plan)})",
        type_name=target,
        unit_id=producer["unit_id"],
        x=site[0],
        y=site[1],
    )


def _placed_site(
    sample: Sample,
    index: int,
    target: str,
    placement: TypePlacement,
    catalogue: Mapping[str, UnitStats],
) -> tuple[float, float] | Decision:
    """Choose where a placed structure goes, or explain why nowhere will do.

    Two placement rules, because the engine has two. A structure bound to a
    resource pool has exactly as many legal sites as there are usable pools, and
    the ring is irrelevant to it -- offering a ring position would produce an
    order the engine refuses without saying so. Everything else takes the next
    ring position around the anchor.

    Args:
        sample: One observation of the world.
        index: The target's position in the plan, which selects the ring offset.
        target: The type being placed, for the failure message.
        placement: The engine's placement rule for it.
        catalogue: Unit stats by type name, for the anchor and pool judgement.

    Returns:
        The world point to build at, or the decision that stops this tick.
    """
    builder = find_builder(sample)
    if builder is None:
        return _decision("blocked", "the player owns no builder")

    # Measure from the most stable reference available. A structure never moves;
    # the builder does, and measuring from it collapses the ring. With no
    # structure owned the builder is the only reference there is, so the
    # collapse is unavoidable rather than chosen -- and a player who has lost
    # every building should still be able to rebuild.
    anchor = find_anchor(sample, catalogue) or builder

    if not placement["needs_pool"]:
        offset = PLACEMENT_RING[index % len(PLACEMENT_RING)]
        return (anchor["x"] + offset[0], anchor["y"] + offset[1])

    survey = survey_pools(sample, anchor, builder, catalogue)
    chosen = survey["pool"]
    if chosen is None:
        # Not "blocked". No pool being usable is a state the world can leave on
        # its own -- fog lifts as units move, a destroyed extractor frees its
        # pool, and a killed enemy stops covering the route to one -- so this is
        # a wait, and the stall detector is what stops it waiting forever.
        return _decision("wait", _no_pool_reason(target, survey))
    return (chosen["x"], chosen["y"])


def _no_pool_reason(target: str, survey: PoolSurvey) -> str:
    """Explain why no pool was chosen, in terms of what was rejected.

    Worth the separate function because the two cases read completely
    differently to whoever is reading the run log. Nothing visible yet is a map
    the bot has not explored; everything visible and rejected is a map where the
    bot is losing ground, and one of those is a reason to keep playing while the
    other is a reason to look at the screen.

    Args:
        target: Type name the plan asks for.
        survey: The rejected counts.

    Returns:
        The wait reason.
    """
    if survey["visible"] == 0:
        return f"{target} needs a resource pool and none is visible yet"
    return (
        f"{target} needs a resource pool: of the {survey['visible']} in sight, "
        f"{survey['occupied']} are built on and {survey['exposed']} can only be "
        "reached through enemy fire"
    )


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
    return Decision(action=action, reason=reason, type_name="", unit_id=0, x=0.0, y=0.0)


__all__ = [
    "BUILDER_TYPE",
    "PLACEHOLDER_TYPE",
    "PLACEMENT_RING",
    "POOL_OCCUPIED_RADIUS",
    "Decision",
    "PoolSurvey",
    "completed_count",
    "decide",
    "find_anchor",
    "find_builder",
    "find_producer",
    "next_unsatisfied_index",
    "survey_pools",
]
