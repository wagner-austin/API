"""Where a structure goes, and why every other site was rejected.

Placement is the one part of the build policy that is not simple, because the
engine's rules are not. Most structures take the next free slot on a ring around
the base. Extractors go on resource pools and nowhere else, and which types
those are is read from the engine rather than assumed
([[mechanics-resource-pools]]). Which pool is *worth* having is a separate
question from which pool is *legal*, and it is answered by who can shoot the way
there ([[policy-threat]]).

Split out of the build policy because the two are different jobs that were only
ever adjacent. The plan decides **what** to make next and whether it can; this
decides **where** it would stand. Both the opening plan and the economy ask
these questions of the same map and must get the same answers, or they collide:
one taking a pool the other is already walking to, or two structures landing on
one ring slot -- both of which the engine refuses in silence ([[policy-loop]]).

Pure, like the rest of the policy layer: geometry and one sample, no clock and
no socket.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.threat import route_is_exposed
from rw_bot.wire.state import Entity, ResourcePool, Sample

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


#: World-unit radius within which a structure is treated as filling a ring slot.
#:
#: Bounded from both sides by measurement. It must exceed a building's own
#: extent -- a factory ordered 84 world units from an existing one was silently
#: refused ([[building-structures]]) -- and it must stay below half the distance
#: between neighbouring ring positions, which is 144 at the closest pair, or one
#: building would fill two slots.
RING_SLOT_RADIUS = 60.0

#: Clearance a ring-placed structure needs around its site, in world units.
#:
#: Wider than the slot radius because the things the ring places are wide: a
#: land factory's footprint exceeds 84 units, and a site that passes a
#: 60-unit occupancy check can still be refused by the engine for a turret
#: standing 80 away. The first Hard batch after the defence cover ring made
#: turrets land showed exactly that -- ten identical factory orders at one
#: ring slot, all silently refused, zero factories and zero army in every
#: match, where the pre-cover-ring era stood two to four factories
#: (log: 2026-07-31). The cover ring densified the base; the clearance test
#: has to see a factory's width, not a turret's.
RING_CLEARANCE = 130.0


def next_ring_site(
    sample: Sample, anchor: Entity, catalogue: Mapping[str, UnitStats]
) -> tuple[float, float] | None:
    """Return the first ring position with nothing standing on it.

    **Read from the world rather than counted, and there is one counter here
    rather than two.** Both callers used to keep their own index into the ring
    and both were wrong in the same way. The build plan indexed by plan
    position, so a plan opening with three pool-placed extractors skipped ring
    slots 0 to 2 and never used them. The economy indexed by how many immobile
    structures were standing, which counts extractors -- so every pool claimed
    advanced the factory's ring index by one, for a structure sitting nowhere
    near the ring. Ten extractors moved it four slots.

    Neither index answers "which position is free", which is the only question a
    site needs. Two of them can also land on the same slot, and the engine
    refuses that silently ([[policy-loop]]).

    Args:
        sample: One observation of the world.
        anchor: The structure the ring is measured around.
        catalogue: Unit stats by type name, for the speed that tells a structure
            from a unit.

    Returns:
        The first free position, or None when every ring slot is taken.
    """
    limit = RING_CLEARANCE**2
    for offset in PLACEMENT_RING:
        site = (anchor["x"] + offset[0], anchor["y"] + offset[1])
        if not any(
            _is_structure(entity, catalogue)
            and (entity["x"] - site[0]) ** 2 + (entity["y"] - site[1]) ** 2 <= limit
            for entity in sample["entities"]
        ):
            return site
    return None


#: Offsets tried around a structure needing cover, nearest shell first.
#:
#: Two shells, both inside the turret's 165-unit reach so the turret placed at
#: any of them actually covers the structure it was bought for -- the base
#: ring's slots sit 233 out, which is why defence cannot borrow it. The first
#: shell is the ring slot radius, the engine's own snapping tolerance
#: ([[building-structures]]); structure centres 60 apart do not overlap.
COVER_RING: tuple[tuple[float, float], ...] = (
    (60.0, 0.0),
    (0.0, 60.0),
    (-60.0, 0.0),
    (0.0, -60.0),
    (60.0, 60.0),
    (-60.0, 60.0),
    (60.0, -60.0),
    (-60.0, -60.0),
    (120.0, 0.0),
    (0.0, 120.0),
    (-120.0, 0.0),
    (0.0, -120.0),
)


def clear_site_near(
    sample: Sample, target: Entity, catalogue: Mapping[str, UnitStats]
) -> tuple[float, float] | None:
    """Return the first cover position around a structure with nothing on it.

    The check defence never had: for its whole prior life the turret site was
    a bare offset reached for without looking, the engine refuses an occupied
    site silently, and the scorecards priced the habit at 27 paid orders for
    about five turrets standing (log: 2026-07-30). This is the same occupancy
    predicate :func:`next_ring_site` already trusts, walked over
    :data:`COVER_RING` instead of the base ring.

    The covered structure itself is exempt from the occupancy test -- it
    stands one ring-slot radius from every first-shell site by construction,
    which is exactly the distance the test would otherwise refuse.

    Args:
        sample: One observation of the world.
        target: The structure being covered.
        catalogue: Unit stats by type name, for telling a structure from a
            unit.

    Returns:
        The first clear position, or None when every cover offset is taken.
    """
    limit = RING_SLOT_RADIUS**2
    for offset in COVER_RING:
        site = (target["x"] + offset[0], target["y"] + offset[1])
        if not any(
            entity["unit_id"] != target["unit_id"]
            and _is_structure(entity, catalogue)
            and (entity["x"] - site[0]) ** 2 + (entity["y"] - site[1]) ** 2 <= limit
            for entity in sample["entities"]
        ):
            return site
    return None


def clear_point_near(
    sample: Sample, point: tuple[float, float], catalogue: Mapping[str, UnitStats]
) -> tuple[float, float] | None:
    """Return the first clear position at or around a bare map point.

    The point-anchored twin of :func:`clear_site_near`, for callers whose
    site is a coordinate rather than a structure -- the turret creep advances
    toward a projected point with nothing of ours standing there yet
    ([[policy-creep]]). The point itself is tried first, then the same cover
    offsets, under the same occupancy predicate the other siting paths trust.

    Args:
        sample: One observation of the world.
        point: Where the caller would like to build.
        catalogue: Unit stats by type name, for telling a structure from a
            unit.

    Returns:
        The first clear position, or None when the point and every offset
        around it are taken.
    """
    limit = RING_SLOT_RADIUS**2
    for offset in ((0.0, 0.0), *COVER_RING):
        site = (point[0] + offset[0], point[1] + offset[1])
        if not any(
            _is_structure(entity, catalogue)
            and (entity["x"] - site[0]) ** 2 + (entity["y"] - site[1]) ** 2 <= limit
            for entity in sample["entities"]
        ):
            return site
    return None


def _is_structure(entity: Entity, catalogue: Mapping[str, UnitStats]) -> bool:
    """Report whether an entity is an immobile thing that occupies ground.

    Ownership is not checked: an opponent's building fills a position exactly as
    firmly as ours, and ordering onto it spends the credits for nothing.

    Args:
        entity: The entity to test.
        catalogue: Unit stats by type name, for the speed field.

    Returns:
        True when the catalogue gives it zero speed.
    """
    stats = catalogue.get(entity["type_name"])
    return stats is not None and stats["speed"] == 0.0


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
        unreachable: How many the builder cannot walk to at all.
        exposed: How many were reachable only through hostile fire.
    """

    pool: ResourcePool | None
    visible: int
    occupied: int
    unreachable: int
    exposed: int


def survey_pools(
    sample: Sample,
    anchor: Entity,
    builder: Entity,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    claimed: Sequence[tuple[float, float]] = (),
) -> PoolSurvey:
    """Choose a resource pool to build on, and account for the rejects.

    Three filters and then a ranking. A pool with something standing on it is
    out. **A pool another worker is already walking to is out** -- see below. A
    pool the builder can only reach by walking through hostile fire is out —
    the route is what matters, not the destination, because the builder this
    rule was written for died in transit rather than on arrival
    ([[policy-threat]]). What survives is ranked by distance.

    **A claim in flight is invisible in the world, and that cost nineteen orders
    in twenty.** Occupancy is judged by what is *standing* on a pool, so a pool
    a builder is merely walking toward still reads as free. With one free worker
    that was nearly harmless -- one order was in flight at a time. The moment
    several workers were freed ([[policy-economy]]) each of them was offered the
    same nearest pool on successive observations, because none had arrived yet:
    an instrumented run granted **23 extractor orders and finished with four
    extractors, having lost none at all**. The credits were not burnt -- a
    granted claim is intent, and the engine simply built one structure -- but
    every duplicate cost a worker its travel time.

    So the sites already under orders are passed in rather than inferred. The
    workforce has recorded them since it began tracking jobs; nothing here is
    new state, only a question that was never asked.

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
        catalogue: Unit stats by type name, for the speed that judges pool
            occupancy.
        profiles: Combat profiles by type name, for the threat filter.
        claimed: Sites workers are already under orders to build on. Counted as
            occupied, because they will be.

    Returns:
        The chosen pool with the counts behind the choice. ``pool`` is None when
        every visible pool is occupied or exposed, and when none is visible at
        all.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe a
            visible type.
    """
    best: ResourcePool | None = None
    best_distance = 0.0
    occupied = 0
    unreachable = 0
    exposed = 0
    origin = (builder["x"], builder["y"])
    limit = POOL_OCCUPIED_RADIUS**2
    for pool in sample["pools"]:
        if _is_occupied(sample, pool, catalogue):
            occupied += 1
            continue
        # Counted under `occupied` rather than given a category of its own: from
        # the caller's side "somebody already has this pool" is one fact, and
        # splitting it would say the map was emptier than it is.
        if any(
            (pool["x"] - site[0]) ** 2 + (pool["y"] - site[1]) ** 2 <= limit for site in claimed
        ):
            occupied += 1
            continue
        if not _can_walk_to(builder, pool):
            unreachable += 1
            continue
        if route_is_exposed(sample, profiles, builder, origin, (pool["x"], pool["y"])):
            exposed += 1
            continue
        distance = (pool["x"] - anchor["x"]) ** 2 + (pool["y"] - anchor["y"]) ** 2
        if best is None or distance < best_distance:
            best = pool
            best_distance = distance
    return PoolSurvey(
        pool=best,
        visible=len(sample["pools"]),
        occupied=occupied,
        unreachable=unreachable,
        exposed=exposed,
    )


def _can_walk_to(builder: Entity, pool: ResourcePool) -> bool:
    """Report whether the builder can reach a pool over land at all.

    The engine precomputes connected components per movement layer and reduces
    reachability to comparing two component ids, which is the whole of this
    function ([[mechanics-movement-layers]]). Both ids ride on the wire, so no
    search happens here.

    Negative ids are rejected rather than compared. The engine uses them for
    "impassable", "off the map" and "the grids were never built", and its own
    predicate has a hole -- it compares two of the last kind for equality and
    answers true. Refusing every negative is strictly more conservative than the
    engine, and costs at most a pool it might have allowed.

    There is deliberately no case for a builder that travels on some other
    layer. Its component id would index a different grid and simply not match
    any land component, so the pool is refused — the bot declines to build
    rather than building somewhere it cannot reach, which is the safe direction
    and needs no branch to arrange.

    Args:
        builder: The unit that would walk there.
        pool: The pool it would walk to.

    Returns:
        True when the builder and the pool share a land component.
    """
    if builder["group"] < 0 or pool["group_land"] < 0:
        return False
    return builder["group"] == pool["group_land"]


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


def no_pool_reason(target: str, survey: PoolSurvey) -> str:
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
        f"{survey['occupied']} are built on, {survey['unreachable']} cannot be "
        f"walked to and {survey['exposed']} can only be reached through enemy fire"
    )


__all__ = [
    "COVER_RING",
    "PLACEMENT_RING",
    "POOL_OCCUPIED_RADIUS",
    "RING_CLEARANCE",
    "RING_SLOT_RADIUS",
    "PoolSurvey",
    "clear_point_near",
    "clear_site_near",
    "find_anchor",
    "next_ring_site",
    "no_pool_reason",
    "survey_pools",
]
