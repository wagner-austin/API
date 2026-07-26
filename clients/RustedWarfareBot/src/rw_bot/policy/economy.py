"""Claiming resource pools for as long as the match lasts.

The bot's economy was a fixed size. ``DEFAULT_GOALS`` asked for three
extractors, the build plan finished, and from roughly thirty seconds in until
the end of the match income never changed again -- on a map carrying **46**
resource pools, of which we held three. Nothing in the code could have taken a
fourth: :func:`rw_bot.policy.production.sustain` orders produced units and
refuses placed structures, and ``reinforcements`` drops anything needing a pool,
both correctly, because choosing a site is a placement decision and a producer
queue cannot express one. So the gap was not a bug in either of them; it was a
policy nobody had written.

Measured against five opponents that expand continuously, the shape of the loss
was never tactical. The army went 4 -> 2 while the opponents went 47 -> 142
visible units over the same window ([[ai-opponent-strategy]]); reinforcement
bought survival, not parity, and target commitment cut order churn 4.3-6.5x
without changing a single outcome. An economy pinned at three extractors is what
those runs were actually measuring.

This module answers one question -- should another pool be claimed, and which --
and answers it from the same survey the build plan uses, so a pool is refused
here for exactly the reasons it would be refused there: something is standing on
it, the builder cannot walk to it, or the walk goes through hostile fire
([[policy-threat]]).

Pure, like the rest of the policy layer. :mod:`rw_bot.policy.campaign` is what
sends the order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.build_order import (
    PLACEMENT_RING,
    find_anchor,
    find_producer,
    survey_pools,
)
from rw_bot.policy.observation import is_rising
from rw_bot.policy.production import production_bound
from rw_bot.wire.state import Entity, Sample

#: The producer the opening plan builds and production expansion keeps adding.
FACTORY_TYPE = "landFactory"

#: The extractor the opening plan builds and expansion keeps building.
#:
#: T1 rather than a higher tier because it is what a Builder can place from the
#: start: the archived capture has the Builder offering ``extractorT1`` and
#: nothing above it, while T2 and T3 appear only on the map editor's placeholder
#: ([[mechanics-build-tree]]). The upgrade path exists -- an ``extractorT1``
#: lists ``extractorT2`` as a build edge, priced at 1400 against T1's 700 -- but
#: no capture has yet shown an owned extractor offering it, so upgrading is
#: deliberately not attempted here rather than guessed at.
EXTRACTOR_TYPE = "extractorT1"


class Expansion(TypedDict):
    """Whether to claim another pool, and the reasoning behind the answer.

    The counts are carried so a match that never expanded can say why. "No pool
    was taken" has at least five distinct causes -- no builder alive, no credits
    above the reserve, every pool occupied, every route exposed, an order still
    in flight -- and they call for opposite responses. A bare count of
    expansions cannot tell them apart.

    Attributes:
        build: Whether to place an extractor now.
        reason: Human-readable justification, for the run log.
        type_name: What to place. Empty unless ``build``.
        unit_id: The producer to order. Zero unless ``build``.
        x: Placement world x. Zero unless ``build``.
        y: Placement world y. Zero unless ``build``.
        owned: Finished extractors the player holds.
        occupied: Pools with something already standing on them.
        exposed: Pools reachable only through hostile fire.
    """

    build: bool
    reason: str
    type_name: str
    unit_id: int
    x: float
    y: float
    owned: int
    occupied: int
    exposed: int


def count_extractors(sample: Sample, type_name: str = EXTRACTOR_TYPE) -> int:
    """Count the finished extractors the player owns.

    Unfinished ones are excluded deliberately. A structure joins the roster the
    moment construction starts, so counting presence would report income the
    player does not have yet -- the same distinction ``completed_count`` draws
    for the build plan ([[policy-loop]]).

    Args:
        sample: One observation of the world.
        type_name: The extractor type to count.

    Returns:
        How many finished extractors of that type the player owns.
    """
    return sum(
        1
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["type_name"] == type_name
    )


def _waiting(reason: str, sample: Sample, type_name: str) -> Expansion:
    """Build a no-expansion answer that still carries its reasoning.

    Args:
        reason: Why nothing is being claimed.
        sample: One observation of the world.
        type_name: The extractor type under consideration.

    Returns:
        An expansion with ``build`` false.
    """
    return Expansion(
        build=False,
        reason=reason,
        type_name="",
        unit_id=0,
        x=0.0,
        y=0.0,
        owned=count_extractors(sample, type_name),
        occupied=0,
        exposed=0,
    )


def _placer(sample: Sample, type_name: str) -> Entity | None:
    """Return the entity the engine says can place an extractor.

    Selected by capability rather than by type name, which is the same rule the
    build plan uses and for the same reason: the map editor's placeholder offers
    nearly every type in the game and is parked off-map, so anything choosing a
    producer by what it can make has to go through :func:`find_producer` to have
    the placeholder excluded ([[policy-loop]]).

    Args:
        sample: One observation of the world.
        type_name: The extractor type to place.

    Returns:
        The producing entity, or None when nothing owned can place one right
        now.
    """
    option = find_producer(sample, type_name)
    if option is None or not option["available"]:
        return None
    for entity in sample["entities"]:
        if entity["unit_id"] == option["unit_id"]:
            return entity
    return None


def expand_economy(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    reaches: Mapping[str, float],
    *,
    reserve: int,
    builder_moved: bool,
    type_name: str = EXTRACTOR_TYPE,
) -> Expansion:
    """Decide whether to claim another resource pool, and which one.

    Three gates, in the order that makes the cheapest check first.

    **An order already in flight blocks another.** There is one builder, and it
    can walk to one pool. A builder in transit is an order still being carried
    out, exactly as the build loop measures it -- so expansion asks the world
    (is it moving, is an extractor going up) rather than counting samples
    ([[policy-loop]]). Without this the fight loop would re-task the builder
    every sample and it would never arrive anywhere, which is the same defect
    that produced 743 attack orders against 24 targets before commitment fixed
    it.

    **The reserve protects the army.** Expansion spends the same credits
    reinforcement does, and an extractor that costs 700 pays back over time
    while a tank replaces a loss now. Holding back a reserve is what keeps a
    long-run investment from starving the short-run one; the caller sets it,
    because what it costs to replace a loss is the caller's business
    ([[policy-economy]]).

    **The pool must be worth having.** That question is not re-answered here.
    :func:`survey_pools` already rejects occupied pools, pools on another land
    mass, and pools whose approach runs through hostile fire, and ranks what
    survives by distance from the base so the economy grows outward rather than
    trailing wherever the builder last walked.

    There is deliberately no cap on how many pools to take. The map's pool
    count, the credit reserve and the threat filter bound this on their own, and
    a number written here would be a guess overriding three measurements.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the extractor's price and for
            the speed that judges pool occupancy.
        reaches: Attack range by type name, for the threat filter.
        reserve: Credits to leave unspent for the army.
        builder_moved: Whether the placing unit moved since the previous
            sample, which is what "still walking to its site" looks like.
        type_name: The extractor type to place.

    Returns:
        What to do, with the reasoning behind it either way.
    """
    if builder_moved:
        return _waiting("builder still walking to its site", sample, type_name)
    if is_rising(sample, type_name):
        return _waiting(f"{type_name} already going up", sample, type_name)

    builder = _placer(sample, type_name)
    if builder is None:
        return _waiting(f"nothing owned can place {type_name}", sample, type_name)

    stats = catalogue.get(type_name)
    if stats is None:
        return _waiting(f"{type_name} is not in the catalogue", sample, type_name)
    needed = stats["price"] + reserve
    if sample["credits"] < needed:
        return _waiting(
            f"{sample['credits']} credits, need {needed} to expand past a {reserve} reserve",
            sample,
            type_name,
        )

    # The anchor is what distance is measured from, so the economy grows out of
    # the base. A player holding no immobile structure measures from the builder
    # instead, which is the build plan's own fallback.
    anchor = find_anchor(sample, catalogue) or builder
    survey = survey_pools(sample, anchor, builder, catalogue, reaches)
    owned = count_extractors(sample, type_name)
    if survey["pool"] is None:
        return Expansion(
            build=False,
            reason=(
                f"no pool free of {survey['visible']}: "
                f"{survey['occupied']} occupied, {survey['unreachable']} unreachable, "
                f"{survey['exposed']} exposed"
            ),
            type_name="",
            unit_id=0,
            x=0.0,
            y=0.0,
            owned=owned,
            occupied=survey["occupied"],
            exposed=survey["exposed"],
        )

    pool = survey["pool"]
    return Expansion(
        build=True,
        reason=f"{type_name} #{owned + 1} at ({pool['x']:.0f}, {pool['y']:.0f})",
        type_name=type_name,
        unit_id=builder["unit_id"],
        x=pool["x"],
        y=pool["y"],
        owned=owned,
        occupied=survey["occupied"],
        exposed=survey["exposed"],
    )


def _immobile_count(sample: Sample, catalogue: Mapping[str, UnitStats]) -> int:
    """Count the finished immobile structures the player holds.

    Immobility comes from the catalogue's speed field rather than a type-name
    guess, the same test the build plan's anchor uses. A type the catalogue does
    not describe is not counted: it cannot be sited from either, so including it
    would shift the ring index for a structure the planner cannot reason about.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the speed field.

    Returns:
        How many finished immobile structures are owned.
    """
    standing = 0
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"]:
            continue
        stats = catalogue.get(entity["type_name"])
        if stats is not None and stats["speed"] == 0.0:
            standing += 1
    return standing


def expand_production(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    *,
    reserve: int,
    factory_type: str = FACTORY_TYPE,
) -> Expansion:
    """Decide whether to add another producer.

    The sibling question to :func:`expand_economy`, and the one that was
    missing. Expansion answered "can I earn more"; nothing answered "can I
    *spend* more". A run measured ten extractors of income against a single
    factory and banked 7,013 credits over 1,500 samples -- not because the
    spending rule failed, but because :func:`~rw_bot.policy.production.sustain`
    can only fill queues that exist ([[policy-combat]]).

    The trigger is observed rather than thresholded: every producer busy means
    the queue is the constraint, and affording another factory means the bank
    is not ([[policy-production]]). Neither alone justifies the spend.

    Sited on the ring around the anchor, indexed by how many immobile
    structures the player already holds. That index advances as the base grows,
    which spreads factories rather than stacking them; a collision with an
    existing building is still possible, and the engine refuses it silently, so
    the caller's retry clock is what absorbs one rather than a cleverer site
    rule here ([[building-structures]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for prices and immobility.
        reserve: Credits to leave unspent for the army.
        factory_type: The producer to add.

    Returns:
        What to do, with the reasoning behind it either way.
    """
    if not production_bound(sample, catalogue, factory_type):
        return _waiting("production is not the constraint", sample, factory_type)

    stats = catalogue[factory_type]
    needed = stats["price"] + reserve
    if sample["credits"] < needed:
        return _waiting(
            f"{sample['credits']} credits, need {needed} past a {reserve} reserve",
            sample,
            factory_type,
        )

    builder = _placer(sample, factory_type)
    if builder is None:
        return _waiting(f"nothing owned can place {factory_type}", sample, factory_type)
    anchor = find_anchor(sample, catalogue) or builder

    standing = _immobile_count(sample, catalogue)
    offset = PLACEMENT_RING[standing % len(PLACEMENT_RING)]
    return Expansion(
        build=True,
        reason=f"every producer busy on {sample['credits']} credits; adding a {factory_type}",
        type_name=factory_type,
        unit_id=builder["unit_id"],
        x=anchor["x"] + offset[0],
        y=anchor["y"] + offset[1],
        owned=count_extractors(sample),
        occupied=0,
        exposed=0,
    )


__all__ = [
    "EXTRACTOR_TYPE",
    "FACTORY_TYPE",
    "Expansion",
    "count_extractors",
    "expand_economy",
    "expand_production",
]
