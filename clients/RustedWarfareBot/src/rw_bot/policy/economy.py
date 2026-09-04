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

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.upgrades import TIER_CHAINS, next_tier, satisfies
from rw_bot.policy.build_order import find_producer
from rw_bot.policy.production import production_bound
from rw_bot.policy.siting import (
    find_anchor,
    next_ring_site,
    survey_pools,
)
from rw_bot.wire.state import Entity, Sample

#: The producer the opening plan builds and production expansion keeps adding.
FACTORY_TYPE = "landFactory"


#: What an owned extractor converts itself into.
#:
#: Tier two pays 12 credits a second against tier one's 8, for 1,400 -- a
#: payback of about 350 seconds against matches lasting roughly 4,400
#: ([[policy-holding-ground]]).
UPGRADE_TYPE = "extractorT2"

#: The extractor the opening plan builds and expansion keeps building.
#:
#: T1 rather than a higher tier because it is what a Builder can *place*: the
#: capture has the Builder offering ``extractorT1`` and nothing above it
#: ([[mechanics-build-tree]]).
#:
#: A higher tier is reached by upgrading rather than by placing, and that path
#: is now taken -- see :data:`UPGRADE_TYPE` and :func:`upgradeable`. This note
#: used to say no capture had shown an owned extractor offering the upgrade, so
#: it was "deliberately not attempted rather than guessed at". The caution was
#: right and the conclusion was wrong: the agent was dropping the action before
#: it reached the wire ([[policy-holding-ground]]).
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
        priced_out: Whether the *only* thing stopping this was credits. It
            separates "the map has nothing left to claim" from "we cannot afford
            it yet" -- two states that look identical in a ``build`` of False
            and call for opposite responses.

            **Without it a cheaper spender jumps the queue every time the
            economy is short.** A turret is 500 and an extractor 700 plus the
            reserve, so on any observation where income is refused for want of
            1,150, defence is offered the same balance, asks for 500, and
            succeeds. Measured at Hard: **29 turrets bought against 4
            extractors, with 43 of 47 extractor claims refused for credits**,
            while income sat at 34/s and never grew ([[policy-holding-ground]]).
        owned: Finished extractors the player holds.
        occupied: Pools with something already standing on them.
        exposed: Pools reachable only through hostile fire.
        visible: Pools the sample carries in total. Zero means no survey ran
            on this answer -- a refusal before the survey, or a decision that
            never asked about pools at all -- and a zero is how the caller
            tells "the map was measured empty" from "the map was not
            measured", two states a shared default would fuse.
        unreachable: Pools the builder cannot walk to at all. With
            ``visible`` this is the map's own answer to how large an economy
            it can ever fund, which is what the expander's protection floor
            is derived from rather than a number measured on one map and
            carried to every other ([[policy-holding-ground]]).
    """

    build: bool
    reason: str
    priced_out: bool
    type_name: str
    unit_id: int
    x: float
    y: float
    owned: int
    occupied: int
    exposed: int
    visible: int
    unreachable: int


def count_extractors(sample: Sample, type_name: str = EXTRACTOR_TYPE) -> int:
    """Count the finished extractors the player owns, at any tier.

    Unfinished ones are excluded deliberately. A structure joins the roster the
    moment construction starts, so counting presence would report income the
    player does not have yet -- the same distinction ``completed_count`` draws
    for the build plan ([[policy-loop]]).

    **Every tier counts, and that correction matters more than it looks.** This
    counted the named type alone, which was right for as long as the bot could
    not upgrade. The moment it could, an upgraded extractor stopped being an
    extractor as far as this was concerned, and a run holding three tier-two
    extractors and earning 54 credits a second reported ``extractors 0 -> 0``.
    A figure that quietly means something other than what it says is how the
    1,500-sample reading went wrong ([[policy-holding-ground]]).

    Args:
        sample: One observation of the world.
        type_name: The base extractor type. Higher tiers of the same family are
            counted with it.

    Returns:
        How many finished extractors of that family the player owns.
    """
    # A set because the upgrade paths share a prefix: every tier below the
    # branch appears on both, and counting membership rather than occurrences
    # is what makes that harmless.
    counted = {name for chain in TIER_CHAINS for name in chain if satisfies(name, type_name)}
    return sum(
        1
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["type_name"] in counted
    )


def waiting(reason: str, sample: Sample, type_name: str, *, priced_out: bool = False) -> Expansion:
    """Build a no-expansion answer that still carries its reasoning.

    Args:
        reason: Why nothing is being claimed.
        sample: One observation of the world.
        type_name: The extractor type under consideration.
        priced_out: Whether credits were the only obstacle. Defaults to False,
            because every other refusal here is a fact about the world rather
            than about the balance.

    Returns:
        An expansion with ``build`` false.
    """
    return Expansion(
        build=False,
        reason=reason,
        priced_out=priced_out,
        type_name="",
        unit_id=0,
        x=0.0,
        y=0.0,
        owned=count_extractors(sample, type_name),
        occupied=0,
        exposed=0,
        visible=0,
        unreachable=0,
    )


def placer(sample: Sample, type_name: str, free: Sequence[Entity]) -> Entity | None:
    """Return the entity the engine says can place an extractor.

    Selected by capability rather than by type name, which is the same rule the
    build plan uses and for the same reason: the map editor's placeholder offers
    nearly every type in the game and is parked off-map, so anything choosing a
    producer by what it can make has to go through :func:`find_producer` to have
    the placeholder excluded ([[policy-loop]]).

    Args:
        sample: One observation of the world.
        type_name: The extractor type to place.
        free: Workers not already carrying out an order.

    Returns:
        The producing entity, or None when no free worker can place one right
        now.
    """
    option = find_producer(sample, type_name, free)
    if option is None or not option["available"]:
        return None
    # Indexed rather than searched: a placed option is only returned for a
    # worker in this very list, so a miss would mean the two reads disagreed.
    return {worker["unit_id"]: worker for worker in free}[option["unit_id"]]


def expand_economy(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    *,
    reserve: int,
    free: Sequence[Entity],
    claimed: Sequence[tuple[float, float]],
    refused: Sequence[tuple[float, float]],
    embargoed: Sequence[tuple[float, float]] = (),
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
        profiles: Combat profiles by type name, for the threat filter.
        reserve: Credits to leave unspent for the army.
        free: Workers not already carrying out an order. Whether a worker is
            free is the loop's judgement, because only the loop can see what
            each was last sent to do -- and keeping it there is what stopped two
            expansion rules re-tasking the same worker off each other
            ([[policy-loop]]).
        claimed: Sites workers are already under orders to build on, so several
            free workers are not all sent to the same pool. A pool is judged
            occupied by what *stands* on it, so one being walked toward reads as
            free -- which cost nineteen orders in twenty the moment more than one
            worker was available at a time ([[policy-holding-ground]]).
        refused: Sites the engine already refused silently, from the
            workforce's ledger, which the pool survey must not offer again.
        embargoed: Sites where a razed extractor stood, withheld while the
            rival's wave holds so the walk back is not into the fire that
            razed it. Temporary, unlike a refusal: the caller passes an
            empty sequence once the wave breaks
            ([[impossible-economy-problem]]).
        type_name: The extractor type to place.

    Returns:
        What to do, with the reasoning behind it either way.
    """
    builder = placer(sample, type_name, free)
    if builder is None:
        return waiting(f"no free worker can place {type_name}", sample, type_name)

    stats = catalogue.get(type_name)
    if stats is None:
        return waiting(f"{type_name} is not in the catalogue", sample, type_name)
    needed = stats["price"] + reserve
    if sample["credits"] < needed:
        # Marked, because a cheaper claimant must not take this balance instead.
        # A turret at 500 was winning 29 purchases to the economy's 4 on exactly
        # these observations ([[policy-holding-ground]]).
        return waiting(
            f"{sample['credits']} credits, need {needed} to expand past a {reserve} reserve",
            sample,
            type_name,
            priced_out=True,
        )

    # The anchor is what distance is measured from, so the economy grows out of
    # the base. A player holding no immobile structure measures from the builder
    # instead, which is the build plan's own fallback.
    anchor = find_anchor(sample, catalogue) or builder
    survey = survey_pools(sample, anchor, builder, catalogue, profiles, claimed, refused, embargoed)
    owned = count_extractors(sample, type_name)
    if survey["pool"] is None:
        return Expansion(
            build=False,
            priced_out=False,
            reason=(
                f"no pool free of {survey['visible']}: "
                f"{survey['occupied']} occupied, {survey['unreachable']} unreachable, "
                f"{survey['exposed']} exposed, {survey['embargoed_blocked']} embargoed"
            ),
            type_name="",
            unit_id=0,
            x=0.0,
            y=0.0,
            owned=owned,
            occupied=survey["occupied"],
            exposed=survey["exposed"],
            visible=survey["visible"],
            unreachable=survey["unreachable"],
        )

    pool = survey["pool"]
    return Expansion(
        build=True,
        priced_out=False,
        reason=f"{type_name} #{owned + 1} at ({pool['x']:.0f}, {pool['y']:.0f})",
        type_name=type_name,
        unit_id=builder["unit_id"],
        x=pool["x"],
        y=pool["y"],
        owned=owned,
        occupied=survey["occupied"],
        exposed=survey["exposed"],
        visible=survey["visible"],
        unreachable=survey["unreachable"],
    )


def expand_production(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    *,
    available: int,
    wanted: Sequence[str],
    free: Sequence[Entity],
    refused: Sequence[tuple[float, float]],
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
        available: Credits still unclaimed after the plan and every producer
            have taken what they can this tick. Surplus nobody could spend is
            the definition of throughput being the constraint
            ([[policy-budget]]).
        wanted: Type names the player is trying to make. Nothing able to make
            any of them makes this the build plan's problem rather than the
            economy's.
        free: Workers not already carrying out an order.
        refused: Sites the engine already refused silently, from the
            workforce's ledger, which the ring chooser must not offer again.
        factory_type: The producer to add.

    Returns:
        What to do, with the reasoning behind it either way.
    """
    if not production_bound(sample, catalogue, factory_type, wanted, available):
        return waiting("production is not the constraint", sample, factory_type)

    builder = placer(sample, factory_type, free)
    if builder is None:
        return waiting(f"no free worker can place {factory_type}", sample, factory_type)
    anchor = find_anchor(sample, catalogue) or builder

    choice = next_ring_site(sample, anchor, catalogue, refused)
    site = choice["site"]
    if site is None:
        # The distinction rides in the reason: a taken slot frees itself when
        # its structure falls, a refused one was empty all along and stays
        # refused, and a run log that said "taken" for both hid 64 doomed
        # re-orders behind a plausible wait (wiki log 2026-08-31,
        # verdict-withheld).
        if choice["refused_blocked"] > 0:
            return waiting(
                f"every ring position is taken or refused "
                f"({choice['refused_blocked']} refused silently)",
                sample,
                factory_type,
            )
        return waiting("every ring position is taken", sample, factory_type)
    return Expansion(
        build=True,
        priced_out=False,
        reason=f"{available} credits nobody could spend; adding a {factory_type}",
        type_name=factory_type,
        unit_id=builder["unit_id"],
        x=site[0],
        y=site[1],
        owned=count_extractors(sample),
        occupied=0,
        exposed=0,
        visible=0,
        unreachable=0,
    )


class UpgradeStep(TypedDict):
    """One owned structure and the tier it would convert itself into.

    The target is carried rather than assumed because it differs per structure:
    a tier one names a tier two and a tier two names a tier three, so a single
    wanted type cannot describe a roster holding both.

    Attributes:
        unit_id: Engine identity of the structure offering the conversion.
        produces: The tier it converts into, taken from the engine's own option
            rather than from the chain, so a type the engine declines to offer
            is never ordered.
    """

    unit_id: int
    produces: str


def upgradeable(sample: Sample) -> tuple[UpgradeStep, ...]:
    """Return the owned structures offering to upgrade themselves, and to what.

    **The walk used to stop at tier two, which capped income at 12 a second.**
    This took a single wanted type and matched options against it, so an
    ``extractorT2`` standing next to 62,146 unspent credits was never asked to
    become an ``extractorT3`` -- 20 a second for 4,000, a 500-second payback in
    a match lasting about 1,130. Each structure is now asked for its *own* next
    tier ([[mechanics-unit-value]]).

    **This was invisible for the whole life of the bot, and the cause was ours.**
    An upgrade is not a build: the asset declares it as ``convertTo``, and the
    engine reports the action with no "makes something" flag and no placement
    type. The agent used to drop exactly that shape, so an owned extractor
    published no options at all and the upgrade path looked unreachable --
    while opponents were observed holding twelve upgraded extractors against
    four un-upgraded ones ([[policy-holding-ground]]).

    With every action published, all four standing extractors offer
    ``extractorT2`` and the engine reports it **available**. No tier-2 builder
    is needed and no prerequisite chain: the reading that it was gated behind
    44,500 credits of experimental units was wrong, and only the probe caught
    it.

    It is the best income in the game per unit of risk. An extractor upgrading
    itself needs no builder, crosses no contested ground, and claims no new
    pool -- on a map where the opponents end up holding 44 of the 46
    ([[policy-holding-ground]]). It pays 12 credits a second against tier one's
    8, for 1,400.

    The engine's own offer is still what authorises the order. The chain says
    which tier is *next*; the option stream says whether it may be built now,
    and that is where tech gating and the unit cap already live
    ([[mechanics-build-actions]]). A structure at a fork -- the tier three,
    which offers both an overclock and a reinforce -- has no single next tier
    and is left alone rather than assigned a preference nobody measured.

    Args:
        sample: One observation of the world.

    Returns:
        One entry per owned structure that can convert itself right now, in
        roster order.
    """
    offered = {
        (option["unit_id"], option["produces"])
        for option in sample["options"]
        if option["available"] and not option["placed"]
    }
    steps: list[UpgradeStep] = []
    for entity in sample["entities"]:
        if not entity["mine"] or not entity["complete"] or entity["queued"] != 0:
            continue
        target = next_tier(entity["type_name"])
        if target is None or (entity["unit_id"], target) not in offered:
            continue
        steps.append(UpgradeStep(unit_id=entity["unit_id"], produces=target))
    return tuple(steps)


__all__ = [
    "EXTRACTOR_TYPE",
    "FACTORY_TYPE",
    "Expansion",
    "count_extractors",
    "expand_economy",
    "expand_production",
    "placer",
    "upgradeable",
    "waiting",
]
