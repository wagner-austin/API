"""Keeping the factories busy for as long as the match lasts.

The planner executes a list and the list ends. That is the difference between
this bot and the opponents beating it: the engine's own AI never finishes,
because it samples from a weighted mix rather than working through an order, so
it never has to decide what comes after the last entry
([[ai-opponent-strategy]]). Measured against that, our bot stopped at four tanks
and banked 21,164 credits while the enemy went from 54 to 126 visible units.

This closes that gap without inventing a value model. What to keep making is the
same thing the goals already asked for -- the plan says a tank is wanted, and
production simply does not stop wanting one. Ranking units by some invented
notion of combat worth would be a guess with a number attached; repeating a
stated goal is not.

Pure, like the rest of the policy layer: a sample goes in and orders come out,
and :mod:`rw_bot.policy.campaign` is what sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.wire.state import Sample


class ProductionOrder(TypedDict):
    """One producer told to make one thing.

    Attributes:
        unit_id: Engine identity of the producing building.
        type_name: What to make.
        price: What it costs, carried so the caller can claim it against the
            tick's budget without a second catalogue lookup.
        reason: Why, for the run log.
    """

    unit_id: int
    type_name: str
    price: int
    reason: str


def idle_producers(sample: Sample) -> tuple[int, ...]:
    """Return the owned producers holding nothing in their queue.

    A building with something queued is already working, and queueing more
    would spend credits now for a unit that starts later anyway. The queue
    depth is the engine's own answer, carried per entity, and it is the same
    signal the stall detector uses to tell a working factory from a refused
    order ([[policy-loop]]).

    **Producer, not merely entity.** Something that makes nothing is not idle;
    it is a wall. Counting every owned unit made "is every producer busy?"
    answer no for as long as the player owned a Command Center or a Builder,
    which is always -- so :func:`production_bound` could not fire and the bot
    banked credits behind a single factory anyway ([[policy-production]]).
    Whether a thing produces is the engine's own answer, carried on the option
    stream: a unit offering a non-placed action it may use right now.

    Args:
        sample: One observation of the world.

    Returns:
        Engine identities of idle owned producers, in roster order.
    """
    producers = {
        option["unit_id"]
        for option in sample["options"]
        if not option["placed"] and option["available"] and option["produces"]
    }
    return tuple(
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"]
        and entity["complete"]
        and entity["queued"] == 0
        and entity["unit_id"] in producers
    )


def _shares(composition: Sequence[str]) -> dict[str, int]:
    """Count how many times each type appears in a composition.

    Multiplicity is the whole notation. ``("c_tank", "c_tank", "hoverTank")``
    is not three preferences, and it is not one preference stated twice: it is
    a ratio, two of the first for every one of the second.

    Args:
        composition: The wanted army mix, with repeats meaningful.

    Returns:
        Wanted count by type name.
    """
    counted: dict[str, int] = {}
    for type_name in composition:
        counted[type_name] = counted.get(type_name, 0) + 1
    return counted


def _furthest_behind(
    target: Mapping[str, int],
    owned: Mapping[str, int],
    order: Sequence[str],
    makeable: Set[str],
) -> str | None:
    """Return the makeable type whose share of the army is furthest below plan.

    The unit of comparison is a *share*, not a count, which is what lets one
    rule cover an army of three and an army of three hundred. A composition
    asking for two tanks per anti-air unit wants tanks at 0.67 of the roster
    forever, not two tanks total.

    Ties break on first appearance in the composition, so the choice is a
    function of the caller's stated order rather than of dictionary iteration.
    An empty roster makes every owned share zero, so the opening pick is simply
    the largest requested share -- which is the right opening.

    Args:
        target: Wanted count by type name, from :func:`_shares`.
        owned: How many of each wanted type the player has, counting orders
            already decided this tick.
        order: Wanted type names in first-appearance order, for the tie-break.
        makeable: What this producer can start right now.

    Returns:
        The type to make, or None when this producer can make nothing in the
        composition.
    """
    wanted_total = sum(target.values())
    if wanted_total == 0:
        return None
    held_total = sum(owned.get(name, 0) for name in target)
    choice: str | None = None
    widest = 0.0
    for type_name in order:
        if type_name not in makeable:
            continue
        held = owned.get(type_name, 0) / held_total if held_total else 0.0
        gap = target[type_name] / wanted_total - held
        if choice is None or gap > widest:
            choice, widest = type_name, gap
    return choice


def sustain(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    composition: Sequence[str],
) -> tuple[ProductionOrder, ...]:
    """Decide what every idle producer should start making.

    Only options the engine reports as **available** are used, which is what
    keeps this honest about two limits it would otherwise have to model itself.
    Availability already accounts for the player's unit cap and for tech gating,
    because the agent asks the engine's own predicate rather than counting units
    here ([[mechanics-build-actions]]).

    Only produced units are ordered, never placed structures. A structure needs
    a site, and choosing one is the build policy's job, not this one's.

    **A composition, not a priority list.** This used to take the first wanted
    type a producer could make and stop, which made a mixed army structurally
    impossible: every idle producer reached the same first entry, so whatever
    stood at the head of the list was the only thing the bot ever built. Three
    1500-sample matches produced armies of 33 identical ``c_tank`` -- a unit
    that cannot shoot at aircraft at all, against opponents fielding 15 visible
    units it could not touch, and whose 130 reach is shorter than every turret
    it walked into ([[mechanics-combat-profile]]). Repeats in the composition
    are therefore read as a ratio and each producer builds whatever the roster
    is furthest short of.

    Orders decided earlier in this same tick count toward the roster. Without
    that, a batch of idle factories would each see the identical shortfall and
    all fill it with the identical unit, which is the old behaviour rediscovered
    one tick at a time.

    **Affordability is deliberately not decided here.** This used to budget
    across the batch against ``sample["credits"]``, which was correct on its own
    and wrong in company: the expansion pass budgeted against the same field in
    the same observation, so the pair committed one credit twice
    ([[policy-budget]]). What a producer *could* start is this module's
    question; what the player can afford this tick has exactly one owner, and it
    is :class:`~rw_bot.policy.budget.Budget`.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for reading which wanted types are
            described at all.
        composition: The army mix to hold, repeats meaningful as a ratio.

    Returns:
        One order per idle producer that has something to make, in roster
        order.
    """
    # Counting is also the de-duplication: a dict keeps insertion order, so the
    # keys are already the wanted types in first-appearance order.
    target = _shares(composition)
    order = tuple(target)
    owned: dict[str, int] = dict.fromkeys(target, 0)
    for entity in sample["entities"]:
        if entity["mine"] and entity["complete"] and entity["type_name"] in owned:
            owned[entity["type_name"]] += 1

    # Indexed by producer so the choice below is made against one producer's
    # whole offer, not against the order the agent enumerated options in.
    offers: dict[int, set[str]] = {}
    for option in sample["options"]:
        if option["placed"] or not option["available"] or not option["produces"]:
            continue
        offers.setdefault(option["unit_id"], set()).add(option["produces"])

    orders: list[ProductionOrder] = []

    for producer in idle_producers(sample):
        # Indexing rather than a guarded lookup: an idle producer is *defined*
        # by offering one of these options, so a miss here would mean the two
        # reads disagreed about the same option stream.
        #
        # A type the engine offers and the catalogue does not price cannot be
        # claimed against a budget, so it is dropped here rather than chosen
        # and then skipped. The catalogue covers every priced type, so that is
        # a stale dump rather than an exotic unit.
        makeable = {name for name in offers[producer] if name in catalogue}
        type_name = _furthest_behind(target, owned, order, makeable)
        if type_name is None:
            # Nothing wanted that this producer can make. It idles, which is
            # the honest answer: a Command Center past the worker ceiling has
            # no useful contribution, and the "make something anyway" channel
            # that used to sit here was reachable by that structure alone
            # ([[policy-production]]).
            continue
        stats = catalogue[type_name]
        orders.append(
            ProductionOrder(
                unit_id=producer,
                type_name=type_name,
                price=stats["price"],
                reason=f"{type_name} at {producer} for {stats['price']}",
            )
        )
        # Unconditional because :func:`_furthest_behind` only ever names a type
        # from the composition, and ``owned`` is keyed by exactly those. The
        # membership test that used to guard this existed for the fallback,
        # which could name a type outside the ratio.
        owned[type_name] += 1
    return tuple(orders)


def wanted_producers(sample: Sample, wanted: Sequence[str]) -> tuple[int, ...]:
    """Return the owned units the engine says can make something wanted.

    Capability only -- whether each is busy is a separate question, asked by the
    caller against the queue depth. Split out because three readers need the
    same set and one of them is the run trace, which exists precisely to answer
    "was there capacity" without re-deriving it ([[policy-production]]).

    Availability is the engine's own predicate, so the unit cap and tech gating
    are already accounted for rather than modelled here
    ([[mechanics-build-actions]]). A type that becomes unavailable therefore
    drops out of this set, which is what makes a cap visible in the trace.

    Args:
        sample: One observation of the world.
        wanted: Type names the player is trying to make.

    Returns:
        Engine identities that can make at least one wanted type, in roster
        order.
    """
    offering = {
        option["unit_id"]
        for option in sample["options"]
        if not option["placed"]
        and option["available"]
        and option["produces"]
        and option["produces"] in wanted
    }
    return tuple(
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["unit_id"] in offering
    )


def production_bound(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    factory_type: str,
    wanted: Sequence[str],
    available: int,
) -> bool:
    """Report whether the player has money it has nowhere to spend.

    The bot banked 7,013 credits over a 1,500-sample run while producing 46
    units, and 18,576 over a later one while producing 29. That was not a
    spending rule failing to fire: every idle producer was already offered work
    every tick. It was arithmetic. Income arriving faster than the queues can
    absorb it piles up no matter how willing the spender is.

    **The question is asked of the budget, not of the queues.** Two earlier
    shapes of this rule both failed, and both failed by asking about queue state
    at an instant:

    * "every producer busy" counted the Command Center, which offers a Builder
      and is therefore idle almost permanently -- so the answer was *no* on every
      observation of every match and the rule never fired at all;
    * restricting that to producers of a wanted type fired exactly once per
      match. A factory is busy for the whole of a build and idle for the single
      tick in which it finishes -- which is precisely the tick this is
      evaluated on, because that is when production had capacity to fill.

    So the surplus test was added: after the plan and every producer have
    claimed all they can this tick, is there still enough left for a factory?
    **On its own that was worse than either.** Measured over three 1500-sample
    matches on one seed, the more factories the economy built, the worse the bot
    did: 0 factories gave 10 extractors, 98 credits/s and an army worth 6,450;
    1 gave 9, 90/s and 4,000; 7 gave 7, 74/s and 3,300. The bank drained and the
    army shrank with it.

    The mechanism is not run-to-run noise, and income is the low-variance figure
    that shows it: **there is one builder, and every factory it places is an
    extractor it does not.** Income gates production, so buying capacity with
    the money that would have bought income trades the thing that was working
    for the thing that was idle.

    Both tests are therefore required, and the conjunction is deliberately
    conservative -- it fires only when the queues are genuinely full *and* money
    is genuinely spare. Whether the surplus has a better use than a factory is
    an open question, and it is not answered by spending it faster
    ([[policy-production]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the factory's price.
        factory_type: Type name of the producer to add.
        wanted: Type names the player is trying to make. Nothing able to make
            any of them means this is a build-order problem, and adding a
            factory is the plan's job rather than a reaction to a queue that
            does not exist yet.
        available: Credits still unclaimed after every higher-priority spender
            has taken what it can this tick.

    Returns:
        True when every producer of a wanted type is busy *and* the surplus
        covers another factory.
    """
    stats = catalogue.get(factory_type)
    if stats is None:
        return False
    producers = set(wanted_producers(sample, wanted))
    if not producers:
        return False
    busy = {
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and entity["queued"] > 0
    }
    if producers - busy:
        # Spare capacity already. Another factory would idle beside the ones
        # that are idling, and the builder that placed it would not have placed
        # an extractor.
        return False
    return available >= stats["price"]


__all__ = [
    "ProductionOrder",
    "idle_producers",
    "production_bound",
    "sustain",
    "wanted_producers",
]
