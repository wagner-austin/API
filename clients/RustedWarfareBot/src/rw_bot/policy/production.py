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

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.wire.state import Sample


class ProductionOrder(TypedDict):
    """One producer told to make one thing.

    Attributes:
        unit_id: Engine identity of the producing building.
        type_name: What to make.
        reason: Why, for the run log.
    """

    unit_id: int
    type_name: str
    reason: str


def idle_producers(sample: Sample) -> tuple[int, ...]:
    """Return the owned buildings holding nothing in their queue.

    A building with something queued is already working, and queueing more
    would spend credits now for a unit that starts later anyway. The queue
    depth is the engine's own answer, carried per entity, and it is the same
    signal the stall detector uses to tell a working factory from a refused
    order ([[policy-loop]]).

    Args:
        sample: One observation of the world.

    Returns:
        Engine identities of idle owned entities, in roster order.
    """
    return tuple(
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["queued"] == 0
    )


def sustain(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    wanted: Sequence[str],
) -> tuple[ProductionOrder, ...]:
    """Decide what every idle producer should start making.

    Only options the engine reports as **available** are used, which is what
    keeps this honest about two limits it would otherwise have to model itself.
    Availability already accounts for the player's unit cap and for tech gating,
    because the agent asks the engine's own predicate rather than counting units
    here ([[mechanics-build-actions]]).

    Only produced units are ordered, never placed structures. A structure needs
    a site, and choosing one is the build policy's job, not this one's.

    Credits are budgeted across the whole batch rather than checked per order.
    Two factories that can each afford a tank cannot always afford two, and
    issuing both would leave the second refused for a reason the run log could
    not explain.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for prices.
        wanted: Type names to keep making, in preference order.

    Returns:
        One order per idle producer that can start something affordable.
    """
    if not wanted:
        return ()

    # Indexed by producer so the preference order below is the caller's, not
    # the order the agent happened to enumerate options in.
    offers: dict[int, set[str]] = {}
    for option in sample["options"]:
        if option["placed"] or not option["available"]:
            continue
        offers.setdefault(option["unit_id"], set()).add(option["produces"])

    budget = sample["credits"]
    orders: list[ProductionOrder] = []

    for producer in idle_producers(sample):
        makeable = offers.get(producer)
        if makeable is None:
            continue
        for type_name in wanted:
            if type_name not in makeable:
                continue
            stats = catalogue.get(type_name)
            if stats is None or stats["price"] > budget:
                continue
            budget -= stats["price"]
            orders.append(
                ProductionOrder(
                    unit_id=producer,
                    type_name=type_name,
                    reason=f"{type_name} at {producer}, {budget} left",
                )
            )
            break
    return tuple(orders)


def production_bound(sample: Sample, catalogue: Mapping[str, UnitStats], factory_type: str) -> bool:
    """Report whether the player has money it has nowhere to spend.

    The bot banked 7,013 credits over a 1,500-sample run while producing 46
    units. That was not a spending rule that failed to fire: :func:`sustain`
    already budgets across every idle producer. It was arithmetic. Ten
    extractors of income and one factory to spend it through means credits pile
    up no matter how willing the spender is.

    So the question is not "should I buy something" but "can I buy *faster*",
    and it is answered from observed state rather than a threshold. Every
    producer busy means the queue is the constraint; affording another factory
    means the bank is not. Both together mean building one converts idle money
    into throughput, and neither alone does.

    **A producer is something the engine says can make a unit**, not merely an
    owned building with an empty queue. :func:`idle_producers` answers the
    looser question and is right to -- its caller cross-references the option
    list anyway -- but reusing it here read the Command Center and the builder
    as spare capacity, so this never fired at all. The option list is the
    engine's own answer to who can produce ([[mechanics-build-actions]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the factory's price.
        factory_type: Type name of the producer to add.

    Returns:
        True when every producer is busy and another factory is affordable.
    """
    stats = catalogue.get(factory_type)
    if stats is None:
        return False
    producers = {option["unit_id"] for option in sample["options"] if not option["placed"]}
    if not producers:
        # Nothing can make a unit at all. That is a build-order problem, and
        # adding a factory is the build plan's job rather than a reaction to a
        # queue that does not exist yet.
        return False
    busy = {
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and entity["queued"] > 0
    }
    if producers - busy:
        # Spare capacity already. Another factory would idle beside the ones
        # that are idling, which spends the bank without raising throughput.
        return False
    return sample["credits"] >= stats["price"]


__all__ = ["ProductionOrder", "idle_producers", "sustain"]
