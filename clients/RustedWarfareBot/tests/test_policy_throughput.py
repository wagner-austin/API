"""Buying capacity when money outruns queues, as the pure function it is.

Income with nowhere to spend it is a bank balance, not an army. This asks one
question of a world: when every producer is busy and credits are still piling
up, is another factory the thing to buy, and where does it stand?

Distinct from ``test_policy_production``, which fills the queues that already
exist -- this is about growing the number of queues. Split from
``test_policy_economy`` with ``test_policy_defence``; the world all three argue
over is :mod:`tests.economy_fixtures`.
"""

from __future__ import annotations

from rw_bot.policy.economy import EXTRACTOR_TYPE, FACTORY_TYPE, expand_production
from rw_bot.policy.siting import PLACEMENT_RING
from rw_bot.wire.state import Sample
from tests.economy_fixtures import CATALOGUE, free, option, sample, unit


def bound_world(credits_held: int = 4000) -> Sample:
    """A player whose only factory is busy, with money in the bank.

    The state that banked 7,013 credits: income arriving with nowhere to spend
    it, because :func:`sustain` can only fill queues that exist.
    """
    return sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=credits_held,
    )


def test_a_busy_queue_and_a_full_bank_buys_another_factory() -> None:
    growth = expand_production(
        bound_world(), CATALOGUE, available=4000, wanted=("c_tank",), free=free(bound_world())
    )
    assert growth["build"] is True
    assert growth["type_name"] == FACTORY_TYPE
    assert growth["unit_id"] == 214


def test_no_surplus_means_throughput_is_not_the_constraint() -> None:
    """Everything was already claimed, so there is no idle money to convert."""
    world = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=100_000,
    )
    growth = expand_production(world, CATALOGUE, available=0, wanted=("c_tank",), free=free(world))
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_a_surplus_short_of_the_price_buys_nothing() -> None:
    """The reserve reaches this through the surplus rather than through a second
    subtraction here: what arrives is already net of everything protected
    ([[policy-budget]]).
    """
    growth = expand_production(
        bound_world(800),
        CATALOGUE,
        available=300,
        wanted=("c_tank",),
        free=free(bound_world(800)),
    )
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_a_factory_nothing_can_place_is_not_proposed() -> None:
    """No builder offers it, so ordering one would go to nobody."""
    world = sample(
        unit(213, "commandCenter"),
        unit(300, FACTORY_TYPE, queued=1),
        options=(option(300, "c_tank", placed=False),),
        credits_held=100_000,
    )
    growth = expand_production(
        world, CATALOGUE, available=4000, wanted=("c_tank",), free=free(world)
    )
    assert growth["build"] is False
    assert growth["reason"] == f"no free worker can place {FACTORY_TYPE}"


def test_successive_factories_take_successive_ring_slots() -> None:
    """The first *free* position, so they spread rather than stack.

    Read from the world rather than counted. The economy used to index the ring
    by how many immobile structures were standing, which counts extractors --
    sitting on pools, nowhere near the ring -- so every pool claimed shifted the
    factory's site by one position ([[policy-loop]]).
    """
    first = expand_production(
        bound_world(), CATALOGUE, available=4000, wanted=("c_tank",), free=free(bound_world())
    )
    taken = unit(302, FACTORY_TYPE, first["x"], first["y"])
    crowded = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        taken,
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    second = expand_production(
        crowded, CATALOGUE, available=4000, wanted=("c_tank",), free=free(crowded)
    )
    assert (second["x"], second["y"]) != (first["x"], first["y"])


def test_an_extractor_on_a_distant_pool_does_not_move_the_factory() -> None:
    """The defect that made this observation-driven.

    An extractor is immobile and it is nowhere near the ring, so counting it as
    ring occupancy moved every later factory for no reason -- ten extractors
    moved it four positions.
    """
    plain = expand_production(
        bound_world(), CATALOGUE, available=4000, wanted=("c_tank",), free=free(bound_world())
    )
    with_economy = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        unit(401, EXTRACTOR_TYPE, 5000.0, 5000.0),
        unit(402, EXTRACTOR_TYPE, 6000.0, 6000.0),
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    grown = expand_production(
        with_economy, CATALOGUE, available=4000, wanted=("c_tank",), free=free(with_economy)
    )
    assert (grown["x"], grown["y"]) == (plain["x"], plain["y"])


def test_a_mobile_unit_does_not_fill_a_ring_position() -> None:
    """It moves; a building does not. Counting it would hide a usable slot."""
    plain = expand_production(
        bound_world(), CATALOGUE, available=4000, wanted=("c_tank",), free=free(bound_world())
    )
    with_army = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        unit(400, "c_tank", plain["x"], plain["y"]),
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    parked = expand_production(
        with_army, CATALOGUE, available=4000, wanted=("c_tank",), free=free(with_army)
    )
    assert (parked["x"], parked["y"]) == (plain["x"], plain["y"])


def test_an_opponents_factory_is_not_our_production_capacity() -> None:
    """Ours to fill or it is not capacity.

    Ring occupancy and production capacity are different questions about the
    same building: an opponent's factory stands on ground we cannot build on,
    and offers a queue we cannot fill ([[policy-production]]).
    """
    theirs = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(302, FACTORY_TYPE, mine=False),
        options=(option(214, FACTORY_TYPE),),
        credits_held=100_000,
    )
    growth = expand_production(
        theirs, CATALOGUE, available=100_000, wanted=("c_tank",), free=free(theirs)
    )
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_a_full_ring_stops_the_factory_rather_than_stacking_one() -> None:
    """A structure destroyed frees its position, so this is a wait, not a block."""
    filled = [unit(500 + i, FACTORY_TYPE, dx, dy) for i, (dx, dy) in enumerate(PLACEMENT_RING)]
    world = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        *filled,
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    growth = expand_production(
        world, CATALOGUE, available=4000, wanted=("c_tank",), free=free(world)
    )
    assert growth["build"] is False
    assert growth["reason"] == "every ring position is taken"


def test_a_factory_already_going_up_blocks_another() -> None:
    """The guard whose absence cost a whole match.

    A factory joins the roster the moment construction starts, so the ring
    position it stands on reads as taken -- but it is not a *producer* until it
    is finished, so the throughput rule fired again and ordered another at the
    next free position. Eight orders walked through all eight ring slots, each
    re-tasking the single builder off the last, and not one was ever completed
    ([[policy-production]]).
    """
    world = sample(
        unit(213, "commandCenter"),
        unit(214, "builder"),
        unit(300, FACTORY_TYPE, queued=1),
        unit(301, FACTORY_TYPE, complete=False),
        options=(option(214, FACTORY_TYPE), option(300, "c_tank", placed=False)),
        credits_held=100_000,
    )
    growth = expand_production(world, CATALOGUE, available=100_000, wanted=("c_tank",), free=())
    assert growth["build"] is False
    assert growth["reason"] == f"no free worker can place {FACTORY_TYPE}"


def test_a_walking_builder_is_left_alone_by_the_factory_rule_too() -> None:
    """One builder, and an order given to it replaces whatever it was doing."""
    growth = expand_production(
        bound_world(), CATALOGUE, available=100_000, wanted=("c_tank",), free=()
    )
    assert growth["build"] is False
    assert growth["reason"] == f"no free worker can place {FACTORY_TYPE}"
