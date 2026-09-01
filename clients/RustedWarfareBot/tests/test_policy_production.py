"""Keeping producers busy, exercised as the pure function it is.

A sample goes in and orders come out. What sends them is
``rw_bot.policy.campaign``; none of that is needed here.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.production import (
    idle_producers,
    production_bound,
    sustain,
    wanted_producers,
)
from rw_bot.wire.state import BuildOption, Entity, Sample
from tests.wire_fixtures import entity


def _unit(type_name: str, price: int) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=1.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    "c_tank": _unit("c_tank", 350),
    "scout": _unit("scout", 700),
    "landFactory": _unit("landFactory", 700),
    "builder": _unit("builder", 500),
}


def _entity(unit_id: int, type_name: str, *, queued: int = 0, complete: bool = True) -> Entity:
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=0.0,
        y=0.0,
        team=0,
        mine=True,
        hostile=False,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=queued,
    )


def _option(
    unit_id: int,
    produces: str,
    *,
    placed: bool = False,
    available: bool = True,
    makes_something: bool = True,
) -> BuildOption:
    return BuildOption(
        index=0,
        unit_id=unit_id,
        produces=produces,
        key="u_x",
        placed=placed,
        available=available,
        makes_something=makes_something,
        price=0,
    )


def _sample(
    entities: tuple[Entity, ...],
    options: tuple[BuildOption, ...],
    credits: int = 4000,
) -> Sample:
    return Sample(
        frame=1,
        clock_ms=10,
        credits=credits,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=entities,
        pools=(),
        players=(),
        options=options,
        refusals=(),
    )


_FACTORY = _entity(300, "landFactory")


def test_an_idle_factory_is_given_something_to_make() -> None:
    world = _sample((_FACTORY,), (_option(300, "c_tank"),))
    orders = sustain(world, _CATALOGUE, ("c_tank",))
    assert [(o["unit_id"], o["type_name"]) for o in orders] == [(300, "c_tank")]


def test_a_busy_factory_is_left_alone() -> None:
    """Queueing more spends credits now for a unit that starts later anyway."""
    busy = _entity(300, "landFactory", queued=1)
    world = _sample((busy,), (_option(300, "c_tank"),))
    assert sustain(world, _CATALOGUE, ("c_tank",)) == ()
    assert idle_producers(world) == ()


def test_an_unfinished_factory_is_not_a_producer() -> None:
    shell = _entity(300, "landFactory", complete=False)
    world = _sample((shell,), (_option(300, "c_tank"),))
    assert sustain(world, _CATALOGUE, ("c_tank",)) == ()


def test_an_unavailable_action_is_skipped() -> None:
    """Availability is where the unit cap and tech gating already live.

    The agent asks the engine's own predicate, so this does not have to count
    units or model tech to respect either.
    """
    world = _sample((_FACTORY,), (_option(300, "c_tank", available=False),))
    assert sustain(world, _CATALOGUE, ("c_tank",)) == ()


def test_a_structure_is_never_produced_this_way() -> None:
    """A structure needs a site, and choosing one is the build policy's job."""
    world = _sample((_FACTORY,), (_option(300, "landFactory", placed=True),))
    assert sustain(world, _CATALOGUE, ("landFactory",)) == ()


def test_affordability_is_not_decided_here() -> None:
    """What a producer *could* start, not what the player can pay for.

    This used to budget against ``sample["credits"]``, which was right on its
    own and wrong in company: the expansion pass budgeted against the same field
    in the same observation and the pair committed one credit twice. Spending
    has one owner now, and it is :class:`~rw_bot.policy.budget.Budget`
    ([[policy-budget]]).
    """
    world = _sample((_FACTORY,), (_option(300, "scout"),), credits=0)
    orders = sustain(world, _CATALOGUE, ("scout",))
    assert [o["unit_id"] for o in orders] == [300]
    assert orders[0]["price"] == 700


def test_every_idle_producer_is_offered_work() -> None:
    """Two factories yield two orders; which of them the player can pay for is
    the budget's question, and the preference order is what makes dropping the
    tail meaningful rather than arbitrary.
    """
    second = _entity(301, "landFactory")
    world = _sample(
        (_FACTORY, second),
        (_option(300, "c_tank"), _option(301, "c_tank")),
        credits=500,
    )
    orders = sustain(world, _CATALOGUE, ("c_tank",))
    assert [o["unit_id"] for o in orders] == [300, 301]


def test_both_factories_run_when_both_are_afforded() -> None:
    second = _entity(301, "landFactory")
    world = _sample(
        (_FACTORY, second),
        (_option(300, "c_tank"), _option(301, "c_tank")),
        credits=700,
    )
    assert [o["unit_id"] for o in sustain(world, _CATALOGUE, ("c_tank",))] == [300, 301]


def test_the_caller_preference_order_decides_not_the_option_order() -> None:
    """The agent enumerates options in its own order, which is not a preference."""
    world = _sample((_FACTORY,), (_option(300, "scout"), _option(300, "c_tank")))
    assert sustain(world, _CATALOGUE, ("c_tank", "scout"))[0]["type_name"] == "c_tank"
    assert sustain(world, _CATALOGUE, ("scout", "c_tank"))[0]["type_name"] == "scout"


def test_a_producer_gets_one_order_per_sample() -> None:
    """It has one queue; a second order would be spending ahead of the first."""
    world = _sample((_FACTORY,), (_option(300, "c_tank"), _option(300, "scout")))
    assert len(sustain(world, _CATALOGUE, ("c_tank", "scout"))) == 1


def test_a_wanted_type_this_producer_cannot_make_is_passed_over() -> None:
    """The preference list is global; what each producer offers is not."""
    world = _sample((_FACTORY,), (_option(300, "c_tank"),))
    assert sustain(world, _CATALOGUE, ("scout", "c_tank"))[0]["type_name"] == "c_tank"


def test_wanting_nothing_orders_nothing() -> None:
    world = _sample((_FACTORY,), (_option(300, "c_tank"),))
    assert sustain(world, _CATALOGUE, ()) == ()


def test_a_type_the_catalogue_cannot_price_is_skipped() -> None:
    """Unpriced means unbudgetable, and spending blind is what the budget prevents."""
    world = _sample((_FACTORY,), (_option(300, "mysteryTank"),))
    assert sustain(world, _CATALOGUE, ("mysteryTank",)) == ()


def test_a_unit_that_makes_nothing_is_not_an_idle_producer() -> None:
    """Something that produces nothing is not idle capacity; it is a wall.

    Counting every owned unit made "is every producer busy?" answer no for as
    long as the player owned a Command Center or a Builder — which is always —
    so the throughput rule could never fire ([[policy-production]]).
    """
    builder = _entity(214, "builder")
    world = _sample((builder, _FACTORY), (_option(300, "c_tank"),))
    assert idle_producers(world) == (300,)
    assert [o["unit_id"] for o in sustain(world, _CATALOGUE, ("c_tank",))] == [300]


def test_a_placed_action_does_not_make_its_owner_a_producer() -> None:
    """A Builder placing structures is not spare production capacity."""
    builder = _entity(214, "builder")
    world = _sample((builder,), (_option(214, "extractorT1", placed=True),))
    assert idle_producers(world) == ()


def _line(
    count: int, produces: tuple[str, ...]
) -> tuple[tuple[Entity, ...], tuple[BuildOption, ...]]:
    """Return ``count`` idle factories, each offering every named type."""
    factories = tuple(_entity(300 + n, "landFactory") for n in range(count))
    options = tuple(_option(300 + n, produces=name) for n in range(count) for name in produces)
    return factories, options


def test_repeats_in_the_composition_are_a_ratio_not_a_priority() -> None:
    """Two tanks per scout means two tanks per scout, not tanks forever.

    The old rule took the first wanted type a producer could make and stopped,
    so whatever stood at the head of the list was the only thing ever built.
    Three 1500-sample matches ended with 33 identical ``c_tank`` and no answer
    to aircraft at all ([[policy-production]]).

    What is asserted is the batch's composition, not its sequence. Filling the
    widest gap each time interleaves the types -- tank, scout, tank -- rather
    than emitting the ratio in blocks, and that is the better of the two: the
    scout exists a build sooner without the mix ever being wrong.
    """
    factories, options = _line(3, ("c_tank", "scout"))
    world = _sample(factories, options)
    orders = sustain(world, _CATALOGUE, ("c_tank", "c_tank", "scout"))
    assert sorted(o["type_name"] for o in orders) == ["c_tank", "c_tank", "scout"]


def test_one_tick_does_not_fill_the_same_gap_three_times() -> None:
    """Orders decided this tick count, or a batch rediscovers the old bug.

    Every idle producer reads the same roster in the same observation. Without
    counting what has already been decided, all of them see the identical
    shortfall and all of them fill it with the identical unit.
    """
    factories, options = _line(3, ("c_tank", "scout"))
    world = _sample(factories, options)
    orders = sustain(world, _CATALOGUE, ("c_tank", "scout"))
    assert sorted(o["type_name"] for o in orders) == ["c_tank", "c_tank", "scout"]


def test_a_type_already_over_its_share_yields_to_the_one_behind() -> None:
    """The roster is read from the world, not assumed empty."""
    owned = tuple(_entity(400 + n, "c_tank") for n in range(3))
    factories, options = _line(1, ("c_tank", "scout"))
    world = _sample((*owned, *factories), options)
    orders = sustain(world, _CATALOGUE, ("c_tank", "scout"))
    assert [o["type_name"] for o in orders] == ["scout"]


def test_a_roster_at_its_target_ratio_keeps_the_ratio() -> None:
    """Two owned tanks to one scout against a 2:1 plan is not a shortfall."""
    owned = (_entity(400, "c_tank"), _entity(401, "c_tank"), _entity(402, "scout"))
    factories, options = _line(3, ("c_tank", "scout"))
    world = _sample((*owned, *factories), options)
    orders = sustain(world, _CATALOGUE, ("c_tank", "c_tank", "scout"))
    assert sorted(o["type_name"] for o in orders) == ["c_tank", "c_tank", "scout"]


def test_an_incomplete_unit_is_not_counted_toward_the_roster() -> None:
    """Half-built is not fielded, and counting it would under-order its type."""
    building = _entity(400, "scout", complete=False)
    factories, options = _line(1, ("c_tank", "scout"))
    world = _sample((building, *factories), options)
    orders = sustain(world, _CATALOGUE, ("c_tank", "scout"))
    assert [o["type_name"] for o in orders] == ["c_tank"]


def test_a_builder_in_the_composition_is_made_by_whichever_producer_can() -> None:
    """The change that unlocked a second worker at all.

    A builder used to be offered through a separate channel reached only by a
    producer that could make *nothing* in the composition -- the Command Center
    and nothing else on this build, because a Land Factory can always make a
    tank and so never fell through. Inside the composition both producers are
    eligible, which is what lets the worker count actually grow
    ([[policy-production]]).
    """
    centre = _entity(1, "commandCenter")
    factories, options = _line(1, ("c_tank",))
    world = _sample(
        (centre, *factories),
        (*options, _option(1, "builder")),
    )
    orders = sustain(world, _CATALOGUE, ("builder", "c_tank"))
    assert [(o["unit_id"], o["type_name"]) for o in orders] == [(1, "builder"), (300, "c_tank")]


def test_a_land_factory_will_build_a_builder_when_one_is_wanted() -> None:
    """The case the old fallback could never reach.

    With the Command Center dead, every remaining Land Factory can build a
    builder and can also build a tank. Under the fallback it always chose the
    tank, so a player that lost its last worker could never make another and
    the run ended ``plan blocked`` with ``workers 0`` ([[policy-production]]).
    """
    factories, options = _line(1, ("c_tank", "builder"))
    world = _sample(factories, options)
    assert [o["type_name"] for o in sustain(world, _CATALOGUE, ("builder",))] == ["builder"]


def test_a_builder_outside_the_composition_is_not_made() -> None:
    """Wanting one is what buys one; the ceiling is expressed by leaving it out.

    :func:`~rw_bot.policy.spending.worker_need` drops the builder from the
    composition once the ceiling is met, and this is what that does downstream:
    a producer offering both builds the army instead.
    """
    factories, options = _line(1, ("c_tank", "builder"))
    world = _sample(factories, options)
    assert [o["type_name"] for o in sustain(world, _CATALOGUE, ("c_tank",))] == ["c_tank"]


def test_a_producer_that_can_make_nothing_wanted_orders_nothing() -> None:
    """It idles rather than being given something nobody asked for."""
    factories, options = _line(1, ("scout",))
    world = _sample(factories, options)
    assert sustain(world, _CATALOGUE, ("c_tank",)) == ()


#: The engine saying unit 300 can make a tank, which is what makes it a producer.
_MAKES_TANKS = (_option(300, "c_tank", placed=False),)


def test_spare_capacity_blocks_it_however_large_the_surplus() -> None:
    """Both tests are required, and this is the one that keeps it conservative.

    Buying capacity beside capacity that is already idle spends the builder's
    time as well as the credits, and the builder is the only thing that places
    extractors. Measured: the more factories the economy bought, the less income
    the bot had and the smaller its army ([[policy-production]]).
    """
    idle = _sample((_entity(300, "landFactory"),), _MAKES_TANKS, credits=100_000)
    assert production_bound(idle, _CATALOGUE, "landFactory", ("c_tank",), 100_000) is False


def test_a_surplus_nobody_could_spend_is_the_bound_case() -> None:
    """The state that banked 7,013 credits, and then 18,576.

    The surplus is what the plan and every producer left behind after claiming
    everything they could this tick. Credits nobody could spend are the
    definition of throughput being the constraint ([[policy-budget]]).
    """
    world = _sample((_entity(300, "landFactory", queued=1),), _MAKES_TANKS, credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory", ("c_tank",), 700) is True


def test_an_idle_building_that_makes_nothing_is_not_spare_capacity() -> None:
    """The bug this rule shipped with.

    A Command Center and a builder both sit with empty queues, and counting
    them as producers meant the constraint never read as reached and the rule
    never fired. Only what the engine lists as able to make a unit counts.
    """
    world = _sample(
        (_entity(213, "commandCenter"), _entity(300, "landFactory", queued=1)),
        _MAKES_TANKS,
        credits=100_000,
    )
    assert production_bound(world, _CATALOGUE, "landFactory", ("c_tank",), 4000) is True


def test_nothing_that_can_produce_at_all_is_a_build_order_problem() -> None:
    world = _sample((_entity(213, "commandCenter"),), (), credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory", ("c_tank",), 4000) is False


def test_no_surplus_is_not_worth_a_factory() -> None:
    """Everything was spent, so there is nothing idle money could be converted from."""
    world = _sample((_entity(300, "landFactory", queued=1),), _MAKES_TANKS, credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory", ("c_tank",), 699) is False


def test_nothing_that_makes_a_wanted_type_is_the_plans_problem() -> None:
    """Adding a factory is the build plan's job, not a reaction to an absent queue."""
    world = _sample((_entity(300, "landFactory", queued=1),), (), credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory", ("c_tank",), 4000) is False


def test_a_factory_the_catalogue_cannot_price_is_not_proposed() -> None:
    world = _sample((_entity(300, "landFactory", queued=1),), _MAKES_TANKS, credits=100_000)
    assert production_bound(world, _CATALOGUE, "someModFactory", ("c_tank",), 4000) is False


def test_capability_is_the_engines_answer_not_a_type_name_guess() -> None:
    """What makes a producer is the option stream, which already accounts for
    the unit cap and tech gating -- so a type that becomes unavailable simply
    drops out, and the trace shows that as capability rather than as reluctance
    ([[mechanics-build-actions]]).
    """
    world = _sample((_FACTORY,), (_option(300, "c_tank"),))
    assert wanted_producers(world, ("c_tank",)) == (300,)
    assert wanted_producers(world, ("scout",)) == ()

    capped = _sample((_FACTORY,), (_option(300, "c_tank", available=False),))
    assert wanted_producers(capped, ("c_tank",)) == ()


def test_a_producer_that_is_still_a_shell_is_not_capacity() -> None:
    shell = _entity(300, "landFactory", complete=False)
    world = _sample((shell,), (_option(300, "c_tank"),))
    assert wanted_producers(world, ("c_tank",)) == ()


def test_a_placed_action_is_not_production_capacity() -> None:
    """A Builder placing structures does not fill a production queue."""
    builder = _entity(214, "builder")
    world = _sample((builder,), (_option(214, "extractorT1", placed=True),))
    assert wanted_producers(world, ("extractorT1",)) == ()
