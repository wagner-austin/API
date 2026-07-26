"""Keeping producers busy, exercised as the pure function it is.

A sample goes in and orders come out. What sends them is
``rw_bot.policy.campaign``; none of that is needed here.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.production import idle_producers, production_bound, sustain
from rw_bot.wire.state import BuildOption, Entity, Sample


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
}


def _entity(unit_id: int, type_name: str, *, queued: int = 0, complete: bool = True) -> Entity:
    return Entity(
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
) -> BuildOption:
    return BuildOption(
        index=0,
        unit_id=unit_id,
        produces=produces,
        action=1,
        placed=placed,
        available=available,
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
        options=options,
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


def test_something_unaffordable_is_not_ordered() -> None:
    world = _sample((_FACTORY,), (_option(300, "scout"),), credits=699)
    assert sustain(world, _CATALOGUE, ("scout",)) == ()


def test_credits_are_budgeted_across_the_whole_batch() -> None:
    """Two factories that can each afford one cannot always afford two.

    Issuing both would leave the second refused for a reason the run log could
    not explain.
    """
    second = _entity(301, "landFactory")
    world = _sample(
        (_FACTORY, second),
        (_option(300, "c_tank"), _option(301, "c_tank")),
        credits=500,
    )
    orders = sustain(world, _CATALOGUE, ("c_tank",))
    assert [o["unit_id"] for o in orders] == [300]


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


def test_an_idle_unit_that_offers_nothing_is_skipped() -> None:
    """Every owned unit is idle by the queue test; most can make nothing."""
    builder = _entity(214, "builder")
    world = _sample((builder, _FACTORY), (_option(300, "c_tank"),))
    assert idle_producers(world) == (214, 300)
    assert [o["unit_id"] for o in sustain(world, _CATALOGUE, ("c_tank",))] == [300]


#: The engine saying unit 300 can make a tank, which is what makes it a producer.
_MAKES_TANKS = (_option(300, "c_tank", placed=False),)


def test_spare_capacity_means_another_factory_would_only_idle() -> None:
    """An idle producer is throughput nobody is using; adding one adds nothing."""
    world = _sample((_entity(300, "landFactory"),), _MAKES_TANKS, credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory") is False


def test_every_producer_busy_and_money_spare_is_the_bound_case() -> None:
    """The state that banked 7,013 credits: the queue is the constraint."""
    world = _sample((_entity(300, "landFactory", queued=1),), _MAKES_TANKS, credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory") is True


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
    assert production_bound(world, _CATALOGUE, "landFactory") is True


def test_nothing_that_can_produce_at_all_is_a_build_order_problem() -> None:
    world = _sample((_entity(213, "commandCenter"),), (), credits=100_000)
    assert production_bound(world, _CATALOGUE, "landFactory") is False


def test_busy_but_broke_is_not_worth_a_factory() -> None:
    world = _sample((_entity(300, "landFactory", queued=1),), _MAKES_TANKS, credits=0)
    assert production_bound(world, _CATALOGUE, "landFactory") is False


def test_a_factory_the_catalogue_cannot_price_is_not_proposed() -> None:
    world = _sample((_entity(300, "landFactory", queued=1),), _MAKES_TANKS, credits=100_000)
    assert production_bound(world, _CATALOGUE, "someModFactory") is False
