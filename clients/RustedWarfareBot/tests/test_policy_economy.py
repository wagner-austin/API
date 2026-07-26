"""Claiming resource pools, exercised as the pure function it is.

No socket and no game: a world state goes in and either a site or a stated
reason comes out. The reasons are asserted as carefully as the sites, because a
match that never expanded has to be able to say which of the five causes it hit
-- and a bare "expansions: 0" cannot.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.economy import (
    EXTRACTOR_TYPE,
    FACTORY_TYPE,
    count_extractors,
    expand_economy,
    expand_production,
)
from rw_bot.wire.state import BuildOption, Entity, ResourcePool, Sample

#: Attack range by type name, as the registry dump gives it. Complete by
#: contract -- every registered type appears, the unarmed at zero.
_REACHES = {
    "commandCenter": 0.0,
    "builder": 0.0,
    EXTRACTOR_TYPE: 0.0,
    "turret": 100.0,
    "editorOrBuilder": 0.0,
}


def _unit(type_name: str, *, price: int = 700, speed: float = 0.0) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    EXTRACTOR_TYPE: _unit(EXTRACTOR_TYPE),
    "builder": _unit("builder", price=200, speed=0.6),
    "commandCenter": _unit("commandCenter", price=0),
    "turret": _unit("turret", price=300),
    FACTORY_TYPE: _unit(FACTORY_TYPE, price=700),
    "c_tank": _unit("c_tank", price=350, speed=1.1),
    "editorOrBuilder": _unit("editorOrBuilder", price=0),
}


def _entity(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    queued: int = 0,
    complete: bool = True,
    group: int = 1,
) -> Entity:
    return Entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=not mine,
        movement="LAND",
        group=group,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=queued,
    )


def _pool(x: float, y: float, *, group_land: int = 1) -> ResourcePool:
    return ResourcePool(
        index=0,
        tile_x=int(x) // 20,
        tile_y=int(y) // 20,
        x=x,
        y=y,
        group_land=group_land,
    )


def _option(
    unit_id: int,
    produces: str = EXTRACTOR_TYPE,
    *,
    available: bool = True,
    placed: bool = True,
) -> BuildOption:
    return BuildOption(
        index=0,
        unit_id=unit_id,
        produces=produces,
        action=1,
        placed=placed,
        available=available,
    )


#: The Builder, and the option by which the engine says it can place one.
_BUILDER = _entity(214, "builder", 0.0, 0.0)
_CAN_PLACE = _option(214)


def _sample(
    *entities: Entity,
    pools: tuple[ResourcePool, ...] = (),
    options: tuple[BuildOption, ...] = (),
    credits_held: int = 4000,
) -> Sample:
    return Sample(
        frame=1,
        clock_ms=10,
        credits=credits_held,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=entities,
        pools=pools,
        options=options,
    )


def test_only_finished_extractors_count_as_income() -> None:
    """A structure joins the roster when construction starts, not when it pays."""
    world = _sample(
        _entity(1, EXTRACTOR_TYPE),
        _entity(2, EXTRACTOR_TYPE, complete=False),
        _entity(3, EXTRACTOR_TYPE, mine=False),
    )
    assert count_extractors(world) == 1


def test_a_free_pool_is_claimed() -> None:
    world = _sample(
        _BUILDER,
        _entity(213, "commandCenter", 0.0, 0.0),
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is True
    assert plan["unit_id"] == 214
    assert plan["type_name"] == EXTRACTOR_TYPE
    assert (plan["x"], plan["y"]) == (300.0, 0.0)
    assert plan["reason"] == f"{EXTRACTOR_TYPE} #1 at (300, 0)"


def test_the_reason_counts_the_extractors_already_standing() -> None:
    """So a run log reads as a sequence rather than a repeated sentence."""
    world = _sample(
        _BUILDER,
        _entity(1, EXTRACTOR_TYPE, 500.0, 500.0),
        _entity(2, EXTRACTOR_TYPE, 600.0, 600.0),
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["reason"] == f"{EXTRACTOR_TYPE} #3 at (300, 0)"
    assert plan["owned"] == 2


def test_a_walking_builder_is_left_alone() -> None:
    """The order it is carrying out is the order we would send again.

    Without this the fight loop re-tasks the builder every sample and it never
    arrives anywhere -- the same churn that cost 743 attack orders on 24
    targets before the combat side learned to commit.
    """
    world = _sample(_BUILDER, pools=(_pool(300.0, 0.0),), options=(_CAN_PLACE,))
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=True)
    assert plan["build"] is False
    assert plan["reason"] == "builder still walking to its site"


def test_an_extractor_already_going_up_blocks_another() -> None:
    world = _sample(
        _BUILDER,
        _entity(9, EXTRACTOR_TYPE, 300.0, 0.0, complete=False),
        pools=(_pool(900.0, 0.0),),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False
    assert plan["reason"] == f"{EXTRACTOR_TYPE} already going up"


def test_an_enemy_extractor_going_up_does_not_block_ours() -> None:
    """Ownership is what makes the rising check about our own construction."""
    world = _sample(
        _BUILDER,
        _entity(9, EXTRACTOR_TYPE, 5000.0, 5000.0, mine=False, complete=False),
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
    )
    assert expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)["build"]


def test_no_builder_alive_stops_the_economy_and_says_so() -> None:
    """The caller reads this reason to know it should make another builder."""
    world = _sample(_entity(213, "commandCenter"), pools=(_pool(300.0, 0.0),))
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False
    assert plan["reason"] == f"nothing owned can place {EXTRACTOR_TYPE}"


def test_an_unavailable_action_is_not_a_placer() -> None:
    """The action exists but cannot be used, which is a wait rather than a site."""
    world = _sample(
        _BUILDER,
        pools=(_pool(300.0, 0.0),),
        options=(_option(214, available=False),),
    )
    assert (
        expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)["build"]
        is False
    )


def test_an_option_from_a_unit_not_in_the_roster_is_not_a_placer() -> None:
    """A dead producer's option can outlive it in a partial observation."""
    world = _sample(_BUILDER, pools=(_pool(300.0, 0.0),), options=(_option(999),))
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False


def test_the_map_editor_placeholder_never_places_anything() -> None:
    """It offers nearly every type in the game and is parked off-map.

    Ordering against it spends nothing and builds nothing, which is the silent
    failure the placeholder exclusion exists to prevent.
    """
    world = _sample(
        _entity(217, "editorOrBuilder", -1000.0, -1000.0),
        pools=(_pool(300.0, 0.0),),
        options=(_option(217),),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False
    assert plan["reason"] == f"nothing owned can place {EXTRACTOR_TYPE}"


def test_a_type_the_catalogue_cannot_price_is_refused_by_name() -> None:
    """A stale dump names itself here rather than freezing the economy silently."""
    world = _sample(_BUILDER, pools=(_pool(300.0, 0.0),), options=(_option(214, "someModRig"),))
    plan = expand_economy(
        world,
        _CATALOGUE,
        _REACHES,
        reserve=0,
        builder_moved=False,
        type_name="someModRig",
    )
    assert plan["build"] is False
    assert plan["reason"] == "someModRig is not in the catalogue"


def test_the_reserve_is_held_back_for_the_army() -> None:
    """Expansion competes with reinforcement for the same credits."""
    world = _sample(
        _BUILDER,
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
        credits_held=1000,
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=350, builder_moved=False)
    assert plan["build"] is False
    assert plan["reason"] == "1000 credits, need 1050 to expand past a 350 reserve"


def test_the_reserve_boundary_is_inclusive() -> None:
    """Exactly enough is enough; refusing here would bank a credit forever."""
    world = _sample(
        _BUILDER,
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
        credits_held=1050,
    )
    assert expand_economy(world, _CATALOGUE, _REACHES, reserve=350, builder_moved=False)["build"]


def test_every_pool_taken_is_reported_with_its_counts() -> None:
    """ "No pool" has several causes and they call for opposite responses."""
    world = _sample(
        _BUILDER,
        _entity(9, EXTRACTOR_TYPE, 300.0, 0.0, mine=False),
        pools=(_pool(300.0, 0.0), _pool(900.0, 0.0, group_land=7)),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False
    assert plan["reason"] == "no pool free of 2: 1 occupied, 1 unreachable, 0 exposed"
    assert plan["occupied"] == 1


def test_a_pool_under_hostile_guns_is_refused() -> None:
    """The builder dies in transit, so the route is what is judged."""
    world = _sample(
        _BUILDER,
        _entity(9, "turret", 150.0, 0.0, mine=False),
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False
    assert plan["exposed"] == 1


def test_the_economy_grows_outward_from_the_base_not_from_the_builder() -> None:
    """Distance is measured from the anchor, so the base stays defensible.

    The builder ends every job standing on the pool it just took, so ranking by
    distance from the builder would walk it steadily off the map.
    """
    centre = _entity(213, "commandCenter", 0.0, 0.0)
    strayed = _entity(214, "builder", 900.0, 0.0)
    world = _sample(
        strayed,
        centre,
        pools=(_pool(200.0, 0.0), _pool(1000.0, 0.0)),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert (plan["x"], plan["y"]) == (200.0, 0.0)


def test_a_player_holding_no_structure_measures_from_the_builder() -> None:
    """The opening fallback, and the build plan's own."""
    world = _sample(
        _entity(214, "builder", 900.0, 0.0),
        pools=(_pool(200.0, 0.0), _pool(1000.0, 0.0)),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert (plan["x"], plan["y"]) == (1000.0, 0.0)


def test_no_pool_in_sight_is_not_an_error() -> None:
    world = _sample(_BUILDER, options=(_CAN_PLACE,))
    plan = expand_economy(world, _CATALOGUE, _REACHES, reserve=0, builder_moved=False)
    assert plan["build"] is False
    assert plan["reason"] == "no pool free of 0: 0 occupied, 0 unreachable, 0 exposed"


def _bound_world(credits_held: int = 4000) -> Sample:
    """A player whose only factory is busy, with money in the bank.

    The state that banked 7,013 credits: income arriving with nowhere to spend
    it, because :func:`sustain` can only fill queues that exist.
    """
    return _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=credits_held,
    )


def test_a_busy_queue_and_a_full_bank_buys_another_factory() -> None:
    growth = expand_production(_bound_world(), _CATALOGUE, reserve=0)
    assert growth["build"] is True
    assert growth["type_name"] == FACTORY_TYPE
    assert growth["unit_id"] == 214


def test_an_idle_producer_means_throughput_is_not_the_constraint() -> None:
    """Another factory would idle beside the one already idling."""
    idle = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=100_000,
    )
    growth = expand_production(idle, _CATALOGUE, reserve=0)
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_the_army_reserve_is_protected_from_a_factory_too() -> None:
    """The same rule pool expansion follows: a tank now beats capacity later."""
    growth = expand_production(_bound_world(credits_held=800), _CATALOGUE, reserve=500)
    assert growth["build"] is False
    assert "need 1200" in growth["reason"]


def test_a_factory_nothing_can_place_is_not_proposed() -> None:
    """No builder offers it, so ordering one would go to nobody."""
    world = _sample(
        _entity(213, "commandCenter"),
        _entity(300, FACTORY_TYPE, queued=1),
        options=(_option(300, "c_tank", placed=False),),
        credits_held=100_000,
    )
    growth = expand_production(world, _CATALOGUE, reserve=0)
    assert growth["build"] is False
    assert growth["reason"] == f"nothing owned can place {FACTORY_TYPE}"


def test_successive_factories_take_successive_ring_slots() -> None:
    """Indexed by what is already standing, so they spread rather than stack."""
    first = expand_production(_bound_world(), _CATALOGUE, reserve=0)
    crowded = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        _entity(301, FACTORY_TYPE, queued=1),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    second = expand_production(crowded, _CATALOGUE, reserve=0)
    assert (first["x"], first["y"]) != (second["x"], second["y"])


def test_a_mobile_unit_does_not_shift_the_ring_index() -> None:
    """Only standing structures crowd the ring; a tank is not a building."""
    plain = expand_production(_bound_world(), _CATALOGUE, reserve=0)
    with_army = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        _entity(400, "c_tank"),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    assert (plain["x"], plain["y"]) == (
        expand_production(with_army, _CATALOGUE, reserve=0)["x"],
        expand_production(with_army, _CATALOGUE, reserve=0)["y"],
    )


def test_an_unfinished_or_enemy_structure_does_not_crowd_the_ring() -> None:
    """The index counts what is standing, not what is starting or theirs.

    A shell counted as a building would shift the next factory a slot early and
    leave a gap; an opponent's would shift it for a structure that is not in
    our way at all.
    """
    plain = expand_production(_bound_world(), _CATALOGUE, reserve=0)
    noisy = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        _entity(301, FACTORY_TYPE, complete=False),
        _entity(302, FACTORY_TYPE, mine=False),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    crowded = expand_production(noisy, _CATALOGUE, reserve=0)
    assert (plain["x"], plain["y"]) == (crowded["x"], crowded["y"])
