"""Claiming resource pools, exercised as the pure function it is.

No socket and no game: a world state goes in and either a site or a stated
reason comes out. The reasons are asserted as carefully as the sites, because a
match that never expanded has to be able to say which of the five causes it hit
-- and a bare "expansions: 0" cannot.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.defence import expand_defence, undefended
from rw_bot.policy.economy import (
    EXTRACTOR_TYPE,
    FACTORY_TYPE,
    count_extractors,
    expand_economy,
    expand_production,
    upgradeable,
)
from rw_bot.policy.siting import COVER_RING, PLACEMENT_RING, RING_SLOT_RADIUS
from rw_bot.wire.state import BuildOption, Entity, ResourcePool, Sample
from tests.wire_fixtures import entity, profile

#: Combat profiles as the registry dump gives them. Complete by contract --
#: every registered type appears, the unarmed at zero reach.
_PROFILES = {
    "commandCenter": profile("commandCenter", 0.0),
    "builder": profile("builder", 0.0),
    EXTRACTOR_TYPE: profile(EXTRACTOR_TYPE, 0.0),
    "turret": profile("turret", 100.0),
    "editorOrBuilder": profile("editorOrBuilder", 0.0),
    "landFactory": profile("landFactory", 0.0),
}


def _bound(credits_held: int = 4000) -> Sample:
    """The throughput-bound world, rebuilt so each caller can read its workers."""
    return _bound_world(credits_held=credits_held)


def _free(world: Sample) -> tuple[Entity, ...]:
    """The workers a loop would report as free: every builder in the world."""
    return tuple(e for e in world["entities"] if e["mine"] and e["type_name"] == "builder")


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
    return entity(
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
    makes_something: bool = True,
) -> BuildOption:
    return BuildOption(
        index=0,
        unit_id=unit_id,
        produces=produces,
        action=1,
        placed=placed,
        available=available,
        makes_something=makes_something,
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
        players=(),
        options=options,
    )


#: A world with a Command Center to anchor on, a Builder, and the option by
#: which the engine says the Builder can place a turret.
_CAN_DEFEND = _option(214, "turret")


def _defence_world(*entities: Entity, cover_base: bool = True) -> Sample:
    """A base with a Command Center, a Builder, and by default a turret on it.

    The Command Center is itself a structure with no cover, so a world without
    the base turret answers "cover the base" to every question. Covering it in
    the fixture is what lets a test isolate the structure it is actually about.
    """
    base = (_entity(299, "turret", 0.0, 0.0),) if cover_base else ()
    return _sample(
        _entity(213, "commandCenter", 0.0, 0.0),
        _BUILDER,
        *base,
        *entities,
        options=(_CAN_DEFEND,),
    )


def _bare(world: Sample) -> Entity:
    """The structure needing cover, failing loudly when there is none.

    Written as a helper so the assertions below can name the structure they
    expect rather than merely checking that *something* came back.
    """
    found = undefended(world, _CATALOGUE, _PROFILES, "turret")
    if found is None:
        raise AssertionError("expected a structure needing cover, got none")
    return found


def test_a_structure_with_no_turret_near_it_is_the_one_to_cover() -> None:
    """The finding this exists for: played to a verdict the bot is defeated, and
    the build policy names the cause -- 44 of the map's 46 pools end up owned by
    the opponents, one run spending 275 expansion orders for a single surviving
    extractor. A turret costs less than the extractor it covers
    ([[policy-holding-ground]]).
    """
    world = _defence_world(_entity(400, EXTRACTOR_TYPE, 900.0, 0.0))
    assert _bare(world)["unit_id"] == 400


def test_cover_is_the_turrets_own_reach_rather_than_a_chosen_radius() -> None:
    """A structure is covered when a turret could actually shoot something
    standing on it, which the registry answers ([[mechanics-combat-profile]]).
    """
    # The fixture turret reaches 100.
    covered = _defence_world(
        _entity(400, EXTRACTOR_TYPE, 900.0, 0.0),
        _entity(401, "turret", 950.0, 0.0),
    )
    assert undefended(covered, _CATALOGUE, _PROFILES, "turret") is None

    out_of_reach = _defence_world(
        _entity(400, EXTRACTOR_TYPE, 900.0, 0.0),
        _entity(401, "turret", 1100.0, 0.0),
    )
    assert _bare(out_of_reach)["unit_id"] == 400


def test_the_base_is_covered_before_the_frontier() -> None:
    """Nearest to the anchor first, and it is kept because covering the
    extractors instead was **measured worse**.

    The argument against this ordering was the strongest the defence policy has
    had. Extractor losses decide every duel; the per-loss table puts each death
    688-1,766 world units from the army's own fighting cloud; and not one unit
    died within 900 units of the base across two traced runs, so nearest-first
    reads as spending where nothing is ever attacked. Restricting cover to the
    extractors, same twelve seeds and same rung: **wins 4 -> 0, drops 21 -> 24,
    and the first two defeats in fifty-two duels** ([[policy-holding-ground]]).

    So the ordering stands on the measurement rather than on the reasoning, and
    the reasoning is recorded because it was good and still lost.
    """
    world = _defence_world(
        _entity(400, EXTRACTOR_TYPE, 4000.0, 0.0),
        _entity(401, EXTRACTOR_TYPE, 400.0, 0.0),
    )
    assert _bare(world)["unit_id"] == 401


def test_a_mobile_unit_is_not_a_structure_to_cover() -> None:
    """A turret cannot follow a tank, so covering one is not a thing to buy."""
    world = _defence_world(_entity(400, "c_tank", 900.0, 0.0), cover_base=False)
    # The Command Center and the Builder are the only other candidates, and the
    # Builder moves; the Command Center is at the anchor and is itself bare.
    assert _bare(world)["type_name"] == "commandCenter"


def test_nothing_to_anchor_on_means_nothing_to_cover() -> None:
    world = _sample(_BUILDER, options=(_CAN_DEFEND,))
    assert undefended(world, _CATALOGUE, _PROFILES, "turret") is None


def test_defence_places_a_turret_beside_what_it_covers() -> None:
    """Beside the structure rather than on the base ring: a turret at the base
    does not defend a pool on the far side of the map, and the pools are what is
    being lost.
    """
    world = _defence_world(_entity(400, EXTRACTOR_TYPE, 900.0, 40.0))
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=4000, free=_free(world), turret_type="turret"
    )
    assert plan["build"]
    assert plan["type_name"] == "turret"
    assert plan["unit_id"] == 214
    assert (plan["x"], plan["y"]) == (900.0 + RING_SLOT_RADIUS, 40.0)


def test_the_editor_placeholder_is_never_a_structure_to_cover() -> None:
    """The placeholder trap, sprung in a new consumer: an owned immobile
    entity parked off-map with 170,000 hp is eternally bare, and the moment
    cover WORKED everywhere real, it became the only bare structure left --
    defence poured turrets at an unbuildable point for whole matches and
    walked 46 of 51 builders to their deaths (log: 2026-07-31)."""
    world = _defence_world(
        _entity(999, "editorOrBuilder", -1000.0, -1000.0),
    )
    assert undefended(world, _CATALOGUE, _PROFILES, "turret") is None


def test_defence_steps_around_an_occupied_cover_slot() -> None:
    """The check defence never had: the old bare offset was reached for
    without looking, the engine refuses an occupied site silently, and one
    scorecard priced the habit at 27 paid orders for about five turrets
    standing. An occupied first slot now means the second, not a refusal."""
    world = _defence_world(
        _entity(400, EXTRACTOR_TYPE, 900.0, 40.0),
        _entity(402, EXTRACTOR_TYPE, 960.0, 40.0),
        _entity(403, "turret", 960.0, 140.0),
    )
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=4000, free=_free(world), turret_type="turret"
    )
    assert plan["build"]
    # (60, 0) is occupied by the neighbouring extractor, so (0, 60) is next.
    assert (plan["x"], plan["y"]) == (900.0, 100.0)


def test_defence_waits_when_every_cover_slot_is_taken() -> None:
    """A refusal with a reason, where the old offset was a silent one."""
    # Hostile, because an opponent's building fills a position exactly as
    # firmly as ours -- and because our own would themselves ask for cover.
    blockers = tuple(
        _entity(500 + i, EXTRACTOR_TYPE, 900.0 + dx, 40.0 + dy, mine=False)
        for i, (dx, dy) in enumerate(COVER_RING)
    )
    world = _defence_world(_entity(400, EXTRACTOR_TYPE, 900.0, 40.0), *blockers)
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=4000, free=_free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert plan["reason"] == "no clear cover position around extractorT1 at 900,40"


def test_defence_waits_when_it_cannot_afford_the_turret() -> None:
    world = _defence_world(_entity(400, EXTRACTOR_TYPE, 900.0, 0.0))
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=100, free=_free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert "300 of 100" in plan["reason"]


def test_defence_waits_when_everything_already_has_cover() -> None:
    world = _defence_world(
        _entity(400, EXTRACTOR_TYPE, 900.0, 0.0),
        _entity(401, "turret", 900.0, 0.0),
        _entity(3, "turret", 0.0, 0.0),
    )
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=4000, free=_free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert plan["reason"] == "every structure already has cover"


def test_defence_waits_when_no_worker_can_place_one() -> None:
    world = _defence_world(_entity(400, EXTRACTOR_TYPE, 900.0, 0.0))
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=4000, free=(), turret_type="turret"
    )
    assert not plan["build"]
    assert "no free worker" in plan["reason"]


def test_a_turret_the_catalogue_cannot_price_is_never_ordered() -> None:
    """Unpriced means unbudgetable, and spending blind is what the budget
    prevents ([[policy-budget]]).
    """
    world = _defence_world(_entity(400, EXTRACTOR_TYPE, 900.0, 0.0))
    plan = expand_defence(
        world, _CATALOGUE, _PROFILES, available=4000, free=_free(world), turret_type="mysteryGun"
    )
    assert not plan["build"]
    assert "does not price" in plan["reason"]


def test_an_upgraded_extractor_still_counts_as_an_extractor() -> None:
    """A figure that quietly means something else is how a reading goes wrong.

    Counting the named type alone was right until the bot could upgrade. The
    moment it could, a run holding three tier-two extractors and earning 54
    credits a second reported ``extractors 0 -> 0``
    ([[policy-holding-ground]]).
    """
    world = _sample(
        _entity(400, "extractorT1"),
        _entity(401, "extractorT2"),
        _entity(402, "extractorT3"),
        _entity(403, "landFactory"),
    )
    assert count_extractors(world) == 3


def test_an_extractor_offering_to_upgrade_itself_is_found() -> None:
    """The action that was invisible for the whole life of the bot.

    An upgrade is declared in the asset as ``convertTo`` and the engine reports
    it with no placement type and no "makes something" flag -- exactly the shape
    the agent used to drop. With every action published, all four standing
    extractors offer ``extractorT2`` and the engine calls it available, with no
    tier-2 builder and no prerequisite chain ([[policy-holding-ground]]).
    """
    world = _sample(
        _entity(400, EXTRACTOR_TYPE),
        _entity(401, EXTRACTOR_TYPE),
        options=(
            _option(400, "extractorT2", placed=False),
            _option(401, "extractorT2", placed=False),
        ),
    )
    assert upgradeable(world) == (
        {"unit_id": 400, "produces": "extractorT2"},
        {"unit_id": 401, "produces": "extractorT2"},
    )


def test_each_structure_is_offered_its_own_next_tier() -> None:
    """The walk that used to stop at tier two.

    A single wanted type cannot describe a roster holding both tiers, so the
    tier two was never asked to become a tier three -- 20 credits a second
    against 12, beside a bank of 62,146 that had nothing to buy
    ([[mechanics-unit-value]]).
    """
    world = _sample(
        _entity(400, EXTRACTOR_TYPE),
        _entity(401, "extractorT2"),
        options=(
            _option(400, "extractorT2", placed=False),
            _option(401, "extractorT3", placed=False),
        ),
    )
    assert upgradeable(world) == (
        {"unit_id": 400, "produces": "extractorT2"},
        {"unit_id": 401, "produces": "extractorT3"},
    )


def test_a_structure_at_the_fork_is_left_alone() -> None:
    """The tier three offers two conversions and neither is an upgrade of the
    other, so choosing between them here would be a preference nobody measured
    ([[mechanics-unit-value]]).
    """
    world = _sample(
        _entity(400, "extractorT3"),
        options=(
            _option(400, "extractorT3_overclocked", placed=False),
            _option(400, "extractorT3_reinforced", placed=False),
        ),
    )
    assert upgradeable(world) == ()


def test_an_extractor_already_upgrading_is_left_alone() -> None:
    """A queued conversion is in progress; ordering it again spends twice."""
    world = _sample(
        _entity(400, EXTRACTOR_TYPE, queued=1),
        options=(_option(400, "extractorT2", placed=False),),
    )
    assert upgradeable(world) == ()


def test_an_upgrade_the_engine_has_not_unlocked_is_not_ordered() -> None:
    """Availability is the engine's own predicate, so tech gating and the unit
    cap are already accounted for rather than modelled here
    ([[mechanics-build-actions]]).
    """
    world = _sample(
        _entity(400, EXTRACTOR_TYPE),
        options=(_option(400, "extractorT2", placed=False, available=False),),
    )
    assert upgradeable(world) == ()


def test_a_placed_option_is_not_an_upgrade() -> None:
    """Placing a second extractor somewhere is expansion, not conversion."""
    world = _sample(
        _entity(400, EXTRACTOR_TYPE),
        options=(_option(400, "extractorT2", placed=True),),
    )
    assert upgradeable(world) == ()


def test_a_type_that_heads_no_chain_offers_no_upgrade() -> None:
    """The chain says which tier is next; a type outside it has no next."""
    world = _sample(
        _entity(400, "landFactory"),
        options=(_option(400, "c_tank", placed=False),),
    )
    assert upgradeable(world) == ()


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
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
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
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
    assert plan["reason"] == f"{EXTRACTOR_TYPE} #3 at (300, 0)"
    assert plan["owned"] == 2


def test_a_worker_the_loop_has_not_offered_is_not_used() -> None:
    """Availability has one owner, and it is not this module.

    A worker walking to a site, or already building one, is not in the free
    list the loop supplies -- which is what stopped two expansion rules
    re-tasking the same worker off each other ([[policy-loop]]).
    """
    world = _sample(_BUILDER, pools=(_pool(300.0, 0.0),), options=(_CAN_PLACE,))
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=())
    assert plan["build"] is False
    assert plan["reason"] == f"no free worker can place {EXTRACTOR_TYPE}"


def test_a_second_worker_can_claim_a_pool_while_the_first_builds() -> None:
    """The point of tracking workers rather than "the" builder.

    One worker mid-build no longer blocks the other. Whether each is busy is
    the loop's judgement, and it hands over only the ones that are not
    ([[policy-loop]]).
    """
    second = _entity(215, "builder", 100.0, 0.0)
    world = _sample(
        _BUILDER,
        second,
        _entity(9, EXTRACTOR_TYPE, 300.0, 0.0, complete=False),
        pools=(_pool(900.0, 0.0),),
        options=(_CAN_PLACE, _option(215)),
    )
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=(second,))
    assert plan["build"] is True
    assert plan["unit_id"] == 215


def test_an_enemy_extractor_going_up_does_not_block_ours() -> None:
    """Ownership is what makes the rising check about our own construction."""
    world = _sample(
        _BUILDER,
        _entity(9, EXTRACTOR_TYPE, 5000.0, 5000.0, mine=False, complete=False),
        pools=(_pool(300.0, 0.0),),
        options=(_CAN_PLACE,),
    )
    assert expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))["build"]


def test_no_builder_alive_stops_the_economy_and_says_so() -> None:
    """The caller reads this reason to know it should make another builder."""
    world = _sample(_entity(213, "commandCenter"), pools=(_pool(300.0, 0.0),))
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
    assert plan["build"] is False
    assert plan["reason"] == f"no free worker can place {EXTRACTOR_TYPE}"


def test_an_unavailable_action_is_not_a_placer() -> None:
    """The action exists but cannot be used, which is a wait rather than a site."""
    world = _sample(
        _BUILDER,
        pools=(_pool(300.0, 0.0),),
        options=(_option(214, available=False),),
    )
    assert (
        expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))["build"] is False
    )


def test_an_option_from_a_unit_not_in_the_roster_is_not_a_placer() -> None:
    """A dead producer's option can outlive it in a partial observation."""
    world = _sample(_BUILDER, pools=(_pool(300.0, 0.0),), options=(_option(999),))
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
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
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
    assert plan["build"] is False
    assert plan["reason"] == f"no free worker can place {EXTRACTOR_TYPE}"


def test_a_type_the_catalogue_cannot_price_is_refused_by_name() -> None:
    """A stale dump names itself here rather than freezing the economy silently."""
    world = _sample(_BUILDER, pools=(_pool(300.0, 0.0),), options=(_option(214, "someModRig"),))
    plan = expand_economy(
        world,
        _CATALOGUE,
        _PROFILES,
        reserve=0,
        free=_free(world),
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
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=350, free=_free(world))
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
    assert expand_economy(world, _CATALOGUE, _PROFILES, reserve=350, free=_free(world))["build"]


def test_every_pool_taken_is_reported_with_its_counts() -> None:
    """ "No pool" has several causes and they call for opposite responses."""
    world = _sample(
        _BUILDER,
        _entity(9, EXTRACTOR_TYPE, 300.0, 0.0, mine=False),
        pools=(_pool(300.0, 0.0), _pool(900.0, 0.0, group_land=7)),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
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
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
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
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
    assert (plan["x"], plan["y"]) == (200.0, 0.0)


def test_a_player_holding_no_structure_measures_from_the_builder() -> None:
    """The opening fallback, and the build plan's own."""
    world = _sample(
        _entity(214, "builder", 900.0, 0.0),
        pools=(_pool(200.0, 0.0), _pool(1000.0, 0.0)),
        options=(_CAN_PLACE,),
    )
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
    assert (plan["x"], plan["y"]) == (1000.0, 0.0)


def test_no_pool_in_sight_is_not_an_error() -> None:
    world = _sample(_BUILDER, options=(_CAN_PLACE,))
    plan = expand_economy(world, _CATALOGUE, _PROFILES, reserve=0, free=_free(world))
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
    growth = expand_production(
        _bound(), _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(_bound())
    )
    assert growth["build"] is True
    assert growth["type_name"] == FACTORY_TYPE
    assert growth["unit_id"] == 214


def test_no_surplus_means_throughput_is_not_the_constraint() -> None:
    """Everything was already claimed, so there is no idle money to convert."""
    world = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=100_000,
    )
    growth = expand_production(
        world, _CATALOGUE, available=0, wanted=("c_tank",), free=_free(world)
    )
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_a_surplus_short_of_the_price_buys_nothing() -> None:
    """The reserve reaches this through the surplus rather than through a second
    subtraction here: what arrives is already net of everything protected
    ([[policy-budget]]).
    """
    growth = expand_production(
        _bound(800),
        _CATALOGUE,
        available=300,
        wanted=("c_tank",),
        free=_free(_bound(800)),
    )
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_a_factory_nothing_can_place_is_not_proposed() -> None:
    """No builder offers it, so ordering one would go to nobody."""
    world = _sample(
        _entity(213, "commandCenter"),
        _entity(300, FACTORY_TYPE, queued=1),
        options=(_option(300, "c_tank", placed=False),),
        credits_held=100_000,
    )
    growth = expand_production(
        world, _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(world)
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
        _bound(), _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(_bound())
    )
    taken = _entity(302, FACTORY_TYPE, first["x"], first["y"])
    crowded = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        taken,
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    second = expand_production(
        crowded, _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(crowded)
    )
    assert (second["x"], second["y"]) != (first["x"], first["y"])


def test_an_extractor_on_a_distant_pool_does_not_move_the_factory() -> None:
    """The defect that made this observation-driven.

    An extractor is immobile and it is nowhere near the ring, so counting it as
    ring occupancy moved every later factory for no reason -- ten extractors
    moved it four positions.
    """
    plain = expand_production(
        _bound(), _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(_bound())
    )
    with_economy = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        _entity(401, EXTRACTOR_TYPE, 5000.0, 5000.0),
        _entity(402, EXTRACTOR_TYPE, 6000.0, 6000.0),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    grown = expand_production(
        with_economy, _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(with_economy)
    )
    assert (grown["x"], grown["y"]) == (plain["x"], plain["y"])


def test_a_mobile_unit_does_not_fill_a_ring_position() -> None:
    """It moves; a building does not. Counting it would hide a usable slot."""
    plain = expand_production(
        _bound(), _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(_bound())
    )
    with_army = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        _entity(400, "c_tank", plain["x"], plain["y"]),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    parked = expand_production(
        with_army, _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(with_army)
    )
    assert (parked["x"], parked["y"]) == (plain["x"], plain["y"])


def test_an_opponents_factory_is_not_our_production_capacity() -> None:
    """Ours to fill or it is not capacity.

    Ring occupancy and production capacity are different questions about the
    same building: an opponent's factory stands on ground we cannot build on,
    and offers a queue we cannot fill ([[policy-production]]).
    """
    theirs = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(302, FACTORY_TYPE, mine=False),
        options=(_option(214, FACTORY_TYPE),),
        credits_held=100_000,
    )
    growth = expand_production(
        theirs, _CATALOGUE, available=100_000, wanted=("c_tank",), free=_free(theirs)
    )
    assert growth["build"] is False
    assert growth["reason"] == "production is not the constraint"


def test_a_full_ring_stops_the_factory_rather_than_stacking_one() -> None:
    """A structure destroyed frees its position, so this is a wait, not a block."""
    filled = [_entity(500 + i, FACTORY_TYPE, dx, dy) for i, (dx, dy) in enumerate(PLACEMENT_RING)]
    world = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        *filled,
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=4000,
    )
    growth = expand_production(
        world, _CATALOGUE, available=4000, wanted=("c_tank",), free=_free(world)
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
    world = _sample(
        _entity(213, "commandCenter"),
        _entity(214, "builder"),
        _entity(300, FACTORY_TYPE, queued=1),
        _entity(301, FACTORY_TYPE, complete=False),
        options=(_option(214, FACTORY_TYPE), _option(300, "c_tank", placed=False)),
        credits_held=100_000,
    )
    growth = expand_production(world, _CATALOGUE, available=100_000, wanted=("c_tank",), free=())
    assert growth["build"] is False
    assert growth["reason"] == f"no free worker can place {FACTORY_TYPE}"


def test_a_walking_builder_is_left_alone_by_the_factory_rule_too() -> None:
    """One builder, and an order given to it replaces whatever it was doing."""
    growth = expand_production(_bound(), _CATALOGUE, available=100_000, wanted=("c_tank",), free=())
    assert growth["build"] is False
    assert growth["reason"] == f"no free worker can place {FACTORY_TYPE}"
