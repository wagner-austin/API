"""The build-order policy, exercised as the pure function it is.

No socket, no game, no clock. Every case here is a world state in and a
decision out, which is the point of keeping the deciding half pure.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.build_order import (
    BUILDER_TYPE,
    completed_count,
    decide,
    next_unsatisfied_index,
)
from rw_bot.policy.siting import (
    PLACEMENT_RING,
    POOL_OCCUPIED_RADIUS,
    find_anchor,
    survey_pools,
)
from rw_bot.wire.state import BuildOption, Entity, ResourcePool, Sample
from tests.wire_fixtures import entity, profile


def _unit(type_name: str, price: int, speed: float = 0.0, attack_range: float = 0.0) -> UnitStats:
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
        weapon=None if attack_range == 0.0 else _weapon(attack_range),
    )


def _weapon(attack_range: float) -> Weapon:
    return Weapon(
        shoot_delay=30.0,
        attack_range=attack_range,
        direct_damage=10.0,
        direct_damage_volley=10.0,
        area_damage=0.0,
        area_damage_volley=0.0,
    )


_CATALOGUE = {
    "landFactory": _unit("landFactory", 300),
    "airFactory": _unit("airFactory", 900),
    "laboratory": _unit("laboratory", 900),
    # The two the live roster always starts with. The Command Center is the
    # anchor placement is measured from; the builder must be mobile, or it
    # would be eligible as an anchor and the ring would follow it again.
    "commandCenter": _unit("commandCenter", 3000),
    "builder": _unit("builder", 500, speed=0.6),
    "extractorT1": _unit("extractorT1", 700),
}


#: Reach of the test turret, in world units.
#:
#: Comfortably wider than :data:`POOL_OCCUPIED_RADIUS` so that "this pool is
#: covered" and "this pool is built on" are always distinguishable — a turret
#: close enough to shoot a pool must not also be close enough to be standing on
#: it, or the tests could not tell which rule rejected it.
_TURRET_RANGE = 100.0

#: The default catalogue plus something that shoots back.
_ARMED = {**_CATALOGUE, "turret": _unit("turret", 400, attack_range=_TURRET_RANGE)}

#: Attack range by type name, as the registry dump gives it.
#:
#: Every type any fixture can name appears, armed or not. That mirrors the real
#: dump, which covers all 173 registered types, and it is the contract
#: :func:`~rw_bot.policy.threat.reach_of` indexes against rather than defaults
#: through ([[policy-threat]]).
_PROFILES: dict[str, CombatProfile] = {name: profile(name, 0.0) for name in _ARMED}
_PROFILES["turret"] = profile("turret", _TURRET_RANGE)
_PROFILES["someModStructure"] = profile("someModStructure", 0.0)


def _place(type_name: str, needs_pool: bool = False) -> TypePlacement:
    return TypePlacement(index=0, type_name=type_name, needs_pool=needs_pool)


#: Where each type may stand, as the engine reports it. Only the extractor is
#: pool-bound; the live dump agrees, and says so of exactly eight types out of
#: 173 ([[mechanics-resource-pools]]).
_PLACEMENTS = {
    "landFactory": _place("landFactory"),
    "airFactory": _place("airFactory"),
    "extractorT1": _place("extractorT1", needs_pool=True),
    "laboratory": _place("laboratory"),
    "commandCenter": _place("commandCenter"),
    "builder": _place("builder"),
    "teleporter": _place("teleporter"),
}


#: Connectivity component every land fixture shares.
#:
#: The engine hands these out per map; the value is arbitrary and only equality
#: matters. Defaulting builders and pools to the same one keeps every test that
#: is not about reachability from having to restate it, and a test that *is*
#: about reachability puts one of them somewhere else.
_MAINLAND = 1

#: A component id no fixture shares, for the far side of water.
_ISLAND = 2


def _pool(index: int, tile_x: int, tile_y: int) -> ResourcePool:
    """Build a pool record at a tile, with the world centre the agent computes."""
    return ResourcePool(
        index=index,
        tile_x=tile_x,
        tile_y=tile_y,
        x=tile_x * 20.0 + 10.0,
        y=tile_y * 20.0 + 10.0,
        group_land=_MAINLAND,
    )


def _entity(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    complete: bool = True,
    hostile: bool | None = None,
    movement: str = "LAND",
    group: int = _MAINLAND,
) -> Entity:
    """Build an entity record.

    ``hostile`` defaults to the opposite of ``mine``, which is what a two-player
    skirmish looks like. It is overridable because the engine does not derive it
    that way: an ally is neither mine nor hostile, and the distinction only
    shows up in a test that sets them independently.
    """
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=(not mine) if hostile is None else hostile,
        movement=movement,
        group=group,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=0,
    )


def _option(
    unit_id: int,
    produces: str,
    *,
    placed: bool = True,
    available: bool = True,
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


#: What the Builder offers by default in these worlds.
#:
#: Mirrors the live capture, where unit 214 reports thirteen placed structures
#: including these. Supplying it by default keeps every test that is about
#: placement or ordering from also having to restate the build tree; a test
#: that is about the build tree passes its own.
_BUILDER_OFFERS = ("landFactory", "airFactory", "extractorT1", "commandCenter", "teleporter")


def _free(world: Sample) -> tuple[Entity, ...]:
    """The workers a loop would report as free: every builder in the world.

    Which workers are free is the loop's judgement in production
    ([[policy-loop]]); a pure test of ``decide`` supplies it directly.
    """
    return tuple(e for e in world["entities"] if e["mine"] and e["type_name"] == BUILDER_TYPE)


def _sample(
    *entities: Entity,
    credits: int = 4000,
    pools: tuple[ResourcePool, ...] = (),
    options: tuple[BuildOption, ...] | None = None,
) -> Sample:
    if options is None:
        options = tuple(_option(214, name) for name in _BUILDER_OFFERS)
    return Sample(
        frame=1,
        clock_ms=10,
        credits=credits,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=tuple(entities),
        pools=pools,
        players=(),
        options=options,
    )


_BUILDER = _entity(214, "builder", 4250.0, 2610.0)

#: Placement is measured from the oldest owned immobile structure, so a world
#: used for placement assertions needs one. The live game always has it: the
#: Command Center, at this position on the sandbox map.
_ANCHOR = _entity(213, "commandCenter", 4250.0, 2550.0)


def test_an_empty_plan_is_immediately_done() -> None:
    assert (
        decide(_sample(_BUILDER), (), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(_sample(_BUILDER)))[
            "action"
        ]
        == "done"
    )


def test_the_first_structure_is_ordered_from_the_builders_position() -> None:
    decision = decide(
        _sample(_BUILDER),
        ("landFactory",),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        _free(_sample(_BUILDER)),
    )
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"
    assert decision["unit_id"] == 214
    assert (decision["x"], decision["y"]) == (
        4250.0 + PLACEMENT_RING[0][0],
        2610.0 + PLACEMENT_RING[0][1],
    )


def test_progress_is_read_from_the_roster_not_from_a_counter() -> None:
    """A structure already standing counts, whoever built it."""
    world = _sample(_BUILDER, _entity(300, "landFactory"))
    decision = decide(
        world, ("landFactory", "airFactory"), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world)
    )
    assert decision["type_name"] == "airFactory"


def test_successive_structures_take_successive_ring_positions() -> None:
    world = _sample(_ANCHOR, _BUILDER, _entity(300, "landFactory", 4450.0, 2670.0))
    decision = decide(
        world, ("landFactory", "airFactory"), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world)
    )
    assert (decision["x"], decision["y"]) == (
        _ANCHOR["x"] + PLACEMENT_RING[1][0],
        _ANCHOR["y"] + PLACEMENT_RING[1][1],
    )


def test_a_destroyed_structure_is_rebuilt_rather_than_counted() -> None:
    """Counting from the roster is what makes this fall out for free."""
    world = _sample(_BUILDER, credits=4000)
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["type_name"] == "landFactory"


def test_a_finished_plan_reports_done() -> None:
    world = _sample(_BUILDER, _entity(300, "landFactory"), _entity(301, "airFactory"))
    decision = decide(
        world, ("landFactory", "airFactory"), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world)
    )
    assert decision["action"] == "done"
    assert decision["reason"] == "all 2 plan entries satisfied"


def test_insufficient_credits_waits_rather_than_ordering() -> None:
    world = _sample(_BUILDER, credits=899)
    decision = decide(world, ("airFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert decision["reason"] == "airFactory costs 900, holding 899"
    assert decision["type_name"] == ""


def test_exactly_enough_credits_orders() -> None:
    """The boundary matters: the engine spends in whole units."""
    world = _sample(_BUILDER, credits=900)
    assert (
        decide(world, ("airFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["action"]
        == "build"
    )


def test_an_unfinished_structure_does_not_satisfy_the_plan() -> None:
    """A building joins the roster when construction starts, not when it ends.

    Counting on presence reported a plan finished while a factory was still a
    shell -- and a shell produces nothing, so the next entry could be ordered
    against a building that could not accept it.
    """
    shell = _entity(300, "landFactory", 4450.0, 2730.0, complete=False)
    world = _sample(_BUILDER, _ANCHOR, shell)
    assert completed_count(world, ("landFactory",)) == 0
    assert next_unsatisfied_index(world, ("landFactory",)) == 0
    assert (
        decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["action"]
        == "build"
    )


def test_a_finished_structure_does_satisfy_the_plan() -> None:
    """The same world, one flag different, is the whole of the distinction."""
    done = _entity(300, "landFactory", 4450.0, 2730.0)
    world = _sample(_BUILDER, _ANCHOR, done)
    assert completed_count(world, ("landFactory",)) == 1
    assert (
        decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["action"]
        == "done"
    )


def test_a_type_nothing_owned_can_make_is_blocked() -> None:
    """The laboratory failure, caught before an order is spent.

    A builder has no action producing a laboratory. The engine refuses the
    waypoint and says so only in its own log, so the old planner ordered it and
    then reported "building laboratory" for three hundred samples. The build
    tree answers the question up front instead.
    """
    decision = decide(
        _sample(_BUILDER),
        ("laboratory",),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        _free(_sample(_BUILDER)),
    )
    assert decision["action"] == "blocked"
    assert "nothing the player owns can make laboratory" in decision["reason"]


def test_the_editor_placeholder_never_counts_as_a_producer() -> None:
    """The map editor's unit answers for nearly every type in the game.

    It is owned, it is in every sample, and in the live capture it offers 108
    types against the real Builder's 13 -- a superset, plus 95 more including
    the laboratory. Counting it would make the check above pass for types
    nothing playable can build, and the resulting order would go to a unit
    parked at (-1000, -1000) and do nothing at all.
    """
    placeholder = _entity(217, "editorOrBuilder", -1000.0, -1000.0)
    world = _sample(
        _BUILDER,
        placeholder,
        options=(_option(217, "laboratory"),),
    )
    assert (
        decide(world, ("laboratory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["action"]
        == "blocked"
    )


def test_an_action_that_exists_but_is_unavailable_waits() -> None:
    """Present-but-unavailable is a world state, not a dead plan.

    A prerequisite can still be built and tech can still be researched, so this
    resolves on its own -- unlike an action that does not exist at all.
    """
    world = _sample(_BUILDER, options=(_option(214, "landFactory", available=False),))
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert "not available yet" in decision["reason"]


def test_the_first_unavailable_option_is_the_one_reported() -> None:
    """Two units both offer it and neither can act yet.

    The wait is reported against one of them rather than collapsing to "nothing
    can make this", which is the answer that would end the run. Which one is
    named is the first seen, so the message stays stable across samples while
    the roster does.
    """
    second = _entity(215, "builder", 4300.0, 2610.0)
    world = _sample(
        _BUILDER,
        second,
        options=(
            _option(214, "landFactory", available=False),
            _option(215, "landFactory", available=False),
        ),
    )
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert "unit 214" in decision["reason"]


def test_an_available_action_is_preferred_over_an_unavailable_one() -> None:
    """Two units offer the same type; only one can act on it now."""
    second = _entity(215, "builder", 4300.0, 2610.0)
    world = _sample(
        _BUILDER,
        second,
        options=(
            _option(214, "landFactory", available=False),
            _option(215, "landFactory"),
        ),
    )
    assert (
        decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["unit_id"]
        == 215
    )


def test_a_unit_that_rolls_out_is_produced_rather_than_placed() -> None:
    """The engine decides where a produced unit appears, so no site is chosen.

    ``placed`` is the engine's own distinction between the two verbs, read from
    the action rather than guessed from the type's speed.
    """
    centre = _entity(213, "commandCenter", 4250.0, 2550.0)
    world = _sample(centre, options=(_option(213, "builder", placed=False),))
    decision = decide(world, ("builder",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "produce"
    assert decision["unit_id"] == 213
    assert decision["type_name"] == "builder"
    assert (decision["x"], decision["y"]) == (0.0, 0.0)


def test_a_produced_unit_still_has_to_be_afforded() -> None:
    centre = _entity(213, "commandCenter", 4250.0, 2550.0)
    world = _sample(centre, credits=499, options=(_option(213, "builder", placed=False),))
    decision = decide(world, ("builder",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert decision["reason"] == "builder costs 500, holding 499"


def test_no_builder_is_blocked_not_a_wait() -> None:
    """Waiting implies it could resolve on its own; this one cannot."""
    world = _sample(_entity(213, "commandCenter"))
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "blocked"
    assert decision["reason"] == (
        "nothing the player owns can make landFactory; the plan is not playable from here"
    )


def test_a_structure_missing_from_the_catalogue_is_blocked() -> None:
    decision = decide(
        _sample(_BUILDER),
        ("teleporter",),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        _free(_sample(_BUILDER)),
    )
    assert decision["action"] == "blocked"
    assert "not in the unit catalogue" in decision["reason"]


def test_credits_are_checked_before_a_builder_is_required() -> None:
    """A plan naming an unknown structure fails on the plan, not the roster."""
    decision = decide(
        _sample(), ("teleporter",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(_sample())
    )
    assert decision["action"] == "blocked"
    assert "catalogue" in decision["reason"]


def test_a_ring_position_already_built_on_is_passed_over() -> None:
    """The site is the first *free* position, read from the world.

    Two schemes used to index this ring by counting -- the plan by its own
    position, the economy by how many immobile structures were standing, which
    counts extractors sitting on pools nowhere near it. Neither answers "which
    position is free", and two counters can land on the same slot, which the
    engine refuses silently ([[policy-loop]]).
    """
    first = (_ANCHOR["x"] + PLACEMENT_RING[0][0], _ANCHOR["y"] + PLACEMENT_RING[0][1])
    world = _sample(_ANCHOR, _BUILDER, _entity(400, "laboratory", first[0], first[1]))
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert (decision["x"], decision["y"]) == (
        _ANCHOR["x"] + PLACEMENT_RING[1][0],
        _ANCHOR["y"] + PLACEMENT_RING[1][1],
    )


def test_an_enemy_building_fills_a_ring_position_too() -> None:
    """It occupies the ground exactly as firmly, and ordering onto it spends
    the credits for nothing.
    """
    first = (_ANCHOR["x"] + PLACEMENT_RING[0][0], _ANCHOR["y"] + PLACEMENT_RING[0][1])
    theirs = _entity(900, "laboratory", first[0], first[1], mine=False)
    world = _sample(_ANCHOR, _BUILDER, theirs)
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert (decision["x"], decision["y"]) != first


def test_a_mobile_unit_standing_on_a_ring_position_does_not_fill_it() -> None:
    """It moves; a building does not. Counting it would hide a usable slot."""
    first = (_ANCHOR["x"] + PLACEMENT_RING[0][0], _ANCHOR["y"] + PLACEMENT_RING[0][1])
    world = _sample(_ANCHOR, _BUILDER, _entity(401, "c_tank", first[0], first[1]))
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert (decision["x"], decision["y"]) == first


def test_a_full_ring_waits_rather_than_ordering_onto_a_building() -> None:
    """A structure destroyed frees its slot, so the world can leave this state."""
    filled = [
        _entity(400 + i, "laboratory", _ANCHOR["x"] + dx, _ANCHOR["y"] + dy)
        for i, (dx, dy) in enumerate(PLACEMENT_RING)
    ]
    world = _sample(_ANCHOR, _BUILDER, *filled)
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert "all are taken" in decision["reason"]


def test_the_free_slot_is_found_through_a_partly_filled_ring() -> None:
    """Not merely the first or the last: the first one actually free."""
    filled = [
        _entity(400 + i, "laboratory", _ANCHOR["x"] + dx, _ANCHOR["y"] + dy)
        for i, (dx, dy) in enumerate(PLACEMENT_RING[:3])
    ]
    world = _sample(_ANCHOR, _BUILDER, *filled)
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert (decision["x"], decision["y"]) == (
        _ANCHOR["x"] + PLACEMENT_RING[3][0],
        _ANCHOR["y"] + PLACEMENT_RING[3][1],
    )


def test_completed_count_matches_each_plan_entry_only_once() -> None:
    """Two factories satisfy two plan entries, not one entry twice."""
    world = _sample(_BUILDER, _entity(300, "landFactory"), _entity(301, "landFactory"))
    assert completed_count(world, ("landFactory", "landFactory")) == 2
    assert completed_count(world, ("landFactory",)) == 1


def test_a_structure_outside_the_plan_does_not_count_as_progress() -> None:
    world = _sample(_BUILDER, _entity(300, "laboratory"))
    assert completed_count(world, ("landFactory",)) == 0


def test_an_entry_with_nowhere_to_stand_defers_to_the_next() -> None:
    """The fault that lost two duels while the bot was level on worth.

    The opening is a sequence -- extractors, then the factory, then the army --
    and it waited on whichever entry it had reached. With every pool taken the
    wait was permanent: the factory was never built, no army was ever produced,
    and a match ended with five idle builders, 60,676 credits and nothing to
    fight with, against an opponent left to do as it liked
    ([[policy-holding-ground]]).

    The extractor here has no pool, so the plan reaches past it to the factory
    rather than stopping.
    """
    world = _sample(_BUILDER, _ANCHOR, pools=())
    decision = decide(
        world,
        ("extractorT1", "landFactory"),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        _free(world),
    )
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_a_deferred_entry_is_still_wanted_once_a_pool_frees() -> None:
    """Deferring is not dropping: progress is read off the roster every
    observation, so the extractor is taken the moment a pool is claimable.
    """
    world = _sample(_BUILDER, _ANCHOR, pools=(_pool(0, 15, 0),))
    decision = decide(
        world,
        ("extractorT1", "landFactory"),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        _free(world),
    )
    assert decision["type_name"] == "extractorT1"


def test_every_entry_unplaceable_still_waits_and_says_why() -> None:
    """Deferring past the last entry is not a decision, so it reports the
    reason the first one could not be placed.
    """
    world = _sample(_BUILDER, _ANCHOR, pools=())
    decision = decide(world, ("extractorT1",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert "resource pool" in decision["reason"]


def test_an_unaffordable_entry_still_stops_the_plan() -> None:
    """Only *placement* defers. Being short of credits is a condition on the
    whole plan, so reaching past it would spend the next entry's credits on the
    strength of this one's being short.
    """
    world = _sample(_BUILDER, _ANCHOR, credits=10, pools=(_pool(0, 15, 0),))
    decision = decide(
        world,
        ("extractorT1", "landFactory"),
        _CATALOGUE,
        _PLACEMENTS,
        _PROFILES,
        _free(world),
    )
    assert decision["action"] == "wait"
    assert "costs 700" in decision["reason"]


def test_an_upgraded_structure_still_satisfies_the_entry_that_built_it() -> None:
    """The regression that cost a match, asked of both readings of progress.

    The plan asks for three ``extractorT1``. They upgrade themselves in place.
    Matching type names exactly, the plan then saw none of them, ordered three
    more, and did that forever -- so the one builder was never free, expansion
    never ran, and the match ended defeated with 41,559 credits banked and the
    plan reading 0 of 8 ([[policy-holding-ground]]).

    Both functions are asserted because they answer different questions off the
    same roster, and it was the disagreement between them that hid the fault:
    a count that advances while the index does not is a plan that reports
    progress it will not act on.
    """
    upgraded = _sample(
        _BUILDER,
        _entity(300, "extractorT2", 100.0, 100.0),
        _entity(301, "extractorT3", 200.0, 200.0),
    )
    plan = ("extractorT1", "extractorT1")
    assert completed_count(upgraded, plan) == 2
    assert next_unsatisfied_index(upgraded, plan) == 2


def test_an_earlier_tier_does_not_satisfy_a_later_one() -> None:
    """The relation is one-way, or the plan would skip work it has not done.

    Holding a tier one is not an answer to a plan asking for a tier two. Were
    it symmetric, a plan naming ``extractorT2`` would call itself satisfied by
    the tier one already standing and never upgrade anything.
    """
    world = _sample(_BUILDER, _entity(300, "extractorT1", 100.0, 100.0))
    assert completed_count(world, ("extractorT2",)) == 0
    assert next_unsatisfied_index(world, ("extractorT2",)) == 0


def test_an_upgrade_satisfies_one_entry_rather_than_every_entry_it_outranks() -> None:
    """One structure, one entry -- the tier rule must not double-count.

    A single ``extractorT3`` outranks both entries of a two-extractor plan. It
    is still one structure earning one structure's income, so it satisfies the
    first entry and leaves the second wanting.
    """
    world = _sample(_BUILDER, _entity(300, "extractorT3", 100.0, 100.0))
    plan = ("extractorT1", "extractorT1")
    assert completed_count(world, plan) == 1
    assert next_unsatisfied_index(world, plan) == 1


def test_no_free_worker_means_no_placed_order() -> None:
    """Which workers are free is the loop's judgement, and this is what it buys.

    A structure is only ever ordered from a worker the loop reports as free, so
    two rules can no longer re-task the same one off each other
    ([[policy-loop]]).
    """
    world = _sample(_BUILDER)
    assert (
        decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, ())["action"]
        == "blocked"
    )
    assert (
        decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["action"]
        == "build"
    )


def test_an_enemy_structure_does_not_advance_the_plan() -> None:
    """The stream carries enemies, so ownership is what makes progress mine."""
    world = _sample(_BUILDER, _entity(900, "landFactory", mine=False))
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_an_enemy_builder_is_never_selected() -> None:
    """Ordering a unit we do not own would be rejected by the engine anyway."""
    world = _sample(_entity(901, "builder", 1.0, 2.0, mine=False))
    assert _free(world) == ()
    assert (
        decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["action"]
        == "blocked"
    )


def test_an_owned_structure_still_counts_when_an_enemy_has_one_too() -> None:
    world = _sample(
        _BUILDER,
        _entity(300, "landFactory"),
        _entity(900, "landFactory", mine=False),
    )
    assert completed_count(world, ("landFactory", "landFactory")) == 1


def test_a_plan_naming_a_starting_unit_does_not_skip_earlier_entries() -> None:
    """The count-as-index conflation, which silently skipped a whole entry.

    Every game starts with a builder. Under the old reading, the plan
    ``("landFactory", "builder")`` counted as one-satisfied, jumped to index 1,
    built a second builder and never built the factory at all.
    """
    world = _sample(_entity(214, "builder"), credits=10_000)
    plan = ("landFactory", "builder")

    assert completed_count(world, plan) == 1
    assert next_unsatisfied_index(world, plan) == 0

    decision = decide(world, plan, _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_the_first_unsatisfied_entry_is_found_past_a_satisfied_one() -> None:
    world = _sample(_entity(214, "builder"), _entity(300, "landFactory"), credits=10_000)
    assert next_unsatisfied_index(world, ("landFactory", "builder")) == 2
    assert next_unsatisfied_index(world, ("landFactory", "builder", "landFactory")) == 2


def test_an_enemy_unit_never_satisfies_a_plan_entry() -> None:
    world = _sample(_entity(900, "landFactory", mine=False), credits=10_000)
    assert next_unsatisfied_index(world, ("landFactory",)) == 0


def test_placement_is_measured_from_a_fixed_structure_not_the_builder() -> None:
    """The ring only spreads if its centre holds still.

    Measuring from the builder collapsed the spread, because the builder walks
    to each site it is sent to. Observed live: the first factory landed at
    (4450, 2730) and the third order went to (4451, 2646) -- 84 apart, close
    enough to overlap, and the engine silently refused it.
    """
    command_centre = _entity(213, "commandCenter", 4250.0, 2550.0)
    builder = _entity(214, "builder", 4250.0, 2610.0)
    plan = ("landFactory", "landFactory", "landFactory")

    owned = [command_centre, builder]
    sites: list[tuple[float, float]] = []
    for _step in range(3):
        world = _sample(*owned, credits=99_999)
        decision = decide(world, plan, _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
        assert decision["action"] == "build"
        sites.append((decision["x"], decision["y"]))
        # The builder ends each build standing at the site it just built.
        owned = [
            command_centre,
            _entity(214, "builder", decision["x"], decision["y"]),
            *[_entity(300 + i, "landFactory", x, y) for i, (x, y) in enumerate(sites)],
        ]

    assert len(set(sites)) == 3
    first_to_third = abs(sites[0][1] - sites[2][1]) + abs(sites[0][0] - sites[2][0])
    assert first_to_third == 240.0


def test_the_anchor_is_the_oldest_owned_immobile_structure() -> None:
    """The factory is listed first, so first-seen would pick the wrong one."""
    command_centre = _entity(213, "commandCenter", 1.0, 2.0)
    world = _sample(
        _entity(400, "landFactory", 9.0, 9.0),
        command_centre,
        _entity(214, "builder", 5.0, 5.0),
        credits=10_000,
    )
    assert find_anchor(world, _CATALOGUE) == command_centre


def test_a_mobile_unit_is_never_the_anchor() -> None:
    """Immobility is read from the catalogue, not guessed from the type name."""
    world = _sample(_entity(214, "builder", 5.0, 5.0), credits=10_000)
    assert find_anchor(world, _CATALOGUE) is None


def test_an_enemy_structure_is_never_the_anchor() -> None:
    world = _sample(
        _entity(1, "commandCenter", 0.0, 0.0, mine=False),
        _entity(214, "builder", 5.0, 5.0),
        credits=10_000,
    )
    assert find_anchor(world, _CATALOGUE) is None


def test_with_no_structure_owned_the_builder_is_the_reference() -> None:
    """A player who has lost every building must still be able to rebuild."""
    world = _sample(_BUILDER, credits=10_000)
    decision = decide(world, ("landFactory",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "build"
    assert (decision["x"], decision["y"]) == (
        _BUILDER["x"] + PLACEMENT_RING[0][0],
        _BUILDER["y"] + PLACEMENT_RING[0][1],
    )


def test_an_extractor_is_placed_on_a_pool_and_not_on_the_ring() -> None:
    """The ring is not a legal site for it, so offering one would be refused."""
    pool = _pool(0, 200, 130)
    world = _sample(_ANCHOR, _BUILDER, pools=(pool,), credits=10_000)
    decision = decide(world, ("extractorT1",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "build"
    assert decision["type_name"] == "extractorT1"
    assert (decision["x"], decision["y"]) == (pool["x"], pool["y"])


def test_the_nearest_free_pool_to_the_anchor_is_chosen() -> None:
    """Nearest to the base, so the economy grows outward rather than wandering."""
    near = _pool(0, 220, 130)
    far = _pool(1, 10, 10)
    world = _sample(_ANCHOR, _BUILDER, pools=(far, near), credits=10_000)
    decision = decide(world, ("extractorT1",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert (decision["x"], decision["y"]) == (near["x"], near["y"])


def test_a_pool_under_a_structure_is_not_offered_again() -> None:
    taken = _pool(0, 220, 130)
    free = _pool(1, 230, 130)
    standing = _entity(400, "extractorT1", taken["x"], taken["y"])
    world = _sample(_ANCHOR, _BUILDER, standing, pools=(taken, free), credits=10_000)
    decision = decide(
        world, ("extractorT1", "extractorT1"), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world)
    )
    assert (decision["x"], decision["y"]) == (free["x"], free["y"])


def test_an_enemy_extractor_holds_a_pool_just_as_firmly() -> None:
    """Ownership is irrelevant to whether the ground is free."""
    taken = _pool(0, 220, 130)
    enemy = _entity(900, "extractorT1", taken["x"], taken["y"], mine=False)
    world = _sample(_ANCHOR, _BUILDER, enemy, pools=(taken,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES) == {
        "pool": None,
        "visible": 1,
        "occupied": 1,
        "unreachable": 0,
        "exposed": 0,
    }


def test_a_pool_another_worker_is_walking_to_is_not_free() -> None:
    """The defect that ate nineteen expansion orders in twenty.

    Occupancy is judged by what is *standing* on a pool, so one a builder is
    merely walking toward reads as free. With a single free worker that was
    nearly harmless -- one order in flight at a time. The moment several workers
    were freed ([[policy-economy]]) each was offered the same nearest pool on
    successive observations, because none had arrived yet: an instrumented run
    granted **23 extractor orders, lost nothing at all, and finished with four
    extractors** ([[policy-holding-ground]]).
    """
    taken = _pool(0, 220, 130)
    spare = _pool(1, 600, 130)
    world = _sample(_ANCHOR, _BUILDER, pools=(taken, spare), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES)["pool"] == taken
    # Somebody is already on their way to the near one, so the next worker is
    # offered the far one rather than the same site again.
    claimed = ((taken["x"], taken["y"]),)
    survey = survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES, claimed)
    assert survey["pool"] == spare
    # Counted as occupied rather than given a category of its own: from here,
    # "somebody already has this pool" is one fact.
    assert survey["occupied"] == 1


def test_every_pool_claimed_leaves_nothing_to_take() -> None:
    """The complement, and what stops the fix hiding the map from itself."""
    only = _pool(0, 220, 130)
    world = _sample(_ANCHOR, _BUILDER, pools=(only,), credits=10_000)
    survey = survey_pools(
        world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES, ((only["x"], only["y"]),)
    )
    assert survey["pool"] is None
    assert survey["occupied"] == 1


def test_a_claim_far_from_a_pool_leaves_it_free() -> None:
    """A worker building a factory on the ring must not burn a pool it is near."""
    pool = _pool(0, 220, 130)
    world = _sample(_ANCHOR, _BUILDER, pools=(pool,), credits=10_000)
    elsewhere = ((pool["x"] + POOL_OCCUPIED_RADIUS + 0.5, pool["y"]),)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES, elsewhere)["pool"] == pool


def test_a_builder_parked_on_a_pool_does_not_occupy_it() -> None:
    """It stands there after every build; counting it would burn the pool."""
    pool = _pool(0, 220, 130)
    parked = _entity(214, "builder", pool["x"], pool["y"])
    world = _sample(_ANCHOR, parked, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, parked, _CATALOGUE, _PROFILES)["pool"] == pool


def test_a_structure_outside_the_occupancy_radius_leaves_a_pool_free() -> None:
    pool = _pool(0, 220, 130)
    beyond = _entity(400, "landFactory", pool["x"] + POOL_OCCUPIED_RADIUS + 0.5, pool["y"])
    world = _sample(_ANCHOR, _BUILDER, beyond, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES)["pool"] == pool


def test_a_structure_exactly_at_the_occupancy_radius_takes_the_pool() -> None:
    """The boundary is inclusive, and which side it falls on is a real choice."""
    pool = _pool(0, 220, 130)
    astride = _entity(400, "landFactory", pool["x"] + POOL_OCCUPIED_RADIUS, pool["y"])
    world = _sample(_ANCHOR, _BUILDER, astride, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES)["pool"] is None


def test_a_type_the_catalogue_does_not_know_leaves_the_pool_free() -> None:
    """A wrong 'free' costs one refused order; a wrong 'taken' hides it for good."""
    pool = _pool(0, 220, 130)
    unknown = _entity(400, "someModStructure", pool["x"], pool["y"])
    world = _sample(_ANCHOR, _BUILDER, unknown, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES)["pool"] == pool


def test_a_pool_across_water_is_not_offered_to_a_land_builder() -> None:
    """The failure the whole reachability check exists for.

    Twelve of the forty-six pools on the archived map sit in components the
    mainland cannot walk to ([[mechanics-movement-layers]]). Distance alone
    would send a builder at one of them the moment the near ground filled up.
    """
    here = _pool(0, 220, 130)
    across = _pool(1, 12, 52)
    across["group_land"] = _ISLAND
    world = _sample(_ANCHOR, _BUILDER, pools=(across, here), credits=10_000)
    survey = survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES)
    assert survey["pool"] == here
    assert survey["unreachable"] == 1


def test_a_pool_with_no_land_component_at_all_is_not_offered() -> None:
    """A negative id is the engine saying there is no component, not an id.

    Comparing two of them for equality is how the engine's own predicate
    answers true for a point it could not place at all; refusing every negative
    is the more conservative reading.
    """
    nowhere = _pool(0, 220, 130)
    nowhere["group_land"] = -1
    world = _sample(_ANCHOR, _BUILDER, pools=(nowhere,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _CATALOGUE, _PROFILES)["unreachable"] == 1


def test_a_builder_off_the_land_grid_is_not_offered_a_pool_it_cannot_be_judged_for() -> None:
    """Its own component id belongs to a different grid, so it has none here."""
    stranded = _entity(214, "builder", 4250.0, 2610.0, group=-1)
    world = _sample(_ANCHOR, stranded, pools=(_pool(0, 220, 130),), credits=10_000)
    assert survey_pools(world, _ANCHOR, stranded, _CATALOGUE, _PROFILES)["unreachable"] == 1


def test_a_builder_on_another_layer_refuses_the_pool_rather_than_guessing() -> None:
    """No special case, and none needed.

    A hover unit's component id indexes the hover grid, so it matches no land
    component and the pool is simply refused. The safe direction falls out of
    the comparison instead of being arranged by a branch.
    """
    hover = _entity(214, "builder", 4250.0, 2610.0, movement="HOVER", group=99)
    world = _sample(_ANCHOR, hover, pools=(_pool(0, 220, 130),), credits=10_000)
    survey = survey_pools(world, _ANCHOR, hover, _CATALOGUE, _PROFILES)
    assert survey["pool"] is None
    assert survey["unreachable"] == 1


def test_a_pool_inside_an_enemy_gun_is_not_offered() -> None:
    pool = _pool(0, 220, 130)
    turret = _entity(900, "turret", pool["x"] + 50.0, pool["y"], mine=False)
    world = _sample(_ANCHOR, _BUILDER, turret, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES) == {
        "pool": None,
        "visible": 1,
        "occupied": 0,
        "unreachable": 0,
        "exposed": 1,
    }


def test_a_pool_is_rejected_for_the_walk_even_when_the_pool_itself_is_safe() -> None:
    """The failure this rule exists for: the builder died in transit, not on arrival.

    The turret sits beside the midpoint of the walk and nowhere near either end,
    so a check that only looked at the destination would send the builder
    straight past it.
    """
    pool = _pool(0, 220, 130)
    midpoint = ((_BUILDER["x"] + pool["x"]) / 2, (_BUILDER["y"] + pool["y"]) / 2)
    ambush = _entity(900, "turret", midpoint[0], midpoint[1] + 50.0, mine=False)
    world = _sample(_ANCHOR, _BUILDER, ambush, pools=(pool,), credits=10_000)

    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] is None
    # ... and the same turret standing at the same distance from the pool, but
    # behind the builder rather than between the two, rules out nothing.
    behind = _entity(900, "turret", _BUILDER["x"], _BUILDER["y"] - 400.0, mine=False)
    clear = _sample(_ANCHOR, _BUILDER, behind, pools=(pool,), credits=10_000)
    assert survey_pools(clear, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] == pool


def test_a_pool_exactly_at_the_edge_of_a_gun_is_rejected() -> None:
    """The boundary is inclusive: a unit at maximum range is a unit in range."""
    pool = _pool(0, 220, 130)
    turret = _entity(900, "turret", pool["x"] + _TURRET_RANGE, pool["y"], mine=False)
    world = _sample(_ANCHOR, _BUILDER, turret, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] is None


def test_a_pool_beyond_every_gun_is_offered() -> None:
    pool = _pool(0, 220, 130)
    turret = _entity(900, "turret", pool["x"] + _TURRET_RANGE + 0.5, pool["y"], mine=False)
    world = _sample(_ANCHOR, _BUILDER, turret, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] == pool


def test_an_unarmed_enemy_standing_on_the_route_is_not_a_threat() -> None:
    """An enemy builder is an obstacle, not a gun, and ruling out ground it
    happens to stand on would concede the map to something that cannot shoot."""
    pool = _pool(0, 220, 130)
    harmless = _entity(900, "builder", pool["x"] + 10.0, pool["y"], mine=False)
    world = _sample(_ANCHOR, _BUILDER, harmless, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] == pool


def test_an_ally_is_not_a_threat_even_though_it_is_not_mine() -> None:
    """Hostility is the engine's answer, not the negation of ownership."""
    pool = _pool(0, 220, 130)
    ally = _entity(900, "turret", pool["x"] + 50.0, pool["y"], mine=False, hostile=False)
    world = _sample(_ANCHOR, _BUILDER, ally, pools=(pool,), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] == pool


def test_the_nearest_safe_pool_beats_a_nearer_exposed_one() -> None:
    """Threat filters before distance ranks, which is the whole ordering.

    The two pools are deliberately not collinear with the builder. When they
    are, the nearer one lies on the walk to the farther one and covering the
    first necessarily covers the route to the second — so a test laid out that
    way could never distinguish the rule from a blanket refusal.
    """
    near = _pool(0, 220, 130)
    far = _pool(1, 220, 90)
    turret = _entity(900, "turret", near["x"], near["y"] + 50.0, mine=False)
    world = _sample(_ANCHOR, _BUILDER, turret, pools=(near, far), credits=10_000)
    assert survey_pools(world, _ANCHOR, _BUILDER, _ARMED, _PROFILES)["pool"] == far


def test_an_extractor_with_every_pool_taken_waits_rather_than_blocking() -> None:
    """Fog lifts and extractors die, so the world can resolve this on its own."""
    taken = _pool(0, 220, 130)
    standing = _entity(400, "extractorT1", taken["x"], taken["y"])
    world = _sample(_ANCHOR, _BUILDER, standing, pools=(taken,), credits=10_000)
    decision = decide(
        world, ("extractorT1", "extractorT1"), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world)
    )
    assert decision["action"] == "wait"
    assert decision["reason"] == (
        "extractorT1 needs a resource pool: of the 1 in sight, 1 are built on, "
        "0 cannot be walked to and 0 can only be reached through enemy fire"
    )


def test_the_wait_reason_separates_a_taken_pool_from_a_covered_one() -> None:
    """Two different games. One is progress, the other is losing ground."""
    taken = _pool(0, 220, 130)
    covered = _pool(1, 260, 130)
    standing = _entity(400, "extractorT1", taken["x"], taken["y"])
    turret = _entity(900, "turret", covered["x"], covered["y"] + 50.0, mine=False)
    world = _sample(_ANCHOR, _BUILDER, standing, turret, pools=(taken, covered), credits=10_000)
    decision = decide(
        world, ("extractorT1", "extractorT1"), _ARMED, _PLACEMENTS, _PROFILES, _free(world)
    )
    assert decision["action"] == "wait"
    assert decision["reason"] == (
        "extractorT1 needs a resource pool: of the 2 in sight, 1 are built on, "
        "0 cannot be walked to and 1 can only be reached through enemy fire"
    )


def test_an_extractor_with_no_pool_visible_waits() -> None:
    world = _sample(_ANCHOR, _BUILDER, credits=10_000)
    decision = decide(world, ("extractorT1",), _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))
    assert decision["action"] == "wait"
    assert decision["reason"] == "extractorT1 needs a resource pool and none is visible yet"


def test_a_type_absent_from_the_placement_dump_is_blocked() -> None:
    """Where it may stand is unknown, which is not the same as unconstrained."""
    catalogue = dict(_CATALOGUE)
    catalogue["teleporter"] = _unit("teleporter", 100)
    placements = {n: p for n, p in _PLACEMENTS.items() if n != "teleporter"}
    decision = decide(_sample(_BUILDER), ("teleporter",), catalogue, placements, _PROFILES)
    assert decision["action"] == "blocked"
    assert decision["reason"] == (
        "'teleporter' is not in the placement dump, so where it may stand is unknown"
    )


def test_a_pool_placement_ignores_the_ring_index_entirely() -> None:
    """Two extractors in a row must not drift apart the way ring entries do."""
    first = _pool(0, 220, 130)
    second = _pool(1, 221, 130)
    world = _sample(_ANCHOR, _BUILDER, pools=(first, second), credits=10_000)
    plan = ("landFactory", "extractorT1")
    built = _sample(
        _ANCHOR,
        _BUILDER,
        _entity(300, "landFactory", 9000.0, 9000.0),
        pools=(first, second),
        credits=10_000,
    )
    decision = decide(built, plan, _CATALOGUE, _PLACEMENTS, _PROFILES, _free(built))
    assert (decision["x"], decision["y"]) == (first["x"], first["y"])
    assert (
        decide(world, plan, _CATALOGUE, _PLACEMENTS, _PROFILES, _free(world))["type_name"]
        == "landFactory"
    )
