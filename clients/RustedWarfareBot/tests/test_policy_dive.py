"""The dive, exercised without a match running.

What is tested: the party closes on the visible hostile ground MOVER whose
land gun outranges every land gun the army fields, nearest the party's
own centre; a gun that does not outrange, a standing structure, a flier,
a ship and an unpriced type are never quarry; an army with no land gun
has no standoff to close; the draft takes the fastest gathered; orders
are never re-sent while the objective holds; survivors below strength
disband home; and with no gun in sight a standing party stands down and
none is raised -- the identity property on a zero-artillery seed
([[policy-raid]], [[issuing-orders]]).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.dive import Diver, outranging_guns
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import enemy, entity, profile, sample


def _stats(type_name: str, speed: float) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=350,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE: dict[str, UnitStats] = {
    "commandCenter": _stats("commandCenter", 0.0),
    "c_tank": _stats("c_tank", 1.1),
    "hoverTank": _stats("hoverTank", 2.0),
    "builder": _stats("builder", 1.0),
    "c_artillery": _stats("c_artillery", 0.7),
    "c_turret_t1_artillery": _stats("c_turret_t1_artillery", 0.0),
    "gunShip": _stats("gunShip", 2.5),
    "battleShip": _stats("battleShip", 0.9),
}

_PROFILES: dict[str, CombatProfile] = {
    "c_tank": profile("c_tank", 130.0),
    "hoverTank": profile("hoverTank", 140.0),
    "builder": profile("builder", 0.0, land=False),
    "c_artillery": profile("c_artillery", 290.0),
    "c_turret_t1_artillery": profile("c_turret_t1_artillery", 400.0),
    "gunShip": profile("gunShip", 300.0),
    "battleShip": profile("battleShip", 350.0),
}

_CENTRE = entity(1, "commandCenter", x=0.0, y=0.0)


def _tank(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


def _hover(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "hoverTank", x=x, y=y)


def _gun(unit_id: int, x: float) -> Entity:
    return enemy(unit_id, "c_artillery", x=x)


def _world(*extra: Entity) -> Sample:
    return sample(_CENTRE, *extra)


def test_only_outranging_ground_movers_are_guns() -> None:
    """The outranged clause's membership test, read for entities: beyond
    the army's longest land gun, moving, on the ground, priced."""
    army = (_tank(20), _hover(21))
    targets = (
        _gun(9, 400.0),
        enemy(8, "c_tank", x=100.0),
        enemy(7, "c_turret_t1_artillery", x=200.0),
        enemy(6, "gunShip", x=300.0, flying=True),
        enemy(5, "battleShip", x=300.0, movement="WATER"),
        enemy(4, "mystery", x=50.0),
    )
    assert [g["unit_id"] for g in outranging_guns(army, targets, _PROFILES, _CATALOGUE, 0.0)] == [9]


def test_a_tie_is_a_fair_fight_not_a_standoff() -> None:
    """A hover at 140 against the tank's 130: the reach is the army's longest."""
    army = (_tank(20), _hover(21))
    targets = (enemy(9, "hoverTank", x=400.0),)
    assert outranging_guns(army, targets, _PROFILES, _CATALOGUE, 0.0) == ()


def test_an_army_with_no_land_gun_has_no_standoff() -> None:
    army = (entity(20, "builder"),)
    assert outranging_guns(army, (_gun(9, 400.0),), _PROFILES, _CATALOGUE, 0.0) == ()


def test_the_nearest_gun_is_closed_on_by_the_fastest_party() -> None:
    """Two hovers behind three tanks by id: the hovers go, at the nearer gun."""
    diver = Diver(size=2)
    army = (_tank(20), _tank(21), _tank(22), _hover(31), _hover(30))
    targets = (_gun(9, 900.0), _gun(8, 400.0))
    orders = diver.dive(_world(*army), army, targets, _CATALOGUE, _PROFILES, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(30, 400.0), (31, 400.0)]
    assert diver.party() == frozenset({30, 31})
    assert diver.objectives == 1
    assert diver.marches == 2


def test_no_gun_in_sight_raises_no_party() -> None:
    """The identity property: a tank in sight is the waves' business."""
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    targets = (enemy(9, "c_tank", x=400.0),)
    assert diver.dive(_world(*army), army, targets, _CATALOGUE, _PROFILES, True) == ()
    assert diver.party() == frozenset()
    assert diver.objectives == 0


def test_no_anchor_is_no_dive() -> None:
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    assert diver.dive(sample(*army), army, (_gun(9, 400.0),), _CATALOGUE, _PROFILES, True) == ()


def test_orders_are_not_resent_while_the_objective_holds() -> None:
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    world = _world(*army)
    targets = (_gun(9, 400.0),)
    diver.dive(world, army, targets, _CATALOGUE, _PROFILES, True)
    assert diver.dive(world, army, targets, _CATALOGUE, _PROFILES, True) == ()
    assert diver.marches == 2


def test_a_nearer_gun_retargets_the_party_measured_from_the_party() -> None:
    diver = Diver(size=2)
    home = (_hover(30), _hover(31))
    diver.dive(_world(*home), home, (_gun(9, 1050.0),), _CATALOGUE, _PROFILES, True)
    # Walked deep: 1050 is 50 from the party and 200 is 800 away, so the
    # objective holds and nothing re-sends.
    away = (_hover(30, x=1000.0), _hover(31, x=1000.0))
    held = diver.dive(
        _world(*away), away, (_gun(9, 1050.0), _gun(7, 200.0)), _CATALOGUE, _PROFILES, True
    )
    assert held == ()
    assert diver.objectives == 1
    # Back home, the gun at 200 is the nearer one and the party turns.
    turned = diver.dive(
        _world(*home), home, (_gun(9, 1050.0), _gun(7, 200.0)), _CATALOGUE, _PROFILES, True
    )
    assert [(o["unit_id"], o["x"]) for o in turned] == [(30, 200.0), (31, 200.0)]
    assert diver.objectives == 2


def test_survivors_below_strength_disband_and_fight_home() -> None:
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    world = _world(*army)
    targets = (_gun(9, 400.0),)
    diver.dive(world, army, targets, _CATALOGUE, _PROFILES, True)
    reduced = (_hover(31),)
    orders = diver.dive(_world(*reduced), reduced, targets, _CATALOGUE, _PROFILES, True)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [(31, 0.0, 0.0)]
    assert diver.party() == frozenset()


def test_a_standing_party_stands_down_when_the_gun_is_gone() -> None:
    """A diversion with no objective goes home and rejoins the reserve."""
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    world = _world(*army)
    diver.dive(world, army, (_gun(9, 400.0),), _CATALOGUE, _PROFILES, True)
    orders = diver.dive(world, army, (), _CATALOGUE, _PROFILES, True)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [(30, 0.0, 0.0), (31, 0.0, 0.0)]
    assert diver.party() == frozenset()
    # The gun returns: a fresh party, a fresh objective.
    again = diver.dive(world, army, (_gun(9, 400.0),), _CATALOGUE, _PROFILES, True)
    assert len(again) == 2
    assert diver.objectives == 2


def test_a_party_that_died_whole_leaves_no_ghosts() -> None:
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    diver.dive(_world(*army), army, (_gun(9, 400.0),), _CATALOGUE, _PROFILES, True)
    rest = (_tank(20),)
    assert diver.dive(_world(*rest), rest, (_gun(9, 400.0),), _CATALOGUE, _PROFILES, False) == ()
    assert diver.party() == frozenset()


def test_no_draft_without_the_campaigns_leave() -> None:
    diver = Diver(size=2)
    army = (_hover(30), _hover(31))
    assert diver.dive(_world(*army), army, (_gun(9, 400.0),), _CATALOGUE, _PROFILES, False) == ()
    assert diver.party() == frozenset()


def test_the_default_size_is_the_engines_first_group() -> None:
    assert Diver().size == 3
    assert Diver().margin == 0.0


def test_the_margin_separates_a_standoff_from_a_technicality() -> None:
    """dive16's lesson: against this fixture's 140 line, a 190 gun
    outranges by a technicality and artillery at 290 by a standoff; at
    margin 100 only the artillery is quarry, and at zero both are."""
    army = (_tank(20), _hover(21))
    targets = (enemy(8, "sniper", x=300.0), _gun(9, 400.0))
    catalogue = {**_CATALOGUE, "sniper": _stats("sniper", 0.8)}
    profiles = {**_PROFILES, "sniper": profile("sniper", 190.0)}
    both = outranging_guns(army, targets, profiles, catalogue, 0.0)
    assert [g["unit_id"] for g in both] == [8, 9]
    artillery_only = outranging_guns(army, targets, profiles, catalogue, 100.0)
    assert [g["unit_id"] for g in artillery_only] == [9]
    # The line here is the hover's 140, so 160 puts the threshold at 300,
    # past the artillery too.
    assert outranging_guns(army, targets, profiles, catalogue, 160.0) == ()


def test_the_diver_dives_by_its_own_margin() -> None:
    """A margin-100 diver raises nothing against the 190 sniper and a
    party against the 290 gun; the sniper never draws it."""
    army = (_hover(30), _hover(31))
    catalogue = {**_CATALOGUE, "sniper": _stats("sniper", 0.8)}
    profiles = {**_PROFILES, "sniper": profile("sniper", 190.0)}
    diver = Diver(size=2, margin=100.0)
    sniper = (enemy(8, "sniper", x=300.0),)
    assert diver.dive(_world(*army), army, sniper, catalogue, profiles, True) == ()
    assert diver.party() == frozenset()
    orders = diver.dive(_world(*army), army, (*sniper, _gun(9, 900.0)), catalogue, profiles, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(30, 900.0), (31, 900.0)]
