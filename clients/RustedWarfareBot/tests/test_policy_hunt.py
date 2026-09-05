"""The hunt, exercised without a match running.

What is tested: the party presses the visible hostile MOVER nearest its
own centre and only movers qualify while any are in sight, the remembered
structure nearest the party is the fallback objective so an empty horizon
walks the party toward the enemy's base, the party discipline is the
raid's (drafted whole from the gathered, disbanded home under strength),
orders are never re-sent while the objective holds, and the stand-down
recall dissolves the party home exactly once ([[issuing-orders]],
[[engine-ai-triggers]]).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.hunt import Hunter
from rw_bot.policy.intel import Intel
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import enemy, entity, sample


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
    "c_turret_t1": _stats("c_turret_t1", 0.0),
}

_CENTRE = entity(1, "commandCenter", x=0.0, y=0.0)


def _tank(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


def _world(*extra: Entity) -> Sample:
    return sample(_CENTRE, *extra)


def _seen(intel: Intel, *hostiles: Entity, frame: int = 100) -> Intel:
    intel.observe(sample(*hostiles, frame=frame))
    return intel


def test_the_nearest_visible_mover_is_pressed_by_a_whole_party() -> None:
    """Movers first, distance from the party's own centre, drafted whole."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    targets = (enemy(9, "c_tank", x=400.0), enemy(8, "c_tank", x=900.0))
    orders = hunter.press(_world(*army), Intel(), army, targets, _CATALOGUE, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(20, 400.0), (21, 400.0)]
    assert hunter.party() == frozenset({20, 21})
    assert hunter.hunts == 1
    assert hunter.marches == 2


def test_buildings_are_not_quarry_while_any_mover_stands() -> None:
    """A nearer turret loses to a farther tank: the hunt bleeds groups, not walls."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    targets = (enemy(9, "c_turret_t1", x=200.0), enemy(8, "c_tank", x=900.0))
    orders = hunter.press(_world(*army), Intel(), army, targets, _CATALOGUE, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(20, 900.0), (21, 900.0)]


def test_an_unknown_type_is_not_a_mover() -> None:
    """A hostile the catalogue cannot price cannot prove it moves."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    targets = (enemy(9, "mystery", x=100.0), enemy(8, "c_tank", x=900.0))
    orders = hunter.press(_world(*army), Intel(), army, targets, _CATALOGUE, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(20, 900.0), (21, 900.0)]


def test_an_empty_horizon_pushes_at_the_nearest_memory() -> None:
    """Nothing moving in sight: the party walks toward the enemy's base."""
    intel = _seen(Intel(), enemy(9, "c_turret_t1", x=1200.0), enemy(8, "c_turret_t1", x=2000.0))
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    orders = hunter.press(_world(*army), intel, army, (), _CATALOGUE, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(20, 1200.0), (21, 1200.0)]
    assert hunter.hunts == 1


def test_nothing_seen_and_nothing_remembered_is_no_hunt() -> None:
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    assert hunter.press(_world(*army), Intel(), army, (), _CATALOGUE, True) == ()
    assert hunter.party() == frozenset({20, 21})
    assert hunter.hunts == 0


def test_no_anchor_is_no_hunt() -> None:
    """Without the gathering ground there is nothing to draft from or return to."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    targets = (enemy(9, "c_tank", x=400.0),)
    assert hunter.press(sample(*army), Intel(), army, targets, _CATALOGUE, True) == ()


def test_orders_are_not_resent_while_the_objective_holds() -> None:
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    targets = (enemy(9, "c_tank", x=400.0),)
    world = _world(*army)
    hunter.press(world, Intel(), army, targets, _CATALOGUE, True)
    assert hunter.press(world, Intel(), army, targets, _CATALOGUE, True) == ()
    assert hunter.marches == 2


def test_a_nearer_mover_retargets_the_party() -> None:
    """Pursuit is per-observation: the quarry is whoever is nearest NOW."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    world = _world(*army)
    hunter.press(world, Intel(), army, (enemy(9, "c_tank", x=400.0),), _CATALOGUE, True)
    orders = hunter.press(
        world,
        Intel(),
        army,
        (enemy(9, "c_tank", x=400.0), enemy(7, "c_tank", x=90.0)),
        _CATALOGUE,
        True,
    )
    assert [(o["unit_id"], o["x"]) for o in orders] == [(20, 90.0), (21, 90.0)]
    assert hunter.hunts == 2
    assert hunter.marches == 4


def test_nearness_is_measured_from_the_party_not_from_home() -> None:
    """A party deep in enemy ground presses what is near IT."""
    hunter = Hunter(size=2)
    home = (_tank(20), _tank(21))
    hunter.press(_world(*home), Intel(), home, (enemy(9, "c_tank", x=1050.0),), _CATALOGUE, True)
    # The same party, now walked deep: 1050 is 50 from the party and the
    # newcomer at 200 is 800 away, so the objective holds and nothing re-sends.
    away = (_tank(20, x=1000.0), _tank(21, x=1000.0))
    orders = hunter.press(
        _world(*away),
        Intel(),
        away,
        (enemy(9, "c_tank", x=1050.0), enemy(7, "c_tank", x=200.0)),
        _CATALOGUE,
        True,
    )
    assert orders == ()
    assert hunter.hunts == 1


def test_survivors_below_strength_disband_and_fight_home() -> None:
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    world = _world(*army)
    targets = (enemy(9, "c_tank", x=400.0),)
    hunter.press(world, Intel(), army, targets, _CATALOGUE, True)
    reduced = (_tank(21),)
    orders = hunter.press(_world(*reduced), Intel(), reduced, targets, _CATALOGUE, True)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [(21, 0.0, 0.0)]
    assert hunter.party() == frozenset()


def test_no_draft_without_the_campaigns_leave() -> None:
    """The wave gate arbitrates drafting; a party already out is managed regardless."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    targets = (enemy(9, "c_tank", x=400.0),)
    assert hunter.press(_world(*army), Intel(), army, targets, _CATALOGUE, False) == ()
    assert hunter.party() == frozenset()


def test_stand_down_recalls_the_party_home_once() -> None:
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    world = _world(*army)
    hunter.press(world, Intel(), army, (enemy(9, "c_tank", x=400.0),), _CATALOGUE, True)
    orders = hunter.stand_down(army, _CATALOGUE, world)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [(20, 0.0, 0.0), (21, 0.0, 0.0)]
    assert hunter.party() == frozenset()
    assert hunter.stand_down(army, _CATALOGUE, world) == ()


def test_stand_down_without_an_anchor_dissolves_silently() -> None:
    """Dead members and a razed base: nothing to order, nothing to hold."""
    hunter = Hunter(size=2)
    army = (_tank(20), _tank(21))
    world = _world(*army)
    hunter.press(world, Intel(), army, (enemy(9, "c_tank", x=400.0),), _CATALOGUE, True)
    assert hunter.stand_down(army, _CATALOGUE, sample(*army)) == ()
    assert hunter.party() == frozenset()


def test_the_default_size_is_the_engines_first_group() -> None:
    """Below it the engine's AI calls a force a trickle, and so does ours."""
    assert Hunter().size == 3
