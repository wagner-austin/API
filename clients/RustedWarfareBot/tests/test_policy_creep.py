"""The turret creep, exercised as the pure controller it is.

The documented human answer to the cheating difficulties: turrets walked to
the enemy's door inside the AI's thousand-tick opening delay. What is pinned
here is the geometry (one turret-reach step from the front, toward the
mirror), the one-at-a-time discipline, and the ordinary ticks -- no worker,
no money, a turret still rising -- that pause the walk without ending it.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.policy.budget import Budget
from rw_bot.policy.creep import Creeper
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.state import Entity, ResourcePool, Sample
from tests.wire_fixtures import entity, pool, profile, sample


def _weapon(reach: float) -> Weapon:
    return Weapon(
        shoot_delay=50.0,
        attack_range=reach,
        direct_damage=17.0,
        direct_damage_volley=17.0,
        area_damage=0.0,
        area_damage_volley=0.0,
    )


def _unit(type_name: str, *, speed: float, price: int, reach: float = 0.0) -> UnitStats:
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
        weapon=_weapon(reach) if reach else None,
    )


_CATALOGUE = {
    "commandCenter": _unit("commandCenter", speed=0.0, price=2000),
    "builder": _unit("builder", speed=0.6, price=500),
    "c_turret_t1": _unit("c_turret_t1", speed=0.0, price=400, reach=165.0),
}

_PROFILES = {
    "commandCenter": profile("commandCenter", 0.0),
    "builder": profile("builder", 0.0),
    "c_turret_t1": profile("c_turret_t1", 165.0, land=True),
}

#: Anchor at (0, 0), pools centred at (500, 0): the mirror of the anchor is
#: (1000, 0), so the creep walks along the x axis and every assertion is a
#: one-dimensional read.
_CENTRE = entity(21, "commandCenter", x=0.0, y=0.0, hp=4000.0)
_BUILDER = entity(24, "builder", x=10.0, y=0.0)


def _world(*extra: Entity, pools: tuple[ResourcePool, ...] = (pool(x=500.0, y=0.0),)) -> Sample:
    return sample(_CENTRE, _BUILDER, *extra, pools=pools)


def test_the_first_turret_steps_out_from_the_anchor_toward_the_mirror() -> None:
    creeper = Creeper()
    budget = Budget(4000, reserve=0)
    workforce = Workforce(45)
    orders = creeper.advance(_world(), _CATALOGUE, _PROFILES, budget, (_BUILDER,), workforce)
    assert len(orders) == 1
    assert orders[0]["unit_id"] == 24
    assert orders[0]["type_name"] == "c_turret_t1"
    # One turret reach along the line from (0,0) toward the mirror (1000,0).
    assert orders[0]["x"] == 165.0
    assert orders[0]["y"] == 0.0
    assert creeper.ordered == 1


def test_the_line_advances_from_the_forward_most_standing_turret() -> None:
    creeper = Creeper()
    front = entity(30, "c_turret_t1", x=330.0, y=0.0)
    orders = creeper.advance(
        _world(front), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert orders[0]["x"] == 495.0


def test_within_one_step_of_the_mirror_the_creep_builds_at_the_mirror() -> None:
    """The kill zone the whole walk exists to reach."""
    creeper = Creeper()
    front = entity(30, "c_turret_t1", x=900.0, y=0.0)
    orders = creeper.advance(
        _world(front), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert (orders[0]["x"], orders[0]["y"]) == (1000.0, 0.0)


def test_a_rising_turret_pauses_the_walk() -> None:
    """One at a time: the next site is chosen from where this one STANDS."""
    creeper = Creeper()
    rising = entity(30, "c_turret_t1", x=165.0, y=0.0, complete=False)
    orders = creeper.advance(
        _world(rising), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert orders == ()


def test_an_occupied_site_is_sidestepped_not_skipped() -> None:
    """The same occupancy discipline every siting path trusts."""
    creeper = Creeper()
    squatter = entity(40, "commandCenter", x=165.0, y=0.0, mine=False, hostile=True)
    orders = creeper.advance(
        _world(squatter), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert len(orders) == 1
    assert (orders[0]["x"], orders[0]["y"]) != (165.0, 0.0)


def test_a_fully_occupied_step_pauses_the_walk() -> None:
    """The point and every cover offset taken: the tick passes, no order."""
    creeper = Creeper()
    squatters = tuple(
        entity(40 + n, "commandCenter", x=165.0 + dx, y=0.0 + dy, mine=False, hostile=True)
        for n, (dx, dy) in enumerate(
            (
                (0.0, 0.0),
                (60.0, 0.0),
                (0.0, 60.0),
                (-60.0, 0.0),
                (0.0, -60.0),
                (60.0, 60.0),
                (-60.0, 60.0),
                (60.0, -60.0),
                (-60.0, -60.0),
                (120.0, 0.0),
                (0.0, 120.0),
                (-120.0, 0.0),
                (0.0, -120.0),
            )
        )
    )
    orders = creeper.advance(
        _world(*squatters),
        _CATALOGUE,
        _PROFILES,
        Budget(4000, reserve=0),
        (_BUILDER,),
        Workforce(45),
    )
    assert orders == ()


def test_the_front_is_the_turret_nearest_the_goal_not_the_first_seen() -> None:
    creeper = Creeper()
    rear = entity(31, "c_turret_t1", x=165.0, y=0.0)
    front = entity(30, "c_turret_t1", x=330.0, y=0.0)
    # The front is listed first so the rear is examined and REJECTED, which
    # is the half of the comparison a single-turret world never exercises.
    orders = creeper.advance(
        _world(front, rear),
        _CATALOGUE,
        _PROFILES,
        Budget(4000, reserve=0),
        (_BUILDER,),
        Workforce(45),
    )
    assert orders[0]["x"] == 495.0


def test_a_refused_budget_pauses_the_walk() -> None:
    creeper = Creeper()
    orders = creeper.advance(
        _world(), _CATALOGUE, _PROFILES, Budget(100, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert orders == ()
    assert creeper.ordered == 0


def test_no_free_worker_pauses_the_walk() -> None:
    creeper = Creeper()
    orders = creeper.advance(
        _world(), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (), Workforce(45)
    )
    assert orders == ()


def test_no_pool_means_no_mirror_and_no_walk() -> None:
    """The goal is geometry over the pools; a poolless world has no mirror."""
    creeper = Creeper()
    orders = creeper.advance(
        _world(pools=()), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert orders == ()


def test_no_anchor_means_no_front_and_no_walk() -> None:
    """A world without our command centre has nowhere to step out from."""
    creeper = Creeper()
    world = sample(_BUILDER, pools=(pool(x=500.0, y=0.0),))
    orders = creeper.advance(
        world, _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), Workforce(45)
    )
    assert orders == ()


def test_the_nearest_free_worker_is_the_one_sent() -> None:
    creeper = Creeper()
    far = entity(25, "builder", x=-400.0, y=0.0)
    orders = creeper.advance(
        _world(far), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (far, _BUILDER), Workforce(45)
    )
    assert orders[0]["unit_id"] == 24


def test_the_worker_is_assigned_so_the_next_tick_sees_it_working() -> None:
    creeper = Creeper()
    workforce = Workforce(45)
    creeper.advance(
        _world(), _CATALOGUE, _PROFILES, Budget(4000, reserve=0), (_BUILDER,), workforce
    )
    assert workforce.claims() == ((165.0, 0.0),)
