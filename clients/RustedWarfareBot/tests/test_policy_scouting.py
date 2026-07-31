"""The scout circuit, exercised without a match running.

What is tested: the scout joins the composition exactly while none is alive,
the route runs farthest-first from the anchor, a leg is ordered once, arrival
advances the circuit, and a replacement scout starts the circuit over.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.scouting import SCOUT_TYPE, ScoutRunner
from rw_bot.wire.state import Entity, ResourcePool, Sample
from tests.wire_fixtures import entity, pool, sample


def _stats(type_name: str, speed: float) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=100,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE: dict[str, UnitStats] = {
    "commandCenter": _stats("commandCenter", 0.0),
    SCOUT_TYPE: _stats(SCOUT_TYPE, 1.4),
}

_CENTRE = entity(1, "commandCenter", x=0.0, y=0.0)

#: Near and far pools; the route must start at the far one.
_NEAR = pool(x=200.0, y=0.0)
_FAR = pool(x=2000.0, y=0.0, index=1)


def _scout(unit_id: int, x: float = 0.0, y: float = 0.0) -> Entity:
    return entity(unit_id, SCOUT_TYPE, x=x, y=y)


def _world(*extra: Entity, pools: tuple[ResourcePool, ...] = (_NEAR, _FAR)) -> Sample:
    return sample(_CENTRE, *extra, pools=pools)


def test_a_scout_is_wanted_exactly_while_none_is_alive() -> None:
    runner = ScoutRunner()
    assert runner.need(_world(), workers=2) == (SCOUT_TYPE,)
    assert runner.need(_world(_scout(7)), workers=2) == ()


def test_the_scout_yields_to_the_economy() -> None:
    """V1 let the scout outrank the builder at the Command Center, and one
    match ran its whole economy on a single builder for it.
    """
    runner = ScoutRunner()
    assert runner.need(_world(), workers=1) == ()
    assert runner.need(_world(), workers=0) == ()


def test_the_route_starts_at_the_farthest_pool() -> None:
    """The far pools are the opponent's side, which is where the intel is."""
    runner = ScoutRunner()
    orders = runner.patrol(_world(_scout(7)), _CATALOGUE)
    assert [(o["x"], o["y"]) for o in orders] == [(2000.0, 0.0)]


def test_a_leg_is_ordered_once_not_once_per_sample() -> None:
    """The engine runs a waypoint until it is replaced; a repeat resets it."""
    runner = ScoutRunner()
    world = _world(_scout(7))
    runner.patrol(world, _CATALOGUE)
    assert runner.patrol(world, _CATALOGUE) == ()


def test_arrival_advances_the_circuit() -> None:
    runner = ScoutRunner()
    runner.patrol(_world(_scout(7)), _CATALOGUE)
    arrived = _world(_scout(7, x=1990.0, y=0.0))
    orders = runner.patrol(arrived, _CATALOGUE)
    assert [(o["x"], o["y"]) for o in orders] == [(200.0, 0.0)]
    assert runner.legs_walked == 1


def test_a_replacement_scout_starts_the_circuit_over() -> None:
    """A dead scout's leg says nothing about where its replacement stands."""
    runner = ScoutRunner()
    runner.patrol(_world(_scout(7, x=1990.0, y=0.0)), _CATALOGUE)
    runner.patrol(_world(_scout(7, x=1990.0, y=0.0)), _CATALOGUE)
    # The scout dies; the runner forgets; a new scout is routed to the far
    # pool again rather than to the dead one's next leg.
    assert runner.patrol(_world(), _CATALOGUE) == ()
    orders = runner.patrol(_world(_scout(8)), _CATALOGUE)
    assert [(o["x"], o["y"]) for o in orders] == [(2000.0, 0.0)]


def test_no_anchor_or_no_pools_means_no_route() -> None:
    runner = ScoutRunner()
    homeless = sample(_scout(7), pools=(_NEAR,))
    assert runner.patrol(homeless, _CATALOGUE) == ()
    poolless = sample(_CENTRE, _scout(7), pools=())
    assert runner.patrol(poolless, _CATALOGUE) == ()
