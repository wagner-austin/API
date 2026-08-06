"""The rush, exercised without a match running.

What is tested: the march target is pure geometry (the anchor reflected
through the pool centroid), released units are ordered at it once each,
contact hands the units back to the engagement policy, and a world with no
anchor or no pools marches nobody.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.rush import Rusher, mirror_point
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import entity, pool, sample


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
}

_CENTRE = entity(1, "commandCenter", x=100.0, y=100.0)


def _tank(unit_id: int) -> Entity:
    return entity(unit_id, "c_tank", x=100.0, y=100.0)


def _world(*extra: Entity, with_pools: bool = True) -> Sample:
    pools = (pool(x=300.0, y=300.0), pool(x=700.0, y=500.0)) if with_pools else ()
    return sample(_CENTRE, *extra, pools=pools)


def test_the_march_target_is_the_anchor_reflected_through_the_pool_centroid() -> None:
    """Skirmish duel maps are symmetric, so the reflection of our Command
    Center through the map's pools is the opponent's -- pure geometry, no fog
    consulted, nothing remembered."""
    # Centroid of (300,300) and (700,500) is (500,400); anchor (100,100)
    # reflects to (900,700).
    assert mirror_point(_world(), _CATALOGUE) == (900.0, 700.0)


def test_no_anchor_or_no_pools_means_no_march() -> None:
    assert mirror_point(sample(_tank(10)), _CATALOGUE) is None
    assert mirror_point(_world(with_pools=False), _CATALOGUE) is None
    rusher = Rusher()
    assert rusher.march(_world(with_pools=False), _CATALOGUE, (_tank(10),), False) == ()


def test_released_units_are_marched_once_each() -> None:
    """Re-issuing a waypoint at the sampling rate resets the walk."""
    rusher = Rusher()
    world = _world()
    first = rusher.march(world, _CATALOGUE, (_tank(10), _tank(11)), False)
    assert [(o["unit_id"], o["x"], o["y"]) for o in first] == [
        (10, 900.0, 700.0),
        (11, 900.0, 700.0),
    ]
    assert rusher.march(world, _CATALOGUE, (_tank(10), _tank(11)), False) == ()
    # A newly released reinforcement joins the march; the veterans are not
    # re-ordered.
    second = rusher.march(world, _CATALOGUE, (_tank(10), _tank(11), _tank(12)), False)
    assert [o["unit_id"] for o in second] == [12]
    assert rusher.marches == 3


def test_contact_hands_the_units_to_the_engagement_policy() -> None:
    """The moment anything is visible to fight, the rush stops ordering: the
    engagement policy re-tasks the released units and the engine runs the
    newest waypoint."""
    rusher = Rusher()
    assert rusher.march(_world(), _CATALOGUE, (_tank(10),), True) == ()
    assert rusher.marches == 0


def test_a_forced_march_deals_itself_across_the_enemy_side_pools() -> None:
    """The income field, not the start point.

    Three time-triggered dumps crossed the map at one point, traded up to
    even against the standing army, and dented six-figure rivals by nothing
    -- the army is the 3.7x-replaceable thing, and the extractors that pay
    for it stand on pools every sample already carries (log 2026-07-31).
    """
    world = _world(_tank(10), _tank(11), _tank(12))
    # The start reflects the anchor through the pool centroid; both pools
    # sit inside income reach of it, so both become posts.
    assert mirror_point(world, _CATALOGUE) == (900.0, 700.0)
    orders = Rusher().march(
        world,
        _CATALOGUE,
        (_tank(10), _tank(11), _tank(12)),
        # Contact at home: only a forced march goes anyway.
        True,
        force=True,
    )
    assert len(orders) == 3
    targets = {(o["x"], o["y"]) for o in orders}
    # Both pools within reach of the start are struck; the round-robin
    # brings the third marcher back to the nearest.
    assert targets == {(300.0, 300.0), (700.0, 500.0)}


def test_an_unforced_march_still_yields_to_contact() -> None:
    world = _world(_tank(10))
    assert Rusher().march(world, _CATALOGUE, (_tank(10),), True) == ()
