"""The lurk cycle: loiter at the enemy start, run from anything that comes.

The verb exists because the raid leash dies with the raiders: the AI
recalls its attack groups home for 500 ticks per intrusion, and a unit that
retreats alive re-arms that recall where a raider pays with its life
([[ai-opponent-strategy]]). What is pinned here is the cycle itself --
orders on mode changes only, flight directly away from the chaser, posts
that do not stack -- against a scripted world with a known geometry.
"""

from __future__ import annotations

from rw_bot.policy.lurk import (
    LOITER_SPREAD,
    POST_STANDOFF,
    RETREAT_RADIUS,
    RETREAT_STEP,
    Lurker,
)
from rw_bot.policy.rush import mirror_point
from rw_bot.wire.state import Entity, Sample
from tests.campaign_fixtures import unit_stats
from tests.wire_fixtures import entity, pool, sample

#: Enough catalogue for the anchor: the Command Center is immobile.
_CATALOGUE = {
    "commandCenter": unit_stats("commandCenter", speed=0.0, price=3000),
    "scout": unit_stats("scout", price=700),
    "c_tank": unit_stats("c_tank"),
}

#: A world whose geometry the assertions can do by hand: the anchor, the
#: pool centroid and the mirrored enemy start share one horizontal line, so
#: the rim post is the enemy start pulled straight back by the standoff.
_CENTRE = entity(213, "commandCenter", x=100.0, y=500.0)

#: Where the rim post lands in this geometry: (900, 500) pulled 380 home.
_POST_X = 900.0 - POST_STANDOFF


def _world(*extra: Entity, credits: int = 4000) -> Sample:
    return sample(
        _CENTRE,
        *extra,
        credits=credits,
        pools=(pool(x=500.0, y=500.0),),
    )


def test_need_counts_missing_lurkers() -> None:
    lurker = Lurker()
    assert lurker.need(_world(), 2) == ("scout", "scout")
    one = _world(entity(300, "scout", x=110.0, y=500.0))
    assert lurker.need(one, 2) == ("scout",)
    assert lurker.need(one, 1) == ()
    assert lurker.need(one, 0) == ()


def test_a_clear_lurker_is_sent_to_its_post_once() -> None:
    """The post is the mirrored enemy start; the order is not repeated.

    The engine runs the newest waypoint, so a stream of identical moves
    would be noise in the run log rather than a different walk.
    """
    world = _world(entity(300, "scout", x=110.0, y=500.0))
    post = mirror_point(world, _CATALOGUE)
    assert post == (900.0, 500.0)
    lurker = Lurker()
    first = lurker.orders(world, _CATALOGUE, 1)
    assert [(o["unit_id"], o["x"], o["y"]) for o in first] == [(300, _POST_X, 500.0)]
    assert lurker.orders(world, _CATALOGUE, 1) == ()


def test_two_lurkers_hold_distinct_posts() -> None:
    """Stacked, they are one target for one wave; spread, each chase is its
    own recall."""
    world = _world(
        entity(300, "scout", x=110.0, y=500.0),
        entity(301, "scout", x=120.0, y=500.0),
    )
    orders = Lurker().orders(world, _CATALOGUE, 2)
    points = {(o["x"], o["y"]) for o in orders}
    assert len(points) == 2
    assert (_POST_X + LOITER_SPREAD, 500.0) in points


def test_a_threatened_lurker_flees_directly_away_from_the_chaser() -> None:
    """Away from the chaser, not toward home: the shortest safe line is the
    one the chaser defines, and a flight through the enemy base's far side
    is still a flight."""
    chaser = entity(400, "c_tank", x=_POST_X, y=400.0, mine=False, hostile=True)
    world = _world(entity(300, "scout", x=_POST_X, y=500.0), chaser)
    orders = Lurker().orders(world, _CATALOGUE, 1)
    assert len(orders) == 1
    # The chaser stands 100 below the lurker, so the flight is straight up.
    assert orders[0]["x"] == _POST_X
    assert orders[0]["y"] == 500.0 + RETREAT_STEP


def test_the_cycle_returns_to_post_when_the_chaser_leaves() -> None:
    lurker = Lurker()
    quiet = _world(entity(300, "scout", x=_POST_X, y=500.0))
    chased = _world(
        entity(300, "scout", x=_POST_X, y=500.0),
        entity(400, "c_tank", x=_POST_X, y=400.0, mine=False, hostile=True),
    )
    assert len(lurker.orders(quiet, _CATALOGUE, 1)) == 1
    assert len(lurker.orders(chased, _CATALOGUE, 1)) == 1
    back = lurker.orders(quiet, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in back] == [(_POST_X, 500.0)]


def test_a_hostile_beyond_the_radius_is_not_a_chaser() -> None:
    beyond = 500.0 - RETREAT_RADIUS - 50.0
    far = entity(400, "c_tank", x=_POST_X, y=beyond, mine=False, hostile=True)
    world = _world(entity(300, "scout", x=_POST_X, y=500.0), far)
    orders = Lurker().orders(world, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in orders] == [(_POST_X, 500.0)]


def test_a_chaser_standing_on_the_lurker_still_forces_a_flight() -> None:
    """Zero distance defines no direction, and any direction beats none."""
    world = _world(
        entity(300, "scout", x=_POST_X, y=500.0),
        entity(400, "c_tank", x=_POST_X, y=500.0, mine=False, hostile=True),
    )
    orders = Lurker().orders(world, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in orders] == [(_POST_X + RETREAT_STEP, 500.0)]


def test_bases_closer_than_the_standoff_post_at_the_midpoint() -> None:
    """A rim pulled back further than the whole gap would stand in our own
    base; halfway is the best available on a cramped map."""
    close = sample(
        entity(213, "commandCenter", x=100.0, y=500.0),
        entity(300, "scout", x=110.0, y=500.0),
        credits=4000,
        # Centroid at (200, 500) mirrors the anchor to (300, 500): a gap of
        # 200, well under the standoff.
        pools=(pool(x=200.0, y=500.0),),
    )
    orders = Lurker().orders(close, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in orders] == [(200.0, 500.0)]


def test_no_pool_means_no_post_and_no_orders() -> None:
    """No geometry, no guess: the mirror is pure arithmetic over the pools,
    and without one the lurker stays wherever it is."""
    bare = sample(_CENTRE, entity(300, "scout", x=110.0, y=500.0), credits=4000)
    assert Lurker().orders(bare, _CATALOGUE, 1) == ()
