"""The scatter line: posts on our own half, flee from anything that comes.

The verb exists because the AI's attack targeting is uniform over ALL our
units with no fog term -- our placement is the distribution of its attacks,
and each decoy is an extra ticket in its lottery
([[ai-opponent-strategy]]). Pinned here: the post geometry, the shared flee
cycle, the one-shortfall-for-all-scout-verbs count, and the allotment that
keeps three verbs off one unit.
"""

from __future__ import annotations

from rw_bot.policy.decoy import POSTS, Decoys, scout_shortfall
from rw_bot.policy.lurk import RETREAT_STEP, Lurker
from rw_bot.wire.state import Entity, Sample
from tests.campaign_fixtures import unit_stats
from tests.wire_fixtures import entity, pool, sample

_CATALOGUE = {
    "commandCenter": unit_stats("commandCenter", speed=0.0, price=3000),
    "scout": unit_stats("scout", price=700),
    "c_tank": unit_stats("c_tank"),
}

#: Anchor at (100, 500), pool centroid (500, 500): the enemy start mirrors
#: to (900, 500), the axis is the x line, and the perpendicular is y -- so
#: every post is hand-computable from the POSTS fractions and the
#: 800-unit base-to-base length.
_CENTRE = entity(213, "commandCenter", x=100.0, y=500.0)
_SPAN = 800.0


def _world(*extra: Entity) -> Sample:
    return sample(_CENTRE, *extra, credits=4000, pools=(pool(x=500.0, y=500.0),))


def test_the_first_post_is_the_forward_flank() -> None:
    world = _world(entity(300, "scout", x=110.0, y=500.0))
    orders = Decoys().orders(world, _CATALOGUE, 1)
    fa, fp = POSTS[0]
    assert [(o["x"], o["y"]) for o in orders] == [
        (100.0 + fa * _SPAN, 500.0 + fp * _SPAN),
    ]


def test_posts_flank_both_sides_and_do_not_stack() -> None:
    world = _world(
        entity(300, "scout", x=110.0, y=500.0),
        entity(301, "scout", x=120.0, y=500.0),
    )
    orders = Decoys().orders(world, _CATALOGUE, 2)
    points = {(o["x"], o["y"]) for o in orders}
    assert len(points) == 2
    # The pair straddles the axis: one post above the line, one below.
    sides = {p[1] > 500.0 for p in points}
    assert sides == {True, False}


def test_a_threatened_decoy_flees_and_returns() -> None:
    """The lurker's cycle, re-used on our side of the map."""
    fa, fp = POSTS[0]
    post = (100.0 + fa * _SPAN, 500.0 + fp * _SPAN)
    runner = Decoys()
    quiet = _world(entity(300, "scout", x=post[0], y=post[1]))
    chased = _world(
        entity(300, "scout", x=post[0], y=post[1]),
        entity(400, "c_tank", x=post[0], y=post[1] - 100.0, mine=False, hostile=True),
    )
    assert len(runner.orders(quiet, _CATALOGUE, 1)) == 1
    flee = runner.orders(chased, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in flee] == [(post[0], post[1] + RETREAT_STEP)]
    back = runner.orders(quiet, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in back] == [post]


def test_the_skip_keeps_the_scatter_off_the_lurkers_scouts() -> None:
    """Patrol first, lurk line next, scatter last: three verbs, one roster,
    no unit serving two of them."""
    world = _world(
        entity(300, "scout", x=110.0, y=500.0),
        entity(301, "scout", x=120.0, y=500.0),
    )
    lurk_orders = Lurker().orders(world, _CATALOGUE, 1)
    decoy_orders = Decoys().orders(world, _CATALOGUE, 1, skip=1)
    assert [o["unit_id"] for o in lurk_orders] == [300]
    assert [o["unit_id"] for o in decoy_orders] == [301]


def test_the_shortfall_counts_all_scout_verbs_together() -> None:
    """Each verb counting the roster against its own figure would leave
    every one satisfied by the others' scouts."""
    world = _world(entity(300, "scout", x=110.0, y=500.0))
    assert scout_shortfall(world, 3) == ("scout", "scout")
    assert scout_shortfall(world, 1) == ()
    assert scout_shortfall(world, 0) == ()


def test_no_geometry_means_no_posts() -> None:
    bare = sample(_CENTRE, entity(300, "scout", x=110.0, y=500.0), credits=4000)
    assert Decoys().orders(bare, _CATALOGUE, 1) == ()


def test_a_held_mode_is_not_reordered() -> None:
    """The engine runs the newest waypoint; repeating it resets the walk."""
    world = _world(entity(300, "scout", x=110.0, y=500.0))
    runner = Decoys()
    assert len(runner.orders(world, _CATALOGUE, 1)) == 1
    assert runner.orders(world, _CATALOGUE, 1) == ()


def test_bases_on_one_point_post_at_the_anchor() -> None:
    """Zero geometry defines no flanks; the anchor is the only post left."""
    degenerate = sample(
        entity(213, "commandCenter", x=500.0, y=500.0),
        entity(300, "scout", x=510.0, y=500.0),
        credits=4000,
        pools=(pool(x=500.0, y=500.0),),
    )
    orders = Decoys().orders(degenerate, _CATALOGUE, 1)
    assert [(o["x"], o["y"]) for o in orders] == [(500.0, 500.0)]
