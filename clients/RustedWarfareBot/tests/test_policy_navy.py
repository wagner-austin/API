"""The shipyard's walk: terrain discovery by attempt, as the channel ships it.

The engine's acceptance is read from the roster, never assumed from the
order -- so the fake world here grants water by growing a factory and
refuses it by silence, exactly the sensor the live probe proved
(log 2026-08-10, the sea probe). Funding is its own early-tick step and
the builder holds the incomplete factory: the battery's pilots measured
both defects in this walk's twin (log 2026-08-14).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.budget import Budget
from rw_bot.policy.navy import FACTORY_TYPE, FRACTIONS, PATIENCE, Shipyard
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, pool, sample


def _stats(name: str, price: int) -> UnitStats:
    return UnitStats(
        type_name=name,
        display_name=name,
        description="",
        price=price,
        hp=100,
        speed=0.0 if name in ("commandCenter", FACTORY_TYPE) else 1.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    "commandCenter": _stats("commandCenter", 0),
    "builder": _stats("builder", 500),
    FACTORY_TYPE: _stats(FACTORY_TYPE, 1000),
}

_ANCHOR = entity(1, "commandCenter", x=0.0, y=0.0)
_BUILDER = entity(2, "builder", x=10.0, y=0.0)
_POOL = pool(x=500.0, y=0.0)


def _world() -> Sample:
    return sample(_ANCHOR, _BUILDER, pools=(_POOL,), credits=4000)


def _step(yard: Shipyard, world: Sample, budget: Budget) -> tuple[float, ...]:
    """One tick of the channel: fund, then walk; the quartermaster's order."""
    yard.fund(world, _CATALOGUE, budget, True)
    return tuple(o["x"] for o in yard.establish(world, _CATALOGUE, budget, True))


def test_the_walk_offers_the_nearest_fraction_first() -> None:
    """Anchor (0,0), mirror (1000,0): the first candidate is 20 percent of
    the way to the reflected start."""
    yard = Shipyard()
    budget = Budget(4000, 0)
    yard.fund(_world(), _CATALOGUE, budget, True)
    orders = yard.establish(_world(), _CATALOGUE, budget, True)
    assert [(o["type_name"], o["unit_id"], o["x"], o["y"]) for o in orders] == [
        (FACTORY_TYPE, 2, 200.0, 0.0)
    ]


def test_one_claim_then_the_order_resends_and_patience_advances() -> None:
    """Both measured failures in one test: claiming per tick consumed
    369,000 credits (navy96), and ordering once let the expander steal
    the builder back (navy96b interim). The price is claimed exactly
    once; the order re-sends every tick so it always lands last; silence
    for PATIENCE samples advances the fraction."""
    yard = Shipyard()
    world = _world()
    ordered: list[float] = []
    claims = 0
    for _ in range(2 * PATIENCE + 1):
        budget = Budget(4000, 0)
        ordered.extend(_step(yard, world, budget))
        follow_up = budget.claim("produce:c_tank", 4000)
        claims += 0 if follow_up["granted"] else 1
    # One claim total, an order every tick, and the fraction advanced
    # exactly once across two patience windows.
    assert claims == 1
    assert len(ordered) == 2 * PATIENCE + 1
    assert set(ordered) == {200.0, 210.0}
    assert ordered[0] == 200.0
    assert ordered[-1] == 210.0


def test_a_finished_factory_ends_the_walk() -> None:
    yard = Shipyard()
    wet = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, FACTORY_TYPE, x=250.0, y=0.0),
        pools=(_POOL,),
        credits=4000,
    )
    assert _step(yard, wet, Budget(4000, 0)) == ()


def test_an_incomplete_factory_holds_the_builder_on_construction() -> None:
    """The battery's sixth pilot: the expander re-tasks a released
    builder and the abandoned construction dies unfinished (log
    2026-08-14). While the factory is incomplete the walk re-sends the
    build at the STANDING factory, winning the builder back every tick;
    completion releases it."""
    yard = Shipyard()
    growing = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, FACTORY_TYPE, x=250.0, y=0.0, complete=False),
        pools=(_POOL,),
        credits=4000,
    )
    budget = Budget(4000, 0)
    yard.fund(growing, _CATALOGUE, budget, True)
    orders = yard.establish(growing, _CATALOGUE, budget, True)
    assert [(o["type_name"], o["unit_id"], o["x"], o["y"]) for o in orders] == [
        (FACTORY_TYPE, 2, 250.0, 0.0)
    ]


def test_an_unfunded_walk_withholds_toward_the_factory() -> None:
    """The saving pattern, now early in the tick where it binds: refused,
    the price is withheld so lesser claims cannot snipe the accrual."""
    yard = Shipyard()
    budget = Budget(300, 0)
    assert _step(yard, _world(), budget) == ()
    lesser = budget.claim("produce:c_tank", 300)
    assert lesser["granted"] is False


def test_a_dead_factory_re_funds_its_rebuild() -> None:
    """The engine charges per attempt, so the books must too (the
    battery's second pilot; log 2026-08-14)."""
    yard = Shipyard()
    claims = 0
    budget = Budget(4000, 0)
    _step(yard, _world(), budget)
    claims += 0 if budget.claim("produce:c_tank", 4000)["granted"] else 1
    stood = sample(
        _ANCHOR, _BUILDER, entity(9, FACTORY_TYPE, x=250.0, y=0.0), pools=(_POOL,), credits=4000
    )
    assert _step(yard, stood, Budget(4000, 0)) == ()
    razed = Budget(4000, 0)
    orders = _step(yard, _world(), razed)
    claims += 0 if razed.claim("produce:c_tank", 4000)["granted"] else 1
    assert len(orders) == 1
    assert claims == 2


def test_the_walk_gives_up_after_the_last_fraction() -> None:
    yard = Shipyard()
    world = _world()
    for _ in range(len(FRACTIONS) * (PATIENCE + 2) + 5):
        _step(yard, world, Budget(4000, 0))
    assert _step(yard, world, Budget(4000, 0)) == ()


def test_the_walk_orders_with_the_newest_builder_not_spoken_for() -> None:
    """Dragging the opening's builder across the map is dragging the
    opening with it; the walk takes the latest hire at pick time -- and
    never one another walk has pinned."""
    yard = Shipyard()
    world = sample(
        _ANCHOR, _BUILDER, entity(7, "builder", x=20.0, y=0.0), pools=(_POOL,), credits=4000
    )
    budget = Budget(4000, 0)
    yard.fund(world, _CATALOGUE, budget, True)
    orders = yard.establish(world, _CATALOGUE, budget, True)
    assert [o["unit_id"] for o in orders] == [7]
    avoided = Shipyard()
    other = Budget(4000, 0)
    avoided.fund(world, _CATALOGUE, other, True)
    assert [
        o["unit_id"] for o in avoided.establish(world, _CATALOGUE, other, True, avoid_builder=7)
    ] == [2]
    starved = Shipyard()
    lone = Budget(4000, 0)
    starved.fund(_world(), _CATALOGUE, lone, True)
    assert starved.establish(_world(), _CATALOGUE, lone, True, avoid_builder=2) == ()


def test_the_walk_keeps_its_builder_while_it_lives() -> None:
    """navy96e's poison: ``builders[-1]`` re-resolved every tick, so each
    hire re-targeted the walk to a fresh builder standing in the base and
    nobody ever reached the shore -- unit 24, then 43, then 55, each with
    forty ticks to live (log 2026-08-10). Once picked, the builder is
    pinned until it dies, however many newer hires appear."""
    yard = Shipyard()
    first = _world()
    budget = Budget(4000, 0)
    yard.fund(first, _CATALOGUE, budget, True)
    assert [o["unit_id"] for o in yard.establish(first, _CATALOGUE, budget, True)] == [2]
    assert yard.pinned_builder() == 2
    crowded = sample(
        _ANCHOR, _BUILDER, entity(7, "builder", x=20.0, y=0.0), pools=(_POOL,), credits=4000
    )
    for _ in range(3):
        orders = yard.establish(crowded, _CATALOGUE, Budget(4000, 0), True)
        assert [o["unit_id"] for o in orders] == [2]


def test_a_dead_builder_is_replaced_and_the_window_restarts() -> None:
    """A replacement starts the trek from the base; inheriting a spent
    patience window would refuse the fraction without ever reaching it."""
    yard = Shipyard()
    world = _world()
    for _ in range(PATIENCE - 1):
        _step(yard, world, Budget(4000, 0))
    survivor = sample(_ANCHOR, entity(7, "builder", x=20.0, y=0.0), pools=(_POOL,), credits=4000)
    ordered: list[tuple[int, float]] = []
    for _ in range(PATIENCE):
        orders = yard.establish(survivor, _CATALOGUE, Budget(4000, 0), True)
        ordered.extend((o["unit_id"], o["x"]) for o in orders)
    # The replacement is ordered for a FULL window at the same fraction:
    # the candidate did not advance on the dead builder's spent clock.
    assert ordered == [(7, 200.0)] * PATIENCE


def test_the_knob_off_a_missing_type_or_a_lost_base_stay_silent() -> None:
    yard = Shipyard()
    off = Budget(4000, 0)
    yard.fund(_world(), _CATALOGUE, off, False)
    assert yard.establish(_world(), _CATALOGUE, off, False) == ()
    bare = {name: stats for name, stats in _CATALOGUE.items() if name != FACTORY_TYPE}
    poor = Budget(4000, 0)
    yard.fund(_world(), bare, poor, True)
    assert yard.establish(_world(), bare, poor, True) == ()
    builderless = sample(_ANCHOR, pools=(_POOL,), credits=4000)
    funded = Budget(4000, 0)
    yard.fund(builderless, _CATALOGUE, funded, True)
    assert yard.establish(builderless, _CATALOGUE, funded, True) == ()
    poolless = sample(_ANCHOR, _BUILDER, credits=4000)
    assert yard.establish(poolless, _CATALOGUE, Budget(4000, 0), True) == ()
