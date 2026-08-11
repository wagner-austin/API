"""The shipyard's walk: terrain discovery by attempt, as the channel ships it.

The engine's acceptance is read from the roster, never assumed from the
order -- so the fake world here grants water by growing a factory and
refuses it by silence, exactly the sensor the live probe proved
(log 2026-08-10, the sea probe).
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


def test_the_walk_offers_the_nearest_fraction_first() -> None:
    """Anchor (0,0), mirror (1000,0): the first candidate is 20 percent of
    the way to the reflected start."""
    yard = Shipyard()
    orders = yard.establish(_world(), _CATALOGUE, Budget(4000, 0), True)
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
        orders = yard.establish(world, _CATALOGUE, budget, True)
        follow_up = budget.claim("produce:c_tank", 4000)
        claims += 0 if follow_up["granted"] else 1
        ordered.extend(o["x"] for o in orders)
    # One claim total, an order every tick, and the fraction advanced
    # exactly once across two patience windows.
    assert claims == 1
    assert len(ordered) == 2 * PATIENCE + 1
    assert set(ordered) == {200.0, 210.0}
    assert ordered[0] == 200.0
    assert ordered[-1] == 210.0


def test_a_standing_or_growing_factory_ends_the_walk() -> None:
    yard = Shipyard()
    wet = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, FACTORY_TYPE, x=250.0, y=0.0, complete=False),
        pools=(_POOL,),
        credits=4000,
    )
    assert yard.establish(wet, _CATALOGUE, Budget(4000, 0), True) == ()


def test_an_unfunded_walk_withholds_toward_the_factory() -> None:
    """The saving pattern every strategic purchase uses: refused, the
    price is withheld so lesser claims cannot snipe the accrual."""
    yard = Shipyard()
    budget = Budget(300, 0)
    assert yard.establish(_world(), _CATALOGUE, budget, True) == ()
    lesser = budget.claim("produce:c_tank", 300)
    assert lesser["granted"] is False


def test_the_walk_gives_up_after_the_last_fraction() -> None:
    yard = Shipyard()
    world = _world()
    for _ in range(len(FRACTIONS) * (PATIENCE + 2) + 5):
        yard.establish(world, _CATALOGUE, Budget(4000, 0), True)
    assert yard.establish(world, _CATALOGUE, Budget(4000, 0), True) == ()


def test_the_walk_orders_with_the_newest_builder() -> None:
    """Dragging the opening's builder across the map is dragging the
    opening with it; the walk takes the latest hire instead."""
    yard = Shipyard()
    world = sample(
        _ANCHOR, _BUILDER, entity(7, "builder", x=20.0, y=0.0), pools=(_POOL,), credits=4000
    )
    orders = yard.establish(world, _CATALOGUE, Budget(4000, 0), True)
    assert [o["unit_id"] for o in orders] == [7]


def test_the_knob_off_a_missing_type_or_a_lost_base_stay_silent() -> None:
    yard = Shipyard()
    assert yard.establish(_world(), _CATALOGUE, Budget(4000, 0), False) == ()
    bare = {name: stats for name, stats in _CATALOGUE.items() if name != FACTORY_TYPE}
    assert yard.establish(_world(), bare, Budget(4000, 0), True) == ()
    builderless = sample(_ANCHOR, pools=(_POOL,), credits=4000)
    assert yard.establish(builderless, _CATALOGUE, Budget(4000, 0), True) == ()
    poolless = sample(_ANCHOR, _BUILDER, credits=4000)
    assert yard.establish(poolless, _CATALOGUE, Budget(4000, 0), True) == ()
