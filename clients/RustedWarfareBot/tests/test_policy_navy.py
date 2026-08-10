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


def test_patience_advances_the_candidate() -> None:
    """PATIENCE silent offers at one fraction move the walk to the next --
    dry land swallows orders without a trace, and the roster's silence is
    the refusal."""
    yard = Shipyard()
    world = _world()
    seen = []
    for _ in range(PATIENCE + 1):
        orders = yard.establish(world, _CATALOGUE, Budget(4000, 0), True)
        seen.append(orders[0]["x"])
    assert seen[0] == 200.0
    assert seen[-1] == 250.0


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
    for _ in range(len(FRACTIONS) * (PATIENCE + 1) + 5):
        yard.establish(world, _CATALOGUE, Budget(4000, 0), True)
    assert yard.establish(world, _CATALOGUE, Budget(4000, 0), True) == ()


def test_the_knob_off_a_missing_type_or_a_lost_base_stay_silent() -> None:
    yard = Shipyard()
    assert yard.establish(_world(), _CATALOGUE, Budget(4000, 0), False) == ()
    bare = {name: stats for name, stats in _CATALOGUE.items() if name != FACTORY_TYPE}
    assert yard.establish(_world(), bare, Budget(4000, 0), True) == ()
    builderless = sample(_ANCHOR, pools=(_POOL,), credits=4000)
    assert yard.establish(builderless, _CATALOGUE, Budget(4000, 0), True) == ()
    poolless = sample(_ANCHOR, _BUILDER, credits=4000)
    assert yard.establish(poolless, _CATALOGUE, Budget(4000, 0), True) == ()
