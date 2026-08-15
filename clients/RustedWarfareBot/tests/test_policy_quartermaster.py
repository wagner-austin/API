"""The quartermaster: the standing purchases dispatched as one block.

The two orderings under test ARE the module's policy: the fleet guard
hires before the submarines, and the battery's fork order comes LAST so
its re-send lands after the flame converter's on a contested holder
(log 2026-08-14, the send-order law).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.battery import BATTERY_TYPE, TURRET_TYPE
from rw_bot.policy.budget import Budget
from rw_bot.policy.convert import FLAME_TYPE
from rw_bot.policy.navy import FACTORY_TYPE
from rw_bot.policy.quartermaster import Quartermaster
from tests.wire_fixtures import entity, option, pool, sample


def _stats(name: str, price: int, speed: float) -> UnitStats:
    return UnitStats(
        type_name=name,
        display_name=name,
        description="",
        price=price,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    "commandCenter": _stats("commandCenter", 0, 0.0),
    "builder": _stats("builder", 500, 1.0),
    TURRET_TYPE: _stats(TURRET_TYPE, 500, 0.0),
    FACTORY_TYPE: _stats(FACTORY_TYPE, 1000, 0.0),
}

_ANCHOR = entity(1, "commandCenter", x=0.0, y=0.0)
_POOL = pool(x=500.0, y=0.0)


def test_the_flame_converter_never_takes_the_batterys_site_turret() -> None:
    """The pilot's void (log 2026-08-14): the $700 flame conversion funds
    before the $1,600 fork and took the shore turret every time it stood.
    With a base turret AND the site turret both offering flame, the flame
    channel converts the base one, the fork takes its own -- and the
    fork's order still comes last in the tuple so its send wins any race
    the exclusion cannot see."""
    quartermaster = Quartermaster(medics=0, navy=0, bunkers=0, flame=1, guns=0, battery=1)
    walking = sample(_ANCHOR, entity(2, "builder", x=10.0, y=0.0), pools=(_POOL,), credits=8000)
    warmup = Budget(8000, 0)
    quartermaster.produces(walking, _CATALOGUE, warmup)
    quartermaster.builds(walking, _CATALOGUE, warmup)
    contested = sample(
        _ANCHOR,
        entity(2, "builder", x=10.0, y=0.0),
        entity(9, TURRET_TYPE, x=228.0, y=0.0),
        entity(4, TURRET_TYPE, x=20.0, y=0.0),
        pools=(_POOL,),
        credits=8000,
        options=(
            option(9, FLAME_TYPE, key="u_flame", index=0, price=1000),
            option(9, BATTERY_TYPE, key="u_arty", index=1, price=1600),
            option(4, FLAME_TYPE, key="u_flame_b", index=2, price=1000),
        ),
    )
    orders = quartermaster.produces(contested, _CATALOGUE, Budget(8000, 0))
    assert [(o["type_name"], o["unit_id"]) for o in orders] == [
        (FLAME_TYPE, 4),
        (BATTERY_TYPE, 9),
    ]


def test_the_fork_claims_ahead_of_the_flame_conversion() -> None:
    """With only enough credits for one conversion, the fork's earlier
    claim wins and the flame channel is refused -- the funding order the
    pilot proved matters."""
    quartermaster = Quartermaster(medics=0, navy=0, bunkers=0, flame=1, guns=0, battery=1)
    walking = sample(_ANCHOR, entity(2, "builder", x=10.0, y=0.0), pools=(_POOL,), credits=8000)
    warmup = Budget(8000, 0)
    quartermaster.produces(walking, _CATALOGUE, warmup)
    quartermaster.builds(walking, _CATALOGUE, warmup)
    contested = sample(
        _ANCHOR,
        entity(2, "builder", x=10.0, y=0.0),
        entity(9, TURRET_TYPE, x=228.0, y=0.0),
        entity(4, TURRET_TYPE, x=20.0, y=0.0),
        pools=(_POOL,),
        credits=2000,
        options=(
            option(9, BATTERY_TYPE, key="u_arty", index=0, price=1600),
            option(4, FLAME_TYPE, key="u_flame_b", index=1, price=1000),
        ),
    )
    orders = quartermaster.produces(contested, _CATALOGUE, Budget(2000, 0))
    assert [(o["type_name"], o["unit_id"]) for o in orders] == [(BATTERY_TYPE, 9)]


def test_the_battery_walk_avoids_the_shipyards_pinned_builder() -> None:
    """Two live walks never share a builder: with navy and battery both
    on and two builders standing, the shipyard pins the newest and the
    battery takes the other."""
    quartermaster = Quartermaster(medics=0, navy=1, bunkers=0, flame=0, guns=0, battery=1)
    world = sample(
        _ANCHOR,
        entity(2, "builder", x=10.0, y=0.0),
        entity(7, "builder", x=20.0, y=0.0),
        pools=(_POOL,),
        credits=8000,
    )
    walk_budget = Budget(8000, 0)
    quartermaster.produces(world, _CATALOGUE, walk_budget)
    orders = quartermaster.builds(world, _CATALOGUE, walk_budget)
    by_type = {o["type_name"]: o["unit_id"] for o in orders}
    assert by_type[FACTORY_TYPE] == 7
    assert by_type[TURRET_TYPE] == 2


def test_every_channel_off_produces_and_builds_nothing() -> None:
    quartermaster = Quartermaster(medics=0, navy=0, bunkers=0, flame=0, guns=0, battery=0)
    world = sample(_ANCHOR, entity(2, "builder", x=10.0, y=0.0), pools=(_POOL,), credits=8000)
    assert quartermaster.produces(world, _CATALOGUE, Budget(8000, 0)) == ()
    assert quartermaster.builds(world, _CATALOGUE, Budget(8000, 0)) == ()
