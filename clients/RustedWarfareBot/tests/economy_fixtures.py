"""The base the spending policies are argued against, shared by all three.

Claiming pools, covering structures and buying throughput are three separate
decisions over one roster (`test_policy_economy`, `test_policy_defence`,
`test_policy_throughput`). They must agree about what a turret costs and what
the Builder can place, so the catalogue, the profiles and the opening roster
live here in one copy.

These are constructors over :mod:`tests.wire_fixtures`, not mocks: what comes
out is the same TypedDict the decoder produces from live bytes.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.economy import EXTRACTOR_TYPE, FACTORY_TYPE
from rw_bot.wire.state import BuildOption, Entity, ResourcePool, Sample
from tests.wire_fixtures import entity, profile

#: Combat profiles as the registry dump gives them. Complete by contract --
#: every registered type appears, the unarmed at zero reach.
PROFILES = {
    "commandCenter": profile("commandCenter", 0.0),
    "builder": profile("builder", 0.0),
    EXTRACTOR_TYPE: profile(EXTRACTOR_TYPE, 0.0),
    "turret": profile("turret", 100.0),
    "editorOrBuilder": profile("editorOrBuilder", 0.0),
    "landFactory": profile("landFactory", 0.0),
}


def unit_stats(type_name: str, *, price: int = 700, speed: float = 0.0) -> UnitStats:
    """Build one unarmed catalogue entry."""
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
        weapon=None,
    )


CATALOGUE = {
    EXTRACTOR_TYPE: unit_stats(EXTRACTOR_TYPE),
    "builder": unit_stats("builder", price=200, speed=0.6),
    "commandCenter": unit_stats("commandCenter", price=0),
    "turret": unit_stats("turret", price=300),
    FACTORY_TYPE: unit_stats(FACTORY_TYPE, price=700),
    "c_tank": unit_stats("c_tank", price=350, speed=1.1),
    "editorOrBuilder": unit_stats("editorOrBuilder", price=0),
}


def unit(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    queued: int = 0,
    complete: bool = True,
    group: int = 1,
) -> Entity:
    """Build an entity record, owned and finished unless said otherwise."""
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=not mine,
        movement="LAND",
        group=group,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=queued,
    )


def pool_at(x: float, y: float, *, group_land: int = 1) -> ResourcePool:
    """Build a pool record at a world position, with the tile it falls in."""
    return ResourcePool(
        index=0,
        tile_x=int(x) // 20,
        tile_y=int(y) // 20,
        x=x,
        y=y,
        group_land=group_land,
    )


def option(
    unit_id: int,
    produces: str = EXTRACTOR_TYPE,
    *,
    available: bool = True,
    placed: bool = True,
    makes_something: bool = True,
) -> BuildOption:
    """Build one action the engine reports a unit as offering."""
    return BuildOption(
        index=0,
        unit_id=unit_id,
        produces=produces,
        key="u_x",
        placed=placed,
        available=available,
        makes_something=makes_something,
        price=0,
    )


#: The Builder, and the option by which the engine says it can place one.
BUILDER = unit(214, "builder", 0.0, 0.0)
CAN_PLACE = option(214)


def sample(
    *entities: Entity,
    pools: tuple[ResourcePool, ...] = (),
    options: tuple[BuildOption, ...] = (),
    credits_held: int = 4000,
) -> Sample:
    """Build one observation from a roster."""
    return Sample(
        frame=1,
        clock_ms=10,
        credits=credits_held,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=entities,
        pools=pools,
        players=(),
        options=options,
        refusals=(),
    )


def free(world: Sample) -> tuple[Entity, ...]:
    """The workers a loop would report as free: every builder in the world."""
    return tuple(e for e in world["entities"] if e["mine"] and e["type_name"] == "builder")
