"""The world the build-order and siting tests decide against, shared by both.

``decide`` answers two questions off one sample -- *what* to build next and
*where* it may stand -- and the two are tested apart (`test_policy_build_order`,
`test_policy_siting`) because they fail for unrelated reasons. They must not
disagree about the ground they stand on, so the catalogue, the placement dump,
the profiles and the opening roster live here in one copy.

These are constructors over :mod:`tests.wire_fixtures`, not mocks: what comes
out is the same TypedDict the decoder produces from live bytes.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.build_order import BUILDER_TYPE
from rw_bot.wire.state import BuildOption, Entity, ResourcePool, Sample
from tests.wire_fixtures import entity, profile


def unit_stats(
    type_name: str, price: int, speed: float = 0.0, attack_range: float = 0.0
) -> UnitStats:
    """Build one catalogue entry, armed only if given a reach."""
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
        weapon=None if attack_range == 0.0 else weapon(attack_range),
    )


def weapon(attack_range: float) -> Weapon:
    """Build a weapon whose only interesting field is its reach."""
    return Weapon(
        shoot_delay=30.0,
        attack_range=attack_range,
        direct_damage=10.0,
        direct_damage_volley=10.0,
        area_damage=0.0,
        area_damage_volley=0.0,
    )


CATALOGUE = {
    "landFactory": unit_stats("landFactory", 300),
    "airFactory": unit_stats("airFactory", 900),
    "laboratory": unit_stats("laboratory", 900),
    # The two the live roster always starts with. The Command Center is the
    # anchor placement is measured from; the builder must be mobile, or it
    # would be eligible as an anchor and the ring would follow it again.
    "commandCenter": unit_stats("commandCenter", 3000),
    "builder": unit_stats("builder", 500, speed=0.6),
    "extractorT1": unit_stats("extractorT1", 700),
}


#: Reach of the test turret, in world units.
#:
#: Comfortably wider than :data:`~rw_bot.policy.siting.POOL_OCCUPIED_RADIUS` so
#: that "this pool is covered" and "this pool is built on" are always
#: distinguishable — a turret close enough to shoot a pool must not also be
#: close enough to be standing on it, or the tests could not tell which rule
#: rejected it.
TURRET_RANGE = 100.0

#: The default catalogue plus something that shoots back.
ARMED = {**CATALOGUE, "turret": unit_stats("turret", 400, attack_range=TURRET_RANGE)}

#: Attack range by type name, as the registry dump gives it.
#:
#: Every type any fixture can name appears, armed or not. That mirrors the real
#: dump, which covers all 173 registered types, and it is the contract
#: :func:`~rw_bot.policy.threat.reach_of` indexes against rather than defaults
#: through ([[policy-threat]]).
PROFILES: dict[str, CombatProfile] = {name: profile(name, 0.0) for name in ARMED}
PROFILES["turret"] = profile("turret", TURRET_RANGE)
PROFILES["someModStructure"] = profile("someModStructure", 0.0)


def _place(type_name: str, needs_pool: bool = False) -> TypePlacement:
    return TypePlacement(index=0, type_name=type_name, needs_pool=needs_pool)


#: Where each type may stand, as the engine reports it. Only the extractor is
#: pool-bound; the live dump agrees, and says so of exactly eight types out of
#: 173 ([[mechanics-resource-pools]]).
PLACEMENTS = {
    "landFactory": _place("landFactory"),
    "airFactory": _place("airFactory"),
    "extractorT1": _place("extractorT1", needs_pool=True),
    "laboratory": _place("laboratory"),
    "commandCenter": _place("commandCenter"),
    "builder": _place("builder"),
    "teleporter": _place("teleporter"),
}


#: Connectivity component every land fixture shares.
#:
#: The engine hands these out per map; the value is arbitrary and only equality
#: matters. Defaulting builders and pools to the same one keeps every test that
#: is not about reachability from having to restate it, and a test that *is*
#: about reachability puts one of them somewhere else.
MAINLAND = 1

#: A component id no fixture shares, for the far side of water.
ISLAND = 2


def pool_at(index: int, tile_x: int, tile_y: int) -> ResourcePool:
    """Build a pool record at a tile, with the world centre the agent computes.

    Named for the tile rather than plainly ``pool`` because the tests that use
    it overwhelmingly bind their result to a local called ``pool``, and a
    fixture a caller shadows on its second line is a fixture that can only be
    used once per test.
    """
    return ResourcePool(
        index=index,
        tile_x=tile_x,
        tile_y=tile_y,
        x=tile_x * 20.0 + 10.0,
        y=tile_y * 20.0 + 10.0,
        group_land=MAINLAND,
    )


def unit(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    complete: bool = True,
    hostile: bool | None = None,
    movement: str = "LAND",
    group: int = MAINLAND,
) -> Entity:
    """Build an entity record.

    ``hostile`` defaults to the opposite of ``mine``, which is what a two-player
    skirmish looks like. It is overridable because the engine does not derive it
    that way: an ally is neither mine nor hostile, and the distinction only
    shows up in a test that sets them independently.
    """
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=(not mine) if hostile is None else hostile,
        movement=movement,
        group=group,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=0,
    )


def option(
    unit_id: int,
    produces: str,
    *,
    placed: bool = True,
    available: bool = True,
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


#: What the Builder offers by default in these worlds.
#:
#: Mirrors the live capture, where unit 214 reports thirteen placed structures
#: including these. Supplying it by default keeps every test that is about
#: placement or ordering from also having to restate the build tree; a test
#: that is about the build tree passes its own.
BUILDER_OFFERS = ("landFactory", "airFactory", "extractorT1", "commandCenter", "teleporter")


def free(world: Sample) -> tuple[Entity, ...]:
    """The workers a loop would report as free: every builder in the world.

    Which workers are free is the loop's judgement in production
    ([[policy-loop]]); a pure test of ``decide`` supplies it directly.
    """
    return tuple(e for e in world["entities"] if e["mine"] and e["type_name"] == BUILDER_TYPE)


def sample(
    *entities: Entity,
    credits: int = 4000,
    pools: tuple[ResourcePool, ...] = (),
    options: tuple[BuildOption, ...] | None = None,
) -> Sample:
    """Build one observation, with the Builder's default offers unless given."""
    if options is None:
        options = tuple(option(214, name) for name in BUILDER_OFFERS)
    return Sample(
        frame=1,
        clock_ms=10,
        credits=credits,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=tuple(entities),
        pools=pools,
        players=(),
        options=options,
        refusals=(),
    )


BUILDER = unit(214, "builder", 4250.0, 2610.0)

#: Placement is measured from the oldest owned immobile structure, so a world
#: used for placement assertions needs one. The live game always has it: the
#: Command Center, at this position on the sandbox map.
ANCHOR = unit(213, "commandCenter", 4250.0, 2550.0)
