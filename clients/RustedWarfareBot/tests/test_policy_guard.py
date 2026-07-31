"""Choosing the raider to turn on, exercised as the pure function it is.

The rule under test: a hostile inside the engine's own outpost radius of one
of our structures is an intruder, the deepest engageable intruder wins, and
ties resolve identically across runs of one seed.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.guard import OUTPOST_RADIUS, deepest_intruder
from rw_bot.wire.state import Entity
from tests.wire_fixtures import entity, profile, sample

_CATALOGUE: dict[str, UnitStats] = {}


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


_CATALOGUE = {
    "commandCenter": _stats("commandCenter", 0.0),
    "extractorT1": _stats("extractorT1", 0.0),
    "c_tank": _stats("c_tank", 1.0),
}

_PROFILES: dict[str, CombatProfile] = {
    "commandCenter": profile("commandCenter", 0.0, land=False),
    "extractorT1": profile("extractorT1", 0.0, land=False),
    "c_tank": profile("c_tank", 110.0),
}

_CENTRE = entity(1, "commandCenter", x=0.0, y=0.0)
_RESERVE = (entity(10, "c_tank", x=50.0, y=0.0),)


def _raider(unit_id: int, x: float, *, flying: bool = False, hp: float = 100.0) -> Entity:
    return entity(unit_id, "c_tank", mine=False, hostile=True, x=x, y=0.0, flying=flying, hp=hp)


def test_a_hostile_beyond_the_outpost_radius_is_not_an_intruder() -> None:
    """Distance is measured to our structures, with the engine's own figure."""
    afar = _raider(9, OUTPOST_RADIUS + 1.0)
    world = sample(_CENTRE, *_RESERVE, afar)
    assert deepest_intruder(world, _CATALOGUE, _PROFILES, _RESERVE, (afar,)) is None


def test_a_hostile_inside_the_radius_is_the_intruder() -> None:
    near = _raider(9, 200.0)
    world = sample(_CENTRE, *_RESERVE, near)
    chosen = deepest_intruder(world, _CATALOGUE, _PROFILES, _RESERVE, (near,))
    assert chosen is not None and chosen["unit_id"] == 9


def test_the_deeper_intruder_wins() -> None:
    """The raider closest to a structure is the one doing damage soonest."""
    deep = _raider(9, 100.0)
    shallow = _raider(8, 300.0)
    world = sample(_CENTRE, *_RESERVE, deep, shallow)
    chosen = deepest_intruder(world, _CATALOGUE, _PROFILES, _RESERVE, (shallow, deep))
    assert chosen is not None and chosen["unit_id"] == 9


def test_an_unengageable_intruder_falls_through_to_the_next() -> None:
    """Turning the reserve toward a helicopter it cannot shoot is the
    engageable-filter failure all over again ([[mechanics-combat-profile]]).
    """
    heli = _raider(9, 100.0, flying=True)
    tank = _raider(8, 300.0)
    world = sample(_CENTRE, *_RESERVE, heli, tank)
    chosen = deepest_intruder(world, _CATALOGUE, _PROFILES, _RESERVE, (heli, tank))
    assert chosen is not None and chosen["unit_id"] == 8


def test_nothing_engageable_means_no_intruder() -> None:
    heli = _raider(9, 100.0, flying=True)
    world = sample(_CENTRE, *_RESERVE, heli)
    assert deepest_intruder(world, _CATALOGUE, _PROFILES, _RESERVE, (heli,)) is None


def test_depth_ties_break_on_health() -> None:
    """Equidistant raiders are ordinary; kill the one closest to dying."""
    hurt = _raider(9, -150.0, hp=20.0)
    fresh = _raider(8, 150.0)
    world = sample(_CENTRE, *_RESERVE, hurt, fresh)
    chosen = deepest_intruder(world, _CATALOGUE, _PROFILES, _RESERVE, (fresh, hurt))
    assert chosen is not None and chosen["unit_id"] == 9


def test_no_reserve_or_no_structure_means_no_intruder() -> None:
    """A player with no buildings has worse problems than raiders."""
    near = _raider(9, 200.0)
    with_structures = sample(_CENTRE, near)
    assert deepest_intruder(with_structures, _CATALOGUE, _PROFILES, (), (near,)) is None
    homeless = sample(*_RESERVE, near)
    assert deepest_intruder(homeless, _CATALOGUE, _PROFILES, _RESERVE, (near,)) is None
