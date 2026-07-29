"""Wave memory between observations, exercised without a match running.

:class:`~rw_bot.policy.dispatch.WaveController` is the state six loop locals
used to be, so what is tested here is exactly what could only be exercised by
playing a whole match before: that gathering happens below the wave size, that
release converts the reserve to attacks, and that neither a rally nor an attack
is ever re-sent to a unit already carrying it -- the engine runs a waypoint
until it is replaced, so a repeat resets the walk ([[issuing-orders]]).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.combat import FIRST_WAVE, Engagement
from rw_bot.policy.dispatch import WaveController, dispatch_attacks, gather_reserve
from rw_bot.wire.command import AttackOrder
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import entity, profile, sample


def _unit(type_name: str, *, speed: float = 1.0, armed: bool = True) -> UnitStats:
    weapon = Weapon(
        shoot_delay=50.0,
        attack_range=110.0,
        direct_damage=17.0,
        direct_damage_volley=17.0,
        area_damage=0.0,
        area_damage_volley=0.0,
    )
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
        weapon=weapon if armed else None,
    )


_CATALOGUE: dict[str, UnitStats] = {
    "commandCenter": _unit("commandCenter", speed=0.0, armed=False),
    "c_tank": _unit("c_tank"),
}

_PROFILES: dict[str, CombatProfile] = {
    "commandCenter": profile("commandCenter", 0.0, land=False),
    "c_tank": profile("c_tank", 110.0),
}


def _tank(unit_id: int, x: float = 500.0, y: float = 500.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


def _world(*army: Entity, hostiles: tuple[Entity, ...] = ()) -> Sample:
    anchor = entity(1, "commandCenter", x=0.0, y=0.0)
    return sample(anchor, *army, *hostiles)


def test_below_the_first_wave_the_army_gathers_and_nobody_attacks() -> None:
    waves = WaveController()
    army = (_tank(10), _tank(11))
    assert len(army) < FIRST_WAVE
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    moves, attacks = waves.command(_world(*army, hostiles=(hostile,)), _CATALOGUE, _PROFILES, army)
    assert [move["unit_id"] for move in moves] == [10, 11]
    assert attacks == ()
    assert waves.rallied == 2
    assert waves.attack_orders == 0


def test_a_rally_is_sent_once_per_stint_not_once_per_sample() -> None:
    waves = WaveController()
    army = (_tank(10), _tank(11))
    world = _world(*army)
    waves.command(world, _CATALOGUE, _PROFILES, army)
    moves, _ = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert moves == ()
    assert waves.rallied == 2


def test_a_full_reserve_is_released_and_attacks_together() -> None:
    waves = WaveController()
    army = tuple(_tank(10 + n) for n in range(FIRST_WAVE))
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    world = _world(*army, hostiles=(hostile,))
    moves, attacks = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert moves == ()
    assert [attack["target_id"] for attack in attacks] == [9, 9, 9]
    assert waves.attack_orders == FIRST_WAVE


def test_an_attack_is_not_re_sent_while_the_pairing_holds() -> None:
    """Re-issuing an identical attack replaces the order with a copy of itself,
    and the unit never closes the distance.
    """
    waves = WaveController()
    army = tuple(_tank(10 + n) for n in range(FIRST_WAVE))
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    world = _world(*army, hostiles=(hostile,))
    waves.command(world, _CATALOGUE, _PROFILES, army)
    _, attacks = waves.command(world, _CATALOGUE, _PROFILES, army)
    assert attacks == ()
    assert waves.attack_orders == FIRST_WAVE


def test_killed_counts_attacked_targets_no_longer_visible() -> None:
    """Named for what was observed: a retreat into fog reads the same way."""
    waves = WaveController()
    army = tuple(_tank(10 + n) for n in range(FIRST_WAVE))
    hostile = entity(9, "c_tank", mine=False, hostile=True, x=600.0, y=600.0)
    waves.command(_world(*army, hostiles=(hostile,)), _CATALOGUE, _PROFILES, army)
    assert waves.killed({9}) == 0
    assert waves.killed(set()) == 1


def test_gather_reserve_marks_each_unit_and_never_repeats_it() -> None:
    rallying: set[int] = set()
    world = _world(_tank(10), _tank(11))
    reserve = (_tank(10), _tank(11))
    first = gather_reserve(world, _CATALOGUE, reserve, rallying)
    assert [order["unit_id"] for order in first] == [10, 11]
    assert gather_reserve(world, _CATALOGUE, reserve, rallying) == ()
    assert rallying == {10, 11}


def test_dispatch_attacks_skips_an_attacker_already_on_that_target() -> None:
    ordered: dict[int, int] = {10: 9}
    attacked: set[int] = set()
    engagements = (
        Engagement(attacker_id=10, target_id=9, reason=""),
        Engagement(attacker_id=11, target_id=9, reason=""),
    )
    sent: tuple[AttackOrder, ...] = dispatch_attacks(engagements, ordered, attacked)
    assert [order["unit_id"] for order in sent] == [11]
    assert ordered == {10: 9, 11: 9}
    assert attacked == {9}
