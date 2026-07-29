"""Choosing what to attack, exercised as the pure function it is.

No socket and no game: a world state goes in and a set of engagements comes
out, which is what keeps the fighting logic arguable without a match running.
"""

from __future__ import annotations

import pytest

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.mechanics.combat_profile import CombatProfileError, is_armed
from rw_bot.policy.combat import (
    FIRST_WAVE,
    RALLY_RADIUS,
    WAVE_SIZES,
    choose_target,
    engageable,
    engagements,
    find_army,
    find_targets,
    is_mobile,
    ladder_to,
    muster,
    rally,
    wave_size,
)
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import entity, profile, profiles_for


def _weapon(damage: float = 17.0, reach: float = 110.0) -> Weapon:
    return Weapon(
        shoot_delay=50.0,
        attack_range=reach,
        direct_damage=damage,
        direct_damage_volley=damage,
        area_damage=0.0,
        area_damage_volley=0.0,
    )


def _unit(
    type_name: str,
    *,
    speed: float = 1.0,
    armed: bool = True,
) -> UnitStats:
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
        weapon=_weapon() if armed else None,
    )


_CATALOGUE = {
    "c_tank": _unit("c_tank"),
    "builder": _unit("builder", speed=0.6, armed=False),
    "commandCenter": _unit("commandCenter", speed=0.0, armed=False),
    "c_turret_t1": _unit("c_turret_t1", speed=0.0),
    "editorOrBuilder": _unit("editorOrBuilder", speed=0.0, armed=False),
}

#: Combat profiles derived from the catalogue above, so a unit cannot be armed
#: in one table and unarmed in the other. Ground-only, like every land unit the
#: base game lets a player build; the layer tests override what they need.
_PROFILES = profiles_for(_CATALOGUE)


def _entity(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    hostile: bool = False,
    complete: bool = True,
) -> Entity:
    return entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=hostile,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=0,
    )


def _sample(*entities: Entity) -> Sample:
    return Sample(
        frame=1,
        clock_ms=10,
        credits=4000,
        defeated=False,
        wiped=False,
        players_left=6,
        entities=tuple(entities),
        pools=(),
        players=(),
        options=(),
    )


def test_an_armed_mobile_unit_is_army() -> None:
    tank = _entity(1, "c_tank")
    assert find_army(_sample(tank), _CATALOGUE, _PROFILES) == (tank,)


def test_a_builder_is_not_army() -> None:
    """Unarmed. Sending it at a tank is a Builder thrown away."""
    assert find_army(_sample(_entity(1, "builder")), _CATALOGUE, _PROFILES) == ()
    assert is_armed(_PROFILES, _entity(1, "builder")) is False


def test_a_turret_is_armed_but_not_army() -> None:
    """It cannot travel, so an order to attack anything distant is undeliverable."""
    turret = _entity(1, "c_turret_t1")
    assert is_armed(_PROFILES, turret) is True
    assert is_mobile(turret, _CATALOGUE) is False
    assert find_army(_sample(turret), _CATALOGUE, _PROFILES) == ()


def test_the_editor_placeholder_is_never_army() -> None:
    """Owned, off-map, and not a unit -- the same exclusion producer selection needs."""
    assert find_army(_sample(_entity(217, "editorOrBuilder")), _CATALOGUE, _PROFILES) == ()


def test_an_unfinished_tank_is_not_army() -> None:
    """It does not exist yet, whatever the roster says."""
    assert find_army(_sample(_entity(1, "c_tank", complete=False)), _CATALOGUE, _PROFILES) == ()


def test_an_enemy_tank_is_not_our_army() -> None:
    enemy = _entity(9, "c_tank", mine=False, hostile=True)
    assert find_army(_sample(enemy), _CATALOGUE, _PROFILES) == ()


def test_a_type_the_combat_dump_does_not_describe_fails_loudly() -> None:
    """A stale dump is a fault, not a unit to skip.

    This used to be absorbed: an unknown type was quietly treated as unarmed and
    left out of the army. That is the same silence that reported every turret as
    harmless, and it is indistinguishable from a real answer. The dump is
    regenerated from the registry and covers every registered type, so a miss
    means the dump and the running game disagree ([[mechanics-combat-profile]]).
    """
    with pytest.raises(CombatProfileError) as caught:
        find_army(_sample(_entity(1, "mysteryTank")), _CATALOGUE, _PROFILES)
    assert caught.value.code == "RW-COMBAT-002"
    assert "mysteryTank" in caught.value.message


def test_a_type_the_catalogue_does_not_price_cannot_travel() -> None:
    """Mobility is still the catalogue's answer, and still fails safe."""
    assert is_mobile(_entity(1, "mysteryTank"), _CATALOGUE) is False


def test_targets_are_the_engines_hostiles_not_merely_the_unowned() -> None:
    """An ally is not mine and is not a target, which "not mine" gets wrong."""
    ally = _entity(5, "c_tank", mine=False, hostile=False)
    enemy = _entity(9, "c_tank", mine=False, hostile=True)
    assert find_targets(_sample(ally, enemy)) == (enemy,)


def test_the_whole_army_commits_to_one_target() -> None:
    """Concentrating fire is the one tactic that matters at this scale."""
    near = _entity(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    far = _entity(10, "c_tank", 900.0, 0.0, mine=False, hostile=True)
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        _entity(2, "c_tank", 10.0, 0.0),
        near,
        far,
    )
    orders = engagements(world, _CATALOGUE, _PROFILES)
    assert [e["attacker_id"] for e in orders] == [1, 2]
    assert {e["target_id"] for e in orders} == {9}


def test_the_target_is_nearest_to_the_army_not_to_one_unit() -> None:
    """A split force converges instead of each unit picking its own enemy."""
    army = (_entity(1, "c_tank", 0.0, 0.0), _entity(2, "c_tank", 1000.0, 0.0))
    near_to_one = _entity(9, "c_tank", 0.0, 400.0, mine=False, hostile=True)
    near_to_centre = _entity(10, "c_tank", 500.0, 0.0, mine=False, hostile=True)
    assert choose_target(army, (near_to_one, near_to_centre)) == near_to_centre


def test_the_current_target_is_kept_while_it_lives() -> None:
    """Commitment, and the whole reason the churn happened.

    Nearest is measured from the army centre, and that centre shifts whenever a
    unit dies or a new one rolls out. Re-choosing every sample re-tasked the
    whole army on a flip that could be a few world units wide.
    """
    held = _entity(9, "c_tank", 900.0, 0.0, mine=False, hostile=True)
    nearer = _entity(10, "c_tank", 10.0, 0.0, mine=False, hostile=True)
    army = (_entity(1, "c_tank", 0.0, 0.0),)
    assert choose_target(army, (held, nearer), holding=9) == held
    assert choose_target(army, (held, nearer)) == nearer


def test_a_target_that_is_gone_is_replaced() -> None:
    """Holding is not clinging: a dead target frees the army to re-commit."""
    nearer = _entity(10, "c_tank", 10.0, 0.0, mine=False, hostile=True)
    army = (_entity(1, "c_tank", 0.0, 0.0),)
    assert choose_target(army, (nearer,), holding=9) == nearer


def test_the_held_target_carries_through_engagements() -> None:
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        _entity(9, "c_tank", 900.0, 0.0, mine=False, hostile=True),
        _entity(10, "c_tank", 10.0, 0.0, mine=False, hostile=True),
    )
    assert engagements(world, _CATALOGUE, _PROFILES, holding=9)[0]["target_id"] == 9
    assert engagements(world, _CATALOGUE, _PROFILES)[0]["target_id"] == 10


def test_no_army_produces_no_orders() -> None:
    enemy = _entity(9, "c_tank", mine=False, hostile=True)
    assert engagements(_sample(enemy), _CATALOGUE, _PROFILES) == ()
    assert choose_target((), (enemy,)) is None


def test_no_enemy_produces_no_orders() -> None:
    tank = _entity(1, "c_tank")
    assert engagements(_sample(tank), _CATALOGUE, _PROFILES) == ()
    assert choose_target((tank,), ()) is None


def test_the_engagement_reason_names_both_sides() -> None:
    """The run log has to be readable back without re-deriving the choice."""
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        _entity(9, "commandCenter", 50.0, 0.0, mine=False, hostile=True),
    )
    assert engagements(world, _CATALOGUE, _PROFILES)[0]["reason"] == "c_tank -> commandCenter 9"


def _wave(size: int) -> tuple[Entity, ...]:
    return tuple(_entity(unit_id, "c_tank") for unit_id in range(1, size + 1))


def test_the_wave_ladder_is_the_engines() -> None:
    """Three, then five, then seven, the last rung repeating."""
    assert [wave_size(n) for n in range(8)] == [3, 3, 5, 5, 5, 7, 7, 7]


def test_massing_more_changes_only_the_sustained_wave() -> None:
    """The early rungs govern the opening, when holding three units back is the
    difference between a first attack and none at all. The final rung governs
    the other twenty-eight minutes, and it is the one worth a question -- an
    experiment that moved both could not say which end mattered
    ([[policy-combat]]).
    """
    assert ladder_to(25) == (3, 3, 5, 5, 5, 25)
    assert [wave_size(n, ladder_to(25)) for n in range(8)] == [3, 3, 5, 5, 5, 25, 25, 25]


def test_the_shipped_ladder_is_reachable_rather_than_a_special_case() -> None:
    assert ladder_to(7) == WAVE_SIZES


def test_massing_less_than_the_fixed_rungs_cannot_lower_them() -> None:
    """A mass below the ladder's own body would make the sustained wave smaller
    than the opening ones, which is the trickle the gate exists to prevent.
    """
    assert ladder_to(1) == (3, 3, 5, 5, 5, 5)


def test_a_bigger_wave_holds_units_back_until_it_is_full() -> None:
    """The behaviour the mass argument buys: an army short of the mass gathers
    rather than trickling in.
    """
    state = muster(_wave(9), frozenset(), 5, ladder_to(25))
    assert state["released"] == frozenset()
    assert state["gathering"] == 9
    assert state["wanted"] == 25


def test_a_bigger_wave_releases_once_it_is_full() -> None:
    state = muster(_wave(25), frozenset(), 5, ladder_to(25))
    assert len(state["released"]) == 25
    assert state["waves"] == 6


def test_a_reserve_short_of_a_wave_releases_nobody() -> None:
    state = muster(_wave(2), frozenset(), 0)
    assert state["released"] == frozenset()
    assert state["gathering"] == 2
    assert state["wanted"] == 3
    assert state["waves"] == 0


def test_a_full_reserve_is_released_as_one_wave() -> None:
    state = muster(_wave(3), frozenset(), 0)
    assert state["released"] == frozenset({1, 2, 3})
    assert state["gathering"] == 0
    assert state["waves"] == 1
    assert state["reason"] == "wave 1 of 3 released"


def test_reinforcements_gather_instead_of_joining_the_fight_alone() -> None:
    """The failure the membership model exists for.

    A plain "have we started" flag latched on the first wave and let every
    later unit walk in one at a time -- 45 reinforcements for a net army growth
    of one, measured over 1,500 samples.
    """
    state = muster(_wave(4), frozenset({1, 2, 3}), 1)
    assert state["released"] == frozenset({1, 2, 3})
    assert state["gathering"] == 1
    assert state["wanted"] == 3


def test_the_second_wave_needs_its_own_full_reserve() -> None:
    state = muster(_wave(6), frozenset({1, 2, 3}), 1)
    assert state["released"] == frozenset({1, 2, 3, 4, 5, 6})
    assert state["waves"] == 2
    assert state["wanted"] == WAVE_SIZES[2]


def test_a_wave_still_worth_the_name_keeps_its_clearance() -> None:
    """Losses do not disband a wave that is still a wave."""
    intact = muster(_wave(3), frozenset({1, 2, 3, 4, 5}), 1)
    assert intact["released"] == frozenset({1, 2, 3})
    assert intact["gathering"] == 0


def test_a_decimated_wave_returns_to_the_reserve() -> None:
    """The trickle this gate exists to prevent, happening on the way out.

    Of 48 units lost in a 1500-sample match, 46 died more than 2,000 world units
    from home and not one died within 900 -- nothing was attacking the base. The
    last survivor of each wave kept its clearance and walked in after the rest,
    alone ([[policy-combat]]). Below the ladder's own first rung the survivors
    go back to the reserve, rally home, and go out with the next wave.
    """
    survivors = muster((_entity(2, "c_tank"),), frozenset({1, 2, 3}), 1)
    assert survivors["released"] == frozenset()
    assert survivors["gathering"] == 1


def test_the_disband_threshold_is_the_ladders_own_first_rung() -> None:
    """Reused rather than reinvented, so there is no new number to justify."""
    assert WAVE_SIZES[0] == FIRST_WAVE
    at_threshold = muster(_wave(FIRST_WAVE), frozenset({1, 2, 3, 4, 5}), 1)
    assert at_threshold["released"] == frozenset({1, 2, 3})
    below = muster(_wave(FIRST_WAVE - 1), frozenset({1, 2, 3, 4, 5}), 1)
    assert below["released"] == frozenset()


def test_a_wiped_wave_leaves_nothing_released() -> None:
    """And the next reserve then re-gathers rather than trickling in."""
    assert muster((), frozenset({1, 2, 3}), 1)["released"] == frozenset()


def test_a_scattered_reserve_is_sent_to_the_rally_point() -> None:
    """Units that rolled out of a factory are wherever the factory is."""
    scattered = (_entity(4, "c_tank", 900.0, 0.0), _entity(5, "c_tank", 0.0, 900.0))
    moves = rally(scattered, (0.0, 0.0))
    assert [m["unit_id"] for m in moves] == [4, 5]
    assert {(m["x"], m["y"]) for m in moves} == {(0.0, 0.0)}


def test_a_unit_already_at_the_rally_point_is_not_re_ordered() -> None:
    """The engine runs a waypoint until it is replaced.

    Re-issuing every sample would reset the walk at the sampling rate and
    nothing would ever arrive -- the failure the attack path already learned.
    """
    assert rally((_entity(4, "c_tank", 10.0, 10.0),), (0.0, 0.0)) == ()


def test_the_rally_boundary_is_the_engines_own_arrival_test() -> None:
    """Sixty world units, which is where its rally group drops a member."""
    just_outside = _entity(4, "c_tank", RALLY_RADIUS + 0.5, 0.0)
    just_inside = _entity(5, "c_tank", RALLY_RADIUS - 0.5, 0.0)
    assert [m["unit_id"] for m in rally((just_outside, just_inside), (0.0, 0.0))] == [4]


def test_an_empty_reserve_is_sent_nowhere() -> None:
    assert rally((), (0.0, 0.0)) == ()


#: A helicopter, and the anti-air turret that can answer it. Neither is in the
#: land catalogue above, so both are stated here with the layer flags the live
#: dump reports for them.
_AIR_PROFILES = {
    **_PROFILES,
    "helicopter": profile("helicopter", 130.0, air=True),
    "antiAirTurret": profile("antiAirTurret", 250.0, land=False, air=True),
}


def test_an_airborne_target_is_not_engageable_by_ground_tanks() -> None:
    """The filter whose absence could hang a whole match.

    ``c_tank`` is the only unit the opening plan builds and it declares
    ``canAttackFlyingUnits: false``. Without this the army committed to a
    helicopter, held it for as long as it stayed visible because commitment
    keeps a visible target, and never fired ([[mechanics-combat-profile]]).
    """
    tanks = (_entity(1, "c_tank"), _entity(2, "c_tank"))
    chopper = entity(9, "helicopter", x=100.0, mine=False, hostile=True, flying=True)
    assert engageable(_AIR_PROFILES, tanks, (chopper,)) == ()


def test_a_landed_gunship_becomes_engageable() -> None:
    """State, not type: the same unit on the ground is an ordinary target."""
    tanks = (_entity(1, "c_tank"),)
    landed = entity(9, "helicopter", x=100.0, mine=False, hostile=True, flying=False)
    assert engageable(_AIR_PROFILES, tanks, (landed,)) == (landed,)


def test_an_unreachable_enemy_is_passed_over_for_one_we_can_hit() -> None:
    """The army does not stall on a target it cannot touch; it picks another."""
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        entity(9, "helicopter", x=10.0, mine=False, hostile=True, flying=True),
        _entity(10, "c_tank", 400.0, 0.0, mine=False, hostile=True),
    )
    orders = engagements(world, _CATALOGUE, _AIR_PROFILES)
    assert [e["target_id"] for e in orders] == [10]


def test_nothing_engageable_produces_no_orders_rather_than_a_doomed_one() -> None:
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        entity(9, "helicopter", x=10.0, mine=False, hostile=True, flying=True),
    )
    assert engagements(world, _CATALOGUE, _AIR_PROFILES) == ()


def test_only_the_units_that_can_reach_the_target_are_ordered() -> None:
    """A mixed force sends the half that can shoot and leaves the half that cannot.

    An order a unit cannot carry out is accepted by the engine and then does
    nothing, which is indistinguishable from a unit that is simply losing.
    """
    catalogue = {**_CATALOGUE, "antiAirTurret": _unit("antiAirTurret")}
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        entity(2, "antiAirTurret", x=0.0),
        entity(9, "helicopter", x=100.0, mine=False, hostile=True, flying=True),
    )
    orders = engagements(world, catalogue, _AIR_PROFILES)
    assert [e["attacker_id"] for e in orders] == [2]


def test_health_breaks_a_tie_between_equidistant_targets() -> None:
    """A tiebreak, not a scoring model.

    Equidistant targets are ordinary on a symmetric map, and resolving them by
    roster order is arbitrary where resolving them by what is closest to dying
    is not.
    """
    army = (_entity(1, "c_tank", 0.0, 0.0),)
    healthy = entity(9, "c_tank", x=100.0, mine=False, hostile=True, hp=100.0)
    hurt = entity(10, "c_tank", x=-100.0, mine=False, hostile=True, hp=12.0)
    assert choose_target(army, (healthy, hurt)) == hurt


def test_distance_still_outranks_health() -> None:
    """The complement: a distant cripple does not outrank a near healthy unit."""
    army = (_entity(1, "c_tank", 0.0, 0.0),)
    near_healthy = entity(9, "c_tank", x=100.0, mine=False, hostile=True, hp=100.0)
    far_cripple = entity(10, "c_tank", x=900.0, mine=False, hostile=True, hp=1.0)
    assert choose_target(army, (near_healthy, far_cripple)) == near_healthy
