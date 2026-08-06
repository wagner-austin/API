"""Choosing what to attack, exercised as the pure function it is.

No socket and no game: a world state goes in and a set of engagements comes
out, which is what keeps the fighting logic arguable without a match running.

Which units are army, which enemies can be reached and hit, and how the
attackers are dealt into groups against them. When a wave is released to go
looking is ``test_policy_muster``; the skirmish both argue over is
:mod:`tests.combat_fixtures`.
"""

from __future__ import annotations

import pytest

from rw_bot.mechanics.combat_profile import CombatProfileError, is_armed
from rw_bot.policy.combat import (
    choose_target,
    engageable,
    engagements,
    find_army,
    find_targets,
    is_mobile,
)
from tests.combat_fixtures import CATALOGUE, PROFILES, sample, unit, unit_stats
from tests.wire_fixtures import entity, profile, profiles_for

#: A helicopter, and the anti-air turret that can answer it. Neither is in the
#: land catalogue the fixtures build, so both are stated here with the layer
#: flags the live dump reports for them.
_AIRPROFILES = {
    **PROFILES,
    "helicopter": profile("helicopter", 130.0, air=True),
    "antiAirTurret": profile("antiAirTurret", 250.0, land=False, air=True),
}


def test_an_armed_mobile_unit_is_army() -> None:
    tank = unit(1, "c_tank")
    assert find_army(sample(tank), CATALOGUE, PROFILES) == (tank,)


def test_a_builder_is_not_army() -> None:
    """Unarmed. Sending it at a tank is a Builder thrown away."""
    assert find_army(sample(unit(1, "builder")), CATALOGUE, PROFILES) == ()
    assert is_armed(PROFILES, unit(1, "builder")) is False


def test_a_turret_is_armed_but_not_army() -> None:
    """It cannot travel, so an order to attack anything distant is undeliverable."""
    turret = unit(1, "c_turret_t1")
    assert is_armed(PROFILES, turret) is True
    assert is_mobile(turret, CATALOGUE) is False
    assert find_army(sample(turret), CATALOGUE, PROFILES) == ()


def test_the_editor_placeholder_is_never_army() -> None:
    """Owned, off-map, and not a unit -- the same exclusion producer selection needs."""
    assert find_army(sample(unit(217, "editorOrBuilder")), CATALOGUE, PROFILES) == ()


def test_an_unfinished_tank_is_not_army() -> None:
    """It does not exist yet, whatever the roster says."""
    assert find_army(sample(unit(1, "c_tank", complete=False)), CATALOGUE, PROFILES) == ()


def test_an_enemy_tank_is_not_our_army() -> None:
    enemy = unit(9, "c_tank", mine=False, hostile=True)
    assert find_army(sample(enemy), CATALOGUE, PROFILES) == ()


def test_a_type_the_combat_dump_does_not_describe_fails_loudly() -> None:
    """A stale dump is a fault, not a unit to skip.

    This used to be absorbed: an unknown type was quietly treated as unarmed and
    left out of the army. That is the same silence that reported every turret as
    harmless, and it is indistinguishable from a real answer. The dump is
    regenerated from the registry and covers every registered type, so a miss
    means the dump and the running game disagree ([[mechanics-combat-profile]]).
    """
    with pytest.raises(CombatProfileError) as caught:
        find_army(sample(unit(1, "mysteryTank")), CATALOGUE, PROFILES)
    assert caught.value.code == "RW-COMBAT-002"
    assert "mysteryTank" in caught.value.message


def test_a_type_the_catalogue_does_not_price_cannot_travel() -> None:
    """Mobility is still the catalogue's answer, and still fails safe."""
    assert is_mobile(unit(1, "mysteryTank"), CATALOGUE) is False


def test_targets_are_the_engines_hostiles_not_merely_the_unowned() -> None:
    """An ally is not mine and is not a target, which "not mine" gets wrong."""
    ally = unit(5, "c_tank", mine=False, hostile=False)
    enemy = unit(9, "c_tank", mine=False, hostile=True)
    assert find_targets(sample(ally, enemy)) == (enemy,)


def test_the_whole_army_commits_to_one_target() -> None:
    """Concentrating fire is the one tactic that matters at this scale."""
    near = unit(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    far = unit(10, "c_tank", 900.0, 0.0, mine=False, hostile=True)
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        unit(2, "c_tank", 10.0, 0.0),
        near,
        far,
    )
    orders = engagements(world, CATALOGUE, PROFILES)
    assert [e["attacker_id"] for e in orders] == [1, 2]
    assert {e["target_id"] for e in orders} == {9}


def test_the_target_is_nearest_to_the_army_not_to_one_unit() -> None:
    """A split force converges instead of each unit picking its own enemy."""
    army = (unit(1, "c_tank", 0.0, 0.0), unit(2, "c_tank", 1000.0, 0.0))
    near_to_one = unit(9, "c_tank", 0.0, 400.0, mine=False, hostile=True)
    near_to_centre = unit(10, "c_tank", 500.0, 0.0, mine=False, hostile=True)
    assert choose_target(army, (near_to_one, near_to_centre)) == near_to_centre


def test_the_current_target_is_kept_while_it_lives() -> None:
    """Commitment, and the whole reason the churn happened.

    Nearest is measured from the army centre, and that centre shifts whenever a
    unit dies or a new one rolls out. Re-choosing every sample re-tasked the
    whole army on a flip that could be a few world units wide.
    """
    held = unit(9, "c_tank", 900.0, 0.0, mine=False, hostile=True)
    nearer = unit(10, "c_tank", 10.0, 0.0, mine=False, hostile=True)
    army = (unit(1, "c_tank", 0.0, 0.0),)
    assert choose_target(army, (held, nearer), holding=9) == held
    assert choose_target(army, (held, nearer)) == nearer


def test_a_target_that_is_gone_is_replaced() -> None:
    """Holding is not clinging: a dead target frees the army to re-commit."""
    nearer = unit(10, "c_tank", 10.0, 0.0, mine=False, hostile=True)
    army = (unit(1, "c_tank", 0.0, 0.0),)
    assert choose_target(army, (nearer,), holding=9) == nearer


def test_the_held_target_carries_through_engagements() -> None:
    """Held per attacker: the unit keeps ITS target, not the army's."""
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        unit(9, "c_tank", 900.0, 0.0, mine=False, hostile=True),
        unit(10, "c_tank", 10.0, 0.0, mine=False, hostile=True),
    )
    assert engagements(world, CATALOGUE, PROFILES, held={1: 9})[0]["target_id"] == 9
    assert engagements(world, CATALOGUE, PROFILES)[0]["target_id"] == 10


def test_a_group_stops_growing_once_one_volley_kills() -> None:
    """Kill-sized, not army-sized: the screening plateau's first lever.

    Six 17-damage volleys cover 100 hp; the seventh tank starts the next
    group instead of piling on ([[policy-combat]], log 2026-07-31).
    """
    near = unit(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    far = unit(10, "c_tank", 300.0, 0.0, mine=False, hostile=True)
    world = sample(
        *(unit(n, "c_tank", 0.0, float(n)) for n in range(1, 8)),
        near,
        far,
    )
    orders = engagements(world, CATALOGUE, PROFILES)
    by_target: dict[int, int] = {}
    for order in orders:
        by_target[order["target_id"]] = by_target.get(order["target_id"], 0) + 1
    assert by_target == {9: 6, 10: 1}


def test_overflow_joins_the_nearest_group_rather_than_idling() -> None:
    """Every group lethal and units to spare: overkill beats watching."""
    near = unit(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    world = sample(
        *(unit(n, "c_tank", 0.0, float(n)) for n in range(1, 8)),
        near,
    )
    orders = engagements(world, CATALOGUE, PROFILES)
    assert len(orders) == 7
    assert {order["target_id"] for order in orders} == {9}


def test_a_nearer_escort_outranks_a_distant_extractor() -> None:
    """The refutation pin for "the wallet outranks the war".

    Ranking visible income structures ahead of distance doubled the drops
    and strangled two of three screening seeds -- the army chased extractors
    past their escorts and ate free damage the whole walk (screen-vh9m, log
    2026-07-31). The fight in front of the army is the fight; the raid party
    owns the wallet.
    """
    catalogue = dict(CATALOGUE)
    catalogue["extractorT1"] = unit_stats("extractorT1", speed=0.0, armed=False)
    profiles = profiles_for(catalogue)
    tank = unit(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    wallet = unit(10, "extractorT1", 500.0, 0.0, mine=False, hostile=True)
    world = sample(unit(1, "c_tank", 0.0, 0.0), tank, wallet)
    orders = engagements(world, catalogue, profiles)
    assert [o["target_id"] for o in orders] == [9]


def test_at_most_two_groups_fill_at_once() -> None:
    """The fist stays a fist: a third target waits until a group is lethal.

    Thirteen tanks and three 100-hp targets: six fill the nearest group, six
    the second, and the thirteenth reinforces the nearest open group rather
    than opening a third ([[policy-combat]]).
    """
    targets = tuple(
        unit(90 + n, "c_tank", 100.0 + 10.0 * n, 0.0, mine=False, hostile=True) for n in range(3)
    )
    world = sample(
        *(unit(n, "c_tank", 0.0, float(n)) for n in range(1, 14)),
        *targets,
    )
    orders = engagements(world, CATALOGUE, PROFILES)
    by_target: dict[int, int] = {}
    for order in orders:
        by_target[order["target_id"]] = by_target.get(order["target_id"], 0) + 1
    assert by_target == {90: 7, 91: 6}


def test_a_unit_armed_in_profile_but_weaponless_in_catalogue_contributes_nothing() -> None:
    """The two dumps are independent sources and may disagree.

    The profiles come from the agent's combat dump and the prices from
    ``-printunits``; a modded type can be armed in one and weaponless in the
    other. Such a unit still fights -- the profile says its fire reaches --
    but its volley counts for nothing, so its group keeps filling.
    """
    catalogue = dict(CATALOGUE)
    catalogue["oddity"] = unit_stats("oddity", armed=False)
    profiles = dict(PROFILES)
    profiles["oddity"] = profile("oddity", 110.0, land=True)
    enemy = unit(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    world = sample(unit(1, "oddity", 0.0, 0.0), unit(2, "c_tank", 0.0, 1.0), enemy)
    orders = engagements(world, catalogue, profiles)
    assert {(o["attacker_id"], o["target_id"]) for o in orders} == {(1, 9), (2, 9)}


def test_a_freed_attacker_is_dealt_into_a_group_afresh() -> None:
    """A dead target frees only ITS group; other assignments stand."""
    near = unit(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    far = unit(10, "c_tank", 300.0, 0.0, mine=False, hostile=True)
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        unit(2, "c_tank", 0.0, 1.0),
        near,
        far,
    )
    # Attacker 1's target 11 is gone; attacker 2 holds 10. Attacker 1 is
    # re-dealt to the nearest open group (9); attacker 2 stays on 10.
    orders = engagements(world, CATALOGUE, PROFILES, held={1: 11, 2: 10})
    assert {(o["attacker_id"], o["target_id"]) for o in orders} == {(1, 9), (2, 10)}


def test_no_army_produces_no_orders() -> None:
    enemy = unit(9, "c_tank", mine=False, hostile=True)
    assert engagements(sample(enemy), CATALOGUE, PROFILES) == ()
    assert choose_target((), (enemy,)) is None


def test_no_enemy_produces_no_orders() -> None:
    tank = unit(1, "c_tank")
    assert engagements(sample(tank), CATALOGUE, PROFILES) == ()
    assert choose_target((tank,), ()) is None


def test_the_engagement_reason_names_both_sides() -> None:
    """The run log has to be readable back without re-deriving the choice."""
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        unit(9, "commandCenter", 50.0, 0.0, mine=False, hostile=True),
    )
    assert engagements(world, CATALOGUE, PROFILES)[0]["reason"] == "c_tank -> commandCenter 9"


def test_an_airborne_target_is_not_engageable_by_ground_tanks() -> None:
    """The filter whose absence could hang a whole match.

    ``c_tank`` is the only unit the opening plan builds and it declares
    ``canAttackFlyingUnits: false``. Without this the army committed to a
    helicopter, held it for as long as it stayed visible because commitment
    keeps a visible target, and never fired ([[mechanics-combat-profile]]).
    """
    tanks = (unit(1, "c_tank"), unit(2, "c_tank"))
    chopper = entity(9, "helicopter", x=100.0, mine=False, hostile=True, flying=True)
    assert engageable(_AIRPROFILES, tanks, (chopper,)) == ()


def test_a_landed_gunship_becomes_engageable() -> None:
    """State, not type: the same unit on the ground is an ordinary target."""
    tanks = (unit(1, "c_tank"),)
    landed = entity(9, "helicopter", x=100.0, mine=False, hostile=True, flying=False)
    assert engageable(_AIRPROFILES, tanks, (landed,)) == (landed,)


def test_an_unreachable_enemy_is_passed_over_for_one_we_can_hit() -> None:
    """The army does not stall on a target it cannot touch; it picks another."""
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        entity(9, "helicopter", x=10.0, mine=False, hostile=True, flying=True),
        unit(10, "c_tank", 400.0, 0.0, mine=False, hostile=True),
    )
    orders = engagements(world, CATALOGUE, _AIRPROFILES)
    assert [e["target_id"] for e in orders] == [10]


def test_nothing_engageable_produces_no_orders_rather_than_a_doomed_one() -> None:
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        entity(9, "helicopter", x=10.0, mine=False, hostile=True, flying=True),
    )
    assert engagements(world, CATALOGUE, _AIRPROFILES) == ()


def test_only_the_units_that_can_reach_the_target_are_ordered() -> None:
    """A mixed force sends the half that can shoot and leaves the half that cannot.

    An order a unit cannot carry out is accepted by the engine and then does
    nothing, which is indistinguishable from a unit that is simply losing.
    """
    catalogue = {**CATALOGUE, "antiAirTurret": unit_stats("antiAirTurret")}
    world = sample(
        unit(1, "c_tank", 0.0, 0.0),
        entity(2, "antiAirTurret", x=0.0),
        entity(9, "helicopter", x=100.0, mine=False, hostile=True, flying=True),
    )
    orders = engagements(world, catalogue, _AIRPROFILES)
    assert [e["attacker_id"] for e in orders] == [2]


def test_health_breaks_a_tie_between_equidistant_targets() -> None:
    """A tiebreak, not a scoring model.

    Equidistant targets are ordinary on a symmetric map, and resolving them by
    roster order is arbitrary where resolving them by what is closest to dying
    is not.
    """
    army = (unit(1, "c_tank", 0.0, 0.0),)
    healthy = entity(9, "c_tank", x=100.0, mine=False, hostile=True, hp=100.0)
    hurt = entity(10, "c_tank", x=-100.0, mine=False, hostile=True, hp=12.0)
    assert choose_target(army, (healthy, hurt)) == hurt


def test_distance_still_outranks_health() -> None:
    """The complement: a distant cripple does not outrank a near healthy unit."""
    army = (unit(1, "c_tank", 0.0, 0.0),)
    near_healthy = entity(9, "c_tank", x=100.0, mine=False, hostile=True, hp=100.0)
    far_cripple = entity(10, "c_tank", x=900.0, mine=False, hostile=True, hp=1.0)
    assert choose_target(army, (near_healthy, far_cripple)) == near_healthy
