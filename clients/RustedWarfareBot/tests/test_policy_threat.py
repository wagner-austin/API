"""The threat model, exercised on its own geometry.

The pool-selection tests in ``test_policy_build_order`` cover what threat does
to a decision. These cover what it computes: the segment arithmetic every answer
rests on, and the layer test that decides whether a given hostile is a threat to
a given traveller at all.
"""

from __future__ import annotations

from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.threat import route_is_exposed
from rw_bot.wire.state import Entity
from tests.wire_fixtures import enemy, entity, sample


def _profile(
    type_name: str,
    attack_range: float,
    *,
    land: bool = True,
    air: bool = False,
    underwater: bool = False,
    out_of_water: bool = True,
) -> CombatProfile:
    return CombatProfile(
        index=0,
        type_name=type_name,
        attack_range=attack_range,
        hits_land=land,
        hits_air=air,
        hits_underwater=underwater,
        hits_land_out_of_water=out_of_water,
    )


#: Combat profiles as the registry dump gives them.
#:
#: Every registered type appears, armed or not — that completeness is the
#: contract the threat model relies on, which is why the unarmed builder is
#: present at zero rather than absent. The figures are the ones the live dump
#: reports for these types.
_PROFILES = {
    "turret": _profile("turret", 100.0),
    "builder": _profile("builder", 0.0, land=False),
    "antiAirTurret": _profile("antiAirTurret", 250.0, land=False, air=True),
    "heavySub": _profile("heavySub", 210.0, underwater=True, out_of_water=False),
}

_BUILDER = entity(214, "builder")


def _walker(x: float, y: float) -> Entity:
    """The unit whose walk is being judged: an ordinary ground builder."""
    return entity(214, "builder", x=x, y=y)


def test_a_walk_of_no_distance_is_still_judged() -> None:
    """The builder standing on its destination is a real state, not a corner case.

    It is where the builder is left after every pool it builds on, so the next
    survey asks about a route whose two ends are the same point.
    """
    standing = (500.0, 500.0)
    covering = sample(enemy(900, "turret", x=550.0, y=500.0))
    assert route_is_exposed(covering, _PROFILES, _BUILDER, standing, standing) is True

    clear = sample(enemy(900, "turret", x=650.5, y=500.0))
    assert route_is_exposed(clear, _PROFILES, _BUILDER, standing, standing) is False


def test_a_hostile_behind_the_start_is_not_on_the_route() -> None:
    """The route is a segment, not an infinite line.

    Projecting onto an unbounded line would put a turret 400 units behind the
    builder squarely 'on' a walk that never goes near it.
    """
    behind = sample(enemy(900, "turret", x=-400.0, y=0.0))
    assert route_is_exposed(behind, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_a_hostile_past_the_end_is_not_on_the_route() -> None:
    beyond = sample(enemy(900, "turret", x=1400.0, y=0.0))
    assert route_is_exposed(beyond, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_a_hostile_beside_the_middle_is_on_the_route() -> None:
    beside = sample(enemy(900, "turret", x=500.0, y=99.0))
    assert route_is_exposed(beside, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is True


def test_the_edge_of_reach_counts_as_covered() -> None:
    """A unit at maximum range is a unit in range."""
    astride = sample(enemy(900, "turret", x=500.0, y=100.0))
    assert route_is_exposed(astride, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is True

    outside = sample(enemy(900, "turret", x=500.0, y=100.5))
    assert route_is_exposed(outside, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_an_empty_world_exposes_nothing() -> None:
    assert route_is_exposed(sample(), _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_an_unarmed_hostile_is_an_obstacle_rather_than_a_threat() -> None:
    """An enemy builder standing on the route does not endanger anything."""
    harmless = sample(enemy(900, "builder", x=500.0, y=0.0))
    assert route_is_exposed(harmless, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_an_ally_on_the_route_is_not_a_threat() -> None:
    """Hostility is the engine's answer, not the negation of ownership."""
    friendly = entity(900, "turret", x=500.0, y=0.0, team=2, mine=False, hostile=False)
    assert (
        route_is_exposed(sample(friendly), _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False
    )


def test_an_anti_air_turret_does_not_threaten_a_ground_builder() -> None:
    """The layer test, and the reason it is not an optimisation.

    ``antiAirTurret`` is armed, has a 250-unit reach, and reports
    ``hits_land: false`` in the live dump. Counting reach alone made it rule out
    every pool within 250 units of one, on a map where it cannot touch the
    builder walking past it ([[mechanics-combat-profile]]).
    """
    covered = sample(enemy(900, "antiAirTurret", x=500.0, y=0.0))
    assert route_is_exposed(covered, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_the_same_turret_does_threaten_something_airborne() -> None:
    """The complement, so the case above is the layer test and not a dead branch."""
    flyer = entity(215, "helicopter", flying=True)
    covered = sample(enemy(900, "antiAirTurret", x=500.0, y=0.0))
    assert route_is_exposed(covered, _PROFILES, flyer, (0.0, 0.0), (1000.0, 0.0)) is True


def test_a_torpedo_boat_does_not_threaten_a_builder_on_dry_land() -> None:
    """``hits_land_out_of_water`` is false for every submarine in the dump.

    A torpedo has to strike something in the water, so a submarine 100 units off
    the shoreline is not a reason to refuse a pool inland.
    """
    lurking = sample(enemy(900, "heavySub", x=500.0, y=0.0))
    assert route_is_exposed(lurking, _PROFILES, _BUILDER, (0.0, 0.0), (1000.0, 0.0)) is False


def test_the_same_torpedo_boat_threatens_a_builder_that_is_in_the_water() -> None:
    wading = entity(214, "builder", touching_water=True)
    lurking = sample(enemy(900, "heavySub", x=500.0, y=0.0))
    assert route_is_exposed(lurking, _PROFILES, wading, (0.0, 0.0), (1000.0, 0.0)) is True
