"""The threat model, exercised on its own geometry.

The pool-selection tests in ``test_policy_build_order`` cover what threat does
to a decision. These cover what it computes: the two cases the policy cannot
reach from a whole-world sample, and the segment arithmetic every other answer
rests on.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.policy.threat import reach_of, route_is_exposed
from rw_bot.wire.state import Entity, Sample


def _unit(type_name: str, attack_range: float | None) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=100,
        hp=100,
        speed=0.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None
        if attack_range is None
        else Weapon(
            shoot_delay=30.0,
            attack_range=attack_range,
            direct_damage=10.0,
            direct_damage_volley=10.0,
            area_damage=0.0,
            area_damage_volley=0.0,
        ),
    )


_CATALOGUE = {
    "turret": _unit("turret", 100.0),
    "builder": _unit("builder", None),
}


def _entity(type_name: str, x: float, y: float, *, hostile: bool = True) -> Entity:
    return Entity(
        index=0,
        unit_id=900,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=1,
        mine=False,
        hostile=hostile,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=True,
        queued=0,
    )


def _sample(*entities: Entity) -> Sample:
    return Sample(
        frame=1,
        clock_ms=10,
        credits=0,
        entities=entities,
        pools=(),
        options=(),
    )


def test_a_type_the_catalogue_does_not_describe_has_no_reach() -> None:
    """There is no honest range to invent for a type that is not in the dump."""
    assert reach_of(_entity("someModTank", 0.0, 0.0), _CATALOGUE) == 0.0


def test_a_unit_with_no_weapon_has_no_reach() -> None:
    assert reach_of(_entity("builder", 0.0, 0.0), _CATALOGUE) == 0.0


def test_reach_is_the_catalogue_attack_range() -> None:
    assert reach_of(_entity("turret", 0.0, 0.0), _CATALOGUE) == 100.0


def test_a_walk_of_no_distance_is_still_judged() -> None:
    """The builder standing on its destination is a real state, not a corner case.

    It is where the builder is left after every pool it builds on, so the next
    survey asks about a route whose two ends are the same point.
    """
    standing = (500.0, 500.0)
    covering = _sample(_entity("turret", 550.0, 500.0))
    assert route_is_exposed(covering, _CATALOGUE, standing, standing) is True

    clear = _sample(_entity("turret", 650.5, 500.0))
    assert route_is_exposed(clear, _CATALOGUE, standing, standing) is False


def test_a_hostile_behind_the_start_is_not_on_the_route() -> None:
    """The route is a segment, not an infinite line.

    Projecting onto an unbounded line would put a turret 400 units behind the
    builder squarely 'on' a walk that never goes near it.
    """
    behind = _sample(_entity("turret", -400.0, 0.0))
    assert route_is_exposed(behind, _CATALOGUE, (0.0, 0.0), (1000.0, 0.0)) is False


def test_a_hostile_past_the_end_is_not_on_the_route() -> None:
    beyond = _sample(_entity("turret", 1400.0, 0.0))
    assert route_is_exposed(beyond, _CATALOGUE, (0.0, 0.0), (1000.0, 0.0)) is False


def test_a_hostile_beside_the_middle_is_on_the_route() -> None:
    beside = _sample(_entity("turret", 500.0, 99.0))
    assert route_is_exposed(beside, _CATALOGUE, (0.0, 0.0), (1000.0, 0.0)) is True


def test_the_edge_of_reach_counts_as_covered() -> None:
    """A unit at maximum range is a unit in range."""
    astride = _sample(_entity("turret", 500.0, 100.0))
    assert route_is_exposed(astride, _CATALOGUE, (0.0, 0.0), (1000.0, 0.0)) is True

    outside = _sample(_entity("turret", 500.0, 100.5))
    assert route_is_exposed(outside, _CATALOGUE, (0.0, 0.0), (1000.0, 0.0)) is False


def test_an_empty_world_exposes_nothing() -> None:
    assert route_is_exposed(_sample(), _CATALOGUE, (0.0, 0.0), (1000.0, 0.0)) is False
