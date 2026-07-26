"""The build-order policy, exercised as the pure function it is.

No socket, no game, no clock. Every case here is a world state in and a
decision out, which is the point of keeping the deciding half pure.
"""

from __future__ import annotations

import pytest

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.build_order import (
    PLACEMENT_RING,
    completed_count,
    decide,
    find_builder,
    next_unsatisfied_index,
)
from rw_bot.wire.state import Entity, Sample


def _unit(type_name: str, price: int) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=price,
        hp=100,
        speed=0.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    "landFactory": _unit("landFactory", 300),
    "extractorT1": _unit("extractorT1", 150),
    "laboratory": _unit("laboratory", 900),
}


def _entity(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
) -> Entity:
    return Entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hp=100.0,
        max_hp=100.0,
    )


def _sample(*entities: Entity, credits: int = 4000) -> Sample:
    return Sample(frame=1, clock_ms=10, credits=credits, entities=tuple(entities))


_BUILDER = _entity(214, "builder", 4250.0, 2610.0)


def test_an_empty_plan_is_immediately_done() -> None:
    assert decide(_sample(_BUILDER), (), _CATALOGUE)["action"] == "done"


def test_the_first_structure_is_ordered_from_the_builders_position() -> None:
    decision = decide(_sample(_BUILDER), ("landFactory",), _CATALOGUE)
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"
    assert decision["unit_id"] == 214
    assert (decision["x"], decision["y"]) == (
        4250.0 + PLACEMENT_RING[0][0],
        2610.0 + PLACEMENT_RING[0][1],
    )


def test_progress_is_read_from_the_roster_not_from_a_counter() -> None:
    """A structure already standing counts, whoever built it."""
    world = _sample(_BUILDER, _entity(300, "landFactory"))
    decision = decide(world, ("landFactory", "extractorT1"), _CATALOGUE)
    assert decision["type_name"] == "extractorT1"


def test_successive_structures_take_successive_ring_positions() -> None:
    world = _sample(_BUILDER, _entity(300, "landFactory"))
    decision = decide(world, ("landFactory", "extractorT1"), _CATALOGUE)
    assert (decision["x"], decision["y"]) == (
        4250.0 + PLACEMENT_RING[1][0],
        2610.0 + PLACEMENT_RING[1][1],
    )


def test_a_destroyed_structure_is_rebuilt_rather_than_counted() -> None:
    """Counting from the roster is what makes this fall out for free."""
    world = _sample(_BUILDER, credits=4000)
    assert decide(world, ("landFactory",), _CATALOGUE)["type_name"] == "landFactory"


def test_a_finished_plan_reports_done() -> None:
    world = _sample(_BUILDER, _entity(300, "landFactory"), _entity(301, "extractorT1"))
    decision = decide(world, ("landFactory", "extractorT1"), _CATALOGUE)
    assert decision["action"] == "done"
    assert decision["reason"] == "all 2 structures built"


def test_insufficient_credits_waits_rather_than_ordering() -> None:
    world = _sample(_BUILDER, credits=899)
    decision = decide(world, ("laboratory",), _CATALOGUE)
    assert decision["action"] == "wait"
    assert decision["reason"] == "laboratory costs 900, holding 899"
    assert decision["type_name"] == ""


def test_exactly_enough_credits_orders() -> None:
    """The boundary matters: the engine spends in whole units."""
    world = _sample(_BUILDER, credits=900)
    assert decide(world, ("laboratory",), _CATALOGUE)["action"] == "build"


def test_no_builder_is_blocked_not_a_wait() -> None:
    """Waiting implies it could resolve on its own; this one cannot."""
    world = _sample(_entity(213, "commandCenter"))
    decision = decide(world, ("landFactory",), _CATALOGUE)
    assert decision["action"] == "blocked"
    assert decision["reason"] == "the player owns no builder"


def test_a_structure_missing_from_the_catalogue_is_blocked() -> None:
    decision = decide(_sample(_BUILDER), ("teleporter",), _CATALOGUE)
    assert decision["action"] == "blocked"
    assert "not in the unit catalogue" in decision["reason"]


def test_credits_are_checked_before_a_builder_is_required() -> None:
    """A plan naming an unknown structure fails on the plan, not the roster."""
    decision = decide(_sample(), ("teleporter",), _CATALOGUE)
    assert decision["action"] == "blocked"
    assert "catalogue" in decision["reason"]


@pytest.mark.parametrize("built", range(len(PLACEMENT_RING) + 2))
def test_the_placement_ring_wraps_rather_than_running_out(built: int) -> None:
    entities = [_BUILDER] + [_entity(400 + i, "landFactory") for i in range(built)]
    plan = ("landFactory",) * (built + 1)
    decision = decide(_sample(*entities), plan, _CATALOGUE)
    expected = PLACEMENT_RING[built % len(PLACEMENT_RING)]
    assert (decision["x"], decision["y"]) == (4250.0 + expected[0], 2610.0 + expected[1])


def test_completed_count_matches_each_plan_entry_only_once() -> None:
    """Two factories satisfy two plan entries, not one entry twice."""
    world = _sample(_BUILDER, _entity(300, "landFactory"), _entity(301, "landFactory"))
    assert completed_count(world, ("landFactory", "landFactory")) == 2
    assert completed_count(world, ("landFactory",)) == 1


def test_a_structure_outside_the_plan_does_not_count_as_progress() -> None:
    world = _sample(_BUILDER, _entity(300, "laboratory"))
    assert completed_count(world, ("landFactory",)) == 0


def test_find_builder_returns_none_when_there_is_none() -> None:
    assert find_builder(_sample(_entity(213, "commandCenter"))) is None


def test_find_builder_returns_the_builder() -> None:
    assert find_builder(_sample(_BUILDER)) == _BUILDER


def test_an_enemy_structure_does_not_advance_the_plan() -> None:
    """The stream carries enemies, so ownership is what makes progress mine."""
    world = _sample(_BUILDER, _entity(900, "landFactory", mine=False))
    decision = decide(world, ("landFactory",), _CATALOGUE)
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_an_enemy_builder_is_never_selected() -> None:
    """Ordering a unit we do not own would be rejected by the engine anyway."""
    world = _sample(_entity(901, "builder", 1.0, 2.0, mine=False))
    assert find_builder(world) is None
    assert decide(world, ("landFactory",), _CATALOGUE)["action"] == "blocked"


def test_an_owned_structure_still_counts_when_an_enemy_has_one_too() -> None:
    world = _sample(
        _BUILDER,
        _entity(300, "landFactory"),
        _entity(900, "landFactory", mine=False),
    )
    assert completed_count(world, ("landFactory", "landFactory")) == 1


def test_a_plan_naming_a_starting_unit_does_not_skip_earlier_entries() -> None:
    """The count-as-index conflation, which silently skipped a whole entry.

    Every game starts with a builder. Under the old reading, the plan
    ``("landFactory", "builder")`` counted as one-satisfied, jumped to index 1,
    built a second builder and never built the factory at all.
    """
    world = _sample(_entity(214, "builder"), credits=10_000)
    plan = ("landFactory", "builder")

    assert completed_count(world, plan) == 1
    assert next_unsatisfied_index(world, plan) == 0

    decision = decide(world, plan, _CATALOGUE)
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_the_first_unsatisfied_entry_is_found_past_a_satisfied_one() -> None:
    world = _sample(_entity(214, "builder"), _entity(300, "landFactory"), credits=10_000)
    assert next_unsatisfied_index(world, ("landFactory", "builder")) == 2
    assert next_unsatisfied_index(world, ("landFactory", "builder", "landFactory")) == 2


def test_an_enemy_unit_never_satisfies_a_plan_entry() -> None:
    world = _sample(_entity(900, "landFactory", mine=False), credits=10_000)
    assert next_unsatisfied_index(world, ("landFactory",)) == 0
