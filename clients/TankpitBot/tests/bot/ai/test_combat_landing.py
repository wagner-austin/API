"""Tests for shared combat landing helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import (
    choose_combat_landing_tile,
    combat_landing_candidates,
    has_cardinal_enemy_adjacency,
)
from tankpit_bot.bot.ai.types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import make_container_state, make_mine_state, make_tank_state
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _world() -> tuple[WorldStateDict, SelfStateDict, EnemyThreatDict]:
    world = make_empty_world_state()
    self_state = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    target = make_enemy_threat(
        tank_id=50,
        x=104,
        y=100,
        distance=4,
        damage_state=0,
        rank=1,
        team=1,
        name="enemy",
        is_bot=False,
        timestamp_ms=1000,
    )
    return world, self_state, target


def test_combat_landing_candidates_orders_by_distance_and_filters_dynamic_tiles() -> None:
    world, self_state, target = _world()
    world["tanks"]["60"] = make_tank_state(
        tank_id=60,
        x=105,
        y=100,
        team=1,
        rank=1,
        damage_state=0,
        name="occupier",
        is_bot=False,
        is_self=False,
    )
    world["containers"]["104,101"] = make_container_state(
        x=104,
        y=101,
        is_fuel=False,
        volume=0,
    )
    world["mines"]["104,99"] = make_mine_state(
        x=104,
        y=99,
        mine_type=0,
        tank_id=-1,
        team=1,
    )

    assert combat_landing_candidates(world, self_state, target) == [(103, 100)]


def test_combat_landing_candidates_skip_out_of_bounds_tiles() -> None:
    world = make_empty_world_state()
    self_state = make_self_state(
        tank_id=1,
        x=1,
        y=1,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    target = make_enemy_threat(
        tank_id=50,
        x=0,
        y=0,
        distance=2,
        damage_state=0,
        rank=1,
        team=1,
        name="enemy",
        is_bot=False,
        timestamp_ms=1000,
    )

    assert combat_landing_candidates(world, self_state, target) == [(1, 0), (0, 1)]


def test_choose_combat_landing_tile_prefers_first_passable_candidate() -> None:
    world, self_state, target = _world()
    terrain = InMemoryTerrainMap(
        {
            (103, 100): InMemoryTerrainMap.ROCK,
            (105, 100): InMemoryTerrainMap.GROUND,
            (104, 101): InMemoryTerrainMap.GROUND,
            (104, 99): InMemoryTerrainMap.GROUND,
        }
    )

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (105, 100)


def test_choose_combat_landing_tile_returns_first_candidate_without_terrain() -> None:
    world, self_state, target = _world()

    assert choose_combat_landing_tile(world, self_state, target, None) == (103, 100)


def test_choose_combat_landing_tile_returns_missing_when_all_candidates_impassable() -> None:
    world, self_state, target = _world()
    terrain = InMemoryTerrainMap(
        {
            (103, 100): InMemoryTerrainMap.ROCK,
            (105, 100): InMemoryTerrainMap.WATER,
            (104, 101): InMemoryTerrainMap.ROCK,
            (104, 99): InMemoryTerrainMap.WATER,
        }
    )

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (-1, -1)


def test_has_cardinal_enemy_adjacency_matches_exact_distance_one() -> None:
    self_state = make_self_state(
        tank_id=1,
        x=103,
        y=100,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=1,
    )
    target = make_enemy_threat(
        tank_id=50,
        x=104,
        y=100,
        distance=1,
        damage_state=0,
        rank=1,
        team=1,
        name="enemy",
        is_bot=False,
        timestamp_ms=1000,
    )

    assert has_cardinal_enemy_adjacency(self_state, target) is True
    self_state["x"] = 102
    assert has_cardinal_enemy_adjacency(self_state, target) is False
