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

    assert combat_landing_candidates(world, self_state, target, None, 100000) == [(103, 100)]


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

    assert combat_landing_candidates(world, self_state, target, None, 100000) == [
        (1, 0),
        (0, 1),
    ]


def test_choose_combat_landing_tile_returns_target_coords_with_terrain() -> None:
    world, self_state, target = _world()
    terrain = InMemoryTerrainMap(
        {
            (103, 100): InMemoryTerrainMap.ROCK,
            (105, 100): InMemoryTerrainMap.GROUND,
            (104, 101): InMemoryTerrainMap.GROUND,
            (104, 99): InMemoryTerrainMap.GROUND,
        }
    )

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (104, 100)


def test_choose_combat_landing_tile_returns_target_coords_without_terrain() -> None:
    world, self_state, target = _world()

    assert choose_combat_landing_tile(world, self_state, target, None) == (104, 100)


def test_choose_combat_landing_tile_returns_target_when_adjacent_impassable() -> None:
    """Server handles displacement when the target tile is occupied or impassable."""
    world, self_state, target = _world()
    terrain = InMemoryTerrainMap(
        {
            (103, 100): InMemoryTerrainMap.ROCK,
            (105, 100): InMemoryTerrainMap.WATER,
            (104, 101): InMemoryTerrainMap.ROCK,
            (104, 99): InMemoryTerrainMap.WATER,
        }
    )

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (104, 100)


def test_choose_combat_landing_tile_standoff_when_target_tile_impassable() -> None:
    """A ferry rider on open water gets the nearest shore stand-off landing.

    Live 2026-07-29: Yuppler rode a ferry at (128,102) -- water on his
    tile and every neighbor -- and the direct-to-target aim would have
    been refused by the server. The chooser must aim at the passable
    tile nearest the target instead (water never blocks the shot).
    """
    world, self_state, target = _world()
    water = {(x, y): InMemoryTerrainMap.WATER for x in range(102, 108) for y in range(97, 104)}
    terrain = InMemoryTerrainMap(water)

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (101, 100)


def test_choose_combat_landing_tile_standoff_skips_occupied_and_prefers_self() -> None:
    """Occupied stand-off tiles are skipped; distance ties break toward self."""
    world, self_state, target = _world()
    water = {(x, y): InMemoryTerrainMap.WATER for x in range(102, 108) for y in range(97, 104)}
    terrain = InMemoryTerrainMap(water)
    world["tanks"]["60"] = make_tank_state(
        tank_id=60,
        x=101,
        y=100,
        team=1,
        rank=1,
        damage_state=0,
        name="occupier",
        is_bot=False,
        is_self=False,
    )

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (100, 100)


def test_choose_combat_landing_tile_returns_target_when_no_standoff_exists() -> None:
    """With no passable tile inside the shot-range diamond, fall back to target coords."""
    world, self_state, target = _world()
    terrain = InMemoryTerrainMap.from_passable_set(set())

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (104, 100)


def test_choose_combat_landing_tile_standoff_clips_map_bounds() -> None:
    """A water-locked target at the map corner only considers in-bounds tiles."""
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
    terrain = InMemoryTerrainMap.from_passable_set({(2, 0)})

    assert choose_combat_landing_tile(world, self_state, target, terrain) == (2, 0)


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


def test_combat_landing_candidates_skip_terrain_blocked_tiles() -> None:
    """Impassable composed terrain removes a candidate (F20).

    Run bot-20260730-110x ticks 904-949: the walk-close re-dispatched
    a move onto unwalkable ground forty-plus consecutive ticks because
    candidates never consulted terrain.
    """
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    world, self_state, target = _world()
    terrain = InMemoryTerrainMap(
        terrain_data={
            (103, 100): InMemoryTerrainMap.ROCK,
            (104, 101): "W",
            (104, 99): "W",
        }
    )

    assert combat_landing_candidates(world, self_state, target, terrain, 100000) == [(105, 100)]


def test_combat_landing_candidates_skip_failed_move_marked_tiles() -> None:
    """A live failed-move mark removes a candidate.

    The server already said "you can't go there" — re-selecting the
    tile inside the mark's TTL re-derives the identical rejected move.
    """
    from tankpit_bot.sniffer.world_state import (
        mark_move_target_failed,
        reset_world_state,
    )

    reset_world_state()
    world, self_state, target = _world()
    mark_move_target_failed(103, 100, 99000)

    try:
        result = combat_landing_candidates(world, self_state, target, None, 100000)
    finally:
        reset_world_state()

    assert result == [(105, 100), (104, 101), (104, 99)]
