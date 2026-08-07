"""Tests for ASCII rendering functions."""

from pathlib import Path

import pytest

from tankpit_bot._pillow import load_pillow_image_module
from tankpit_bot.state import (
    add_mine,
    apply_tank_observation,
    make_empty_world_state,
    render_world_ascii,
    terrain_to_ascii,
    update_container_from_radar,
    update_self_from_movement_response,
)
from tankpit_bot.state.types import make_tank_observation
from tankpit_bot.terrain import TerrainMap
from tankpit_bot.types.constants import (
    ASCII_ALLY,
    ASCII_BRIDGE,
    ASCII_ENEMY,
    ASCII_EQUIPMENT,
    ASCII_FERRY,
    ASCII_FUEL,
    ASCII_GROUND,
    ASCII_MINE,
    ASCII_ROCK,
    ASCII_SELF,
    ASCII_UNKNOWN,
    TEAM_BLUE,
    TEAM_RED,
    TERRAIN_BLOCK_BRIDGE,
    TERRAIN_BLOCK_LAND,
    TERRAIN_BLOCK_STACKED,
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
)

_IMAGE = load_pillow_image_module()


class TestTerrainToAscii:
    """Tests for terrain_to_ascii."""

    def test_ground(self) -> None:
        """Ground terrain returns dot."""
        assert terrain_to_ascii(TERRAIN_GROUND) == ASCII_GROUND

    def test_block_types(self) -> None:
        """Bridges render as '='; land and stacked blocks as rock.

        Wire values 1/2/3 are movable concrete blocks (2026-07-20,
        [[movable-blocks]]): a bridge is walkable and gets its own
        glyph so viewport dumps show it; land/stacked blocks are
        obstacles and render as rock.
        """
        assert terrain_to_ascii(TERRAIN_BLOCK_BRIDGE) == ASCII_BRIDGE
        assert terrain_to_ascii(TERRAIN_BLOCK_LAND) == ASCII_ROCK
        assert terrain_to_ascii(TERRAIN_BLOCK_STACKED) == ASCII_ROCK

    def test_ferry(self) -> None:
        """Ferry terrain returns tilde."""
        assert terrain_to_ascii(TERRAIN_FERRY) == ASCII_FERRY

    def test_ferry_rock(self) -> None:
        """Ferry + rock returns hash."""
        assert terrain_to_ascii(TERRAIN_FERRY_ROCK) == ASCII_ROCK

    def test_unknown(self) -> None:
        """Unknown terrain returns question mark."""
        assert terrain_to_ascii(99) == ASCII_UNKNOWN


@pytest.fixture()
def terrain_map(tmp_path: Path) -> TerrainMap:
    """Create a test TerrainMap with uniform ground."""
    gif_path = tmp_path / "test.gif"
    img = _IMAGE.new("RGB", (256, 256), (60, 129, 85))  # Dark green = ground
    img.save(gif_path)
    return TerrainMap(gif_path)


class TestRenderWorldAscii:
    """Tests for render_world_ascii."""

    def test_renders_empty_state(self, terrain_map: TerrainMap) -> None:
        """Renders empty state with ground tiles."""
        state = make_empty_world_state()
        output = render_world_ascii(state, terrain_map)

        assert "Viewport:" in output
        assert "Legend:" in output
        assert ASCII_GROUND in output

    def test_renders_self(self, terrain_map: TerrainMap) -> None:
        """Renders self position."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state, tank_id=1, x=5, y=5, team=0, rank=0, leaderboard_position=1, timestamp_ms=1000
        )
        output = render_world_ascii(state, terrain_map)

        assert ASCII_SELF in output
        assert "Self:" in output

    def test_renders_fuel_container(self, terrain_map: TerrainMap) -> None:
        """Renders fuel container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=5, y=5, volume=500, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        assert ASCII_FUEL in output

    def test_renders_equipment_container(self, terrain_map: TerrainMap) -> None:
        """Renders equipment container."""
        state = make_empty_world_state()
        state = update_container_from_radar(state, x=5, y=5, volume=-1, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        assert ASCII_EQUIPMENT in output

    def test_renders_mine(self, terrain_map: TerrainMap) -> None:
        """Renders mine."""
        state = make_empty_world_state()
        state = add_mine(state, x=5, y=5, mine_type=1, tank_id=42, team=0, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        assert ASCII_MINE in output

    def test_renders_enemy_tank(self, terrain_map: TerrainMap) -> None:
        """Renders enemy tank."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=0,
            y=0,
            team=TEAM_BLUE,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=42,
                timestamp_ms=1000,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(5, 5),
                team=TEAM_RED,
                rank=1,
                name="Enemy",
                is_bot=False,
            ),
        )
        output = render_world_ascii(state, terrain_map)

        assert ASCII_ENEMY in output

    def test_renders_ally_tank(self, terrain_map: TerrainMap) -> None:
        """Renders ally tank."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=0,
            y=0,
            team=TEAM_BLUE,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=42,
                timestamp_ms=1000,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(5, 5),
                team=TEAM_BLUE,
                rank=1,
                name="Ally",
                is_bot=False,
            ),
        )
        output = render_world_ascii(state, terrain_map)

        assert ASCII_ALLY in output

    def test_renders_terrain_from_map(self, terrain_map: TerrainMap) -> None:
        """Renders terrain from TerrainMap."""
        state = make_empty_world_state()
        output = render_world_ascii(state, terrain_map)

        assert ASCII_GROUND in output

    def test_self_takes_priority(self, terrain_map: TerrainMap) -> None:
        """Self position takes priority over other entities."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=5,
            y=5,
            team=0,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        # Add container at same position
        state = update_container_from_radar(state, x=5, y=5, volume=500, timestamp_ms=1000)
        output = render_world_ascii(state, terrain_map)

        # Should show @ not F at position 5,5
        assert ASCII_SELF in output

    def test_shows_tank_counts(self, terrain_map: TerrainMap) -> None:
        """Shows tank count summary."""
        state = make_empty_world_state()
        state = update_self_from_movement_response(
            state,
            tank_id=1,
            x=0,
            y=0,
            team=TEAM_BLUE,
            rank=0,
            leaderboard_position=1,
            timestamp_ms=500,
        )
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=10,
                timestamp_ms=600,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(5, 5),
                team=TEAM_BLUE,
                rank=1,
                name="Ally",
                is_bot=False,
            ),
        )
        state = apply_tank_observation(
            state,
            make_tank_observation(
                tank_id=20,
                timestamp_ms=700,
                is_wire_sourced=True,
                storage_source="viewport",
                position=(6, 6),
                team=TEAM_RED,
                rank=1,
                name="Enemy",
                is_bot=False,
            ),
        )
        output = render_world_ascii(state, terrain_map)

        assert "Tanks:" in output
        assert "allies=" in output
        assert "enemies=" in output
