"""Tests for ASCII rendering functions."""

from pathlib import Path

import pytest
from PIL import Image

from tankpit_bot.state import (
    ASCII_ALLY,
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
    TERRAIN_FERRY,
    TERRAIN_FERRY_ROCK,
    TERRAIN_GROUND,
    TERRAIN_ROCK_A,
    TERRAIN_ROCK_AB,
    TERRAIN_ROCK_B,
    add_mine,
    make_empty_world_state,
    render_world_ascii,
    terrain_to_ascii,
    update_container_from_radar,
    update_self_from_movement_response,
    update_tank_from_registry,
)
from tankpit_bot.terrain import TerrainMap


class TestTerrainToAscii:
    """Tests for terrain_to_ascii."""

    def test_ground(self) -> None:
        """Ground terrain returns dot."""
        assert terrain_to_ascii(TERRAIN_GROUND) == ASCII_GROUND

    def test_rock_types(self) -> None:
        """Rock terrain types return hash."""
        assert terrain_to_ascii(TERRAIN_ROCK_A) == ASCII_ROCK
        assert terrain_to_ascii(TERRAIN_ROCK_B) == ASCII_ROCK
        assert terrain_to_ascii(TERRAIN_ROCK_AB) == ASCII_ROCK

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
    img = Image.new("RGB", (256, 256), (60, 129, 85))  # Dark green = ground
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
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=TEAM_RED,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=5,
            y=5,
            timestamp_ms=1000,
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
        state = update_tank_from_registry(
            state,
            tank_id=42,
            team=TEAM_BLUE,
            name="Ally",
            rank=1,
            is_bot=False,
            x=5,
            y=5,
            timestamp_ms=1000,
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
        state = update_tank_from_registry(
            state,
            tank_id=10,
            team=TEAM_BLUE,
            name="Ally",
            rank=1,
            is_bot=False,
            x=5,
            y=5,
            timestamp_ms=600,
        )
        state = update_tank_from_registry(
            state,
            tank_id=20,
            team=TEAM_RED,
            name="Enemy",
            rank=1,
            is_bot=False,
            x=6,
            y=6,
            timestamp_ms=700,
        )
        output = render_world_ascii(state, terrain_map)

        assert "Tanks:" in output
        assert "allies=" in output
        assert "enemies=" in output
