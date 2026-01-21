"""Tests for sniffer world state core operations (terrain, position, radar)."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer import (
    reset_world_state,
    update_world_state_from_position,
    update_world_state_from_radar,
    world_state,
)
from tests.fakes import FakeTerrainMap


class TestWorldStateCore:
    """Tests for sniffer world state core operations."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_reset_world_state_clears_state(self) -> None:
        """Test reset_world_state clears world state and terrain map."""
        update_world_state_from_position(100, 100)
        reset_world_state()

        assert world_state._world_state["self_state"] is None
        assert world_state._terrain_map is None

    def test_load_terrain_map_returns_none_if_no_file(self) -> None:
        """Test returns None when no terrain file exists."""
        from tankpit_bot.sniffer.world_state import _load_terrain_map_if_needed

        _test_hooks.path_exists = lambda path: False

        result = _load_terrain_map_if_needed()
        assert result is None

    def test_load_terrain_map_caches_result(self) -> None:
        """Test terrain map is cached after first load."""
        from tankpit_bot.sniffer.world_state import _load_terrain_map_if_needed

        fake_terrain = FakeTerrainMap()

        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        result1 = _load_terrain_map_if_needed()
        assert result1 is fake_terrain
        assert world_state._terrain_map is fake_terrain

        result2 = _load_terrain_map_if_needed()
        assert result2 is fake_terrain

    def test_update_world_state_from_position(self) -> None:
        """Test updates self position in world state."""
        update_world_state_from_position(128, 64)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after position update")
        assert self_state["x"] == 128
        assert self_state["y"] == 64

    def test_update_world_state_from_position_updates_existing(self) -> None:
        """Test updates existing self position in world state."""
        # First call creates self_state
        update_world_state_from_position(100, 100)
        # Second call updates existing self_state
        update_world_state_from_position(200, 150)

        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None after position update")
        assert self_state["x"] == 200
        assert self_state["y"] == 150

    def test_update_world_state_from_radar_containers(self) -> None:
        """Test updates containers from radar."""
        from tankpit_bot.container import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=50, y=60, volume=100),  # fuel with 100 units
            RadarContainerDict(x=55, y=65, volume=-1),  # equipment (volume=-1)
        ]
        mines: list[RadarMineDict] = []

        update_world_state_from_radar(containers, mines)

        assert "50,60" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["50,60"]["is_fuel"] is True
        assert "55,65" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["55,65"]["is_fuel"] is False

    def test_update_world_state_from_radar_mines(self) -> None:
        """Test updates mines from radar."""
        from tankpit_bot.container import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = []
        mines: list[RadarMineDict] = [
            RadarMineDict(x=70, y=80, team=1),
            RadarMineDict(x=75, y=85, team=2),
        ]

        update_world_state_from_radar(containers, mines)

        assert "70,80" in world_state._world_state["mines"]
        assert world_state._world_state["mines"]["70,80"]["team"] == 1
        assert "75,85" in world_state._world_state["mines"]


class TestWorldStateRendering:
    """Tests for world state ASCII rendering."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_render_world_state_ascii_returns_none_without_terrain(self) -> None:
        """Test returns None when no terrain file exists."""
        from tankpit_bot.sniffer import render_world_state_ascii

        _test_hooks.path_exists = lambda path: False

        result = render_world_state_ascii()
        assert result is None

    def test_render_world_state_ascii_with_terrain(self) -> None:
        """Test renders ASCII with terrain map."""
        from tankpit_bot.sniffer import render_world_state_ascii

        fake_terrain = FakeTerrainMap()
        _test_hooks.path_exists = lambda path: True
        _test_hooks.load_terrain_map = lambda path: fake_terrain

        update_world_state_from_position(128, 128)

        result = render_world_state_ascii()
        if result is None:
            raise AssertionError("expected string, got None")
        assert "Viewport:" in result
        assert "@" in result
