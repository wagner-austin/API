"""Tests for _apply_waypoints helper function."""

from __future__ import annotations

from tankpit_bot.sniffer import world_state_tiles


class TestApplyWaypoints:
    """Tests for _apply_waypoints helper function."""

    def test_apply_waypoints_empty(self) -> None:
        """Test empty waypoints returns start position."""
        x, y = world_state_tiles.apply_waypoints(100, 100, "")
        assert x == 100
        assert y == 100

    def test_apply_waypoints_north(self) -> None:
        """Test north direction decreases y."""
        x, y = world_state_tiles.apply_waypoints(100, 100, "nnn")
        assert x == 100
        assert y == 97

    def test_apply_waypoints_south(self) -> None:
        """Test south direction increases y."""
        x, y = world_state_tiles.apply_waypoints(100, 100, "sss")
        assert x == 100
        assert y == 103

    def test_apply_waypoints_east(self) -> None:
        """Test east direction increases x."""
        x, y = world_state_tiles.apply_waypoints(100, 100, "eee")
        assert x == 103
        assert y == 100

    def test_apply_waypoints_west(self) -> None:
        """Test west direction decreases x."""
        x, y = world_state_tiles.apply_waypoints(100, 100, "www")
        assert x == 97
        assert y == 100

    def test_apply_waypoints_mixed(self) -> None:
        """Test mixed waypoints."""
        # wsss = west, south, south, south
        x, y = world_state_tiles.apply_waypoints(100, 100, "wsss")
        assert x == 99
        assert y == 103

    def test_apply_waypoints_complex_path(self) -> None:
        """Test complex path from actual game data."""
        # eeeessssssseeeeeeeeennnnnnn = 4e + 7s + 9e + 7n
        # Final: (100 + 4 + 9, 100 + 7 - 7) = (113, 100)
        x, y = world_state_tiles.apply_waypoints(100, 100, "eeeessssssseeeeeeeeennnnnnn")
        assert x == 113
        assert y == 100

    def test_apply_waypoints_west_then_continue(self) -> None:
        """Test west followed by other directions (ensures loop continuation after w)."""
        # wne = west, north, east -> back to start
        x, y = world_state_tiles.apply_waypoints(100, 100, "wne")
        assert x == 100
        assert y == 99

    def test_apply_waypoints_ignores_unknown_characters(self) -> None:
        """Test unknown characters are ignored (covers else branch)."""
        # "nXs" = north, unknown 'X', south -> net y stays same
        x, y = world_state_tiles.apply_waypoints(100, 100, "nXs")
        assert x == 100
        assert y == 100
