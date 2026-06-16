"""Tests for sniffer formatter function edge cases."""

from __future__ import annotations

from tankpit_bot.sniffer import viewport
from tankpit_bot.sniffer.formatters import (
    format_container_pickup,
    format_movement,
    format_radar_response,
    format_tank_registry_details,
)


class TestFormatFunctionsEdgeCases:
    """Tests for format function edge cases."""

    def test_format_tank_registry_details_container_with_viewport(self) -> None:
        """Test format_tank_registry_details for container with viewport position."""
        viewport.reset_viewport_tracking()
        viewport.update_viewport_origin(100, 200)
        try:
            result = format_tank_registry_details(
                tid=42,
                name="",
                team="",
                rank=0,
                badges=0,
                is_bot=False,
                is_container=True,
                container_y=50,
                container_viewport_x=10,
            )
            assert "container id=42" in result
            assert "pos=(110,50)" in result  # 100 + 10 = 110
        finally:
            viewport.reset_viewport_tracking()

    def test_format_tank_registry_details_container_no_viewport(self) -> None:
        """Test format_tank_registry_details for container without viewport left."""
        viewport.reset_viewport_tracking()
        try:
            result = format_tank_registry_details(
                tid=42,
                name="",
                team="",
                rank=0,
                badges=0,
                is_bot=False,
                is_container=True,
                container_y=50,
                container_viewport_x=10,
            )
            assert "container id=42" in result
            assert "y=50" in result
            assert "vx=10" in result
        finally:
            viewport.reset_viewport_tracking()

    def test_format_tank_registry_details_container_no_position(self) -> None:
        """Test format_tank_registry_details for container without position data."""
        result = format_tank_registry_details(
            tid=42,
            name="",
            team="",
            rank=0,
            badges=0,
            is_bot=False,
            is_container=True,
            container_y=None,
            container_viewport_x=None,
        )
        assert result == "container id=42"

    def test_format_movement(self) -> None:
        """Test format_movement formats movement details."""
        result = format_movement(sx=10, sy=20, pid=1, waypoints="RRDD", is_self=True)
        assert "self" in result
        assert "from=(10,20)" in result
        assert 'path="RRDD"' in result
        assert "(4 tiles)" in result

    def test_format_movement_enemy(self) -> None:
        """Test format_movement formats enemy movement."""
        result = format_movement(sx=30, sy=40, pid=2, waypoints="LL", is_self=False)
        assert "enemy" in result
        assert "from=(30,40)" in result

    def test_format_radar_response_with_containers_and_mines(self) -> None:
        """Test format_radar_response formats both containers and mines."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=10, y=20, volume=100),  # fuel
            RadarContainerDict(x=30, y=40, volume=-1),  # equipment
        ]
        mines: list[RadarMineDict] = [
            RadarMineDict(x=50, y=60, team=0),  # red
            RadarMineDict(x=70, y=80, team=2),  # blue
        ]
        result = format_radar_response(containers, mines)
        assert "(10,20):fuel=100" in result
        assert "(30,40):equip" in result
        assert "(50,60):mine[red]" in result
        assert "(70,80):mine[blue]" in result

    def test_format_radar_response_empty_lists(self) -> None:
        """Test format_radar_response handles empty lists."""
        result = format_radar_response([], [])
        assert result == ""

    def test_format_radar_response_unknown_team(self) -> None:
        """Test format_radar_response handles unknown team numbers."""
        from tankpit_bot.protocol import RadarMineDict

        mines: list[RadarMineDict] = [RadarMineDict(x=10, y=20, team=99)]
        result = format_radar_response([], mines)
        assert "(10,20):mine[team99]" in result

    def test_format_radar_response_all_teams(self) -> None:
        """Test format_radar_response shows all team names."""
        from tankpit_bot.protocol import RadarMineDict

        mines: list[RadarMineDict] = [
            RadarMineDict(x=1, y=1, team=0),  # red
            RadarMineDict(x=2, y=2, team=1),  # purple
            RadarMineDict(x=3, y=3, team=2),  # blue
            RadarMineDict(x=4, y=4, team=3),  # orange
        ]
        result = format_radar_response([], mines)
        assert "mine[red]" in result
        assert "mine[purple]" in result
        assert "mine[blue]" in result
        assert "mine[orange]" in result

    def test_format_container_pickup_fuel(self) -> None:
        """Test format_container_pickup for fuel container."""
        result = format_container_pickup(x=10, y=20, vol=100, is_fuel=True)
        assert "pos=(10,20)" in result
        assert "FUEL vol=100" in result

    def test_format_container_pickup_equipment(self) -> None:
        """Test format_container_pickup for equipment container."""
        result = format_container_pickup(x=30, y=40, vol=-1, is_fuel=False)
        assert "pos=(30,40)" in result
        assert "EQUIPMENT" in result
