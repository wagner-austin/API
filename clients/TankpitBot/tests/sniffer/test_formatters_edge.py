"""Tests for sniffer formatter function edge cases."""

from __future__ import annotations

from tankpit_bot.container.types import ContainerPickupRecordDict
from tankpit_bot.sniffer.formatters import (
    format_container_pickup,
    format_radar_response,
)


class TestFormatFunctionsEdgeCases:
    """Tests for format function edge cases."""

    # format_tank_registry_details tests deleted 2026-06-20: the
    # underlying TankRegistryDict + formatter were removed after corpus
    # sweep proved zero production fires for the container path.

    # format_movement was deleted 2026-06-19 along with the container
    # MovementDict / PlayerIdMapper. Protocol 0x47 Movement is formatted
    # by format_decoded_message via the 0x47 branch in format_combat_details.

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

    def test_format_container_pickup_partial_remaining(self) -> None:
        """Partial fuel pickup leaves volume in the container; format flags it."""
        result = format_container_pickup(
            (ContainerPickupRecordDict(x=10, y=20, remaining_volume=283),),
        )
        assert "pos=(10,20)" in result
        assert "FUEL partial remaining=283" in result

    def test_format_container_pickup_empty(self) -> None:
        """An emptied container (vol=0) is either equipment or a fully consumed
        fuel container; the formatter is agnostic and shows ``container emptied``.
        """
        result = format_container_pickup(
            (ContainerPickupRecordDict(x=30, y=40, remaining_volume=0),),
        )
        assert "pos=(30,40)" in result
        assert "container emptied" in result

    def test_format_container_pickup_multi_record(self) -> None:
        """Multi-record bodies render a comma-joined summary with a count.

        A 3-record pickup (the empirical 13-byte 0x43 corpus case) prints
        ``pickups=N: <record1>, <record2>, <record3>``.
        """
        pickups = (
            ContainerPickupRecordDict(x=240, y=150, remaining_volume=0),
            ContainerPickupRecordDict(x=239, y=149, remaining_volume=0),
            ContainerPickupRecordDict(x=240, y=149, remaining_volume=846),
        )
        result = format_container_pickup(pickups)
        assert "pickups=3" in result
        assert "pos=(240,150) container emptied" in result
        assert "pos=(240,149) FUEL partial remaining=846" in result
