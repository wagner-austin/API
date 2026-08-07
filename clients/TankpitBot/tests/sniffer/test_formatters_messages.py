"""Tests for per-message formatter output."""

from __future__ import annotations

from tankpit_bot.sniffer.formatters import (
    format_container_details,
    format_decoded_message,
    format_misc_details,
    format_position_details,
    format_radar_details,
    format_tank_details,
)


class TestFormatMessageFunctions:
    """Tests for per-message formatter output."""

    def test_format_position_details_mine_detonation(self) -> None:
        """Test format_position_details for mine detonation (0x45)."""
        from tankpit_bot.container import MineDetonationDict

        msg = MineDetonationDict(
            msg_type=0x45,
            positions=[(10, 20), (30, 40), (50, 60)],
        )
        result = format_position_details(msg)
        assert "count=3" in result

    def test_format_position_details_unknown(self) -> None:
        """Test format_position_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_position_details
        msg = SyncDict(msg_type=0x3F)
        result = format_position_details(msg)
        assert result == ""

    def test_format_radar_details_radar_ack(self) -> None:
        """Test format_radar_details for radar ack (0x46)."""
        from tankpit_bot.protocol import RadarResultDict

        msg = RadarResultDict(msg_type=0x46, detection_type=1, found=True)
        result = format_radar_details(msg)
        assert "type=1" in result
        assert "found=True" in result

    def test_format_radar_details_radar_result(self) -> None:
        """Test format_radar_details for radar result (0x4F)."""
        from tankpit_bot.protocol import (
            RadarContainerDict,
            RadarMineClearDict,
            RadarScanResultDict,
        )

        container = RadarContainerDict(x=10, y=20, volume=100)
        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[container],
            mines=[],
            mine_clears=[RadarMineClearDict(x=11, y=21)],
        )
        result = format_radar_details(msg)
        assert result == "containers=1 mines=0 clears=1"

    def test_format_radar_details_viewport_update(self) -> None:
        """Test format_radar_details for viewport update (0x5A)."""
        from tankpit_bot.protocol import ViewportEntityDict, ViewportUpdateDict

        entity1 = ViewportEntityDict(
            col=10,
            row=20,
            cache_value=0,
            overlay_value=0,
            terrain_type=0,
        )
        entity2 = ViewportEntityDict(
            col=30,
            row=40,
            cache_value=0,
            overlay_value=0,
            terrain_type=0,
        )
        msg = ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=3,
            viewport_top=7,
            entities=[entity1, entity2],
        )
        result = format_radar_details(msg)
        assert "viewport=(3,7)" in result
        assert "entities=2" in result

    def test_format_radar_details_unknown(self) -> None:
        """Test format_radar_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_radar_details
        msg = SyncDict(msg_type=0x3F)
        result = format_radar_details(msg)
        assert result == ""

    def test_format_misc_details_equip_gain(self) -> None:
        """Test format_misc_details for equipment gain (0x67)."""
        from tankpit_bot.protocol import EquipmentGainDict

        msg = EquipmentGainDict(msg_type=0x67, show_message=True, gained=[1, 0, 0, 0, 0])
        result = format_misc_details(msg)
        assert "gained=" in result

    def test_format_misc_details_equip_toggle(self) -> None:
        """Test format_misc_details for equipment toggle (0x74)."""
        from tankpit_bot.protocol import EquipmentToggleDict

        msg = EquipmentToggleDict(msg_type=0x74, enabled=[True, False, True, False, True])
        result = format_misc_details(msg)
        assert "enabled=" in result

    def test_format_misc_details_statistics(self) -> None:
        """Test format_misc_details for statistics (0x56)."""
        from tankpit_bot.protocol import StatisticsDict

        msg = StatisticsDict(
            msg_type=0x56,
            playtime_hours=5,
            playtime_minutes=30,
            playtime_seconds=0,
            destroyed=0,
            deactivated=0,
            score=0,
        )
        result = format_misc_details(msg)
        assert "time=5h30m" in result

    def test_format_misc_details_supervisor(self) -> None:
        """Test format_misc_details for supervisor (0x52)."""
        from tankpit_bot.protocol import SupervisorDict

        msg = SupervisorDict(msg_type=0x52, reset_action=4, close_map=0, error_code=5)
        result = format_misc_details(msg)
        assert "reset=4" in result
        assert "err=5" in result

    def test_format_misc_details_player_msg(self) -> None:
        """Test format_misc_details for player message (0x4D)."""
        from tankpit_bot.protocol import ChatMessageDict

        msg = ChatMessageDict(msg_type=0x4D, sender_id=100, message_type=2, x=None, y=None)
        result = format_misc_details(msg)
        assert "sender=100" in result
        assert "type=2" in result
        assert "text='Attack the blue'" in result

    def test_format_misc_details_chat_unknown_id(self) -> None:
        """0x4D formatting marks IDs outside the preset table."""
        from tankpit_bot.protocol import ChatMessageDict

        msg = ChatMessageDict(msg_type=0x4D, sender_id=100, message_type=99, x=None, y=None)
        result = format_misc_details(msg)
        assert "text='unknown_99'" in result

    def test_format_misc_details_unknown(self) -> None:
        """Test format_misc_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_misc_details
        msg = SyncDict(msg_type=0x3F)
        result = format_misc_details(msg)
        assert result == ""

    def test_format_misc_details_decoration(self) -> None:
        """Test format_misc_details for decoration (0x4E)."""
        from tankpit_bot.protocol import DecorationDict

        msg = DecorationDict(msg_type=0x4E, tank_id=99, slot=2, level=3)
        result = format_misc_details(msg)
        assert "tank=99" in result
        assert "slot=2" in result
        assert "level=3" in result

    def test_format_misc_details_supervisor_text(self) -> None:
        """Test format_misc_details for SupervisorText (0x3C)."""
        from tankpit_bot.protocol import SupervisorTextDict

        msg = SupervisorTextDict(msg_type=0x3C, message="Server down\nin 5 mins")
        result = format_misc_details(msg)
        assert "Server down" in result
        # newlines are flattened in the preview
        assert "\\n" not in result

    def test_format_misc_details_map_data(self) -> None:
        """Test format_misc_details for MapData (0x4C)."""
        from tankpit_bot.protocol import MapDataDict, MapTankEntry

        msg = MapDataDict(
            msg_type=0x4C,
            tanks=[MapTankEntry(x=1, y=2, tank_id=5, rank=0, damage=0, team=0)],
            fuel_dots=[],
        )
        result = format_misc_details(msg)
        assert "tanks=1" in result

    def test_format_tank_details_build_pickup_bridge(self) -> None:
        """obstacle_type == 1 renders as the bridge variant."""
        from tankpit_bot.protocol import BuildPickupDict

        msg = BuildPickupDict(
            msg_type=0x42,
            tank_id=44,
            source_x=10,
            source_y=20,
            drop_x=11,
            drop_y=20,
            direction=4,
            obstacle_type=1,
            flag=0,
        )
        result = format_tank_details(msg)
        assert "tank=44" in result
        assert "bridge" in result
        assert "(10,20)" in result
        assert "(11,20)" in result

    def test_format_tank_details_build_pickup_obstacle(self) -> None:
        """Any non-1 obstacle_type renders as a generic obstacle drop/pickup."""
        from tankpit_bot.protocol import BuildPickupDict

        msg = BuildPickupDict(
            msg_type=0x42,
            tank_id=44,
            source_x=10,
            source_y=20,
            drop_x=11,
            drop_y=20,
            direction=4,
            obstacle_type=2,
            flag=1,
        )
        result = format_tank_details(msg)
        assert "obstacle" in result

    def test_format_decoded_message_protocol(self) -> None:
        """Format decoded ShootEvent (0x53) message."""
        from tankpit_bot.protocol import ShootEventDict

        msg = ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=1301,
            source_x=155,
            source_y=154,
            target_x=155,
            target_y=155,
            aim_x=155,
            aim_y=155,
            weapon=1,
        )
        result = format_decoded_message(0x53, msg)
        assert "Shooting" in result or "Msg0x53" in result

    def test_format_decoded_message_no_details(self) -> None:
        """Test format_decoded_message with no details."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which has no specific details formatter
        msg = SyncDict(msg_type=0x3F)
        result = format_decoded_message(0x3F, msg)
        assert "[" in result  # Just has type name in brackets

    def test_format_container_details_unknown_container(self) -> None:
        """Test format_container_details for unknown container."""
        from tankpit_bot.container import UnknownContainerDict

        msg = UnknownContainerDict(
            msg_type="unknown_container",
            subtype=0x99,
            length=50,
            data=b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a" * 5,
        )
        result = format_container_details(msg)
        assert "len=50" in result
        assert "data=" in result

    def test_format_container_details_unmatched_returns_empty(self) -> None:
        """Test format_container_details returns empty for unmatched pattern.

        TeleportLandedDict (string msg_type, not handled by
        format_container_simple) routes through the empty fallback.
        """
        from tankpit_bot.container import TeleportLandedDict

        msg = TeleportLandedDict(msg_type="teleport_landed", subtype=0x0C)
        result = format_container_details(msg)
        assert result == ""
