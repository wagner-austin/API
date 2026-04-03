"""Tests for sniffer detail formatter functions."""

from __future__ import annotations

from tankpit_bot.sniffer import (
    format_combat_details,
    format_container_details,
    format_decoded_message,
    format_misc_details,
    format_position_details,
    format_radar_details,
    format_resource_details,
    format_tank_details,
)


class TestFormatFunctions:
    """Tests for message format functions using proper TypedDict instances."""

    def test_format_combat_details_shooting(self) -> None:
        """Test format_combat_details for shooting message (0x53)."""
        from tankpit_bot.protocol import ShootEventDict

        msg = ShootEventDict(
            msg_type=0x53,
            shooter_id=100,
            target_x=50,
            target_y=60,
            projectile_x=0,
            projectile_y=0,
            fuel=0,
            weapon=0,
            ammo=0,
            friendly_fire=False,
        )
        result = format_combat_details(msg)
        assert "shooter=100" in result
        assert "tgt=(50,60)" in result

    def test_format_combat_details_deactivation(self) -> None:
        """Test format_combat_details for deactivation message (0x41)."""
        from tankpit_bot.protocol import DeactivationDict

        msg = DeactivationDict(
            msg_type=0x41,
            victim_id=50,
            killer_id=100,
            rank=0,
            points=0,
        )
        result = format_combat_details(msg)
        assert "victim=50" in result
        assert "killer=100" in result

    def test_format_combat_details_unknown(self) -> None:
        """Test format_combat_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_combat_details
        msg = SyncDict(msg_type=0x3F)
        result = format_combat_details(msg)
        assert result == ""

    def test_format_tank_details_entry(self) -> None:
        """Test format_tank_details for tank entry (0x28)."""
        from tankpit_bot.protocol import TankEntryDict

        msg = TankEntryDict(
            msg_type=0x28,
            tank_id=100,
            x=50,
            y=60,
            name="TestTank",
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "TestTank" in result

    def test_format_tank_details_exit(self) -> None:
        """Test format_tank_details for tank exit (0x58)."""
        from tankpit_bot.protocol import TankExitDict

        msg = TankExitDict(msg_type=0x58, tank_id=100)
        result = format_tank_details(msg)
        assert "tank=100 left" in result

    def test_format_tank_details_status_sync(self) -> None:
        """Test format_tank_details for tank status sync (0x2E)."""
        from tankpit_bot.protocol import TankStatusSyncDict

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=1,
            tank_id=100,
            damage_state=1,
            rank=3,
            flags=b"\x00\x00\x00",
            leaderboard_position=5,
            fuel=None,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "sergeant" in result  # rank 3
        assert "light" in result  # damage_state 1
        assert "lb=5" in result

    def test_format_tank_details_status(self) -> None:
        """Test format_tank_details for tank status (0x3E)."""
        from tankpit_bot.protocol import TankStatusDict

        msg = TankStatusDict(
            msg_type=0x3E,
            team=2,
            rank=5,
            tank_id=100,
            decoration_state=b"\x00\x00\x00\x00",
            leaderboard_score=1000,
            leaderboard_position=0,
            name="",
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "captain" in result  # rank 5
        assert "blue" in result  # team 2
        assert "lb=1000" in result

    def test_format_tank_details_info(self) -> None:
        """Test format_tank_details for tank info (0x21)."""
        from tankpit_bot.protocol import TankInfoDict

        msg = TankInfoDict(
            msg_type=0x21,
            tank_id=100,
            team=1,
            decoration_state=b"\x00\x00\x00\x00",
            score=0,
            name="InfoTank",
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "purple" in result  # team 1
        assert "InfoTank" in result

    def test_format_tank_details_movement(self) -> None:
        """Test format_tank_details for movement (0x47)."""
        from tankpit_bot.protocol import MovementDict

        msg = MovementDict(
            msg_type=0x47,
            tank_id=100,
            start_x=50,
            start_y=60,
            direction=2,
            flag=0,
            leaderboard_position=500,
            waypoints=[],
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "dir=2" in result
        assert "lb=500" in result

    def test_format_tank_details_movement_response(self) -> None:
        """Test format_tank_details for movement response (0x3D)."""
        from tankpit_bot.protocol import MovementResponseDict

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=100,
            x=50,
            y=60,
            direction=2,
            rank=4,
            leaderboard_position=10,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "dir=2" in result
        assert "lieutenant" in result  # rank 4
        assert "lb=10" in result

    def test_format_tank_details_0x48(self) -> None:
        """Test format_tank_details for 0x48 message (EnemyDetection)."""
        from tankpit_bot.protocol import EnemyDetectionDict

        msg = EnemyDetectionDict(
            msg_type=0x48,
            tank_id=100,
            x=50,
            y=60,
            rank=6,
            team=0,
        )
        result = format_tank_details(msg)
        assert "tank=100" in result
        assert "(50,60)" in result
        assert "major" in result  # rank 6

    def test_format_tank_details_unknown(self) -> None:
        """Test format_tank_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_tank_details
        msg = SyncDict(msg_type=0x3F)
        result = format_tank_details(msg)
        assert result == ""

    def test_format_resource_details_fuel_refill(self) -> None:
        """Test format_resource_details for fuel refill (0x44)."""
        from tankpit_bot.protocol import FuelGainDict

        msg = FuelGainDict(msg_type=0x44, fuel_total=500, is_free=True)
        result = format_resource_details(msg)
        assert "fuel=500" in result
        assert "free=True" in result

    def test_format_resource_details_fuel_deposit(self) -> None:
        """Test format_resource_details for fuel deposit (0x64)."""
        from tankpit_bot.protocol import FuelDepositDict

        msg = FuelDepositDict(msg_type=0x64, fuel_total=1000)
        result = format_resource_details(msg)
        assert "fuel=1000" in result

    def test_format_resource_details_item_pickup(self) -> None:
        """Test format_resource_details for item pickup (0x49)."""
        from tankpit_bot.protocol import InventoryDict

        msg = InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=[1, 2, 3, 0, 1],
            enabled=[True, True, True, True, True],
        )
        result = format_resource_details(msg)
        assert "counts=" in result

    def test_format_resource_details_cache_update(self) -> None:
        """Test format_resource_details for cache update (0x43)."""
        from tankpit_bot.protocol import CacheUpdateDict

        msg = CacheUpdateDict(msg_type=0x43, updates=[(10, 20, 100), (30, 40, -1)])
        result = format_resource_details(msg)
        assert "updates=2" in result

    def test_format_resource_details_unknown(self) -> None:
        """Test format_resource_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_resource_details
        msg = SyncDict(msg_type=0x3F)
        result = format_resource_details(msg)
        assert result == ""

    def test_format_position_details_mine_placement(self) -> None:
        """Test format_position_details for mine placement (0x4B)."""
        from tankpit_bot.protocol import MinePlacementDict

        msg = MinePlacementDict(
            msg_type=0x4B,
            mine_type=0,
            tank_id=100,
            positions=[(10, 20), (30, 40)],
        )
        result = format_position_details(msg)
        assert "tank=100" in result
        assert "count=2" in result

    def test_format_position_details_mine_detonation(self) -> None:
        """Test format_position_details for mine detonation (0x45)."""
        from tankpit_bot.protocol import MineDetonationDict

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
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.protocol import RadarScanResultDict

        container = RadarContainerDict(x=10, y=20, volume=100)
        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[container],
            mines=[],
        )
        result = format_radar_details(msg)
        assert "containers=1" in result or "entities=1" in result

    def test_format_radar_details_combined_tile_update(self) -> None:
        """Test format_radar_details for top-level combined tile update (0x4F)."""
        from tankpit_bot.protocol import CombinedTileUpdateDict

        msg = CombinedTileUpdateDict(
            msg_type=0x4F,
            cache_updates=[(10, 20, 300)],
            overlay_updates=[(11, 21, 7), (12, 22, 255)],
        )
        result = format_radar_details(msg)
        assert result == "cache_updates=1 overlay_updates=2"

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

        msg = SupervisorDict(msg_type=0x52, status=4, reserved=0, data=5)
        result = format_misc_details(msg)
        assert "status=4" in result
        assert "data=5" in result

    def test_format_misc_details_player_msg(self) -> None:
        """Test format_misc_details for player message (0x4D)."""
        from tankpit_bot.protocol import ChatMessageDict

        msg = ChatMessageDict(msg_type=0x4D, sender_id=100, message_type=2, x=None, y=None)
        result = format_misc_details(msg)
        assert "sender=100" in result
        assert "type=2" in result

    def test_format_misc_details_unknown(self) -> None:
        """Test format_misc_details returns empty for unknown type."""
        from tankpit_bot.protocol import SyncDict

        # SyncDict has msg_type=0x3F which is not handled by format_misc_details
        msg = SyncDict(msg_type=0x3F)
        result = format_misc_details(msg)
        assert result == ""

    def test_format_decoded_message_container(self) -> None:
        """Test format_decoded_message for container message."""
        from tankpit_bot.container import TankStatusSyncDict

        msg = TankStatusSyncDict(msg_type="tank_status_sync", sync_data=b"\x01\x02")
        result = format_decoded_message(0x2E, msg)
        assert "TankStatusSync" in result
        assert "0102" in result

    def test_format_decoded_message_protocol(self) -> None:
        """Test format_decoded_message for protocol message."""
        from tankpit_bot.protocol import ShootEventDict

        msg = ShootEventDict(
            msg_type=0x53,
            shooter_id=100,
            target_x=50,
            target_y=60,
            projectile_x=0,
            projectile_y=0,
            fuel=0,
            weapon=0,
            ammo=0,
            friendly_fire=False,
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

    def test_format_container_details_combat_hit_outgoing(self) -> None:
        """Test format_container_details for combat hit outgoing (direction 0x09)."""
        from tankpit_bot.container import CombatHitDict

        msg = CombatHitDict(
            msg_type="combat_hit",
            direction=0x09,
            attacker_id=100,
            combat_data=b"\x00\x00\x00\x00\x00\x00",
            is_outgoing=True,
        )
        result = format_container_details(msg)
        assert "attacker=100" in result
        assert "dir=out" in result

    def test_format_container_details_combat_hit_incoming(self) -> None:
        """Test format_container_details for combat hit incoming."""
        from tankpit_bot.container import CombatHitDict

        msg = CombatHitDict(
            msg_type="combat_hit",
            direction=0x05,  # Not 0x09, so incoming
            attacker_id=50,
            combat_data=b"\x00\x00\x00\x00\x00\x00",
            is_outgoing=False,
        )
        result = format_container_details(msg)
        assert "attacker=50" in result
        assert "dir=in" in result

    def test_format_container_details_tank_registry(self) -> None:
        """Test format_container_details for tank registry."""
        from tankpit_bot.container import TankRegistryDict

        msg = TankRegistryDict(
            msg_type="tank_registry",
            flags=0x12,
            tank_id=100,
            info_bytes=b"\x01\x02\x03\x04",
            team="blue",
            tank_name="TestTank",
            military_rank=2,
            badge_count=3,
            is_bot=False,
            is_container=False,
            container_x=None,
            container_y=None,
            container_viewport_x=None,
            tank_y=None,
            tank_viewport_x=None,
        )
        result = format_container_details(msg)
        assert "tank=100" in result
        assert '"TestTank"' in result
        assert "blue" in result
        assert "corporal" in result
        assert "badges=3" in result

    def test_format_container_details_position_update(self) -> None:
        """Test format_container_details for position update."""
        from tankpit_bot.container import PositionUpdateDict

        msg = PositionUpdateDict(
            msg_type="position_update",
            flags=0xAB,
            tank_id=200,
            x=50,
            y=75,
            extra_data=b"\x01\x02\x03\x04\x05\x06\x07",
        )
        result = format_container_details(msg)
        assert "tank=200" in result
        assert "flags=0xAB" in result or "x=" in result

    def test_format_container_details_tank_status_short(self) -> None:
        """Test format_container_details for tank status short."""
        from tankpit_bot.container import TankStatusShortDict

        msg = TankStatusShortDict(
            msg_type="tank_status_short",
            flags=0,
            tank_id=150,
            damage_state=2,
            rank=4,
            leaderboard_position=25,
        )
        result = format_container_details(msg)
        assert "tank=150" in result
        assert "lieutenant" in result  # rank 4
        assert "hp=medium" in result  # damage_state 2
        assert "lb=25" in result

    def test_format_container_details_tank_update_compact(self) -> None:
        """Test format_container_details for tank update compact."""
        from tankpit_bot.container import TankUpdateCompactDict

        msg = TankUpdateCompactDict(
            msg_type="tank_update_compact",
            flags=0x05,
            tank_id=75,
            status_data=b"\xaa\xbb\xcc\xdd\xee\xff",
        )
        result = format_container_details(msg)
        assert "tank=75" in result
        assert "flags=0x05" in result
        assert "data=aabbccddeeff" in result

    def test_format_container_details_tank_update_extended(self) -> None:
        """Test format_container_details for tank update extended."""
        from tankpit_bot.container import TankUpdateExtendedDict

        msg = TankUpdateExtendedDict(
            msg_type="tank_update_extended",
            flags=0x07,
            tank_id=80,
            status_data=b"\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa",
        )
        result = format_container_details(msg)
        assert "tank=80" in result
        assert "flags=0x07" in result
        assert "data=112233445566778899aa" in result

    def test_format_container_details_tank_update_full(self) -> None:
        """Test format_container_details for tank update full."""
        from tankpit_bot.container import TankUpdateFullDict

        msg = TankUpdateFullDict(
            msg_type="tank_update_full",
            flags=0x0F,
            tank_id=90,
            status_data=b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b",
        )
        result = format_container_details(msg)
        assert "tank=90" in result
        assert "flags=0x0F" in result
        assert "data=0102030405060708090a0b" in result

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
        """Test format_container_details returns empty for unmatched pattern."""
        from tankpit_bot.container import TankLeaveDict

        # TankLeaveDict is not handled by format_container_details
        msg = TankLeaveDict(
            msg_type="tank_leave",
            tank_id=100,
            flags=0,
            extra_data=b"\x00\x00",
        )
        result = format_container_details(msg)
        assert result == ""
