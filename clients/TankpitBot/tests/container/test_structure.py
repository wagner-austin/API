"""Tests for container structure detection functions.

Tests for is_*_structure pattern matching functions.
"""

from __future__ import annotations

from tankpit_bot.container import (
    is_chunk_data_structure,
    is_container_pickup_structure,
    is_deactivation_death_structure,
    is_player_list_extended_structure,
    is_player_list_short_structure,
    is_position_update_structure,
    is_tank_leave_structure,
    is_tank_registry_structure,
    is_tank_status_short_structure,
    is_teleport_landed_structure,
    is_tip_notification_structure,
    is_world_state_structure,
)
from tests.container.test_data import (
    CHUNK_DATA_80,
    CHUNK_DATA_95,
    CHUNK_DATA_130,
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    DEACTIVATION_DEATH_7,
    MOVEMENT_16_SSSS,
    MOVEMENT_18_ENNNW,
    MOVEMENT_19_WWWWWW,
    MOVEMENT_20_NESW,
    PLAYER_LIST_EXTENDED_7,
    PLAYER_LIST_SHORT_4,
    POSITION_UPDATE_13,
    TANK_LEAVE_6,
    TANK_LEAVE_LARGE_ID,
    TANK_REGISTRY_16,
    TANK_REGISTRY_20,
    TANK_STATUS_SHORT_9,
    TELEPORT_LANDED_1,
    TIP_NOTIFICATION_29,
    TIP_NOTIFICATION_55,
    TIP_NOTIFICATION_79,
    WORLD_STATE_500,
    WORLD_STATE_650,
)


class TestIsTankRegistryStructure:
    """Tests for tank registry structure detection.

    TankRegistry messages are 16-20 bytes with tank names ending in alphanumeric chars.
    Movement messages overlap this range but end with waypoint directions (w/s/n/e).
    The structure check must reject Movement messages to prevent misclassification.
    """

    def test_matches_16_bytes(self) -> None:
        """Matches 16-byte TankRegistry message."""
        assert is_tank_registry_structure(TANK_REGISTRY_16) is True

    def test_matches_20_bytes(self) -> None:
        """Matches 20-byte TankRegistry message."""
        assert is_tank_registry_structure(TANK_REGISTRY_20) is True

    def test_matches_17_18_19_bytes_without_waypoint_tail(self) -> None:
        """Matches 17-19 byte messages when tail is not all direction chars."""
        # 0x01 is NOT a direction char (w=0x77, s=0x73, n=0x6e, e=0x65)
        assert is_tank_registry_structure(bytes([0x01] * 17)) is True
        assert is_tank_registry_structure(bytes([0x01] * 18)) is True
        assert is_tank_registry_structure(bytes([0x01] * 19)) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 16-20 byte range."""
        assert is_tank_registry_structure(bytes([0x01] * 15)) is False
        assert is_tank_registry_structure(bytes([0x01] * 21)) is False

    def test_rejects_movement_with_ennnw_waypoints(self) -> None:
        """Rejects Movement ending with 'ennnw' waypoints (real capture)."""
        assert is_tank_registry_structure(MOVEMENT_18_ENNNW) is False

    def test_rejects_movement_with_wwwwww_waypoints(self) -> None:
        """Rejects Movement ending with 'wwwwww' waypoints (real capture)."""
        assert is_tank_registry_structure(MOVEMENT_19_WWWWWW) is False

    def test_rejects_movement_at_minimum_length(self) -> None:
        """Rejects 16-byte Movement with 'ssss' tail (boundary test)."""
        assert is_tank_registry_structure(MOVEMENT_16_SSSS) is False

    def test_rejects_movement_at_maximum_length(self) -> None:
        """Rejects 20-byte Movement with 'nesw' tail (boundary test)."""
        assert is_tank_registry_structure(MOVEMENT_20_NESW) is False

    def test_accepts_partial_direction_tail(self) -> None:
        """Accepts TankRegistry when only some tail bytes are directions.

        Tank names like 'west' have 't' (0x74) which breaks the all-directions pattern.
        Uses 0x21 ('!') subtype since 0x47 ('G') is now rejected as Movement.
        """
        # 16 bytes with TankRegistry subtype 0x21, ending with "nnnt"
        # 't' (0x74) is not a direction char, breaking the all-directions pattern
        data_with_t = bytes.fromhex("2101020304050607" + "08091011" + "6e6e6e74")
        assert is_tank_registry_structure(data_with_t) is True

    def test_rejects_movement_subtype_regardless_of_tail(self) -> None:
        """Rejects messages with Movement subtype 0x47 even without direction tail.

        The subtype check rejects 0x47 ('G') before examining the tail pattern.
        This prevents misclassifying short Movement messages as TankRegistry.
        """
        # 16 bytes with Movement subtype 0x47, non-direction tail "nnnt"
        data = bytes.fromhex("4701020304050607" + "08091011" + "6e6e6e74")
        assert is_tank_registry_structure(data) is False


# is_movement_structure tests deleted 2026-06-19: container Movement
# decoder removed. The unified `decode_0x2e_message` dispatches 0x47
# Movement to the protocol decoder via subtype + length-gate matching.


class TestIsPositionUpdateStructure:
    """Tests for position update structure detection."""

    def test_matches_exactly_13_bytes(self) -> None:
        """Matches verified 13-byte position update message."""
        assert is_position_update_structure(POSITION_UPDATE_13) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_position_update_structure(bytes([0x01] * 12)) is False
        assert is_position_update_structure(bytes([0x01] * 14)) is False

    def test_rejects_wrong_subtype(self) -> None:
        """Rejects 13-byte messages that do not use the position subtype."""
        assert is_position_update_structure(bytes([0x01] * 13)) is False

    def test_rejects_real_capture_false_positive(self) -> None:
        """Rejects the captured 13-byte packet that was misread as position."""
        data = bytes.fromhex("45938292839182938191839181")
        assert is_position_update_structure(data) is False


class TestIsTankStatusShortStructure:
    """Tests for tank status short structure detection."""

    def test_matches_9_bytes(self) -> None:
        """Matches exactly 9-byte message."""
        assert is_tank_status_short_structure(TANK_STATUS_SHORT_9) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_tank_status_short_structure(bytes([0x01] * 8)) is False
        assert is_tank_status_short_structure(bytes([0x01] * 10)) is False


# Container TankStatusSync structure check was a 2-3 byte length-only
# catch-all -- deleted 2026-06-19. Real 0x2E TankStatusSync (8+ bytes
# per JS Og.h) is tested at the protocol layer.


class TestIsTankLeaveStructure:
    """Tests for tank leave structure detection."""

    def test_matches_6_bytes_with_tank_id_pattern(self) -> None:
        """Matches 6-byte message with valid tank_id pattern."""
        assert is_tank_leave_structure(TANK_LEAVE_6) is True
        assert is_tank_leave_structure(TANK_LEAVE_LARGE_ID) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_tank_leave_structure(bytes([0x01] * 5)) is False
        assert is_tank_leave_structure(bytes([0x01] * 7)) is False


class TestIsTeleportLandedStructure:
    """Tests for teleport landed structure detection."""

    def test_matches_single_byte_0x0c(self) -> None:
        """Matches single byte 0x0C subtype."""
        assert is_teleport_landed_structure(TELEPORT_LANDED_1) is True

    def test_matches_any_single_byte(self) -> None:
        """Matches any single byte (length-only check)."""
        assert is_teleport_landed_structure(bytes([0x0A])) is True
        assert is_teleport_landed_structure(bytes([0x0D])) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_teleport_landed_structure(bytes([0x0C, 0x00])) is False
        assert is_teleport_landed_structure(bytes([])) is False


class TestIsContainerPickupStructure:
    """Tests for container pickup structure detection."""

    def test_matches_5_bytes_with_subtype(self) -> None:
        """Matches 5-byte message with 0x43 subtype."""
        assert is_container_pickup_structure(CONTAINER_PICKUP_EQUIPMENT) is True
        assert is_container_pickup_structure(CONTAINER_PICKUP_FUEL) is True

    def test_rejects_wrong_subtype(self) -> None:
        """Rejects 5-byte message with wrong subtype."""
        data = bytes([0x42, 0x88, 0x5E, 0x00, 0x00])  # subtype 0x42 instead of 0x43
        assert is_container_pickup_structure(data) is False

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_container_pickup_structure(bytes([0x43] * 4)) is False
        assert is_container_pickup_structure(bytes([0x43] * 6)) is False


# 0x4F RadarResponse structure check moved to protocol layer
# (decode_radar_scan_result handles structural validation inline).


class TestIsPlayerListShortStructure:
    """Tests for player list short structure detection."""

    def test_matches_4_bytes(self) -> None:
        """Matches exactly 4-byte message."""
        assert is_player_list_short_structure(PLAYER_LIST_SHORT_4) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_player_list_short_structure(bytes([0x79] * 3)) is False
        assert is_player_list_short_structure(bytes([0x79] * 5)) is False


class TestIsPlayerListExtendedStructure:
    """Tests for player list extended structure detection."""

    def test_matches_7_bytes(self) -> None:
        """Matches exactly 7-byte message."""
        assert is_player_list_extended_structure(PLAYER_LIST_EXTENDED_7) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_player_list_extended_structure(bytes([0x79] * 6)) is False
        assert is_player_list_extended_structure(bytes([0x79] * 8)) is False


# 0x41 DeactivationKill structure tests moved to the protocol layer
# tests/protocol/test_combat.py:TestDecodeDeactivation -- container
# path was deleted 2026-06-19.


class TestIsDeactivationDeathStructure:
    """Tests for deactivation death structure detection."""

    def test_matches_7_bytes_with_0x43_subtype(self) -> None:
        """Matches 7-byte message with 0x43 subtype."""
        assert is_deactivation_death_structure(DEACTIVATION_DEATH_7) is True

    def test_rejects_wrong_subtype(self) -> None:
        """Rejects 7-byte message with wrong subtype."""
        data = bytes([0x42, 0x07, 0x86, 0x16, 0x0C, 0x7F, 0x1F])
        assert is_deactivation_death_structure(data) is False

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_deactivation_death_structure(bytes([0x43] * 6)) is False
        assert is_deactivation_death_structure(bytes([0x43] * 8)) is False


class TestIsTipNotificationStructure:
    """Tests for tip notification structure detection."""

    def test_matches_29_bytes_minimum(self) -> None:
        """Matches 29-byte message (minimum)."""
        assert is_tip_notification_structure(TIP_NOTIFICATION_29) is True

    def test_matches_79_bytes_maximum(self) -> None:
        """Matches 79-byte message (maximum)."""
        assert is_tip_notification_structure(TIP_NOTIFICATION_79) is True

    def test_matches_55_bytes_middle(self) -> None:
        """Matches 55-byte message (middle of range)."""
        assert is_tip_notification_structure(TIP_NOTIFICATION_55) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 29-79 byte range."""
        assert is_tip_notification_structure(bytes([0x68] * 28)) is False
        assert is_tip_notification_structure(bytes([0x68] * 80)) is False

    def test_accepts_any_subtype_in_range(self) -> None:
        """Accepts any subtype if length is 29-79 (length-only check)."""
        data = bytes([0x67] + [0x00] * 28)  # 29 bytes, different subtype
        assert is_tip_notification_structure(data) is True


class TestIsChunkDataStructure:
    """Tests for chunk data structure detection."""

    def test_matches_80_bytes_minimum(self) -> None:
        """Matches 80-byte message (minimum)."""
        assert is_chunk_data_structure(CHUNK_DATA_80) is True

    def test_matches_130_bytes_maximum(self) -> None:
        """Matches 130-byte message (maximum)."""
        assert is_chunk_data_structure(CHUNK_DATA_130) is True

    def test_matches_95_bytes_middle(self) -> None:
        """Matches 95-byte message (middle of range)."""
        assert is_chunk_data_structure(CHUNK_DATA_95) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 80-130 byte range."""
        assert is_chunk_data_structure(bytes([0x14] * 79)) is False
        assert is_chunk_data_structure(bytes([0x14] * 131)) is False


class TestIsWorldStateStructure:
    """Tests for world state structure detection."""

    def test_matches_500_bytes_minimum(self) -> None:
        """Matches 500-byte message (minimum)."""
        assert is_world_state_structure(WORLD_STATE_500) is True

    def test_matches_650_bytes(self) -> None:
        """Matches 650-byte message (common size)."""
        assert is_world_state_structure(WORLD_STATE_650) is True

    def test_rejects_below_minimum(self) -> None:
        """Rejects messages below 500 bytes."""
        assert is_world_state_structure(bytes([0x14] * 499)) is False
        assert is_world_state_structure(bytes([0x14] * 130)) is False
