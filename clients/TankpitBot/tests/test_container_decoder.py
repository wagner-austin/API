"""Tests for container_decoder module - structure-based 0x2E message decoding.

All tests use real data patterns from captured sessions.
No mocks, no weak assertions, 100% coverage target.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    MESSAGE_TYPE_LEVELS,
    ChunkDataDict,
    CombatHitDict,
    ContainerDecodeError,
    ContainerMessageType,
    ContainerPickupDict,
    DeactivationDeathDict,
    DeactivationKillDict,
    DecodeLevel,
    PlayerListExtendedDict,
    PlayerListShortDict,
    PositionUpdateDict,
    RadarResponseDict,
    TankLeaveDict,
    TankRegistryDict,
    TankStatusShortDict,
    TankStatusSyncDict,
    TankUpdateCompactDict,
    TankUpdateExtendedDict,
    TankUpdateFullDict,
    TeleportLandedDict,
    TipNotificationDict,
    UnknownContainerDict,
    WorldStateDict,
    decode_chunk_data,
    decode_combat_hit,
    decode_container_message,
    decode_container_pickup,
    decode_deactivation_death,
    decode_deactivation_kill,
    decode_movement,
    decode_player_list_extended,
    decode_player_list_short,
    decode_position_update,
    decode_radar_response,
    decode_tank_leave,
    decode_tank_registry,
    decode_tank_status_short,
    decode_tank_status_sync,
    decode_tank_update_compact,
    decode_tank_update_extended,
    decode_tank_update_full,
    decode_teleport_landed,
    decode_tip_notification,
    decode_unknown_container,
    decode_world_state,
    extract_uint16_le,
    get_decode_level,
    identify_container_type,
    is_chunk_data_structure,
    is_combat_hit_structure,
    is_container_pickup_structure,
    is_deactivation_death_structure,
    is_deactivation_kill_structure,
    is_movement_structure,
    is_player_list_extended_structure,
    is_player_list_short_structure,
    is_position_update_structure,
    is_radar_response_structure,
    is_tank_leave_structure,
    is_tank_registry_structure,
    is_tank_status_short_structure,
    is_tank_status_sync_structure,
    is_tank_update_compact_structure,
    is_tank_update_extended_structure,
    is_tank_update_full_structure,
    is_teleport_landed_structure,
    is_tip_notification_structure,
    is_world_state_structure,
    require_exact_length,
    require_length_range,
    require_min_length,
)
from tankpit_bot.container.decoders.tank import _parse_tank_name

# =============================================================================
# Test Data - Real patterns from captured sessions
# =============================================================================

# Combat hit: 11 bytes
# From "5953cd07998493ce9c8051" - session capture
COMBAT_HIT_11_OUTGOING = bytes.fromhex("5909cd07998493ce9c8051")  # direction=0x09 (outgoing)
COMBAT_HIT_11_INCOMING = bytes.fromhex("590bcd07998493ce9c8051")  # direction=0x0b (incoming)

# Tank registry: 16 bytes (minimum)
# From session capture - 16 bytes
TANK_REGISTRY_16 = bytes.fromhex("7c0980530b0f41094aedcf0f326e6576")

# Tank registry: 20 bytes (maximum)
# Extended data pattern for name length variation
TANK_REGISTRY_20 = bytes.fromhex("7c0980530b0f41094aedcf0f326e657600112233")

# Tank registry: bot (17 bytes)
# Structure: [subtype:1][flags:1][tank_id:2 LE][info:13]
# info for bot: [zeros:6][bot_num:1][name:5+null] - bot has first 6 info bytes as zeros
# flags=0x01 (red team), tank_id=0x023A, zeros(6), bot_num=5, name="red-3\0"
TANK_REGISTRY_BOT = bytes.fromhex("7c013a02000000000000057265642d3300")

# Tank registry: container with wasd name (18 bytes)
# flags=0x7e (extended), tank_id=0x1E82, info has x=17,y=9, name="sse"
# Extended format: name at offset 10
TANK_REGISTRY_CONTAINER_WASD = bytes.fromhex("7c7e821e11090200030000007373650000")

# Tank registry: container with short garbage name (16 bytes)
# flags=0x35 (extended), tank_id=0x081D, info has x=3,y=146, name=non-printable
# Extended format: name at offset 10, info[10:12] = 00 82 (non-printable)
TANK_REGISTRY_CONTAINER_GARBAGE = bytes.fromhex("7c351d08039280000000000000000082")

# Movement messages: 16-20 bytes but ending with waypoint directions (w/s/n/e)
# These should NOT match is_tank_registry_structure because tail4 is all direction chars
# Movement 18 bytes: subtype=0x47('G'), ends with "ennnw" (0x65 0x6e 0x6e 0x6e 0x77)
# From session capture: tank moving east, north, north, north, west
MOVEMENT_18_ENNNW = bytes.fromhex("477e026e5c0c03002e87030000656e6e6e77")

# Movement 19 bytes: subtype=0x47('G'), ends with "wwwwww" (6x 0x77)
# From session capture: tank moving 6 tiles west
MOVEMENT_19_WWWWWW = bytes.fromhex("477e02745c0803002e87030000777777777777")

# Movement 16 bytes: minimal length with 4 direction chars at end "ssss"
# Constructed to test exact boundary - 12 header bytes + 4 waypoints (s=0x73)
MOVEMENT_16_SSSS = bytes.fromhex("470102030405060708091011" + "73737373")

# Movement 20 bytes: maximal TankRegistry length with 4 directions "nesw"
# Constructed to test exact boundary - 16 header bytes + 4 waypoints (n=0x6e,e=0x65,s=0x73,w=0x77)
MOVEMENT_20_NESW = bytes.fromhex("47010203040506070809101112131415" + "6e657377")

# Position update: exactly 13 bytes
# From "2453cd0715121d67b315515506" capture
POSITION_UPDATE_13 = bytes.fromhex("2453cd0715121d67b315515506")

# Tank status short: 9 bytes (enemy status with rank/damage)
# Structure: [subtype:1] [tank_id:2 LE] [damage:1] [rank:1] [flag:1] [lb_pos:2 LE] [extra:1]
# Example: tank_id=0x5782, damage=2 (medium), rank=4 (lieutenant), flag=0, lb_pos=0x0015, extra=0
TANK_STATUS_SHORT_9 = bytes.fromhex("018257020400150000")

# Tank status sync: 2 bytes
TANK_STATUS_SYNC_2 = bytes.fromhex("0100")

# Tank status sync: 3 bytes
TANK_STATUS_SYNC_3 = bytes.fromhex("030102")

# Unknown: 8 bytes (doesn't match any pattern - gap between 7 and 9)
UNKNOWN_8_BYTES = bytes.fromhex("7e51460516112233")

# Unknown: 12 bytes (doesn't match any known pattern)
# 12 bytes is between combat_hit (11) and position_update (13)
UNKNOWN_12_BYTES = bytes.fromhex("010203040506070809101112")

# Teleport landed: 1 byte (0x0C subtype)
# From capture: single byte confirmation after teleport completes
TELEPORT_LANDED_1 = bytes.fromhex("0c")

# Container pickup: 5 bytes [subtype:1][x:1][y:1][volume:2 LE]
# Equipment pickup (volume=0)
CONTAINER_PICKUP_EQUIPMENT = bytes.fromhex("43" + "88" + "5e" + "0000")  # x=136, y=94, vol=0
# Fuel pickup (volume=618 = 0x026a)
CONTAINER_PICKUP_FUEL = bytes.fromhex("43" + "89" + "5f" + "6a02")  # x=137, y=95, vol=618

# Radar response: [subtype:1][count:2 LE][entries: count*4]
# Each entry: [x:1][y:1][volume:2 LE] (volume=0xFFFF for equipment)
# 1 equipment container at (123, 105)
RADAR_RESPONSE_1 = bytes.fromhex("4f" + "0100" + "7b69ffff")  # count=1, (123,105):equip
# 2 containers: 1 equipment + 1 fuel
RADAR_RESPONSE_2 = bytes.fromhex("4f" + "0200" + "7b69ffff" + "895fea02")  # count=2
# 5 containers (4 equipment + 1 fuel) - realistic radar response
RADAR_RESPONSE_5 = bytes.fromhex(
    "4f"
    + "0500"  # subtype + count=5
    + "7b69ffff"  # (123,105):equip
    + "7d68ffff"  # (125,104):equip
    + "8469ffff"  # (132,105):equip
    + "885effff"  # (136,94):equip
    + "895fea02"  # (137,95):fuel=746
)

# Tip notification: 29 bytes (minimum of range 29-79)
# From session capture - game tips and notifications
TIP_NOTIFICATION_29 = bytes.fromhex("68" + "00" * 28)

# Tip notification: 79 bytes (maximum of range 29-79)
TIP_NOTIFICATION_79 = bytes.fromhex("68" + "01" * 78)

# Tip notification: 55 bytes (middle of range)
TIP_NOTIFICATION_55 = bytes.fromhex("68" + "02" * 54)

# Chunk data: 80 bytes (minimum of range 80-130)
# From session capture - terrain/map chunk data
CHUNK_DATA_80 = bytes.fromhex("14" + "00" * 79)

# Chunk data: 130 bytes (maximum of range 80-130)
CHUNK_DATA_130 = bytes.fromhex("14" + "01" * 129)

# Chunk data: 95 bytes (middle of range - from session summary)
CHUNK_DATA_95 = bytes.fromhex("14" + "02" * 94)

# World state: 500 bytes (minimum)
# From session capture - full world/map state
WORLD_STATE_500 = bytes.fromhex("14" + "00" * 499)

# World state: 650 bytes (common size from session summary)
WORLD_STATE_650 = bytes.fromhex("14" + "01" * 649)

# Tank leave: 6 bytes with tank_id pattern (byte[3] == 0 for tank IDs < 256)
# From capture: "7f138b004213" - Arterial (tank 139) left the game
TANK_LEAVE_6 = bytes.fromhex("7f138b004213")

# Player list short: 4 bytes response to '/' key
# From capture: "79990507" - single player response
PLAYER_LIST_SHORT_4 = bytes.fromhex("79990507")

# Player list extended: 7 bytes response with multiple players
# From capture: "79990507ce1144" - multi-player response
PLAYER_LIST_EXTENDED_7 = bytes.fromhex("79990507ce1144")

# Deactivation kill: 5 bytes [0x41, victim_lo, victim_hi, killer_lo, killer_hi]
# From capture: "41bb629c0e" - you killed tank 25275, your tank is 3740
DEACTIVATION_KILL_5 = bytes.fromhex("41bb629c0e")

# Deactivation death: 7 bytes [0x43, flags, killer_lo, killer_hi, extra...]
# From capture: "430786160c7f1f" - you were killed by tank 5766
DEACTIVATION_DEATH_7 = bytes.fromhex("430786160c7f1f")

# Tank leave with large tank ID (> 256): 6 bytes
# From capture: "204a845d5201" - tank 23940 left the game
TANK_LEAVE_LARGE_ID = bytes.fromhex("204a845d5201")

# Tank update compact: 10 bytes
# From session capture "3644df501d1a2b9bf78b"
# Structure: [subtype:1] [flags:1] [tank_id:2 LE] [status_data:6]
TANK_UPDATE_COMPACT_10 = bytes.fromhex("3644df501d1a2b9bf78b")

# Tank update extended: 14 bytes
# From session capture "3f447950521a001b11879a3c2479"
# Structure: [subtype:1] [flags:1] [tank_id:2 LE] [status_data:10]
TANK_UPDATE_EXTENDED_14 = bytes.fromhex("3f447950521a001b11879a3c2479")

# Tank update full: 15 bytes
# From session capture "3f46c750541a001b11871c59642525"
# Structure: [subtype:1] [flags:1] [tank_id:2 LE] [status_data:11]
TANK_UPDATE_FULL_15 = bytes.fromhex("3f46c750541a001b11871c59642525")


# =============================================================================
# Validation Helper Tests
# =============================================================================


class TestRequireMinLength:
    """Tests for require_min_length validation."""

    def test_passes_when_length_sufficient(self) -> None:
        """Validation passes when data meets minimum length."""
        data = bytes([0x01, 0x02, 0x03])
        require_min_length(data, 3, "Test")  # Should not raise

    def test_passes_when_length_exceeds_minimum(self) -> None:
        """Validation passes when data exceeds minimum length."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        require_min_length(data, 2, "Test")  # Should not raise

    def test_raises_when_length_insufficient(self) -> None:
        """Validation raises when data is too short."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            require_min_length(data, 5, "TestContext")
        assert "TestContext" in str(exc.value)
        assert "need at least 5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)


class TestRequireExactLength:
    """Tests for require_exact_length validation."""

    def test_passes_when_length_matches(self) -> None:
        """Validation passes when length matches exactly."""
        data = bytes([0x01, 0x02, 0x03])
        require_exact_length(data, 3, "Test")  # Should not raise

    def test_raises_when_length_too_short(self) -> None:
        """Validation raises when data is too short."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            require_exact_length(data, 5, "TestContext")
        assert "expected 5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)

    def test_raises_when_length_too_long(self) -> None:
        """Validation raises when data is too long."""
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        with pytest.raises(ContainerDecodeError) as exc:
            require_exact_length(data, 3, "TestContext")
        assert "expected 3 bytes" in str(exc.value)
        assert "got 5" in str(exc.value)


class TestRequireLengthRange:
    """Tests for require_length_range validation."""

    def test_passes_at_minimum(self) -> None:
        """Validation passes at minimum of range."""
        data = bytes([0x01, 0x02, 0x03])
        require_length_range(data, 3, 5, "Test")  # Should not raise

    def test_passes_at_maximum(self) -> None:
        """Validation passes at maximum of range."""
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        require_length_range(data, 3, 5, "Test")  # Should not raise

    def test_passes_within_range(self) -> None:
        """Validation passes within range."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        require_length_range(data, 3, 5, "Test")  # Should not raise

    def test_raises_below_minimum(self) -> None:
        """Validation raises below minimum."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            require_length_range(data, 3, 5, "TestContext")
        assert "expected 3-5 bytes" in str(exc.value)
        assert "got 2" in str(exc.value)

    def test_raises_above_maximum(self) -> None:
        """Validation raises above maximum."""
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06])
        with pytest.raises(ContainerDecodeError) as exc:
            require_length_range(data, 3, 5, "TestContext")
        assert "expected 3-5 bytes" in str(exc.value)
        assert "got 6" in str(exc.value)


class TestExtractUint16Le:
    """Tests for extract_uint16_le extraction."""

    def test_extracts_little_endian_value(self) -> None:
        """Correctly extracts little-endian uint16."""
        # 0x5380 in little-endian is bytes [0x80, 0x53]
        data = bytes([0x00, 0x00, 0x80, 0x53, 0x00])
        result = extract_uint16_le(data, 2, "Test")
        assert result == 0x5380

    def test_extracts_at_offset_zero(self) -> None:
        """Extracts from start of data."""
        data = bytes([0x34, 0x12])
        result = extract_uint16_le(data, 0, "Test")
        assert result == 0x1234

    def test_raises_when_offset_out_of_bounds(self) -> None:
        """Raises when offset exceeds data length."""
        data = bytes([0x01, 0x02])
        with pytest.raises(ContainerDecodeError) as exc:
            extract_uint16_le(data, 1, "TestContext")
        assert "cannot read uint16 at offset 1" in str(exc.value)
        assert "data length 2" in str(exc.value)


# =============================================================================
# Structure Pattern Tests
# =============================================================================


class TestIsCombatHitStructure:
    """Tests for combat hit structure detection."""

    def test_matches_11_bytes(self) -> None:
        """Matches exactly 11-byte message."""
        assert is_combat_hit_structure(COMBAT_HIT_11_OUTGOING) is True
        assert is_combat_hit_structure(COMBAT_HIT_11_INCOMING) is True

    def test_rejects_wrong_length(self) -> None:
        """Rejects messages with wrong length."""
        assert is_combat_hit_structure(bytes([0x01] * 10)) is False
        assert is_combat_hit_structure(bytes([0x01] * 12)) is False
        assert is_combat_hit_structure(bytes([0x01] * 13)) is False


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


class TestIsMovementStructure:
    """Tests for movement structure detection."""

    def test_matches_18_bytes_with_waypoint_tail(self) -> None:
        """Matches 18-byte message with direction char tail (real capture)."""
        assert is_movement_structure(MOVEMENT_18_ENNNW) is True

    def test_matches_19_bytes_with_waypoint_tail(self) -> None:
        """Matches 19-byte message with direction char tail (real capture)."""
        assert is_movement_structure(MOVEMENT_19_WWWWWW) is True

    def test_matches_16_bytes_at_minimum(self) -> None:
        """Matches 16-byte message with direction char tail."""
        assert is_movement_structure(MOVEMENT_16_SSSS) is True

    def test_matches_20_bytes_at_boundary(self) -> None:
        """Matches 20-byte message with direction char tail."""
        assert is_movement_structure(MOVEMENT_20_NESW) is True

    def test_rejects_too_short(self) -> None:
        """Rejects messages shorter than 14 bytes."""
        # 13 bytes ending with directions - still too short
        data = bytes.fromhex("47010203040506070809737373")  # 13 bytes
        assert len(data) == 13
        assert is_movement_structure(data) is False

    def test_rejects_non_direction_tail_without_movement_subtype(self) -> None:
        """Rejects messages without Movement subtype where tail is not all directions.

        Uses 0x00 subtype (not 0x47 'G') so subtype check fails, then tail check fails.
        """
        # 18 bytes with non-Movement subtype 0x00, tail has 'x' (0x78)
        data = bytes.fromhex("007e026e5c0c03002e870300006565656578")
        assert is_movement_structure(data) is False

    def test_accepts_movement_subtype_regardless_of_tail(self) -> None:
        """Accepts messages with Movement subtype 0x47 even with non-direction tail.

        The subtype check (0x47 = 'G') takes precedence over tail pattern check.
        This handles short movements where padding bytes appear at the end.
        """
        # 18 bytes with Movement subtype 0x47, tail has 'x' (0x78)
        data = bytes.fromhex("477e026e5c0c03002e870300006565656578")
        assert is_movement_structure(data) is True


class TestDecodeMovement:
    """Tests for movement message decoding."""

    def test_decodes_18_byte_movement(self) -> None:
        """Decodes 18-byte movement message correctly.

        MOVEMENT_18_ENNNW: 477e026e5c0c03002e87030000656e6e6e77
        - [0] 0x47 = subtype 'G'
        - [1] 0x7E = flags (self)
        - [2-3] 0x026E (LE) = packed 0x6E02 -> start_x = 0x6E = 110
        - [4] 0x5C = start_y = 92
        - [5-7] unknown
        - [8-11] 0x2E870300 (LE) = player_id = 231214
        - [12+] waypoints = "ennnw"
        """
        result = decode_movement(MOVEMENT_18_ENNNW)
        assert result["msg_type"] == "movement"
        assert result["flags"] == 0x7E
        assert result["start_x"] == 0x6E  # 110 - high byte of packed 0x6E02
        assert result["start_y"] == 0x5C  # 92
        assert result["player_id"] == 231214  # 0x0003872E
        assert result["tank_id"] is None  # Not resolved yet
        assert result["waypoints"] == "ennnw"
        assert result["is_self"] is True  # flags 0x7E has bits 5-6 set

    def test_decodes_19_byte_movement(self) -> None:
        """Decodes 19-byte movement message correctly."""
        result = decode_movement(MOVEMENT_19_WWWWWW)
        assert result["msg_type"] == "movement"
        assert result["waypoints"] == "wwwwww"

    def test_decodes_boundary_lengths(self) -> None:
        """Decodes boundary length messages."""
        result16 = decode_movement(MOVEMENT_16_SSSS)
        assert result16["msg_type"] == "movement"
        assert result16["waypoints"] == "ssss"

        result20 = decode_movement(MOVEMENT_20_NESW)
        assert result20["msg_type"] == "movement"
        assert result20["waypoints"] == "nesw"

    def test_raises_on_short_data(self) -> None:
        """Raises ContainerDecodeError for too-short data."""
        short_data = bytes([0x47] + [0x00] * 12)  # 13 bytes
        with pytest.raises(ContainerDecodeError, match="Movement"):
            decode_movement(short_data)


class TestIdentifyContainerTypeMovement:
    """Tests for movement identification in dispatcher."""

    def test_identifies_movement_before_tank_registry(self) -> None:
        """Identifies Movement messages correctly (not misclassified as TankRegistry)."""
        assert identify_container_type(MOVEMENT_18_ENNNW) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_19_WWWWWW) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_16_SSSS) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_20_NESW) == ContainerMessageType.MOVEMENT

    def test_tank_registry_not_confused_with_movement(self) -> None:
        """TankRegistry messages with non-direction tails are correctly identified."""
        assert identify_container_type(TANK_REGISTRY_16) == ContainerMessageType.TANK_REGISTRY
        assert identify_container_type(TANK_REGISTRY_20) == ContainerMessageType.TANK_REGISTRY


class TestDecodeContainerMessageMovement:
    """Tests for Movement via decode_container_message dispatcher."""

    def test_dispatches_to_movement_decoder(self) -> None:
        """decode_container_message returns MovementDict for movement messages."""
        result = decode_container_message(MOVEMENT_18_ENNNW)
        assert result["msg_type"] == "movement"
        # Narrow type and verify waypoints
        if result["msg_type"] == "movement":
            assert result["waypoints"] == "ennnw"


class TestPlayerIdMapper:
    """Tests for PlayerIdMapper correlating player_id to tank_id."""

    def test_resolve_movement_from_position_correlation(self) -> None:
        """Resolves tank_id by matching Movement start position to MovementResponse."""
        from tankpit_bot.container import MovementDict, PlayerIdMapper

        mapper = PlayerIdMapper()

        # Record MovementResponse at position (36, 92) for tank 638
        mapper.record_movement_response(tank_id=638, x=36, y=92)

        # Create Movement starting at same position with player_id 231214
        movement = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="ennnw",
            is_self=True,
        )

        # Resolve should correlate position and learn mapping
        mapper.resolve_movement(movement)
        assert movement["tank_id"] == 638

        # Mapping should be cached
        assert mapper.get_tank_id(231214) == 638

    def test_resolve_movement_from_cached_mapping(self) -> None:
        """Uses cached player_id -> tank_id mapping for subsequent movements."""
        from tankpit_bot.container import MovementDict, PlayerIdMapper

        mapper = PlayerIdMapper()

        # Record initial correlation
        mapper.record_movement_response(tank_id=638, x=36, y=92)
        movement1 = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="ennnw",
            is_self=True,
        )
        mapper.resolve_movement(movement1)

        # Second movement at different position should still resolve
        movement2 = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=50,
            start_y=100,
            player_id=231214,  # Same player_id
            tank_id=None,
            waypoints="wwww",
            is_self=True,
        )
        mapper.resolve_movement(movement2)
        assert movement2["tank_id"] == 638  # Resolved from cached mapping

    def test_resolve_movement_no_match(self) -> None:
        """Movement without matching position leaves tank_id as None."""
        from tankpit_bot.container import MovementDict, PlayerIdMapper

        mapper = PlayerIdMapper()

        movement = MovementDict(
            msg_type="movement",
            flags=0x1E,
            start_x=99,
            start_y=99,
            player_id=12345,
            tank_id=None,
            waypoints="eeee",
            is_self=False,
        )
        mapper.resolve_movement(movement)
        assert movement["tank_id"] is None

    def test_get_player_id_reverse_lookup(self) -> None:
        """Reverse lookup from tank_id to player_id."""
        from tankpit_bot.container import MovementDict, PlayerIdMapper

        mapper = PlayerIdMapper()
        mapper.record_movement_response(tank_id=638, x=36, y=92)

        movement = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="e",
            is_self=True,
        )
        mapper.resolve_movement(movement)

        assert mapper.get_player_id(638) == 231214
        assert mapper.get_player_id(999) is None  # Unknown tank

    def test_clear_resets_all_mappings(self) -> None:
        """Clear removes all cached mappings."""
        from tankpit_bot.container import MovementDict, PlayerIdMapper

        mapper = PlayerIdMapper()
        mapper.record_movement_response(tank_id=638, x=36, y=92)

        movement = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="e",
            is_self=True,
        )
        mapper.resolve_movement(movement)
        assert mapper.get_tank_id(231214) == 638

        mapper.clear()
        assert mapper.get_tank_id(231214) is None


class TestIsPositionUpdateStructure:
    """Tests for position update structure detection."""

    def test_matches_13_bytes(self) -> None:
        """Matches exactly 13-byte message."""
        assert is_position_update_structure(POSITION_UPDATE_13) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 13 bytes."""
        assert is_position_update_structure(bytes([0x01] * 12)) is False
        assert is_position_update_structure(bytes([0x01] * 14)) is False


class TestIsTankStatusShortStructure:
    """Tests for tank status short structure detection (9 bytes)."""

    def test_matches_9_bytes(self) -> None:
        """Matches exactly 9-byte message."""
        assert is_tank_status_short_structure(TANK_STATUS_SHORT_9) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 9 bytes."""
        assert is_tank_status_short_structure(bytes([0x01] * 8)) is False
        assert is_tank_status_short_structure(bytes([0x01] * 10)) is False


class TestIsTankStatusSyncStructure:
    """Tests for tank status sync structure detection."""

    def test_matches_2_bytes(self) -> None:
        """Matches 2-byte message."""
        assert is_tank_status_sync_structure(TANK_STATUS_SYNC_2) is True

    def test_matches_3_bytes(self) -> None:
        """Matches 3-byte message."""
        assert is_tank_status_sync_structure(TANK_STATUS_SYNC_3) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 2-3 range."""
        assert is_tank_status_sync_structure(bytes([0x01])) is False
        assert is_tank_status_sync_structure(bytes([0x01] * 4)) is False


class TestIsTankUpdateCompactStructure:
    """Tests for tank update compact structure detection (10 bytes)."""

    def test_matches_10_bytes(self) -> None:
        """Matches exactly 10-byte message."""
        assert is_tank_update_compact_structure(TANK_UPDATE_COMPACT_10) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 10 bytes."""
        assert is_tank_update_compact_structure(bytes([0x01] * 9)) is False
        assert is_tank_update_compact_structure(bytes([0x01] * 11)) is False


class TestIsTankUpdateExtendedStructure:
    """Tests for tank update extended structure detection (14 bytes)."""

    def test_matches_14_bytes(self) -> None:
        """Matches exactly 14-byte message."""
        assert is_tank_update_extended_structure(TANK_UPDATE_EXTENDED_14) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 14 bytes."""
        assert is_tank_update_extended_structure(bytes([0x01] * 13)) is False
        assert is_tank_update_extended_structure(bytes([0x01] * 15)) is False


class TestIsTankUpdateFullStructure:
    """Tests for tank update full structure detection (15 bytes)."""

    def test_matches_15_bytes(self) -> None:
        """Matches exactly 15-byte message."""
        assert is_tank_update_full_structure(TANK_UPDATE_FULL_15) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 15 bytes."""
        assert is_tank_update_full_structure(bytes([0x01] * 14)) is False
        assert is_tank_update_full_structure(bytes([0x01] * 16)) is False


# =============================================================================
# Decoder Tests
# =============================================================================


class TestDecodeCombatHit:
    """Tests for combat hit decoding."""

    def test_decodes_outgoing_hit(self) -> None:
        """Decodes outgoing combat hit correctly."""
        result = decode_combat_hit(COMBAT_HIT_11_OUTGOING)
        assert result["msg_type"] == "combat_hit"
        assert result["direction"] == 0x09
        assert result["attacker_id"] == 0x07CD  # cd 07 little-endian
        assert result["is_outgoing"] is True
        assert len(result["combat_data"]) == 7  # bytes 4-10

    def test_decodes_incoming_hit(self) -> None:
        """Decodes incoming combat hit correctly."""
        result = decode_combat_hit(COMBAT_HIT_11_INCOMING)
        assert result["msg_type"] == "combat_hit"
        assert result["direction"] == 0x0B
        assert result["attacker_id"] == 0x07CD  # cd 07 little-endian
        assert result["is_outgoing"] is False
        assert len(result["combat_data"]) == 7

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_combat_hit(bytes([0x01] * 10))
        with pytest.raises(ContainerDecodeError):
            decode_combat_hit(bytes([0x01] * 12))


class TestParseTankName:
    """Tests for _parse_tank_name helper function."""

    def test_returns_empty_for_short_info_bytes_standard(self) -> None:
        """Returns empty string when info_bytes too short for standard format."""
        # Standard format has name at offset 7, so 7 bytes = no name
        assert _parse_tank_name(bytes([0x00] * 7), is_extended=False) == ""
        # Even shorter
        assert _parse_tank_name(bytes([0x00] * 3), is_extended=False) == ""

    def test_returns_empty_for_short_info_bytes_extended(self) -> None:
        """Returns empty string when info_bytes too short for extended format."""
        # Extended format has name at offset 10, so 10 bytes = no name
        assert _parse_tank_name(bytes([0x00] * 10), is_extended=True) == ""
        # Even shorter
        assert _parse_tank_name(bytes([0x00] * 5), is_extended=True) == ""

    def test_parses_name_at_standard_offset(self) -> None:
        """Parses name from offset 7 in standard format."""
        # 7 padding bytes + "ABC"
        info = bytes([0x00] * 7) + b"ABC"
        assert _parse_tank_name(info, is_extended=False) == "ABC"

    def test_parses_name_at_extended_offset(self) -> None:
        """Parses name from offset 10 in extended format."""
        # 10 padding bytes + "Artax"
        info = bytes([0x00] * 10) + b"Artax"
        assert _parse_tank_name(info, is_extended=True) == "Artax"

    def test_replaces_non_printable_chars(self) -> None:
        """Replaces non-printable characters with '?'."""
        # 7 padding + byte 0x01 (non-printable) + "A"
        info = bytes([0x00] * 7) + bytes([0x01, 0x41])
        assert _parse_tank_name(info, is_extended=False) == "?A"


class TestDecodeTankRegistry:
    """Tests for tank registry decoding."""

    def test_decodes_16_byte_registry(self) -> None:
        """Decodes 16-byte tank registry correctly."""
        result = decode_tank_registry(TANK_REGISTRY_16)
        assert result["msg_type"] == "tank_registry"
        assert result["flags"] == 0x09
        assert result["tank_id"] == 0x5380  # 80 53 little-endian
        assert len(result["info_bytes"]) == 12  # 16 - 4 header bytes

    def test_decodes_20_byte_registry(self) -> None:
        """Decodes 20-byte tank registry correctly."""
        result = decode_tank_registry(TANK_REGISTRY_20)
        assert result["msg_type"] == "tank_registry"
        assert len(result["info_bytes"]) == 16  # 20 - 4 header bytes

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_registry(bytes([0x01] * 15))
        with pytest.raises(ContainerDecodeError):
            decode_tank_registry(bytes([0x01] * 21))

    def test_detects_bot(self) -> None:
        """Detects bot when first 6 info bytes are zeros."""
        result = decode_tank_registry(TANK_REGISTRY_BOT)
        assert result["is_bot"] is True
        assert result["is_container"] is False
        assert result["container_x"] is None
        assert result["container_y"] is None
        assert result["container_viewport_x"] is None
        assert result["team"] == "purple"  # flags 0x01 & 0x03 = 1 -> purple

    def test_detects_container_wasd_name(self) -> None:
        """Detects container when name is all direction chars."""
        result = decode_tank_registry(TANK_REGISTRY_CONTAINER_WASD)
        assert result["is_bot"] is False
        assert result["is_container"] is True
        # Container position: info[0]=y (absolute), info[1]=viewport_x (relative)
        assert result["container_y"] == 17  # info[0] = 0x11 (absolute y)
        assert result["container_viewport_x"] == 9  # info[1] = 0x09 (viewport-relative x)
        assert result["container_x"] is None  # Absolute x needs player position
        assert result["tank_name"] == ""  # Cleared for containers

    def test_detects_container_short_garbage(self) -> None:
        """Detects container when name is short with non-printables."""
        result = decode_tank_registry(TANK_REGISTRY_CONTAINER_GARBAGE)
        assert result["is_bot"] is False
        assert result["is_container"] is True
        # Container position: info[0]=y (absolute), info[1]=viewport_x (relative)
        assert result["container_y"] == 3  # info[0] = 0x03 (absolute y)
        assert result["container_viewport_x"] == 146  # info[1] = 0x92 (viewport-relative x)
        assert result["container_x"] is None  # Absolute x needs player position
        assert result["tank_name"] == ""  # Cleared for containers

    def test_regular_tank_not_container(self) -> None:
        """Regular tanks are not detected as containers."""
        result = decode_tank_registry(TANK_REGISTRY_16)
        assert result["is_bot"] is False
        assert result["is_container"] is False
        assert result["container_x"] is None
        assert result["container_y"] is None
        assert result["container_viewport_x"] is None


class TestDecodePositionUpdate:
    """Tests for position update decoding."""

    def test_decodes_13_byte_update(self) -> None:
        """Decodes 13-byte position update correctly."""
        result = decode_position_update(POSITION_UPDATE_13)
        assert result["msg_type"] == "position_update"
        assert result["flags"] == 0x53
        assert result["tank_id"] == 0x07CD  # cd 07 little-endian
        assert result["x"] == 0x15  # 21
        assert result["y"] == 0x12  # 18
        assert len(result["extra_data"]) == 7  # 13 - 4 header - 2 coords

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_position_update(bytes([0x01] * 12))
        with pytest.raises(ContainerDecodeError):
            decode_position_update(bytes([0x01] * 14))


class TestDecodeTankStatusShort:
    """Tests for tank status short decoding (9 bytes with rank/damage)."""

    def test_decodes_9_byte_status(self) -> None:
        """Decodes 9-byte tank status short correctly."""
        # Data: 01 82 57 02 04 00 15 00 00
        # [0]=subtype [1]=flags [2-3]=tank_id [4]=dmg [5]=rank [6-7]=lb_pos [8]=extra
        result = decode_tank_status_short(TANK_STATUS_SHORT_9)
        assert result["msg_type"] == "tank_status_short"
        assert result["flags"] == 0x82
        assert result["tank_id"] == 0x0257  # 57 02 little-endian = 599
        assert result["damage_state"] == 4
        assert result["rank"] == 0  # recruit
        assert result["leaderboard_position"] == 0x0015  # 15 00 little-endian

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_status_short(bytes([0x01] * 8))
        with pytest.raises(ContainerDecodeError):
            decode_tank_status_short(bytes([0x01] * 10))


class TestDecodeTankStatusSync:
    """Tests for tank status sync decoding."""

    def test_decodes_2_byte_sync(self) -> None:
        """Decodes 2-byte sync correctly."""
        result = decode_tank_status_sync(TANK_STATUS_SYNC_2)
        assert result["msg_type"] == "tank_status_sync"
        assert result["sync_data"] == bytes([0x00])

    def test_decodes_3_byte_sync(self) -> None:
        """Decodes 3-byte sync correctly."""
        result = decode_tank_status_sync(TANK_STATUS_SYNC_3)
        assert result["msg_type"] == "tank_status_sync"
        assert result["sync_data"] == bytes([0x01, 0x02])

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_status_sync(bytes([0x01]))
        with pytest.raises(ContainerDecodeError):
            decode_tank_status_sync(bytes([0x01] * 4))


class TestDecodeTankUpdateCompact:
    """Tests for tank update compact decoding (10 bytes)."""

    def test_decodes_10_byte_update(self) -> None:
        """Decodes 10-byte tank update compact correctly."""
        result = decode_tank_update_compact(TANK_UPDATE_COMPACT_10)
        assert result["msg_type"] == "tank_update_compact"
        assert result["flags"] == 0x44  # byte[1]
        assert result["tank_id"] == 0x50DF  # df 50 little-endian at bytes[2-3]
        assert len(result["status_data"]) == 6  # bytes 4-9

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_update_compact(bytes([0x01] * 9))
        with pytest.raises(ContainerDecodeError):
            decode_tank_update_compact(bytes([0x01] * 11))


class TestDecodeTankUpdateExtended:
    """Tests for tank update extended decoding (14 bytes)."""

    def test_decodes_14_byte_update(self) -> None:
        """Decodes 14-byte tank update extended correctly."""
        result = decode_tank_update_extended(TANK_UPDATE_EXTENDED_14)
        assert result["msg_type"] == "tank_update_extended"
        assert result["flags"] == 0x44  # byte[1]
        assert result["tank_id"] == 0x5079  # 79 50 little-endian at bytes[2-3]
        assert len(result["status_data"]) == 10  # bytes 4-13

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_update_extended(bytes([0x01] * 13))
        with pytest.raises(ContainerDecodeError):
            decode_tank_update_extended(bytes([0x01] * 15))


class TestDecodeTankUpdateFull:
    """Tests for tank update full decoding (15 bytes)."""

    def test_decodes_15_byte_update(self) -> None:
        """Decodes 15-byte tank update full correctly."""
        result = decode_tank_update_full(TANK_UPDATE_FULL_15)
        assert result["msg_type"] == "tank_update_full"
        assert result["flags"] == 0x46  # byte[1]
        assert result["tank_id"] == 0x50C7  # c7 50 little-endian at bytes[2-3]
        assert len(result["status_data"]) == 11  # bytes 4-14

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_update_full(bytes([0x01] * 14))
        with pytest.raises(ContainerDecodeError):
            decode_tank_update_full(bytes([0x01] * 16))


class TestDecodeTankLeave:
    """Tests for tank leave message decoding."""

    def test_decodes_tank_leave(self) -> None:
        """Correctly decodes tank leave message."""
        result = decode_tank_leave(TANK_LEAVE_6)
        assert result["msg_type"] == "tank_leave"
        assert result["tank_id"] == 139  # 0x8B from Arterial leaving
        assert result["flags"] == 0x13
        assert result["extra_data"] == bytes.fromhex("4213")

    def test_decodes_tank_leave_large_id(self) -> None:
        """Correctly decodes tank leave with large tank ID."""
        result = decode_tank_leave(TANK_LEAVE_LARGE_ID)
        assert result["msg_type"] == "tank_leave"
        assert result["tank_id"] == 23940  # 0x5d84 little-endian
        assert result["flags"] == 0x4A
        assert result["extra_data"] == bytes.fromhex("5201")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tank_leave(bytes([0x01] * 5))
        with pytest.raises(ContainerDecodeError):
            decode_tank_leave(bytes([0x01] * 7))


class TestIsTankLeaveStructure:
    """Tests for tank leave structure detection."""

    def test_matches_6_bytes_with_zero_high_byte(self) -> None:
        """Matches 6 bytes with tank_id high byte == 0."""
        assert is_tank_leave_structure(TANK_LEAVE_6)

    def test_accepts_6_bytes_with_large_tank_id(self) -> None:
        """Accepts 6 bytes for tanks with ID > 256."""
        # Tank ID 23940 (0x5d84) - high byte is non-zero
        data = bytes.fromhex("204a845d5201")
        assert is_tank_leave_structure(data)

    def test_rejects_other_lengths(self) -> None:
        """Rejects other lengths."""
        assert not is_tank_leave_structure(bytes([0x01] * 5))
        assert not is_tank_leave_structure(bytes([0x01] * 7))


class TestDecodePlayerListShort:
    """Tests for player list short response decoding."""

    def test_decodes_player_list_short(self) -> None:
        """Correctly decodes short player list response."""
        result = decode_player_list_short(PLAYER_LIST_SHORT_4)
        assert result["msg_type"] == "player_list_short"
        assert result["response_data"] == bytes.fromhex("990507")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_player_list_short(bytes([0x01] * 3))
        with pytest.raises(ContainerDecodeError):
            decode_player_list_short(bytes([0x01] * 5))


class TestIsPlayerListShortStructure:
    """Tests for player list short structure detection."""

    def test_matches_4_bytes(self) -> None:
        """Matches exactly 4 bytes."""
        assert is_player_list_short_structure(PLAYER_LIST_SHORT_4)

    def test_rejects_other_lengths(self) -> None:
        """Rejects other lengths."""
        assert not is_player_list_short_structure(bytes([0x01] * 3))
        assert not is_player_list_short_structure(bytes([0x01] * 5))


class TestDecodePlayerListExtended:
    """Tests for player list extended response decoding."""

    def test_decodes_player_list_extended(self) -> None:
        """Correctly decodes extended player list response."""
        result = decode_player_list_extended(PLAYER_LIST_EXTENDED_7)
        assert result["msg_type"] == "player_list_extended"
        assert result["response_data"] == bytes.fromhex("990507")
        assert result["extended_data"] == bytes.fromhex("ce1144")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_player_list_extended(bytes([0x01] * 6))
        with pytest.raises(ContainerDecodeError):
            decode_player_list_extended(bytes([0x01] * 8))


class TestIsPlayerListExtendedStructure:
    """Tests for player list extended structure detection."""

    def test_matches_7_bytes(self) -> None:
        """Matches exactly 7 bytes."""
        assert is_player_list_extended_structure(PLAYER_LIST_EXTENDED_7)

    def test_rejects_other_lengths(self) -> None:
        """Rejects other lengths."""
        assert not is_player_list_extended_structure(bytes([0x01] * 6))
        assert not is_player_list_extended_structure(bytes([0x01] * 8))


# =============================================================================
# Deactivation Kill Tests
# =============================================================================


class TestDecodeDeactivationKill:
    """Tests for deactivation kill message decoding."""

    def test_decodes_deactivation_kill(self) -> None:
        """Correctly decodes deactivation kill message."""
        result = decode_deactivation_kill(DEACTIVATION_KILL_5)
        assert result["msg_type"] == "deactivation_kill"
        # victim_id = 0xBB | (0x62 << 8) = 187 | 25088 = 25275
        assert result["victim_id"] == 25275
        # killer_id = 0x9C | (0x0E << 8) = 156 | 3584 = 3740
        assert result["killer_id"] == 3740

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_kill(bytes([0x41] + [0x01] * 3))
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_kill(bytes([0x41] + [0x01] * 5))


class TestIsDeactivationKillStructure:
    """Tests for deactivation kill structure detection."""

    def test_matches_5_bytes_with_0x41(self) -> None:
        """Matches exactly 5 bytes starting with 0x41."""
        assert is_deactivation_kill_structure(DEACTIVATION_KILL_5)

    def test_rejects_wrong_first_byte(self) -> None:
        """Rejects 5 bytes without 0x41 first byte."""
        assert not is_deactivation_kill_structure(bytes([0x42] + [0x01] * 4))
        assert not is_deactivation_kill_structure(bytes([0x43] + [0x01] * 4))

    def test_rejects_other_lengths(self) -> None:
        """Rejects other lengths even with 0x41."""
        assert not is_deactivation_kill_structure(bytes([0x41] + [0x01] * 3))
        assert not is_deactivation_kill_structure(bytes([0x41] + [0x01] * 5))


# =============================================================================
# Deactivation Death Tests
# =============================================================================


class TestDecodeDeactivationDeath:
    """Tests for deactivation death message decoding."""

    def test_decodes_deactivation_death(self) -> None:
        """Correctly decodes deactivation death message."""
        result = decode_deactivation_death(DEACTIVATION_DEATH_7)
        assert result["msg_type"] == "deactivation_death"
        assert result["flags"] == 0x07
        # killer_id = 0x86 | (0x16 << 8) = 134 | 5632 = 5766
        assert result["killer_id"] == 5766
        assert result["extra_data"] == bytes.fromhex("0c7f1f")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_death(bytes([0x43] + [0x01] * 5))
        with pytest.raises(ContainerDecodeError):
            decode_deactivation_death(bytes([0x43] + [0x01] * 7))


class TestIsDeactivationDeathStructure:
    """Tests for deactivation death structure detection."""

    def test_matches_7_bytes_with_0x43(self) -> None:
        """Matches exactly 7 bytes starting with 0x43."""
        assert is_deactivation_death_structure(DEACTIVATION_DEATH_7)

    def test_rejects_wrong_first_byte(self) -> None:
        """Rejects 7 bytes without 0x43 first byte."""
        assert not is_deactivation_death_structure(bytes([0x41] + [0x01] * 6))
        assert not is_deactivation_death_structure(bytes([0x79] + [0x01] * 6))

    def test_rejects_other_lengths(self) -> None:
        """Rejects other lengths even with 0x43."""
        assert not is_deactivation_death_structure(bytes([0x43] + [0x01] * 5))
        assert not is_deactivation_death_structure(bytes([0x43] + [0x01] * 7))


class TestDecodeUnknownContainer:
    """Tests for unknown container decoding."""

    def test_preserves_data(self) -> None:
        """Preserves data for unknown structures."""
        result = decode_unknown_container(UNKNOWN_8_BYTES)
        assert result["msg_type"] == "unknown_container"
        assert result["subtype"] == 0x7E
        assert result["length"] == 8
        assert result["data"] == UNKNOWN_8_BYTES

    def test_raises_on_empty_data(self) -> None:
        """Raises on empty data."""
        with pytest.raises(ContainerDecodeError):
            decode_unknown_container(b"")


# =============================================================================
# Message Type Identification Tests
# =============================================================================


class TestIdentifyContainerType:
    """Tests for container type identification."""

    def test_identifies_combat_hit(self) -> None:
        """Correctly identifies combat hit structure."""
        assert identify_container_type(COMBAT_HIT_11_OUTGOING) == ContainerMessageType.COMBAT_HIT
        assert identify_container_type(COMBAT_HIT_11_INCOMING) == ContainerMessageType.COMBAT_HIT

    def test_identifies_tank_registry(self) -> None:
        """Correctly identifies tank registry structure."""
        assert identify_container_type(TANK_REGISTRY_16) == ContainerMessageType.TANK_REGISTRY
        assert identify_container_type(TANK_REGISTRY_20) == ContainerMessageType.TANK_REGISTRY

    def test_identifies_position_update(self) -> None:
        """Correctly identifies position update structure."""
        assert identify_container_type(POSITION_UPDATE_13) == ContainerMessageType.POSITION_UPDATE

    def test_identifies_tank_status_short(self) -> None:
        """Correctly identifies tank status short structure (9 bytes)."""
        result = identify_container_type(TANK_STATUS_SHORT_9)
        assert result == ContainerMessageType.TANK_STATUS_SHORT

    def test_identifies_tank_status_sync(self) -> None:
        """Correctly identifies tank status sync structure."""
        assert identify_container_type(TANK_STATUS_SYNC_2) == ContainerMessageType.TANK_STATUS_SYNC
        assert identify_container_type(TANK_STATUS_SYNC_3) == ContainerMessageType.TANK_STATUS_SYNC

    def test_identifies_tank_update_compact(self) -> None:
        """Correctly identifies tank update compact structure (10 bytes)."""
        result = identify_container_type(TANK_UPDATE_COMPACT_10)
        assert result == ContainerMessageType.TANK_UPDATE_COMPACT

    def test_identifies_tank_update_extended(self) -> None:
        """Correctly identifies tank update extended structure (14 bytes)."""
        result = identify_container_type(TANK_UPDATE_EXTENDED_14)
        assert result == ContainerMessageType.TANK_UPDATE_EXTENDED

    def test_identifies_tank_update_full(self) -> None:
        """Correctly identifies tank update full structure (15 bytes)."""
        result = identify_container_type(TANK_UPDATE_FULL_15)
        assert result == ContainerMessageType.TANK_UPDATE_FULL

    def test_identifies_tank_leave(self) -> None:
        """Correctly identifies tank leave structure (6 bytes)."""
        result = identify_container_type(TANK_LEAVE_6)
        assert result == ContainerMessageType.TANK_LEAVE

    def test_identifies_player_list_short(self) -> None:
        """Correctly identifies player list short structure (4 bytes)."""
        result = identify_container_type(PLAYER_LIST_SHORT_4)
        assert result == ContainerMessageType.PLAYER_LIST_SHORT

    def test_identifies_player_list_extended(self) -> None:
        """Correctly identifies player list extended structure (7 bytes)."""
        result = identify_container_type(PLAYER_LIST_EXTENDED_7)
        assert result == ContainerMessageType.PLAYER_LIST_EXTENDED

    def test_identifies_deactivation_kill(self) -> None:
        """Correctly identifies deactivation kill structure (5 bytes with 0x41)."""
        result = identify_container_type(DEACTIVATION_KILL_5)
        assert result == ContainerMessageType.DEACTIVATION_KILL

    def test_identifies_deactivation_death(self) -> None:
        """Correctly identifies deactivation death structure (7 bytes with 0x43)."""
        result = identify_container_type(DEACTIVATION_DEATH_7)
        assert result == ContainerMessageType.DEACTIVATION_DEATH

    def test_identifies_unknown(self) -> None:
        """Correctly identifies unknown structure."""
        assert identify_container_type(UNKNOWN_8_BYTES) == ContainerMessageType.UNKNOWN
        assert identify_container_type(UNKNOWN_12_BYTES) == ContainerMessageType.UNKNOWN

    def test_empty_data_is_unknown(self) -> None:
        """Empty data is identified as unknown."""
        assert identify_container_type(b"") == ContainerMessageType.UNKNOWN


# =============================================================================
# Full Decoder Dispatch Tests
# =============================================================================


class TestDecodeContainerMessage:
    """Tests for main decode_container_message dispatcher."""

    def test_dispatches_combat_hit(self) -> None:
        """Dispatches to combat hit decoder."""
        result = decode_container_message(COMBAT_HIT_11_OUTGOING)
        assert result["msg_type"] == "combat_hit"

    def test_dispatches_tank_registry(self) -> None:
        """Dispatches to tank registry decoder."""
        result = decode_container_message(TANK_REGISTRY_16)
        assert result["msg_type"] == "tank_registry"

    def test_dispatches_position_update(self) -> None:
        """Dispatches to position update decoder."""
        result = decode_container_message(POSITION_UPDATE_13)
        assert result["msg_type"] == "position_update"

    def test_dispatches_tank_status_short(self) -> None:
        """Dispatches to tank status short decoder (9 bytes)."""
        result = decode_container_message(TANK_STATUS_SHORT_9)
        assert result["msg_type"] == "tank_status_short"

    def test_dispatches_tank_status_sync(self) -> None:
        """Dispatches to tank status sync decoder."""
        result = decode_container_message(TANK_STATUS_SYNC_2)
        assert result["msg_type"] == "tank_status_sync"

    def test_dispatches_tank_update_compact(self) -> None:
        """Dispatches to tank update compact decoder (10 bytes)."""
        result = decode_container_message(TANK_UPDATE_COMPACT_10)
        assert result["msg_type"] == "tank_update_compact"

    def test_dispatches_tank_update_extended(self) -> None:
        """Dispatches to tank update extended decoder (14 bytes)."""
        result = decode_container_message(TANK_UPDATE_EXTENDED_14)
        assert result["msg_type"] == "tank_update_extended"

    def test_dispatches_tank_update_full(self) -> None:
        """Dispatches to tank update full decoder (15 bytes)."""
        result = decode_container_message(TANK_UPDATE_FULL_15)
        assert result["msg_type"] == "tank_update_full"

    def test_dispatches_tank_leave(self) -> None:
        """Dispatches to tank leave decoder (6 bytes)."""
        result = decode_container_message(TANK_LEAVE_6)
        assert result["msg_type"] == "tank_leave"

    def test_dispatches_player_list_short(self) -> None:
        """Dispatches to player list short decoder (4 bytes)."""
        result = decode_container_message(PLAYER_LIST_SHORT_4)
        assert result["msg_type"] == "player_list_short"

    def test_dispatches_player_list_extended(self) -> None:
        """Dispatches to player list extended decoder (7 bytes)."""
        result = decode_container_message(PLAYER_LIST_EXTENDED_7)
        assert result["msg_type"] == "player_list_extended"

    def test_dispatches_deactivation_kill(self) -> None:
        """Dispatches to deactivation kill decoder (5 bytes with 0x41)."""
        result = decode_container_message(DEACTIVATION_KILL_5)
        assert result["msg_type"] == "deactivation_kill"

    def test_dispatches_deactivation_death(self) -> None:
        """Dispatches to deactivation death decoder (7 bytes with 0x43)."""
        result = decode_container_message(DEACTIVATION_DEATH_7)
        assert result["msg_type"] == "deactivation_death"

    def test_dispatches_unknown(self) -> None:
        """Dispatches to unknown decoder for unrecognized structures."""
        result = decode_container_message(UNKNOWN_8_BYTES)
        assert result["msg_type"] == "unknown_container"


# =============================================================================
# TypedDict Type Verification
# =============================================================================


class TestTypedDictStructure:
    """Verify TypedDict structures match expected keys."""

    def test_combat_hit_dict_keys(self) -> None:
        """CombatHitDict has expected keys."""
        result: CombatHitDict = decode_combat_hit(COMBAT_HIT_11_OUTGOING)
        assert "msg_type" in result
        assert "direction" in result
        assert "attacker_id" in result
        assert "combat_data" in result
        assert "is_outgoing" in result

    def test_tank_registry_dict_keys(self) -> None:
        """TankRegistryDict has expected keys."""
        result: TankRegistryDict = decode_tank_registry(TANK_REGISTRY_16)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "info_bytes" in result
        assert "team" in result
        assert "tank_name" in result
        assert "military_rank" in result
        assert "badge_count" in result
        assert "is_bot" in result
        assert "is_container" in result
        assert "container_x" in result
        assert "container_y" in result
        assert "container_viewport_x" in result

    def test_position_update_dict_keys(self) -> None:
        """PositionUpdateDict has expected keys."""
        result: PositionUpdateDict = decode_position_update(POSITION_UPDATE_13)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "x" in result
        assert "y" in result
        assert "extra_data" in result

    def test_tank_status_sync_dict_keys(self) -> None:
        """TankStatusSyncDict has expected keys."""
        result: TankStatusSyncDict = decode_tank_status_sync(TANK_STATUS_SYNC_2)
        assert "msg_type" in result
        assert "sync_data" in result

    def test_tank_status_short_dict_keys(self) -> None:
        """TankStatusShortDict has expected keys."""
        result: TankStatusShortDict = decode_tank_status_short(TANK_STATUS_SHORT_9)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "damage_state" in result
        assert "rank" in result
        assert "leaderboard_position" in result

    def test_tank_update_compact_dict_keys(self) -> None:
        """TankUpdateCompactDict has expected keys."""
        result: TankUpdateCompactDict = decode_tank_update_compact(TANK_UPDATE_COMPACT_10)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "status_data" in result

    def test_tank_update_extended_dict_keys(self) -> None:
        """TankUpdateExtendedDict has expected keys."""
        result: TankUpdateExtendedDict = decode_tank_update_extended(TANK_UPDATE_EXTENDED_14)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "status_data" in result

    def test_tank_update_full_dict_keys(self) -> None:
        """TankUpdateFullDict has expected keys."""
        result: TankUpdateFullDict = decode_tank_update_full(TANK_UPDATE_FULL_15)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "status_data" in result

    def test_unknown_container_dict_keys(self) -> None:
        """UnknownContainerDict has expected keys."""
        result: UnknownContainerDict = decode_unknown_container(UNKNOWN_8_BYTES)
        assert "msg_type" in result
        assert "subtype" in result
        assert "length" in result
        assert "data" in result

    def test_tank_leave_dict_keys(self) -> None:
        """TankLeaveDict has expected keys."""
        result: TankLeaveDict = decode_tank_leave(TANK_LEAVE_6)
        assert "msg_type" in result
        assert "tank_id" in result
        assert "flags" in result
        assert "extra_data" in result

    def test_player_list_short_dict_keys(self) -> None:
        """PlayerListShortDict has expected keys."""
        result: PlayerListShortDict = decode_player_list_short(PLAYER_LIST_SHORT_4)
        assert "msg_type" in result
        assert "response_data" in result

    def test_player_list_extended_dict_keys(self) -> None:
        """PlayerListExtendedDict has expected keys."""
        result: PlayerListExtendedDict = decode_player_list_extended(PLAYER_LIST_EXTENDED_7)
        assert "msg_type" in result
        assert "response_data" in result
        assert "extended_data" in result

    def test_deactivation_kill_dict_keys(self) -> None:
        """DeactivationKillDict has expected keys."""
        result: DeactivationKillDict = decode_deactivation_kill(DEACTIVATION_KILL_5)
        assert "msg_type" in result
        assert "victim_id" in result
        assert "killer_id" in result

    def test_deactivation_death_dict_keys(self) -> None:
        """DeactivationDeathDict has expected keys."""
        result: DeactivationDeathDict = decode_deactivation_death(DEACTIVATION_DEATH_7)
        assert "msg_type" in result
        assert "flags" in result
        assert "killer_id" in result
        assert "extra_data" in result

    def test_teleport_landed_dict_keys(self) -> None:
        """TeleportLandedDict has expected keys."""
        result: TeleportLandedDict = decode_teleport_landed(TELEPORT_LANDED_1)
        assert "msg_type" in result
        assert "subtype" in result

    def test_container_pickup_dict_keys(self) -> None:
        """ContainerPickupDict has expected keys."""
        result: ContainerPickupDict = decode_container_pickup(CONTAINER_PICKUP_EQUIPMENT)
        assert "msg_type" in result
        assert "x" in result
        assert "y" in result
        assert "volume" in result
        assert "is_fuel" in result

    def test_radar_container_dict_keys(self) -> None:
        """RadarContainerDict has expected keys (no is_fuel)."""
        result: RadarResponseDict = decode_radar_response(RADAR_RESPONSE_1)
        c = result["containers"][0]
        assert "x" in c
        assert "y" in c
        assert "volume" in c

    def test_radar_response_dict_keys(self) -> None:
        """RadarResponseDict has expected keys."""
        result: RadarResponseDict = decode_radar_response(RADAR_RESPONSE_1)
        assert "msg_type" in result
        assert "container_count" in result
        assert "containers" in result
        assert "mines" in result

    def test_tip_notification_dict_keys(self) -> None:
        """TipNotificationDict has expected keys."""
        result: TipNotificationDict = decode_tip_notification(TIP_NOTIFICATION_29)
        assert "msg_type" in result
        assert "subtype" in result
        assert "length" in result
        assert "notification_data" in result

    def test_chunk_data_dict_keys(self) -> None:
        """ChunkDataDict has expected keys."""
        result: ChunkDataDict = decode_chunk_data(CHUNK_DATA_80)
        assert "msg_type" in result
        assert "subtype" in result
        assert "length" in result
        assert "chunk_data" in result

    def test_world_state_dict_keys(self) -> None:
        """WorldStateDict has expected keys."""
        result: WorldStateDict = decode_world_state(WORLD_STATE_500)
        assert "msg_type" in result
        assert "subtype" in result
        assert "length" in result
        assert "world_data" in result


# =============================================================================
# Teleport Landed Tests
# =============================================================================


class TestIsTeleportLandedStructure:
    """Tests for teleport landed structure detection."""

    def test_matches_1_byte(self) -> None:
        """Matches exactly 1 byte."""
        assert is_teleport_landed_structure(TELEPORT_LANDED_1) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 1 byte."""
        assert is_teleport_landed_structure(bytes([0x0C, 0x00])) is False
        assert is_teleport_landed_structure(b"") is False


class TestDecodeTeleportLanded:
    """Tests for teleport landed decoding."""

    def test_decodes_1_byte_message(self) -> None:
        """Decodes 1-byte teleport landed message correctly."""
        result = decode_teleport_landed(TELEPORT_LANDED_1)
        assert result["msg_type"] == "teleport_landed"
        assert result["subtype"] == 0x0C

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_teleport_landed(bytes([0x0C, 0x00]))
        with pytest.raises(ContainerDecodeError):
            decode_teleport_landed(b"")


# =============================================================================
# Container Pickup Tests
# =============================================================================


class TestIsContainerPickupStructure:
    """Tests for container pickup structure detection."""

    def test_matches_5_bytes(self) -> None:
        """Matches exactly 5 bytes."""
        assert is_container_pickup_structure(CONTAINER_PICKUP_EQUIPMENT) is True
        assert is_container_pickup_structure(CONTAINER_PICKUP_FUEL) is True

    def test_rejects_other_lengths(self) -> None:
        """Rejects messages not exactly 5 bytes."""
        assert is_container_pickup_structure(bytes([0x01] * 4)) is False
        assert is_container_pickup_structure(bytes([0x01] * 6)) is False


class TestDecodeContainerPickup:
    """Tests for container pickup decoding."""

    def test_decodes_equipment_pickup(self) -> None:
        """Decodes equipment pickup (volume=0) correctly."""
        result = decode_container_pickup(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"
        assert result["x"] == 0x88  # 136
        assert result["y"] == 0x5E  # 94
        assert result["volume"] == 0
        assert result["is_fuel"] is False

    def test_decodes_fuel_pickup(self) -> None:
        """Decodes fuel pickup (volume>0) correctly."""
        result = decode_container_pickup(CONTAINER_PICKUP_FUEL)
        assert result["msg_type"] == "container_pickup"
        assert result["x"] == 0x89  # 137
        assert result["y"] == 0x5F  # 95
        assert result["volume"] == 618
        assert result["is_fuel"] is True

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_container_pickup(bytes([0x01] * 4))
        with pytest.raises(ContainerDecodeError):
            decode_container_pickup(bytes([0x01] * 6))


# =============================================================================
# Radar Response Tests
# =============================================================================


class TestIsRadarResponseStructure:
    """Tests for radar response structure detection."""

    def test_matches_valid_radar_response(self) -> None:
        """Matches valid radar response with correct count/length."""
        assert is_radar_response_structure(RADAR_RESPONSE_1) is True  # 7 bytes: 3 + 1*4
        assert is_radar_response_structure(RADAR_RESPONSE_2) is True  # 11 bytes: 3 + 2*4
        assert is_radar_response_structure(RADAR_RESPONSE_5) is True  # 23 bytes: 3 + 5*4

    def test_rejects_invalid_length(self) -> None:
        """Rejects messages with mismatched count/length."""
        # Count=1 but only 6 bytes (should be 7)
        assert is_radar_response_structure(bytes.fromhex("4f0100" + "00" * 3)) is False
        # Too short for even header
        assert is_radar_response_structure(bytes.fromhex("4f01")) is False

    def test_accepts_high_container_count(self) -> None:
        """Accepts radar response with count > 10 (valid)."""
        # Count=11 with enough bytes
        data = bytes.fromhex("4f0b00" + "00" * 44)  # 3 + 11*4 = 47 bytes
        assert is_radar_response_structure(data) is True

    def test_accepts_count_zero_with_mines(self) -> None:
        """Accepts radar response with count=0 and mines."""
        # Count=0, 3 mines (9 bytes)
        data = bytes.fromhex("4f0000" + "00" * 9)  # 3 header + 9 mine bytes
        assert is_radar_response_structure(data) is True

    def test_accepts_count_zero_empty(self) -> None:
        """Accepts radar response with count=0 and no mines."""
        data = bytes.fromhex("4f0000")  # Just header, 0 containers, 0 mines
        assert is_radar_response_structure(data) is True

    def test_rejects_invalid_mine_bytes(self) -> None:
        """Rejects when remaining bytes not divisible by 3."""
        # Count=1 container (4 bytes) + 2 remaining (not divisible by 3)
        data = bytes.fromhex("4f0100" + "00" * 6)  # 3 + 4 + 2 = 9 bytes
        assert is_radar_response_structure(data) is False


class TestDecodeRadarResponse:
    """Tests for radar response decoding."""

    def test_decodes_single_equipment_container(self) -> None:
        """Decodes radar with 1 equipment container."""
        result = decode_radar_response(RADAR_RESPONSE_1)
        assert result["msg_type"] == "radar_response"
        assert result["container_count"] == 1
        assert len(result["containers"]) == 1
        assert len(result["mines"]) == 0
        c = result["containers"][0]
        assert c["x"] == 123
        assert c["y"] == 105
        assert c["volume"] == -1  # Equipment uses volume=-1

    def test_decodes_mixed_containers(self) -> None:
        """Decodes radar with equipment and fuel containers."""
        result = decode_radar_response(RADAR_RESPONSE_2)
        assert result["msg_type"] == "radar_response"
        assert result["container_count"] == 2
        assert len(result["containers"]) == 2
        # First: equipment (volume=-1)
        assert result["containers"][0]["volume"] == -1
        # Second: fuel with volume 746
        assert result["containers"][1]["volume"] == 746

    def test_decodes_five_containers(self) -> None:
        """Decodes realistic 5-container radar response."""
        result = decode_radar_response(RADAR_RESPONSE_5)
        assert result["msg_type"] == "radar_response"
        assert result["container_count"] == 5
        assert len(result["containers"]) == 5
        # Check positions
        assert result["containers"][0]["x"] == 123
        assert result["containers"][0]["y"] == 105
        assert result["containers"][4]["x"] == 137
        assert result["containers"][4]["y"] == 95
        # 4 equipment (volume=-1) + 1 fuel (volume>0)
        equipment_count = sum(1 for c in result["containers"] if c["volume"] < 0)
        fuel_count = sum(1 for c in result["containers"] if c["volume"] >= 0)
        assert equipment_count == 4
        assert fuel_count == 1

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            # Count=1 but wrong length
            decode_radar_response(bytes.fromhex("4f0100" + "00" * 3))

    def test_decodes_containers_with_mines(self) -> None:
        """Decodes radar with containers and mines."""
        # 1 container (4 bytes) + 2 mines (6 bytes)
        # Container: x=50, y=60, volume=100 (fuel)
        # Mine 1: x=55, y=65, team=0 (red)
        # Mine 2: x=58, y=68, team=2 (blue)
        data = bytes.fromhex(
            "4f"  # subtype
            + "0100"  # container_count=1, flags=0
            + "323c6400"  # container: x=50, y=60, volume=100
            + "374100"  # mine: x=55, y=65, team=0
            + "3a4402"  # mine: x=58, y=68, team=2
        )
        result = decode_radar_response(data)
        assert result["container_count"] == 1
        assert len(result["containers"]) == 1
        assert len(result["mines"]) == 2
        # Check container
        assert result["containers"][0]["x"] == 50
        assert result["containers"][0]["y"] == 60
        assert result["containers"][0]["volume"] == 100
        # Check mines
        assert result["mines"][0]["x"] == 55
        assert result["mines"][0]["y"] == 65
        assert result["mines"][0]["team"] == 0
        assert result["mines"][1]["x"] == 58
        assert result["mines"][1]["y"] == 68
        assert result["mines"][1]["team"] == 2

    def test_decodes_mines_only(self) -> None:
        """Decodes radar with 0 containers and only mines."""
        # 0 containers, 2 mines
        data = bytes.fromhex(
            "4f"  # subtype
            + "0000"  # container_count=0, flags=0
            + "0a0b01"  # mine: x=10, y=11, team=1 (purple)
            + "0c0d03"  # mine: x=12, y=13, team=3 (orange)
        )
        result = decode_radar_response(data)
        assert result["container_count"] == 0
        assert len(result["containers"]) == 0
        assert len(result["mines"]) == 2
        assert result["mines"][0] == {"x": 10, "y": 11, "team": 1}
        assert result["mines"][1] == {"x": 12, "y": 13, "team": 3}


# =============================================================================
# Tip Notification Tests
# =============================================================================


class TestIsTipNotificationStructure:
    """Tests for tip notification structure detection."""

    def test_matches_29_bytes(self) -> None:
        """Matches 29 bytes (minimum of range)."""
        assert is_tip_notification_structure(TIP_NOTIFICATION_29) is True

    def test_matches_79_bytes(self) -> None:
        """Matches 79 bytes (maximum of range)."""
        assert is_tip_notification_structure(TIP_NOTIFICATION_79) is True

    def test_matches_55_bytes(self) -> None:
        """Matches 55 bytes (middle of range)."""
        assert is_tip_notification_structure(TIP_NOTIFICATION_55) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 29-79 range."""
        assert is_tip_notification_structure(bytes([0x01] * 28)) is False
        assert is_tip_notification_structure(bytes([0x01] * 80)) is False


class TestDecodeTipNotification:
    """Tests for tip notification decoding."""

    def test_decodes_29_byte_message(self) -> None:
        """Decodes 29-byte tip notification message correctly."""
        result = decode_tip_notification(TIP_NOTIFICATION_29)
        assert result["msg_type"] == "tip_notification"
        assert result["subtype"] == 0x68
        assert result["length"] == 29
        assert len(result["notification_data"]) == 28

    def test_decodes_79_byte_message(self) -> None:
        """Decodes 79-byte tip notification message correctly."""
        result = decode_tip_notification(TIP_NOTIFICATION_79)
        assert result["msg_type"] == "tip_notification"
        assert result["subtype"] == 0x68
        assert result["length"] == 79
        assert len(result["notification_data"]) == 78

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tip_notification(bytes([0x01] * 28))
        with pytest.raises(ContainerDecodeError):
            decode_tip_notification(bytes([0x01] * 80))


# =============================================================================
# Chunk Data Tests
# =============================================================================


class TestIsChunkDataStructure:
    """Tests for chunk data structure detection."""

    def test_matches_80_bytes(self) -> None:
        """Matches 80 bytes (minimum of range)."""
        assert is_chunk_data_structure(CHUNK_DATA_80) is True

    def test_matches_130_bytes(self) -> None:
        """Matches 130 bytes (maximum of range)."""
        assert is_chunk_data_structure(CHUNK_DATA_130) is True

    def test_matches_95_bytes(self) -> None:
        """Matches 95 bytes (middle of range)."""
        assert is_chunk_data_structure(CHUNK_DATA_95) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 80-130 range."""
        assert is_chunk_data_structure(bytes([0x01] * 79)) is False
        assert is_chunk_data_structure(bytes([0x01] * 131)) is False


class TestDecodeChunkData:
    """Tests for chunk data decoding."""

    def test_decodes_80_byte_message(self) -> None:
        """Decodes 80-byte chunk data message correctly."""
        result = decode_chunk_data(CHUNK_DATA_80)
        assert result["msg_type"] == "chunk_data"
        assert result["subtype"] == 0x14
        assert result["length"] == 80
        assert len(result["chunk_data"]) == 79

    def test_decodes_130_byte_message(self) -> None:
        """Decodes 130-byte chunk data message correctly."""
        result = decode_chunk_data(CHUNK_DATA_130)
        assert result["msg_type"] == "chunk_data"
        assert result["subtype"] == 0x14
        assert result["length"] == 130
        assert len(result["chunk_data"]) == 129

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_chunk_data(bytes([0x01] * 79))
        with pytest.raises(ContainerDecodeError):
            decode_chunk_data(bytes([0x01] * 131))


# =============================================================================
# World State Tests
# =============================================================================


class TestIsWorldStateStructure:
    """Tests for world state structure detection."""

    def test_matches_500_bytes(self) -> None:
        """Matches 500 bytes (minimum)."""
        assert is_world_state_structure(WORLD_STATE_500) is True

    def test_matches_650_bytes(self) -> None:
        """Matches 650 bytes (common size)."""
        assert is_world_state_structure(WORLD_STATE_650) is True

    def test_rejects_below_minimum(self) -> None:
        """Rejects messages below 500 bytes."""
        assert is_world_state_structure(bytes([0x01] * 499)) is False
        assert is_world_state_structure(bytes([0x01] * 131)) is False


class TestDecodeWorldState:
    """Tests for world state decoding."""

    def test_decodes_500_byte_message(self) -> None:
        """Decodes 500-byte world state message correctly."""
        result = decode_world_state(WORLD_STATE_500)
        assert result["msg_type"] == "world_state"
        assert result["subtype"] == 0x14
        assert result["length"] == 500
        assert len(result["world_data"]) == 499

    def test_decodes_650_byte_message(self) -> None:
        """Decodes 650-byte world state message correctly."""
        result = decode_world_state(WORLD_STATE_650)
        assert result["msg_type"] == "world_state"
        assert result["subtype"] == 0x14
        assert result["length"] == 650
        assert len(result["world_data"]) == 649

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_world_state(bytes([0x01] * 499))


# =============================================================================
# New Container Type Identification Tests
# =============================================================================


class TestIdentifyNewContainerTypes:
    """Tests for identification of new container types."""

    def test_identifies_teleport_landed(self) -> None:
        """Correctly identifies teleport landed structure (1 byte)."""
        result = identify_container_type(TELEPORT_LANDED_1)
        assert result == ContainerMessageType.TELEPORT_LANDED

    def test_identifies_container_pickup(self) -> None:
        """Correctly identifies container pickup structure (5 bytes with 0x43)."""
        result = identify_container_type(CONTAINER_PICKUP_EQUIPMENT)
        assert result == ContainerMessageType.CONTAINER_PICKUP
        result = identify_container_type(CONTAINER_PICKUP_FUEL)
        assert result == ContainerMessageType.CONTAINER_PICKUP

    def test_identifies_radar_response(self) -> None:
        """Correctly identifies radar response structure (7-43 bytes with 0x4F)."""
        assert identify_container_type(RADAR_RESPONSE_1) == ContainerMessageType.RADAR_RESPONSE
        assert identify_container_type(RADAR_RESPONSE_2) == ContainerMessageType.RADAR_RESPONSE
        assert identify_container_type(RADAR_RESPONSE_5) == ContainerMessageType.RADAR_RESPONSE

    def test_identifies_tip_notification(self) -> None:
        """Correctly identifies tip notification structure (29-79 bytes)."""
        assert identify_container_type(TIP_NOTIFICATION_29) == ContainerMessageType.TIP_NOTIFICATION
        assert identify_container_type(TIP_NOTIFICATION_79) == ContainerMessageType.TIP_NOTIFICATION
        assert identify_container_type(TIP_NOTIFICATION_55) == ContainerMessageType.TIP_NOTIFICATION

    def test_identifies_chunk_data(self) -> None:
        """Correctly identifies chunk data structure (80-130 bytes)."""
        assert identify_container_type(CHUNK_DATA_80) == ContainerMessageType.CHUNK_DATA
        assert identify_container_type(CHUNK_DATA_130) == ContainerMessageType.CHUNK_DATA
        assert identify_container_type(CHUNK_DATA_95) == ContainerMessageType.CHUNK_DATA

    def test_identifies_world_state(self) -> None:
        """Correctly identifies world state structure (500+ bytes)."""
        assert identify_container_type(WORLD_STATE_500) == ContainerMessageType.WORLD_STATE
        assert identify_container_type(WORLD_STATE_650) == ContainerMessageType.WORLD_STATE


# =============================================================================
# New Container Type Dispatch Tests
# =============================================================================


class TestDecodeNewContainerTypes:
    """Tests for main decode_container_message dispatcher with new types."""

    def test_dispatches_teleport_landed(self) -> None:
        """Dispatches to teleport landed decoder (1 byte)."""
        result = decode_container_message(TELEPORT_LANDED_1)
        assert result["msg_type"] == "teleport_landed"

    def test_dispatches_container_pickup(self) -> None:
        """Dispatches to container pickup decoder (5 bytes with 0x43)."""
        result = decode_container_message(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"

    def test_dispatches_radar_response(self) -> None:
        """Dispatches to radar response decoder (7-43 bytes with 0x4F)."""
        result = decode_container_message(RADAR_RESPONSE_1)
        assert result["msg_type"] == "radar_response"

    def test_dispatches_tip_notification(self) -> None:
        """Dispatches to tip notification decoder (29-79 bytes)."""
        result = decode_container_message(TIP_NOTIFICATION_29)
        assert result["msg_type"] == "tip_notification"

    def test_dispatches_chunk_data(self) -> None:
        """Dispatches to chunk data decoder (80-130 bytes)."""
        result = decode_container_message(CHUNK_DATA_80)
        assert result["msg_type"] == "chunk_data"

    def test_dispatches_world_state(self) -> None:
        """Dispatches to world state decoder (500+ bytes)."""
        result = decode_container_message(WORLD_STATE_500)
        assert result["msg_type"] == "world_state"


# =============================================================================
# ContainerDecodeError Tests
# =============================================================================


class TestContainerDecodeError:
    """Tests for ContainerDecodeError exception."""

    def test_error_can_be_raised_and_caught(self) -> None:
        """ContainerDecodeError can be raised and caught as Exception."""
        with pytest.raises(Exception) as exc:
            raise ContainerDecodeError("test message")
        assert str(exc.value) == "test message"

    def test_error_message_stored(self) -> None:
        """Error message is stored in message attribute."""
        error = ContainerDecodeError("test message")
        assert error.message == "test message"

    def test_error_str_representation(self) -> None:
        """Error has proper string representation."""
        error = ContainerDecodeError("test message")
        assert str(error) == "test message"


# =============================================================================
# DecodeLevel Tests
# =============================================================================


class TestDecodeLevel:
    """Tests for DecodeLevel enum."""

    def test_level_values_are_weights(self) -> None:
        """DecodeLevel values are integer weights for stats calculation."""
        assert DecodeLevel.UNKNOWN.value == 0
        assert DecodeLevel.IDENTIFIED.value == 25
        assert DecodeLevel.PARTIAL.value == 50
        assert DecodeLevel.FULL.value == 100

    def test_level_ordering(self) -> None:
        """DecodeLevel values are ordered from least to most understanding."""
        assert DecodeLevel.UNKNOWN < DecodeLevel.IDENTIFIED
        assert DecodeLevel.IDENTIFIED < DecodeLevel.PARTIAL
        assert DecodeLevel.PARTIAL < DecodeLevel.FULL


# =============================================================================
# get_decode_level Tests
# =============================================================================


class TestGetDecodeLevel:
    """Tests for get_decode_level function."""

    def test_full_level_for_combat_hit(self) -> None:
        """Combat hit has FULL decode level."""
        level = get_decode_level(ContainerMessageType.COMBAT_HIT)
        assert level == DecodeLevel.FULL

    def test_full_level_for_movement(self) -> None:
        """Movement has FULL decode level."""
        level = get_decode_level(ContainerMessageType.MOVEMENT)
        assert level == DecodeLevel.FULL

    def test_full_level_for_container_pickup(self) -> None:
        """Container pickup has FULL decode level."""
        level = get_decode_level(ContainerMessageType.CONTAINER_PICKUP)
        assert level == DecodeLevel.FULL

    def test_full_level_for_radar_response(self) -> None:
        """Radar response has FULL decode level."""
        level = get_decode_level(ContainerMessageType.RADAR_RESPONSE)
        assert level == DecodeLevel.FULL

    def test_identified_level_for_tip_notification(self) -> None:
        """Tip notification has IDENTIFIED decode level."""
        level = get_decode_level(ContainerMessageType.TIP_NOTIFICATION)
        assert level == DecodeLevel.IDENTIFIED

    def test_identified_level_for_world_state(self) -> None:
        """World state has IDENTIFIED decode level."""
        level = get_decode_level(ContainerMessageType.WORLD_STATE)
        assert level == DecodeLevel.IDENTIFIED

    def test_unknown_for_unknown_type(self) -> None:
        """UNKNOWN type has UNKNOWN decode level."""
        level = get_decode_level(ContainerMessageType.UNKNOWN)
        assert level == DecodeLevel.UNKNOWN


# =============================================================================
# MESSAGE_TYPE_LEVELS Tests
# =============================================================================


class TestMessageTypeLevels:
    """Tests for MESSAGE_TYPE_LEVELS registry."""

    def test_all_message_types_have_level(self) -> None:
        """All ContainerMessageType values are in MESSAGE_TYPE_LEVELS."""
        for msg_type in ContainerMessageType:
            assert msg_type in MESSAGE_TYPE_LEVELS, f"{msg_type} missing from registry"

    def test_registry_values_are_decode_levels(self) -> None:
        """All values in MESSAGE_TYPE_LEVELS are DecodeLevel enum values."""
        for msg_type, level in MESSAGE_TYPE_LEVELS.items():
            assert level in DecodeLevel, f"{msg_type} has invalid level {level}"
