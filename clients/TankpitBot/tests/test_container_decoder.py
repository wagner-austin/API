"""Tests for container_decoder module - structure-based 0x2E message decoding.

All tests use real data patterns from captured sessions.
No mocks, no weak assertions, 100% coverage target.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container_decoder import (
    CombatHitDict,
    ContainerDecodeError,
    ContainerMessageType,
    DeactivationDeathDict,
    DeactivationKillDict,
    PlayerListExtendedDict,
    PlayerListShortDict,
    PositionUpdateDict,
    TankLeaveDict,
    TankRegistryDict,
    TankStatusShortDict,
    TankStatusSyncDict,
    TankUpdateCompactDict,
    TankUpdateExtendedDict,
    TankUpdateFullDict,
    UnknownContainerDict,
    decode_combat_hit,
    decode_container_message,
    decode_deactivation_death,
    decode_deactivation_kill,
    decode_player_list_extended,
    decode_player_list_short,
    decode_position_update,
    decode_tank_leave,
    decode_tank_registry,
    decode_tank_status_short,
    decode_tank_status_sync,
    decode_tank_update_compact,
    decode_tank_update_extended,
    decode_tank_update_full,
    decode_unknown_container,
    extract_uint16_le,
    identify_container_type,
    is_combat_hit_structure,
    is_deactivation_death_structure,
    is_deactivation_kill_structure,
    is_player_list_extended_structure,
    is_player_list_short_structure,
    is_position_update_structure,
    is_tank_leave_structure,
    is_tank_registry_structure,
    is_tank_status_short_structure,
    is_tank_status_sync_structure,
    is_tank_update_compact_structure,
    is_tank_update_extended_structure,
    is_tank_update_full_structure,
    require_exact_length,
    require_length_range,
    require_min_length,
)

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

# Unknown: 5 bytes (doesn't match any pattern - gap between 4 and 6)
UNKNOWN_5_BYTES = bytes.fromhex("0102030405")

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
    """Tests for tank registry structure detection."""

    def test_matches_16_bytes(self) -> None:
        """Matches 16-byte message (minimum)."""
        assert is_tank_registry_structure(TANK_REGISTRY_16) is True

    def test_matches_20_bytes(self) -> None:
        """Matches 20-byte message (maximum)."""
        assert is_tank_registry_structure(TANK_REGISTRY_20) is True

    def test_matches_17_18_19_bytes(self) -> None:
        """Matches 17, 18, and 19 byte messages."""
        assert is_tank_registry_structure(bytes([0x01] * 17)) is True
        assert is_tank_registry_structure(bytes([0x01] * 18)) is True
        assert is_tank_registry_structure(bytes([0x01] * 19)) is True

    def test_rejects_outside_range(self) -> None:
        """Rejects messages outside 16-20 range."""
        assert is_tank_registry_structure(bytes([0x01] * 15)) is False
        assert is_tank_registry_structure(bytes([0x01] * 21)) is False


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


class TestDecodePositionUpdate:
    """Tests for position update decoding."""

    def test_decodes_13_byte_update(self) -> None:
        """Decodes 13-byte position update correctly."""
        result = decode_position_update(POSITION_UPDATE_13)
        assert result["msg_type"] == "position_update"
        assert result["flags"] == 0x53
        assert result["tank_id"] == 0x07CD  # cd 07 little-endian
        assert len(result["status_bytes"]) == 9  # 13 - 4 header bytes

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
        result = decode_tank_status_short(TANK_STATUS_SHORT_9)
        assert result["msg_type"] == "tank_status_short"
        assert result["tank_id"] == 0x5782  # 82 57 little-endian
        assert result["damage_state"] == 2  # medium damage
        assert result["rank"] == 4  # lieutenant
        assert result["flag"] == 0
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
        assert identify_container_type(UNKNOWN_5_BYTES) == ContainerMessageType.UNKNOWN

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

    def test_position_update_dict_keys(self) -> None:
        """PositionUpdateDict has expected keys."""
        result: PositionUpdateDict = decode_position_update(POSITION_UPDATE_13)
        assert "msg_type" in result
        assert "flags" in result
        assert "tank_id" in result
        assert "status_bytes" in result

    def test_tank_status_sync_dict_keys(self) -> None:
        """TankStatusSyncDict has expected keys."""
        result: TankStatusSyncDict = decode_tank_status_sync(TANK_STATUS_SYNC_2)
        assert "msg_type" in result
        assert "sync_data" in result

    def test_tank_status_short_dict_keys(self) -> None:
        """TankStatusShortDict has expected keys."""
        result: TankStatusShortDict = decode_tank_status_short(TANK_STATUS_SHORT_9)
        assert "msg_type" in result
        assert "tank_id" in result
        assert "damage_state" in result
        assert "rank" in result
        assert "flag" in result
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
