"""Tests for tank-related container decoders.

Tests for tank registry, status, update, and leave decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    MinePlacementDict,
    TankLeaveDict,
    TankRegistryDict,
    TankStatusShortDict,
    TankStatusSyncDict,
    TankUpdateCompactDict,
    TankUpdateExtendedDict,
    TankUpdateFullDict,
    decode_tank_leave,
    decode_tank_registry,
    decode_tank_status_short,
    decode_tank_status_sync,
    decode_tank_update_compact,
    decode_tank_update_extended,
    decode_tank_update_full,
)
from tankpit_bot.container.decoders.combat import decode_mine_placement
from tankpit_bot.container.decoders.tank import _parse_tank_name
from tests.container.test_data import (
    MINE_PLACEMENT_15,
    TANK_LEAVE_6,
    TANK_LEAVE_LARGE_ID,
    TANK_REGISTRY_16,
    TANK_REGISTRY_20,
    TANK_REGISTRY_BOT,
    TANK_REGISTRY_CONTAINER_GARBAGE,
    TANK_REGISTRY_CONTAINER_WASD,
    TANK_STATUS_SHORT_9,
    TANK_STATUS_SYNC_2,
    TANK_STATUS_SYNC_3,
    TANK_UPDATE_COMPACT_10,
    TANK_UPDATE_EXTENDED_14,
    TANK_UPDATE_FULL_15,
)


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


class TestDecodeMinePlacement:
    """Tests for tunneled mine placement decoding."""

    def test_decodes_captured_mine_placement(self) -> None:
        """Decodes captured 15-byte tunneled mine placement correctly."""
        result = decode_mine_placement(MINE_PLACEMENT_15)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 2
        assert result["tank_id"] == 1301
        assert result["positions"] == [
            (131, 126),
            (131, 125),
            (132, 125),
            (132, 126),
            (132, 127),
        ]

    def test_mine_placement_dict_keys(self) -> None:
        """MinePlacementDict has expected keys."""
        result: MinePlacementDict = decode_mine_placement(MINE_PLACEMENT_15)
        assert result["msg_type"] == 0x4B
        assert result["mine_type"] == 2
        assert result["tank_id"] == 1301
        assert len(result["positions"]) == 5
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

    def test_tank_registry_dict_keys(self) -> None:
        """TankRegistryDict has expected keys."""
        result: TankRegistryDict = decode_tank_registry(TANK_REGISTRY_16)
        assert result["msg_type"] == "tank_registry"
        assert result["flags"] == 0x09
        assert result["tank_id"] == 0x5380
        assert len(result["info_bytes"]) == 12
        assert result["team"] == "purple"  # flags 0x09 & 0x03 = 1 -> purple
        assert result["tank_name"] == "ev"  # extracted from info_bytes
        assert result["military_rank"] == 3
        assert result["badge_count"] == 1
        assert result["is_bot"] is False
        assert result["is_container"] is False


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

    def test_tank_status_short_dict_keys(self) -> None:
        """TankStatusShortDict has expected keys."""
        result: TankStatusShortDict = decode_tank_status_short(TANK_STATUS_SHORT_9)
        assert result["msg_type"] == "tank_status_short"
        assert result["flags"] == 0x82
        assert result["tank_id"] == 0x0257
        assert result["damage_state"] == 4
        assert result["rank"] == 0
        assert result["leaderboard_position"] == 0x0015


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

    def test_tank_status_sync_dict_keys(self) -> None:
        """TankStatusSyncDict has expected keys."""
        result: TankStatusSyncDict = decode_tank_status_sync(TANK_STATUS_SYNC_2)
        assert result["msg_type"] == "tank_status_sync"
        assert result["sync_data"] == bytes([0x00])


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

    def test_tank_update_compact_dict_keys(self) -> None:
        """TankUpdateCompactDict has expected keys."""
        result: TankUpdateCompactDict = decode_tank_update_compact(TANK_UPDATE_COMPACT_10)
        assert result["msg_type"] == "tank_update_compact"
        assert result["flags"] == 0x44
        assert result["tank_id"] == 0x50DF
        assert len(result["status_data"]) == 6


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

    def test_tank_update_extended_dict_keys(self) -> None:
        """TankUpdateExtendedDict has expected keys."""
        result: TankUpdateExtendedDict = decode_tank_update_extended(TANK_UPDATE_EXTENDED_14)
        assert result["msg_type"] == "tank_update_extended"
        assert result["flags"] == 0x44
        assert result["tank_id"] == 0x5079
        assert len(result["status_data"]) == 10


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

    def test_tank_update_full_dict_keys(self) -> None:
        """TankUpdateFullDict has expected keys."""
        result: TankUpdateFullDict = decode_tank_update_full(TANK_UPDATE_FULL_15)
        assert result["msg_type"] == "tank_update_full"
        assert result["flags"] == 0x46
        assert result["tank_id"] == 0x50C7
        assert len(result["status_data"]) == 11


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

    def test_tank_leave_dict_keys(self) -> None:
        """TankLeaveDict has expected keys."""
        result: TankLeaveDict = decode_tank_leave(TANK_LEAVE_6)
        assert result["msg_type"] == "tank_leave"
        assert result["tank_id"] == 139
        assert result["flags"] == 0x13
        assert result["extra_data"] == bytes.fromhex("4213")
