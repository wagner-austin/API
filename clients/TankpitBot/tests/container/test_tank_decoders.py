"""Tests for tank-related container decoders.

Tests for tank registry, status, update, and leave decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    MineDetonationDict,
    MinePlacementDict,
    TankLeaveDict,
    TankRegistryDict,
    decode_tank_leave,
    decode_tank_registry,
)
from tankpit_bot.container.decoders.combat import decode_mine_detonation, decode_mine_placement
from tankpit_bot.container.decoders.tank import _parse_tank_name
from tests.container.test_data import (
    MINE_DETONATION_3,
    MINE_DETONATION_15,
    MINE_PLACEMENT_15,
    TANK_LEAVE_6,
    TANK_LEAVE_LARGE_ID,
    TANK_REGISTRY_16,
    TANK_REGISTRY_20,
    TANK_REGISTRY_BOT,
    TANK_REGISTRY_CONTAINER_GARBAGE,
    TANK_REGISTRY_CONTAINER_WASD,
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


class TestDecodeMineDetonation:
    """Tests for tunneled mine detonation decoding."""

    def test_decodes_solitary_mine_detonation(self) -> None:
        """Decodes 3-byte tunneled mine detonation correctly."""
        result = decode_mine_detonation(MINE_DETONATION_3)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [(44, 59)]

    def test_decodes_chain_reaction_mine_detonation(self) -> None:
        """Decodes 15-byte tunneled mine detonation correctly."""
        result = decode_mine_detonation(MINE_DETONATION_15)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [
            (38, 52),
            (39, 53),
            (38, 54),
            (37, 53),
            (39, 54),
            (39, 52),
            (37, 54),
        ]

    def test_mine_detonation_dict_keys(self) -> None:
        """MineDetonationDict has expected keys."""
        result: MineDetonationDict = decode_mine_detonation(MINE_DETONATION_3)
        assert result["msg_type"] == 0x45
        assert result["positions"] == [(44, 59)]

    def test_raises_on_invalid_mine_detonation(self) -> None:
        """Raises on invalid tunneled mine detonation payload."""
        with pytest.raises(ContainerDecodeError):
            decode_mine_detonation(bytes.fromhex("442c3b"))

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


# Container TankStatusShort decoder was deleted 2026-06-19. Crack
# confirmed 0/74 production samples produced a valid container rank;
# all 74 are Og.h-shaped (decode_tank_status_sync). Real 0x2E
# TankStatusSync is tested in
# tests/protocol/test_tank.py::TestDecodeTankStatusSync.


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
