"""Tests for container structure detection functions.

Tests for is_*_structure pattern matching functions.

Container TankRegistry / PositionUpdate / TankLeave / PlayerListShort /
PlayerListExtended / DeactivationDeath structure checks all deleted
2026-06-20 after empirical proof of zero production fires. The unified
`decode_0x2e_message` dispatches the corresponding protocol subtypes
(0x21, 0x3D, etc.) via subtype-first matching.
"""

from __future__ import annotations

from tankpit_bot.container import (
    is_container_pickup_structure,
    is_teleport_landed_structure,
)
from tests.container.test_data import (
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    TELEPORT_LANDED_1,
)

# is_movement_structure tests deleted 2026-06-19: container Movement
# decoder removed. The unified `decode_0x2e_message` dispatches 0x47
# Movement to the protocol decoder via subtype + length-gate matching.


# Container TankStatusSync structure check was a 2-3 byte length-only
# catch-all -- deleted 2026-06-19. Real 0x2E TankStatusSync (8+ bytes
# per JS Og.h) is tested at the protocol layer.


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


# 0x41 DeactivationKill structure tests moved to the protocol layer
# tests/protocol/test_combat.py:TestDecodeDeactivation -- container
# path was deleted 2026-06-19.


# TipNotification / ChunkData / WorldState structure-check tests
# deleted 2026-06-19 -- all three length-based types had zero
# production traffic across 150 corpus sessions after 0x4C MapData
# was tunneled correctly inside 0x2E.
