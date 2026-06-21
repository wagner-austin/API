"""Tests for container message type identification and dispatch.

Tests for identify_container_type and decode_container_message dispatcher.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerMessageType,
    decode_container_message,
    identify_container_type,
)
from tests.container.test_data import (
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    MINE_DETONATION_3,
    MINE_DETONATION_15,
    MINE_PLACEMENT_15,
    TANK_STATUS_SYNC_2,
    TANK_STATUS_SYNC_3,
    TELEPORT_LANDED_1,
    UNKNOWN_8_BYTES,
    UNKNOWN_12_BYTES,
)


class TestIdentifyContainerType:
    """Tests for container type identification."""

    def test_identifies_mine_placement(self) -> None:
        """Correctly identifies captured tunneled mine placement."""
        assert identify_container_type(MINE_PLACEMENT_15) == ContainerMessageType.MINE_PLACEMENT

    def test_identifies_mine_detonation(self) -> None:
        """Correctly identifies captured tunneled mine detonation."""
        assert identify_container_type(MINE_DETONATION_3) == ContainerMessageType.MINE_DETONATION
        assert identify_container_type(MINE_DETONATION_15) == ContainerMessageType.MINE_DETONATION

    # Container Movement identification was deleted 2026-06-19.
    # Container TankRegistry / PositionUpdate / TankLeave / PlayerListShort
    # / PlayerListExtended / DeactivationDeath identification deleted
    # 2026-06-20 after corpus sweep proved zero production fires.
    # The unified `decode_0x2e_message` dispatches the corresponding
    # protocol subtypes (0x21, 0x3D, etc.) via subtype-first matching.

    def test_rejects_false_positive_unknown_packet(self) -> None:
        """Does not classify an unrelated packet as a known container subtype."""
        data = bytes.fromhex("46938292839182938191839181")
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN

    def test_short_bodies_are_unknown(self) -> None:
        """2-3 byte container bodies resolve to UNKNOWN_CONTAINER."""
        assert identify_container_type(TANK_STATUS_SYNC_2) == ContainerMessageType.UNKNOWN
        assert identify_container_type(TANK_STATUS_SYNC_3) == ContainerMessageType.UNKNOWN

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

    def test_identifies_unknown(self) -> None:
        """Correctly identifies unknown structure."""
        assert identify_container_type(UNKNOWN_8_BYTES) == ContainerMessageType.UNKNOWN
        assert identify_container_type(UNKNOWN_12_BYTES) == ContainerMessageType.UNKNOWN

    def test_empty_data_is_unknown(self) -> None:
        """Empty data is identified as unknown."""
        assert identify_container_type(b"") == ContainerMessageType.UNKNOWN


class TestDecodeContainerMessage:
    """Tests for main decode_container_message dispatcher."""

    def test_dispatches_mine_placement(self) -> None:
        """Dispatches to tunneled mine placement decoder."""
        result = decode_container_message(MINE_PLACEMENT_15)
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

    def test_dispatches_mine_detonation(self) -> None:
        """Dispatches to tunneled mine detonation decoder."""
        result = decode_container_message(MINE_DETONATION_15)
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

    def test_dispatches_short_body_as_unknown(self) -> None:
        """Short bodies (2-3 bytes) dispatch as unknown_container."""
        result = decode_container_message(TANK_STATUS_SYNC_2)
        assert result["msg_type"] == "unknown_container"

    def test_dispatches_teleport_landed(self) -> None:
        """Dispatches to teleport landed decoder (1 byte)."""
        result = decode_container_message(TELEPORT_LANDED_1)
        assert result["msg_type"] == "teleport_landed"

    def test_dispatches_container_pickup(self) -> None:
        """Dispatches to container pickup decoder (5 bytes with 0x43)."""
        result = decode_container_message(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"

    def test_dispatches_unknown(self) -> None:
        """Dispatches to unknown decoder for unrecognized structures."""
        result = decode_container_message(UNKNOWN_8_BYTES)
        assert result["msg_type"] == "unknown_container"

    def test_empty_data_raises(self) -> None:
        """Empty body raises ContainerDecodeError (no subtype byte)."""
        from tankpit_bot.container.helpers import ContainerDecodeError

        with pytest.raises(ContainerDecodeError):
            decode_container_message(b"")

    def test_subtype_0x43_with_8_bytes_falls_through(self) -> None:
        """0x43 with length other than 5 falls through.

        The 8-byte body misses container_pickup (5); 8 isn't a known
        length match either, so unknown_container.
        """
        result = decode_container_message(bytes([0x43, 1, 2, 3, 4, 5, 6, 7]))
        assert result["msg_type"] == "unknown_container"

    def test_subtype_0x79_falls_through(self) -> None:
        """0x79 bodies always fall through to unknown.

        Container PlayerListShort/Extended were deleted 2026-06-20 after
        corpus proof of zero production fires.
        """
        result = decode_container_message(bytes([0x79, 1, 2, 3, 4]))
        assert result["msg_type"] == "unknown_container"
        result = decode_container_message(bytes([0x79, 1, 2, 3, 4, 5, 6]))
        assert result["msg_type"] == "unknown_container"
