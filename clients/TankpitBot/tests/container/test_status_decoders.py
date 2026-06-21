"""Tests for 0x2E container body identification.

The container-only SelfStatus decoder was deleted 2026-06-19; the
13-byte nested 0x2E form is now decoded by the protocol path's
TankStatusSync (decode_tank_status_sync), which extracts the fuel
field at offsets 10-11. Tests for the protocol path live in
tests/protocol/test_tank.py.

The 0x3D TankPositionStatus moved to the protocol layer as
MovementResponseDict (with the carrying byte restored). Tests for
that path live in tests/protocol/test_movement.py.

Container PositionUpdate (0x24 13-byte) was deleted 2026-06-20 after
empirical proof of zero production fires; 13-byte 0x2E bodies are now
all 0x3D MovementResponse via the protocol tunnel.
"""

from __future__ import annotations

from tankpit_bot.container.identification import identify_container_type
from tankpit_bot.container.types import ContainerMessageType


class TestIdentification:
    """Tests for the slimmed-down container identification."""

    def test_zero_24_no_longer_identified_as_position_update(self) -> None:
        """0x24 13-byte was POSITION_UPDATE; now falls through (decoder deleted)."""
        data = bytes([0x24, 0x02, 0x15, 0x05, 0xA5, 0x4A, 0x08, 0x01, 0x01, 0x00, 0x00, 0x97, 0x00])
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN

    def test_zero_2e_no_longer_identified_as_self_status(self) -> None:
        """0x2E 13-byte was SELF_STATUS in the container path; now falls through.

        The 13-byte nested 0x2E body is dispatched by the protocol path's
        TankStatusSync via the unified `decode_0x2e_message` entrypoint.
        Calling `identify_container_type` directly returns UNKNOWN.
        """
        data = bytes([0x2E, 0x02, 0x15, 0x05, 0x01, 0x01, 0x00, 0x00, 0x97, 0x0A, 0x01, 0x4C, 0x04])
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN

    def test_zero_3d_no_longer_identified_as_tank_position_status(self) -> None:
        """0x3D 13-byte was TANK_POSITION_STATUS; now falls through (moved to protocol)."""
        data = bytes([0x3D, 0x02, 0x15, 0x05, 0xA5, 0x4A, 0x08, 0x01, 0x01, 0x00, 0x00, 0x97, 0x00])
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN

    def test_unknown_subtype_13_bytes(self) -> None:
        """Unknown subtype at 13 bytes falls through to UNKNOWN."""
        data = bytes([0x99, 0x02, 0x15, 0x05, 0xA5, 0x4A, 0x08, 0x01, 0x01, 0x00, 0x00, 0x97, 0x00])
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN
