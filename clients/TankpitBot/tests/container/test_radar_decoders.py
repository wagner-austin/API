"""Tests for radar-related container decoders.

0x4F RadarResponse moved to the protocol layer 2026-06-19; the canonical
decoder is tankpit_bot.protocol.decode_radar_scan_result. Tests for
container pickup and teleport-landed remain here.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    ContainerPickupDict,
    TeleportLandedDict,
    decode_container_pickup,
    decode_teleport_landed,
)
from tests.container.test_data import (
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    TELEPORT_LANDED_1,
)


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

    def test_teleport_landed_dict_keys(self) -> None:
        """TeleportLandedDict has expected keys."""
        result: TeleportLandedDict = decode_teleport_landed(TELEPORT_LANDED_1)
        assert result["msg_type"] == "teleport_landed"
        assert result["subtype"] == 0x0C


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

    def test_container_pickup_dict_keys(self) -> None:
        """ContainerPickupDict has expected keys."""
        result: ContainerPickupDict = decode_container_pickup(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"
        assert result["x"] == 0x88
        assert result["y"] == 0x5E
        assert result["volume"] == 0
        assert result["is_fuel"] is False


# RadarResponse decoder tests moved to tests/protocol/test_radar.py
# (decode_radar_scan_result). Container path deleted 2026-06-19.
