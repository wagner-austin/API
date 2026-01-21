"""Tests for radar-related container decoders.

Tests for radar response and container pickup decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    ContainerPickupDict,
    RadarResponseDict,
    TeleportLandedDict,
    decode_container_pickup,
    decode_radar_response,
    decode_teleport_landed,
)
from tests.container.test_data import (
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    RADAR_RESPONSE_1,
    RADAR_RESPONSE_2,
    RADAR_RESPONSE_5,
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

    def test_radar_response_dict_keys(self) -> None:
        """RadarResponseDict has expected keys."""
        result: RadarResponseDict = decode_radar_response(RADAR_RESPONSE_1)
        assert result["msg_type"] == "radar_response"
        assert result["container_count"] == 1
        assert len(result["containers"]) == 1
        assert len(result["mines"]) == 0
        c = result["containers"][0]
        assert c["x"] == 123
        assert c["y"] == 105
        assert c["volume"] == -1
