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

    def test_decodes_emptied_container(self) -> None:
        """Decodes a pickup whose container is now empty (remaining_volume=0).

        Covers both equipment containers (no fuel attribute, always 0)
        and fuel containers whose entire stored volume was transferred
        to the picker (the empirically common case when the picker's
        tank had room for all of it).
        """
        result = decode_container_pickup(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"
        assert len(result["pickups"]) == 1
        assert result["pickups"][0]["x"] == 0x88  # 136
        assert result["pickups"][0]["y"] == 0x5E  # 94
        assert result["pickups"][0]["remaining_volume"] == 0

    def test_decodes_partial_fuel_pickup(self) -> None:
        """Decodes a pickup that left fuel behind in the container.

        Per the user-annotated 2026-06-20 multi-pickup capture
        (runs/sniff/sniff-20260620-155103), the wire ``remaining_volume``
        is the fuel that REMAINS in the container after this pickup --
        not the fuel transferred. Picker's tank was near the 1100 cap
        and only had room for a partial transfer.
        """
        result = decode_container_pickup(CONTAINER_PICKUP_FUEL)
        assert result["msg_type"] == "container_pickup"
        assert len(result["pickups"]) == 1
        assert result["pickups"][0]["x"] == 0x89  # 137
        assert result["pickups"][0]["y"] == 0x5F  # 95
        assert result["pickups"][0]["remaining_volume"] == 618

    def test_decodes_two_record_body(self) -> None:
        """Two-record bodies fire when a tank touches two containers in one tick.

        Real corpus sample (bot-20260407-000551 t+134.25s):
        body bytes ``43 c6 af 00 00 c7 b0 25 00`` -> two pickups:
        (198, 175, 0) and (199, 176, 37). 80 such samples seen across
        the 2026-06-20 corpus sweep.
        """
        body = bytes([0x43, 0xC6, 0xAF, 0x00, 0x00, 0xC7, 0xB0, 0x25, 0x00])
        result = decode_container_pickup(body)
        assert result["msg_type"] == "container_pickup"
        assert len(result["pickups"]) == 2
        assert result["pickups"][0] == {"x": 0xC6, "y": 0xAF, "remaining_volume": 0}
        assert result["pickups"][1] == {"x": 0xC7, "y": 0xB0, "remaining_volume": 0x25}

    def test_decodes_three_record_body(self) -> None:
        """Three-record bodies fire on triple-pickup ticks.

        Real corpus sample (bot-20260611-155750 t+247.29s):
        body bytes ``43 f0 96 00 00 ef 95 00 00 f0 95 4e 03`` -> three
        pickups including one with remaining_volume=846. 2 such samples
        in the 2026-06-20 corpus sweep.
        """
        body = bytes(
            [
                0x43,
                0xF0,
                0x96,
                0x00,
                0x00,
                0xEF,
                0x95,
                0x00,
                0x00,
                0xF0,
                0x95,
                0x4E,
                0x03,
            ]
        )
        result = decode_container_pickup(body)
        assert result["msg_type"] == "container_pickup"
        assert len(result["pickups"]) == 3
        assert result["pickups"][2]["remaining_volume"] == 0x034E  # 846

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_container_pickup(bytes([0x01] * 4))
        with pytest.raises(ContainerDecodeError):
            decode_container_pickup(bytes([0x43, 0x01]))

    def test_raises_on_partial_record(self) -> None:
        """Raises when the body has a trailing partial 4-byte record."""
        with pytest.raises(ContainerDecodeError):
            decode_container_pickup(bytes([0x43] + [0x01] * 6))

    def test_raises_on_wrong_subtype(self) -> None:
        """Raises when the subtype byte is not 0x43."""
        with pytest.raises(ContainerDecodeError):
            decode_container_pickup(bytes([0x44, 0x01, 0x02, 0x03, 0x04]))

    def test_container_pickup_dict_keys(self) -> None:
        """ContainerPickupDict has expected keys."""
        result: ContainerPickupDict = decode_container_pickup(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"
        assert "pickups" in result
        assert result["pickups"][0]["x"] == 0x88
        assert result["pickups"][0]["y"] == 0x5E
        assert result["pickups"][0]["remaining_volume"] == 0


# RadarResponse decoder tests moved to tests/protocol/test_radar.py
# (decode_radar_scan_result). Container path deleted 2026-06-19.
