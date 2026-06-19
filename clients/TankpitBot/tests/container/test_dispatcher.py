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
    CHUNK_DATA_80,
    CHUNK_DATA_95,
    CHUNK_DATA_130,
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    DEACTIVATION_DEATH_7,
    MINE_DETONATION_3,
    MINE_DETONATION_15,
    MINE_PLACEMENT_15,
    PLAYER_LIST_EXTENDED_7,
    PLAYER_LIST_SHORT_4,
    POSITION_UPDATE_13,
    TANK_LEAVE_6,
    TANK_REGISTRY_16,
    TANK_REGISTRY_20,
    TANK_STATUS_SHORT_9,
    TANK_STATUS_SYNC_2,
    TANK_STATUS_SYNC_3,
    TANK_UPDATE_COMPACT_10,
    TANK_UPDATE_EXTENDED_14,
    TANK_UPDATE_FULL_15,
    TELEPORT_LANDED_1,
    TIP_NOTIFICATION_29,
    TIP_NOTIFICATION_55,
    TIP_NOTIFICATION_79,
    UNKNOWN_8_BYTES,
    UNKNOWN_12_BYTES,
    WORLD_STATE_500,
    WORLD_STATE_650,
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

    def test_identifies_tank_registry(self) -> None:
        """Correctly identifies tank registry structure."""
        assert identify_container_type(TANK_REGISTRY_16) == ContainerMessageType.TANK_REGISTRY
        assert identify_container_type(TANK_REGISTRY_20) == ContainerMessageType.TANK_REGISTRY

    # Container Movement identification was deleted 2026-06-19.
    # The unified `decode_0x2e_message` dispatches 0x47 Movement to the
    # protocol decoder via subtype-first matching.

    def test_identifies_position_update(self) -> None:
        """Correctly identifies position update structure."""
        assert identify_container_type(POSITION_UPDATE_13) == ContainerMessageType.POSITION_UPDATE

    def test_rejects_false_positive_unknown_packet(self) -> None:
        """Does not classify an unrelated packet as a known container subtype."""
        data = bytes.fromhex("46938292839182938191839181")
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN

    def test_identifies_tank_status_short(self) -> None:
        """Correctly identifies tank status short structure (9 bytes)."""
        result = identify_container_type(TANK_STATUS_SHORT_9)
        assert result == ContainerMessageType.TANK_STATUS_SHORT

    def test_short_bodies_are_unknown(self) -> None:
        """2-3 byte container bodies resolve to UNKNOWN_CONTAINER.

        The prior TankStatusSync identifier was a length-only catch-all
        misidentifying short bodies (0x4F/0x46/0x58/0x3F at len=2-3).
        Deleted 2026-06-19.
        """
        assert identify_container_type(TANK_STATUS_SYNC_2) == ContainerMessageType.UNKNOWN
        assert identify_container_type(TANK_STATUS_SYNC_3) == ContainerMessageType.UNKNOWN

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

    def test_identifies_deactivation_death(self) -> None:
        """Correctly identifies deactivation death structure (7 bytes with 0x43)."""
        result = identify_container_type(DEACTIVATION_DEATH_7)
        assert result == ContainerMessageType.DEACTIVATION_DEATH

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

    # 0x4F RadarResponse moved to protocol layer 2026-06-19; container
    # identification no longer returns RADAR_RESPONSE for these.

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

    def test_dispatches_tank_registry(self) -> None:
        """Dispatches to tank registry decoder."""
        result = decode_container_message(TANK_REGISTRY_16)
        assert result["msg_type"] == "tank_registry"

    # Container Movement dispatch deleted 2026-06-19.

    def test_dispatches_position_update(self) -> None:
        """Dispatches to position update decoder."""
        result = decode_container_message(POSITION_UPDATE_13)
        assert result["msg_type"] == "position_update"

    def test_position_update_rejects_wrong_subtype(self) -> None:
        """Position update decoder rejects 13-byte data with non-0x24 subtype."""
        import pytest

        from tankpit_bot.container import ContainerDecodeError, decode_position_update

        with pytest.raises(ContainerDecodeError, match="expected subtype 0x24"):
            decode_position_update(bytes([0x99] + [0x00] * 12))

    def test_dispatches_tank_status_short(self) -> None:
        """Dispatches to tank status short decoder (9 bytes)."""
        result = decode_container_message(TANK_STATUS_SHORT_9)
        assert result["msg_type"] == "tank_status_short"

    def test_dispatches_short_body_as_unknown(self) -> None:
        """Short bodies (2-3 bytes) dispatch as unknown_container."""
        result = decode_container_message(TANK_STATUS_SYNC_2)
        assert result["msg_type"] == "unknown_container"

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

    def test_dispatches_deactivation_death(self) -> None:
        """Dispatches to deactivation death decoder (7 bytes with 0x43)."""
        result = decode_container_message(DEACTIVATION_DEATH_7)
        assert result["msg_type"] == "deactivation_death"

    def test_dispatches_teleport_landed(self) -> None:
        """Dispatches to teleport landed decoder (1 byte)."""
        result = decode_container_message(TELEPORT_LANDED_1)
        assert result["msg_type"] == "teleport_landed"

    def test_dispatches_container_pickup(self) -> None:
        """Dispatches to container pickup decoder (5 bytes with 0x43)."""
        result = decode_container_message(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"

    # 0x4F RadarResponse dispatched via protocol layer 2026-06-19.

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
        """0x43 with length other than 5 or 7 falls through to length matching.

        The 8-byte body misses container_pickup (5) and deactivation_death
        (7); 8 isn't a known length match either, so unknown_container.
        """
        result = decode_container_message(bytes([0x43, 1, 2, 3, 4, 5, 6, 7]))
        assert result["msg_type"] == "unknown_container"

    def test_subtype_0x79_with_5_bytes_falls_through(self) -> None:
        """0x79 with length other than 4 or 7 falls through.

        5-byte 0x79 misses player_list_short (4) and player_list_extended
        (7); length doesn't match a known container fallback either.
        """
        result = decode_container_message(bytes([0x79, 1, 2, 3, 4]))
        assert result["msg_type"] == "unknown_container"
