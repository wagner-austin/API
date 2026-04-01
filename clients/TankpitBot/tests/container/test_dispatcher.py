"""Tests for container message type identification and dispatch.

Tests for identify_container_type and decode_container_message dispatcher.
"""

from __future__ import annotations

from tankpit_bot.container import (
    ContainerMessageType,
    decode_container_message,
    identify_container_type,
)
from tests.container.test_data import (
    CHUNK_DATA_80,
    CHUNK_DATA_95,
    CHUNK_DATA_130,
    COMBAT_HIT_11_INCOMING,
    COMBAT_HIT_11_OUTGOING,
    CONTAINER_PICKUP_EQUIPMENT,
    CONTAINER_PICKUP_FUEL,
    DEACTIVATION_DEATH_7,
    DEACTIVATION_KILL_5,
    MOVEMENT_16_SSSS,
    MOVEMENT_18_ENNNW,
    MOVEMENT_19_WWWWWW,
    MOVEMENT_20_NESW,
    PLAYER_LIST_EXTENDED_7,
    PLAYER_LIST_SHORT_4,
    POSITION_UPDATE_13,
    RADAR_RESPONSE_1,
    RADAR_RESPONSE_2,
    RADAR_RESPONSE_5,
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

    def test_identifies_combat_hit(self) -> None:
        """Correctly identifies combat hit structure."""
        assert identify_container_type(COMBAT_HIT_11_OUTGOING) == ContainerMessageType.COMBAT_HIT
        assert identify_container_type(COMBAT_HIT_11_INCOMING) == ContainerMessageType.COMBAT_HIT

    def test_identifies_tank_registry(self) -> None:
        """Correctly identifies tank registry structure."""
        assert identify_container_type(TANK_REGISTRY_16) == ContainerMessageType.TANK_REGISTRY
        assert identify_container_type(TANK_REGISTRY_20) == ContainerMessageType.TANK_REGISTRY

    def test_identifies_movement(self) -> None:
        """Correctly identifies movement structure."""
        assert identify_container_type(MOVEMENT_18_ENNNW) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_19_WWWWWW) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_16_SSSS) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_20_NESW) == ContainerMessageType.MOVEMENT

    def test_identifies_position_update(self) -> None:
        """Correctly identifies position update structure."""
        assert identify_container_type(POSITION_UPDATE_13) == ContainerMessageType.POSITION_UPDATE

    def test_rejects_false_positive_13_byte_packet(self) -> None:
        """Does not classify an unrelated captured 13-byte packet as position."""
        data = bytes.fromhex("45938292839182938191839181")
        assert identify_container_type(data) == ContainerMessageType.UNKNOWN

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

    def test_identifies_radar_response(self) -> None:
        """Correctly identifies radar response structure (7-43 bytes with 0x4F)."""
        assert identify_container_type(RADAR_RESPONSE_1) == ContainerMessageType.RADAR_RESPONSE
        assert identify_container_type(RADAR_RESPONSE_2) == ContainerMessageType.RADAR_RESPONSE
        assert identify_container_type(RADAR_RESPONSE_5) == ContainerMessageType.RADAR_RESPONSE

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

    def test_dispatches_combat_hit(self) -> None:
        """Dispatches to combat hit decoder."""
        result = decode_container_message(COMBAT_HIT_11_OUTGOING)
        assert result["msg_type"] == "combat_hit"

    def test_dispatches_tank_registry(self) -> None:
        """Dispatches to tank registry decoder."""
        result = decode_container_message(TANK_REGISTRY_16)
        assert result["msg_type"] == "tank_registry"

    def test_dispatches_movement(self) -> None:
        """Dispatches to movement decoder."""
        result = decode_container_message(MOVEMENT_18_ENNNW)
        assert result["msg_type"] == "movement"

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

    def test_dispatches_teleport_landed(self) -> None:
        """Dispatches to teleport landed decoder (1 byte)."""
        result = decode_container_message(TELEPORT_LANDED_1)
        assert result["msg_type"] == "teleport_landed"

    def test_dispatches_container_pickup(self) -> None:
        """Dispatches to container pickup decoder (5 bytes with 0x43)."""
        result = decode_container_message(CONTAINER_PICKUP_EQUIPMENT)
        assert result["msg_type"] == "container_pickup"

    def test_dispatches_radar_response(self) -> None:
        """Dispatches to radar response decoder (7-43 bytes with 0x4F)."""
        result = decode_container_message(RADAR_RESPONSE_1)
        assert result["msg_type"] == "radar_response"

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
