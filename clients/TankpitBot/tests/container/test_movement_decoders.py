"""Tests for movement-related container decoders.

Tests for movement decoding and PlayerIdMapper correlation.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ContainerDecodeError,
    ContainerMessageType,
    MovementDict,
    PlayerIdMapper,
    PositionUpdateDict,
    decode_container_message,
    decode_movement,
    decode_position_update,
    identify_container_type,
)
from tests.container.test_data import (
    MOVEMENT_16_SSSS,
    MOVEMENT_18_ENNNW,
    MOVEMENT_19_WWWWWW,
    MOVEMENT_20_NESW,
    POSITION_UPDATE_13,
    TANK_REGISTRY_16,
    TANK_REGISTRY_20,
)


class TestDecodeMovement:
    """Tests for movement message decoding."""

    def test_decodes_18_byte_movement(self) -> None:
        """Decodes 18-byte movement message correctly.

        MOVEMENT_18_ENNNW: 477e026e5c0c03002e87030000656e6e6e77
        - [0] 0x47 = subtype 'G'
        - [1] 0x7E = flags (self)
        - [2-3] 0x026E (LE) = packed 0x6E02 -> start_x = 0x6E = 110
        - [4] 0x5C = start_y = 92
        - [5-7] unknown
        - [8-11] 0x2E870300 (LE) = player_id = 231214
        - [12+] waypoints = "ennnw"
        """
        result = decode_movement(MOVEMENT_18_ENNNW)
        assert result["msg_type"] == "movement"
        assert result["flags"] == 0x7E
        assert result["start_x"] == 0x6E  # 110 - high byte of packed 0x6E02
        assert result["start_y"] == 0x5C  # 92
        assert result["player_id"] == 231214  # 0x0003872E
        assert result["tank_id"] is None  # Not resolved yet
        assert result["waypoints"] == "ennnw"
        assert result["is_self"] is True  # flags 0x7E has bits 5-6 set

    def test_decodes_19_byte_movement(self) -> None:
        """Decodes 19-byte movement message correctly."""
        result = decode_movement(MOVEMENT_19_WWWWWW)
        assert result["msg_type"] == "movement"
        assert result["waypoints"] == "wwwwww"

    def test_decodes_boundary_lengths(self) -> None:
        """Decodes boundary length messages."""
        result16 = decode_movement(MOVEMENT_16_SSSS)
        assert result16["msg_type"] == "movement"
        assert result16["waypoints"] == "ssss"

        result20 = decode_movement(MOVEMENT_20_NESW)
        assert result20["msg_type"] == "movement"
        assert result20["waypoints"] == "nesw"

    def test_raises_on_short_data(self) -> None:
        """Raises ContainerDecodeError for too-short data."""
        short_data = bytes([0x47] + [0x00] * 12)  # 13 bytes
        with pytest.raises(ContainerDecodeError, match="Movement"):
            decode_movement(short_data)


class TestIdentifyContainerTypeMovement:
    """Tests for movement identification in dispatcher."""

    def test_identifies_movement_before_tank_registry(self) -> None:
        """Identifies Movement messages correctly (not misclassified as TankRegistry)."""
        assert identify_container_type(MOVEMENT_18_ENNNW) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_19_WWWWWW) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_16_SSSS) == ContainerMessageType.MOVEMENT
        assert identify_container_type(MOVEMENT_20_NESW) == ContainerMessageType.MOVEMENT

    def test_tank_registry_not_confused_with_movement(self) -> None:
        """TankRegistry messages with non-direction tails are correctly identified."""
        assert identify_container_type(TANK_REGISTRY_16) == ContainerMessageType.TANK_REGISTRY
        assert identify_container_type(TANK_REGISTRY_20) == ContainerMessageType.TANK_REGISTRY


class TestDecodeContainerMessageMovement:
    """Tests for Movement via decode_container_message dispatcher."""

    def test_dispatches_to_movement_decoder(self) -> None:
        """decode_container_message returns MovementDict for movement messages."""
        result = decode_container_message(MOVEMENT_18_ENNNW)
        assert result["msg_type"] == "movement"
        # Narrow type and verify waypoints
        if result["msg_type"] == "movement":
            assert result["waypoints"] == "ennnw"


class TestPlayerIdMapper:
    """Tests for PlayerIdMapper correlating player_id to tank_id."""

    def test_resolve_movement_from_position_correlation(self) -> None:
        """Resolves tank_id by matching Movement start position to MovementResponse."""
        mapper = PlayerIdMapper()

        # Record MovementResponse at position (36, 92) for tank 638
        mapper.record_movement_response(tank_id=638, x=36, y=92)

        # Create Movement starting at same position with player_id 231214
        movement = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="ennnw",
            is_self=True,
        )

        # Resolve should correlate position and learn mapping
        mapper.resolve_movement(movement)
        assert movement["tank_id"] == 638

        # Mapping should be cached
        assert mapper.get_tank_id(231214) == 638

    def test_resolve_movement_from_cached_mapping(self) -> None:
        """Uses cached player_id -> tank_id mapping for subsequent movements."""
        mapper = PlayerIdMapper()

        # Record initial correlation
        mapper.record_movement_response(tank_id=638, x=36, y=92)
        movement1 = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="ennnw",
            is_self=True,
        )
        mapper.resolve_movement(movement1)

        # Second movement at different position should still resolve
        movement2 = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=50,
            start_y=100,
            player_id=231214,  # Same player_id
            tank_id=None,
            waypoints="wwww",
            is_self=True,
        )
        mapper.resolve_movement(movement2)
        assert movement2["tank_id"] == 638  # Resolved from cached mapping

    def test_resolve_movement_no_match(self) -> None:
        """Movement without matching position leaves tank_id as None."""
        mapper = PlayerIdMapper()

        movement = MovementDict(
            msg_type="movement",
            flags=0x1E,
            start_x=99,
            start_y=99,
            player_id=12345,
            tank_id=None,
            waypoints="eeee",
            is_self=False,
        )
        mapper.resolve_movement(movement)
        assert movement["tank_id"] is None

    def test_get_player_id_reverse_lookup(self) -> None:
        """Reverse lookup from tank_id to player_id."""
        mapper = PlayerIdMapper()
        mapper.record_movement_response(tank_id=638, x=36, y=92)

        movement = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="e",
            is_self=True,
        )
        mapper.resolve_movement(movement)

        assert mapper.get_player_id(638) == 231214
        assert mapper.get_player_id(999) is None  # Unknown tank

    def test_clear_resets_all_mappings(self) -> None:
        """Clear removes all cached mappings."""
        mapper = PlayerIdMapper()
        mapper.record_movement_response(tank_id=638, x=36, y=92)

        movement = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=36,
            start_y=92,
            player_id=231214,
            tank_id=None,
            waypoints="e",
            is_self=True,
        )
        mapper.resolve_movement(movement)
        assert mapper.get_tank_id(231214) == 638

        mapper.clear()
        assert mapper.get_tank_id(231214) is None


class TestDecodePositionUpdate:
    """Tests for position update decoding."""

    def test_decodes_13_byte_update(self) -> None:
        """Decodes 13-byte position update correctly."""
        result = decode_position_update(POSITION_UPDATE_13)
        assert result["msg_type"] == "position_update"
        assert result["flags"] == 0x53
        assert result["tank_id"] == 0x07CD  # cd 07 little-endian
        assert result["x"] == 0x15  # 21
        assert result["y"] == 0x12  # 18
        assert len(result["extra_data"]) == 7  # 13 - 4 header - 2 coords

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_position_update(bytes([0x01] * 12))
        with pytest.raises(ContainerDecodeError):
            decode_position_update(bytes([0x01] * 14))

    def test_raises_on_wrong_subtype(self) -> None:
        """Raises when a 13-byte packet lacks the position subtype."""
        with pytest.raises(ContainerDecodeError, match="expected subtype 0x24"):
            decode_position_update(bytes([0x45] + [0x00] * 12))

    def test_position_update_dict_keys(self) -> None:
        """PositionUpdateDict has expected keys."""
        result: PositionUpdateDict = decode_position_update(POSITION_UPDATE_13)
        assert result["msg_type"] == "position_update"
        assert result["flags"] == 0x53
        assert result["tank_id"] == 0x07CD
        assert result["x"] == 0x15
        assert result["y"] == 0x12
        assert len(result["extra_data"]) == 7
