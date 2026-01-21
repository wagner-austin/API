"""Tests for world-related container decoders.

Tests for world state, chunk data, tip notification, and player list decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.container import (
    ChunkDataDict,
    ContainerDecodeError,
    PlayerListExtendedDict,
    PlayerListShortDict,
    TipNotificationDict,
    WorldStateDict,
    decode_chunk_data,
    decode_player_list_extended,
    decode_player_list_short,
    decode_tip_notification,
    decode_world_state,
)
from tests.container.test_data import (
    CHUNK_DATA_80,
    CHUNK_DATA_130,
    PLAYER_LIST_EXTENDED_7,
    PLAYER_LIST_SHORT_4,
    TIP_NOTIFICATION_29,
    TIP_NOTIFICATION_79,
    WORLD_STATE_500,
    WORLD_STATE_650,
)


class TestDecodeTipNotification:
    """Tests for tip notification decoding."""

    def test_decodes_29_byte_message(self) -> None:
        """Decodes 29-byte tip notification message correctly."""
        result = decode_tip_notification(TIP_NOTIFICATION_29)
        assert result["msg_type"] == "tip_notification"
        assert result["subtype"] == 0x68
        assert result["length"] == 29
        assert len(result["notification_data"]) == 28

    def test_decodes_79_byte_message(self) -> None:
        """Decodes 79-byte tip notification message correctly."""
        result = decode_tip_notification(TIP_NOTIFICATION_79)
        assert result["msg_type"] == "tip_notification"
        assert result["subtype"] == 0x68
        assert result["length"] == 79
        assert len(result["notification_data"]) == 78

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_tip_notification(bytes([0x01] * 28))
        with pytest.raises(ContainerDecodeError):
            decode_tip_notification(bytes([0x01] * 80))

    def test_tip_notification_dict_keys(self) -> None:
        """TipNotificationDict has expected keys."""
        result: TipNotificationDict = decode_tip_notification(TIP_NOTIFICATION_29)
        assert result["msg_type"] == "tip_notification"
        assert result["subtype"] == 0x68
        assert result["length"] == 29
        assert len(result["notification_data"]) == 28


class TestDecodeChunkData:
    """Tests for chunk data decoding."""

    def test_decodes_80_byte_message(self) -> None:
        """Decodes 80-byte chunk data message correctly."""
        result = decode_chunk_data(CHUNK_DATA_80)
        assert result["msg_type"] == "chunk_data"
        assert result["subtype"] == 0x14
        assert result["length"] == 80
        assert len(result["chunk_data"]) == 79

    def test_decodes_130_byte_message(self) -> None:
        """Decodes 130-byte chunk data message correctly."""
        result = decode_chunk_data(CHUNK_DATA_130)
        assert result["msg_type"] == "chunk_data"
        assert result["subtype"] == 0x14
        assert result["length"] == 130
        assert len(result["chunk_data"]) == 129

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_chunk_data(bytes([0x01] * 79))
        with pytest.raises(ContainerDecodeError):
            decode_chunk_data(bytes([0x01] * 131))

    def test_chunk_data_dict_keys(self) -> None:
        """ChunkDataDict has expected keys."""
        result: ChunkDataDict = decode_chunk_data(CHUNK_DATA_80)
        assert result["msg_type"] == "chunk_data"
        assert result["subtype"] == 0x14
        assert result["length"] == 80
        assert len(result["chunk_data"]) == 79


class TestDecodeWorldState:
    """Tests for world state decoding."""

    def test_decodes_500_byte_message(self) -> None:
        """Decodes 500-byte world state message correctly."""
        result = decode_world_state(WORLD_STATE_500)
        assert result["msg_type"] == "world_state"
        assert result["subtype"] == 0x14
        assert result["length"] == 500
        assert len(result["world_data"]) == 499

    def test_decodes_650_byte_message(self) -> None:
        """Decodes 650-byte world state message correctly."""
        result = decode_world_state(WORLD_STATE_650)
        assert result["msg_type"] == "world_state"
        assert result["subtype"] == 0x14
        assert result["length"] == 650
        assert len(result["world_data"]) == 649

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_world_state(bytes([0x01] * 499))

    def test_world_state_dict_keys(self) -> None:
        """WorldStateDict has expected keys."""
        result: WorldStateDict = decode_world_state(WORLD_STATE_500)
        assert result["msg_type"] == "world_state"
        assert result["subtype"] == 0x14
        assert result["length"] == 500
        assert len(result["world_data"]) == 499


class TestDecodePlayerListShort:
    """Tests for player list short response decoding."""

    def test_decodes_player_list_short(self) -> None:
        """Correctly decodes short player list response."""
        result = decode_player_list_short(PLAYER_LIST_SHORT_4)
        assert result["msg_type"] == "player_list_short"
        assert result["response_data"] == bytes.fromhex("990507")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_player_list_short(bytes([0x01] * 3))
        with pytest.raises(ContainerDecodeError):
            decode_player_list_short(bytes([0x01] * 5))

    def test_player_list_short_dict_keys(self) -> None:
        """PlayerListShortDict has expected keys."""
        result: PlayerListShortDict = decode_player_list_short(PLAYER_LIST_SHORT_4)
        assert result["msg_type"] == "player_list_short"
        assert result["response_data"] == bytes.fromhex("990507")


class TestDecodePlayerListExtended:
    """Tests for player list extended response decoding."""

    def test_decodes_player_list_extended(self) -> None:
        """Correctly decodes extended player list response."""
        result = decode_player_list_extended(PLAYER_LIST_EXTENDED_7)
        assert result["msg_type"] == "player_list_extended"
        assert result["response_data"] == bytes.fromhex("990507")
        assert result["extended_data"] == bytes.fromhex("ce1144")

    def test_raises_on_wrong_length(self) -> None:
        """Raises on invalid length."""
        with pytest.raises(ContainerDecodeError):
            decode_player_list_extended(bytes([0x01] * 6))
        with pytest.raises(ContainerDecodeError):
            decode_player_list_extended(bytes([0x01] * 8))

    def test_player_list_extended_dict_keys(self) -> None:
        """PlayerListExtendedDict has expected keys."""
        result: PlayerListExtendedDict = decode_player_list_extended(PLAYER_LIST_EXTENDED_7)
        assert result["msg_type"] == "player_list_extended"
        assert result["response_data"] == bytes.fromhex("990507")
        assert result["extended_data"] == bytes.fromhex("ce1144")
