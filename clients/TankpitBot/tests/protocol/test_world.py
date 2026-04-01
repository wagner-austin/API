"""Tests for world message decoders.

Tests for sync, container, terrain, viewport, chat, statistics, and active forces decoders.
"""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    DecodeError,
    ViewportEntityDict,
    decode_0x2e_message,
    decode_active_forces,
    decode_chat_message,
    decode_container,
    decode_statistics,
    decode_sync,
    decode_terrain_update,
    decode_viewport_update,
    viewport_entity_is_empty,
    viewport_entity_is_equipment,
    viewport_entity_is_fuel,
)


class TestDecodeSync:
    """Tests for decode_sync function."""

    def test_decodes_sync(self) -> None:
        """Decodes sync message (always succeeds)."""
        result = decode_sync(b"")
        assert result["msg_type"] == 0x3F

    def test_ignores_extra_data(self) -> None:
        """Ignores any extra data in sync message."""
        result = decode_sync(bytes([1, 2, 3, 4, 5]))
        assert result["msg_type"] == 0x3F


class TestDecodeContainer:
    """Tests for decode_container function."""

    def test_decodes_container(self) -> None:
        """Decodes container fuel message."""
        # container_id=0x0102, fuel=0x0304
        data = bytes([0x02, 0x01, 0x04, 0x03])
        result = decode_container(data)
        assert result["msg_type"] == 0x43
        assert result["container_id"] == 0x0102
        assert result["fuel"] == 0x0304

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_container(bytes([1, 2]))


class TestDecodeChatMessage:
    """Tests for decode_chat_message function."""

    def test_decodes_chat_with_coords(self) -> None:
        """Decodes chat message with coordinates."""
        # sender_id=0x0102, message_type=1, x=50, y=60
        data = bytes([0x02, 0x01, 1, 50, 60])
        result = decode_chat_message(data)
        assert result["msg_type"] == 0x4D
        assert result["sender_id"] == 0x0102
        assert result["message_type"] == 1
        assert result["x"] == 50
        assert result["y"] == 60

    def test_decodes_chat_without_coords(self) -> None:
        """Decodes chat message without coordinates."""
        data = bytes([0x02, 0x01, 1])
        result = decode_chat_message(data)
        assert result["x"] is None
        assert result["y"] is None

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_chat_message(bytes([1, 2]))


class TestDecodeStatistics:
    """Tests for decode_statistics function."""

    def test_decodes_statistics(self) -> None:
        """Decodes statistics message."""
        data = (
            bytes([0x10, 0x00, 30, 45])  # hours=16, mins=30, secs=45
            + (100).to_bytes(4, "little")  # destroyed
            + (50).to_bytes(4, "little")  # deactivated
            + (5000).to_bytes(4, "little")  # score
        )
        result = decode_statistics(data)
        assert result["msg_type"] == 0x56
        assert result["playtime_hours"] == 16
        assert result["playtime_minutes"] == 30
        assert result["playtime_seconds"] == 45
        assert result["destroyed"] == 100
        assert result["deactivated"] == 50
        assert result["score"] == 5000

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_statistics(bytes([1, 2, 3, 4]))


class TestDecodeActiveForces:
    """Tests for decode_active_forces function."""

    def test_decodes_active_forces(self) -> None:
        """Decodes active forces message."""
        data = bytes([10, 15, 8, 12])  # Team counts
        result = decode_active_forces(data)
        assert result["msg_type"] == 0x2A
        assert result["team_counts"] == [10, 15, 8, 12]

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_active_forces(bytes([1, 2]))


class TestDecodeTerrainUpdate:
    """Tests for decode_terrain_update function."""

    def test_decodes_terrain_updates(self) -> None:
        """Decodes terrain update triplets."""
        # Two updates: (10, 20, 5) and (30, 40, 0)
        data = bytes([10, 20, 5, 30, 40, 0])
        result = decode_terrain_update(data)
        assert result["msg_type"] == 0x4A
        assert result["updates"] == [(10, 20, 5), (30, 40, 0)]

    def test_handles_empty_data(self) -> None:
        """Handles empty terrain data."""
        result = decode_terrain_update(b"")
        assert result["updates"] == []


class TestDecodeViewportUpdate:
    """Tests for decode_viewport_update function."""

    def test_decodes_viewport_header(self) -> None:
        """Decodes viewport update header."""
        data = bytes([3, 0x0F])  # viewport_left=3, viewport_top=15
        result = decode_viewport_update(data)
        assert result["msg_type"] == 0x5A
        assert result["viewport_left"] == 3
        assert result["viewport_top"] == 0x0F
        assert result["entities"] == []

    def test_decodes_viewport_with_entities(self) -> None:
        """Decodes viewport with entity data."""
        # viewport_left=0, viewport_top=0, delta=1 (col=1, row=0), entity data (3 bytes)
        # z = (entity_id << 8) | (value << 4) | terrain_type
        # Let's encode: terrain=5, value=2, entity_id=100
        # z = (100 << 8) | (2 << 4) | 5 = 0x6425
        # Big endian 3 bytes: 0x00, 0x64, 0x25
        data = bytes([0, 0, 1, 0x00, 0x64, 0x25])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 1
        entity = result["entities"][0]
        assert entity["col"] == 1
        assert entity["row"] == 0
        assert entity["terrain_type"] == 5
        assert entity["entity_id"] == 100

    def test_handles_skip_marker(self) -> None:
        """Handles delta 255 as skip marker."""
        data = bytes([0, 0, 255])  # Skip marker, no entity data follows
        result = decode_viewport_update(data)
        assert result["entities"] == []

    def test_handles_column_wrap(self) -> None:
        """Handles column wraparound to next row."""
        # Delta of 20 should wrap: col += 20 -> col=20, then col>=18 so col-=18=2, row+=1
        data = bytes([0, 0, 20, 0x00, 0x00, 0x00])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["col"] == 2
        assert result["entities"][0]["row"] == 1

    def test_handles_column_wrap_multiple_entities(self) -> None:
        """Handles column accumulation across entities requiring normalization."""
        # First entity: delta=10 -> col=10, row=0
        # Second entity: delta=10 -> col=20, triggers while loop: col=2, row=1
        data = bytes([0, 0, 10, 0x00, 0x00, 0x00, 10, 0x00, 0x00, 0x00])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 2
        assert result["entities"][0]["col"] == 10
        assert result["entities"][0]["row"] == 0
        assert result["entities"][1]["col"] == 2
        assert result["entities"][1]["row"] == 1

    def test_handles_truncated_entity_data(self) -> None:
        """Handles truncated data gracefully by breaking early."""
        # Non-255 delta but only 1 byte of entity data (need 3)
        data = bytes([0, 0, 5, 0x00])
        result = decode_viewport_update(data)
        # Should break early, no entities parsed
        assert result["entities"] == []

    def test_handles_tank_entity_id(self) -> None:
        """Handles special tank entity ID (65535 -> -1)."""
        # entity_id=65535 (0xFFFF) means tank
        # z = (0xFFFF << 8) | 0 | 0 = 0xFFFF00
        data = bytes([0, 0, 1, 0xFF, 0xFF, 0x00])
        result = decode_viewport_update(data)
        assert result["entities"][0]["entity_id"] == -1

    def test_handles_high_value(self) -> None:
        """Handles value >= 8 becoming 255."""
        # value = 8 -> becomes 255
        # z = (0 << 8) | (8 << 4) | 0 = 0x80
        data = bytes([0, 0, 1, 0x00, 0x00, 0x80])
        result = decode_viewport_update(data)
        assert result["entities"][0]["value"] == 255

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_viewport_update(bytes([1]))


class TestViewportEntityHelpers:
    """Tests for viewport entity helper functions."""

    def test_viewport_entity_is_equipment(self) -> None:
        """Checks if row marks equipment."""
        equipment: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": -1,
            "value": 0,
            "terrain_type": 0,
        }
        not_equipment: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 100,
            "value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_is_equipment(equipment) is True
        assert viewport_entity_is_equipment(not_equipment) is False

    def test_viewport_entity_is_fuel(self) -> None:
        """Checks if row marks fuel."""
        fuel: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 100,
            "value": 0,
            "terrain_type": 0,
        }
        not_fuel: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 0,
            "value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_is_fuel(fuel) is True
        assert viewport_entity_is_fuel(not_fuel) is False

    def test_viewport_entity_is_empty(self) -> None:
        """Checks if tile is empty."""
        empty: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 0,
            "value": 0,
            "terrain_type": 0,
        }
        not_empty: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "entity_id": 100,
            "value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_is_empty(empty) is True
        assert viewport_entity_is_empty(not_empty) is False


class TestDecode0x2eMessage:
    """Tests for decode_0x2e_message function."""

    def test_dispatches_to_container_decoder(self) -> None:
        """Dispatches to container decoder module."""
        # 11 bytes = combat hit
        data = bytes([0x59, 0x09, 0xCD, 0x07, 0x99, 0x84, 0x93, 0xCE, 0x9C, 0x80, 0x51])
        result = decode_0x2e_message(data)
        assert result["msg_type"] == "combat_hit"
