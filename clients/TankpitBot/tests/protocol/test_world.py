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
    decode_cache_update,
    decode_chat_message,
    decode_combined_tile_update,
    decode_overlay_update,
    decode_promotion,
    decode_statistics,
    decode_sync,
    decode_terrain_update,
    decode_viewport_update,
    viewport_entity_has_equipment_cache,
    viewport_entity_has_fuel_cache,
    viewport_entity_has_no_cache,
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


class TestDecodeCacheUpdate:
    """Tests for decode_cache_update function."""

    def test_decodes_cache_updates(self) -> None:
        """Decodes cache update entries."""
        data = bytes([10, 20, 0x04, 0x03, 30, 40, 0xFF, 0xFF])
        result = decode_cache_update(data)
        assert result["msg_type"] == 0x43
        assert result["updates"] == [(10, 20, 0x0304), (30, 40, -1)]

    def test_raises_on_invalid_length(self) -> None:
        """Raises DecodeError on invalid entry length."""
        with pytest.raises(DecodeError):
            decode_cache_update(bytes([1, 2, 3]))


class TestDecodeOverlayUpdate:
    """Tests for decode_overlay_update function."""

    def test_decodes_overlay_updates(self) -> None:
        """Decodes overlay update entries."""
        data = bytes([10, 20, 7, 30, 40, 255])
        result = decode_overlay_update(data)
        assert result["msg_type"] == 0x40
        assert result["updates"] == [(10, 20, 7), (30, 40, 255)]

    def test_raises_on_invalid_length(self) -> None:
        """Raises DecodeError on invalid entry length."""
        with pytest.raises(DecodeError):
            decode_overlay_update(bytes([1, 2]))


class TestDecodeCombinedTileUpdate:
    """Tests for decode_combined_tile_update function."""

    def test_decodes_combined_tile_updates(self) -> None:
        """Decodes cache and overlay sections."""
        data = bytes([1, 0, 10, 20, 0x04, 0x03, 30, 40, 7])
        result = decode_combined_tile_update(data)
        assert result["msg_type"] == 0x4F
        assert result["cache_updates"] == [(10, 20, 0x0304)]
        assert result["overlay_updates"] == [(30, 40, 7)]

    def test_raises_on_short_header(self) -> None:
        """Raises DecodeError on missing cache count header."""
        with pytest.raises(DecodeError):
            decode_combined_tile_update(bytes([1]))

    def test_raises_on_truncated_cache_section(self) -> None:
        """Raises DecodeError when cache section exceeds payload length."""
        with pytest.raises(DecodeError):
            decode_combined_tile_update(bytes([1, 0, 10, 20]))

    def test_raises_on_invalid_overlay_length(self) -> None:
        """Raises DecodeError when overlay section is not triplet-aligned."""
        with pytest.raises(DecodeError):
            decode_combined_tile_update(bytes([0, 0, 10]))


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

    def test_decodes_statistics_long_format(self) -> None:
        """Decodes long-format statistics (>12 bytes)."""
        data = (
            bytes([0x10, 0x00, 30, 45])  # hours=16, mins=30, secs=45
            + (100).to_bytes(4, "big")  # destroyed (32-bit BE)
            + bytes([50, 0x00])  # deactivated (LE u16 = 50)
            + (5000).to_bytes(4, "big")  # promo_points (32-bit BE)
        )
        result = decode_statistics(data)
        assert result["msg_type"] == 0x56
        assert result["playtime_hours"] == 16
        assert result["playtime_minutes"] == 30
        assert result["playtime_seconds"] == 45
        assert result["destroyed"] == 100
        assert result["deactivated"] == 50
        assert result["score"] == 5000

    def test_decodes_statistics_short_format(self) -> None:
        """Decodes short-format statistics (<=12 bytes)."""
        data = (
            bytes([0x10, 0x00, 30, 45])  # hours=16, mins=30, secs=45
            + bytes([100, 0x00])  # destroyed (LE u16 = 100)
            + bytes([50, 0x00])  # deactivated (LE u16 = 50)
            + (5000).to_bytes(4, "big")  # promo_points (32-bit BE)
        )
        result = decode_statistics(data)
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


class TestDecodePromotion:
    """Tests for decode_promotion function.

    0x2B '+' Rf binary form: ``Rf(a[0], 1===a[1])``. a[0] is the new
    rank index; a[1]==1 sets the "You have been promoted!" banner.
    """

    def test_decodes_promotion_with_banner(self) -> None:
        """Banner-on promotion: rank=4, was_promoted=True."""
        data = bytes([4, 1])
        result = decode_promotion(data)
        assert result["msg_type"] == 0x2B
        assert result["new_rank"] == 4
        assert result["was_promoted"] is True

    def test_decodes_silent_rank_set(self) -> None:
        """Silent rank set (e.g. join-time): a[1]=0 -> was_promoted=False."""
        data = bytes([2, 0])
        result = decode_promotion(data)
        assert result["new_rank"] == 2
        assert result["was_promoted"] is False

    def test_banner_flag_only_true_on_exact_one(self) -> None:
        """JS uses ``1===a[1]``: byte 2 is False (not truthy)."""
        data = bytes([1, 2])
        result = decode_promotion(data)
        assert result["was_promoted"] is False

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data (require 2 bytes)."""
        with pytest.raises(DecodeError):
            decode_promotion(bytes([1]))


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
        # z = (cache_value << 8) | (overlay_value << 4) | terrain_type
        # Let's encode: terrain=5, overlay=2, cache=100
        # z = (100 << 8) | (2 << 4) | 5 = 0x6425
        # Big endian 3 bytes: 0x00, 0x64, 0x25
        data = bytes([0, 0, 1, 0x00, 0x64, 0x25])
        result = decode_viewport_update(data)
        assert len(result["entities"]) == 1
        entity = result["entities"][0]
        assert entity["col"] == 1
        assert entity["row"] == 0
        assert entity["terrain_type"] == 5
        assert entity["cache_value"] == 100
        assert entity["overlay_value"] == 2

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

    def test_handles_equipment_cache_sentinel(self) -> None:
        """Handles special cache sentinel (65535 -> -1)."""
        # cache_value=65535 (0xFFFF) means equipment cache
        # z = (0xFFFF << 8) | 0 | 0 = 0xFFFF00
        data = bytes([0, 0, 1, 0xFF, 0xFF, 0x00])
        result = decode_viewport_update(data)
        assert result["entities"][0]["cache_value"] == -1

    def test_handles_high_overlay_value(self) -> None:
        """Handles overlay nibble >= 8 becoming 255."""
        # overlay_value = 8 -> becomes 255
        # z = (0 << 8) | (8 << 4) | 0 = 0x80
        data = bytes([0, 0, 1, 0x00, 0x00, 0x80])
        result = decode_viewport_update(data)
        assert result["entities"][0]["overlay_value"] == 255

    def test_raises_on_short_data(self) -> None:
        """Raises DecodeError on insufficient data."""
        with pytest.raises(DecodeError):
            decode_viewport_update(bytes([1]))


class TestViewportEntityHelpers:
    """Tests for viewport entity helper functions."""

    def test_viewport_entity_has_equipment_cache(self) -> None:
        """Checks if row marks equipment."""
        equipment: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "cache_value": -1,
            "overlay_value": 0,
            "terrain_type": 0,
        }
        not_equipment: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "cache_value": 100,
            "overlay_value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_has_equipment_cache(equipment) is True
        assert viewport_entity_has_equipment_cache(not_equipment) is False

    def test_viewport_entity_has_fuel_cache(self) -> None:
        """Checks if row marks fuel."""
        fuel: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "cache_value": 100,
            "overlay_value": 0,
            "terrain_type": 0,
        }
        not_fuel: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "cache_value": 0,
            "overlay_value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_has_fuel_cache(fuel) is True
        assert viewport_entity_has_fuel_cache(not_fuel) is False

    def test_viewport_entity_has_no_cache(self) -> None:
        """Checks if tile has no cache update."""
        empty: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "cache_value": 0,
            "overlay_value": 0,
            "terrain_type": 0,
        }
        not_empty: ViewportEntityDict = {
            "col": 0,
            "row": 0,
            "cache_value": 100,
            "overlay_value": 0,
            "terrain_type": 0,
        }
        assert viewport_entity_has_no_cache(empty) is True
        assert viewport_entity_has_no_cache(not_empty) is False


class TestDecode0x2eMessage:
    """Tests for decode_0x2e_message function."""

    def test_dispatches_to_container_decoder(self) -> None:
        """Dispatches to container length-fallback for 1-byte 0x54.

        After unification (2026-06-19), `decode_0x2e_message` is the
        single subtype-first + length-fallback entrypoint. Subtype 0x54
        with empty inner falls through to length-based teleport_landed
        (production captures confirm bare 0x54 bodies are teleport
        confirmations, not ActionDone payloads).
        """
        data = bytes([0x54])
        result = decode_0x2e_message(data)
        assert result["msg_type"] == "teleport_landed"

    def test_empty_data_raises(self) -> None:
        """Empty body propagates ContainerDecodeError from the container path."""
        from tankpit_bot.container.helpers import ContainerDecodeError

        with pytest.raises(ContainerDecodeError):
            decode_0x2e_message(b"")
