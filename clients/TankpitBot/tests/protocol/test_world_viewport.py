"""Tests for viewport and terrain message decoders."""

from __future__ import annotations

import pytest

from tankpit_bot.protocol import (
    ViewportEntityDict,
    decode_0x2e_message,
    decode_cache_update,
    decode_terrain_update,
    decode_viewport_update,
    viewport_entity_has_equipment_cache,
    viewport_entity_has_fuel_cache,
    viewport_entity_has_no_cache,
)
from tankpit_bot.wire.helpers import DecodeError


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

    def test_tunneled_statistics_routes_through_misc(self) -> None:
        """Tunneled 0x56 inside 0x2E decodes as a Statistics body.

        Verified against 239 production samples
        (analysis_scripts/crack_tank_update.py): every 15-byte 0x2E
        body with subtype 0x56 in 150 capture sessions decodes to a
        sane Statistics record (minutes/seconds within bounds,
        monotonic playtime/destroyed/score).
        """
        # First production sample, exactly as captured:
        # 56 28 00 12 1f 00 00 00 1e 00 00 00 00 da 7b
        # -> hours=40 minutes=18 seconds=31 destroyed=30
        #    deactivated=0 score=55931
        data = bytes.fromhex("562800121f0000001e00000000da7b")
        result = decode_0x2e_message(data)
        assert result["msg_type"] == 0x56
        assert result["playtime_hours"] == 40
        assert result["playtime_minutes"] == 18
        assert result["playtime_seconds"] == 31
        assert result["destroyed"] == 30
        assert result["deactivated"] == 0
        assert result["score"] == 55931

    def test_9byte_0x43_body_routes_to_multi_record_container_pickup(self) -> None:
        """9-byte 0x43 0x2E bodies are 2-record ContainerPickup, not Og.h.

        Corpus correction 2026-06-20: the prior version of this test
        claimed 9-byte bodies routed to TankStatusSync short form,
        based on a corpus sweep that read each body as Og.h and noted
        the resulting (damage, rank, lb_score, promo) fell inside sane
        bounds. The interpretation was wrong: every one of those 80
        samples has byte 0 = 0x43, and the JS Mf dispatcher
        (tpclient.pretty.js:4837) re-dispatches on byte 0 -- so a body
        starting with 0x43 goes to V.C = $g (CacheUpdate), reading
        repeating 4-byte ``[x, y, cache_lo, cache_hi]`` records. Each
        record is a container pickup notification. The Og.h reading
        produced in-range numbers by coincidence; team=67 (from
        byte 0 = 0x43) was the giveaway that something was off.

        Fixture below is the first production sample, byte-for-byte:
        ``43 c6 af 00 00 c7 b0 25 00``
          pickup 1: x=198, y=175, remaining=0     (container emptied)
          pickup 2: x=199, y=176, remaining=37    (partial)
        """
        body = bytes.fromhex("43c6af0000c7b02500")
        result = decode_0x2e_message(body)
        assert result["msg_type"] == "container_pickup"
        assert len(result["pickups"]) == 2
        assert result["pickups"][0] == {"x": 198, "y": 175, "remaining_volume": 0}
        assert result["pickups"][1] == {"x": 199, "y": 176, "remaining_volume": 37}

    def test_tunneled_chat_routes_through_misc(self) -> None:
        """Tunneled 0x4D inside 0x2E decodes as a ChatMessage body.

        Corpus sweep 2026-07-29 (320 sessions): chat arrives ONLY
        0x2E-tunneled, always exactly 5 inner bytes, never top-level.
        Fixture is the first production echo of
        sniff-20260729-214411, byte-for-byte: ``4d 15 05 0c 61 d4``
        — sender 1301 (Artax) saying 12 "Base is here" at (97, 212).
        """
        data = bytes.fromhex("4d15050c61d4")
        result = decode_0x2e_message(data)
        assert result["msg_type"] == 0x4D
        assert result["sender_id"] == 1301
        assert result["message_type"] == 12
        assert result["x"] == 97
        assert result["y"] == 212

    def test_short_0x4d_body_falls_through_to_container(self) -> None:
        """A sub-5-byte 0x4D body is NOT claimed by the chat route.

        The 5-byte gate mirrors the corpus (every tunneled chat body
        is exactly 5 inner bytes); anything shorter falls through to
        length-based container identification.
        """
        result = decode_0x2e_message(bytes.fromhex("4d150501"))
        assert result["msg_type"] != 0x4D

    def test_tunneled_map_data_routes_through_world(self) -> None:
        """Tunneled 0x4C inside 0x2E decodes as MapData.

        Verified against 2941 production samples in the 150-session
        corpus -- previously misidentified as length-based
        ``WorldState`` blobs because no tunneled 0x4C dispatch
        existed. The bot's world_state was being broadcast at us with
        full tank-position info on every map open and we were
        ignoring it.
        """
        # Body: subtype 0x4C + LE u16 RLE count (=0, no dots) + one
        # 5-byte tank entry at x=10,y=20,tid=0x0102,packed=0x12.
        data = bytes.fromhex("4c0000" + "0a14" + "0201" + "12")
        result = decode_0x2e_message(data)
        assert result["msg_type"] == 0x4C
        assert len(result["tanks"]) == 1
        assert result["tanks"][0]["tank_id"] == 0x0102

    def test_tunneled_build_pickup_routes_through_misc(self) -> None:
        """Tunneled 0x42 inside 0x2E decodes as a BuildPickup body.

        Verified against 2 production samples (own-tank obstacle drop
        events). The actor is tank 1301 dropping an obstacle on the
        adjacent tile west of the source.
        """
        # Production sample with subtype 0x42 prepended:
        # 42 15 05 c7 76 c6 76 77 02 00
        # -> tid=1301 src=(199,118) drop=(198,118) obstacle_type=2
        data = bytes.fromhex("421505c776c676770200")
        result = decode_0x2e_message(data)
        assert result["msg_type"] == 0x42
        assert result["tank_id"] == 1301
        assert result["source_x"] == 199
        assert result["source_y"] == 118
        assert result["drop_x"] == 198
        assert result["drop_y"] == 118
        assert result["obstacle_type"] == 2


def test_cache_update_rejects_partial_entries() -> None:
    """0x43 tile patches are whole 4-byte entries; a lone byte raises.

    The overloaded plaintext chat ack never reaches this decoder — it
    travels un-XORed and is discriminated at the framing layer by
    ``try_decode_plaintext_ack``.
    """
    with pytest.raises(DecodeError):
        decode_cache_update(bytes([1]))
