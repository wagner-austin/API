"""Tests for raw viewport-entity extraction."""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture.viewport_entities import (
    ViewportEntityDumpDict,
    ViewportEntityRowDict,
    ViewportEntityUpdateDict,
    analyze_viewport_entities,
    decode_viewport_entity_dump,
    decode_viewport_entity_row,
    decode_viewport_entity_update,
    encode_viewport_entity_dump,
    encode_viewport_entity_row,
    encode_viewport_entity_update,
    format_viewport_entity_dump,
)
from tankpit_bot.capture.xor import (
    XorStaticKeyUnavailableError,
    reset_static_key_cache,
)
from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH, build_xor_table
from tankpit_bot.types import CapturedMessage, CaptureSession
from tests.conftest import FakeFileSystem
from tests.wire_builders import encode_wire_frame


def _encode_entity_data(entity_id: int, value: int, terrain_type: int) -> bytes:
    """Encode one 0x5A entity payload word.

    Args:
        entity_id: Raw entity id or ``-1`` sentinel.
        value: Decoded value field. ``255`` is stored using nibble ``8``.
        terrain_type: Terrain nibble.

    Returns:
        Three big-endian bytes for the packed entity word.
    """
    raw_id = 0xFFFF if entity_id == -1 else entity_id
    value_nibble = 8 if value == 255 else value
    z = (raw_id << 8) | (value_nibble << 4) | terrain_type
    return bytes([(z >> 16) & 0xFF, (z >> 8) & 0xFF, z & 0xFF])


def _make_viewport_payload(xor_table: bytes) -> str:
    """Create a viewport update with three raw entity classes.

    Args:
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    decoded_data = (
        bytes([10, 20, 1])
        + _encode_entity_data(-1, 255, 0)
        + bytes([2])
        + _encode_entity_data(777, 255, 0)
        + bytes([3])
        + _encode_entity_data(0, 2, 0)
    )
    return encode_wire_frame(0x5A, decoded_data, xor_table)


def _make_session(messages: list[CapturedMessage], magic: str | None) -> CaptureSession:
    """Create a capture session for viewport-entity tests.

    Args:
        messages: Capture messages for the session.
        magic: Session magic or ``None``.

    Returns:
        Capture session.
    """
    return CaptureSession(
        session_id="viewport-entities-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )


class TestViewportEntityEncoding:
    """Tests for viewport-entity encode/decode helpers."""

    def test_round_trips_row_update_and_dump(self) -> None:
        """Encodes and decodes nested viewport-entity structures."""
        row = ViewportEntityRowDict(
            abs_x=11,
            abs_y=20,
            col=1,
            row=0,
            cache_value=-1,
            overlay_value=255,
            terrain_type=0,
        )
        update = ViewportEntityUpdateDict(
            message_index=3,
            timestamp_ms=1100,
            viewport_left=10,
            viewport_top=20,
            entity_count=1,
            equipment_cache_count=1,
            positive_cache_count=0,
            zero_cache_count=0,
            entities=[row],
        )
        dump = ViewportEntityDumpDict(update_count=1, updates=[update])

        assert decode_viewport_entity_row(encode_viewport_entity_row(row)) == row
        assert decode_viewport_entity_update(encode_viewport_entity_update(update)) == update
        assert decode_viewport_entity_dump(encode_viewport_entity_dump(dump)) == dump

    def test_rejects_non_object_nested_entries(self) -> None:
        """Rejects invalid nested JSON entries precisely."""
        with pytest.raises(JSONTypeError, match="updates\\[0\\] must be an object"):
            decode_viewport_entity_dump({"update_count": 1, "updates": [1]})

        with pytest.raises(JSONTypeError, match="entities\\[0\\] must be an object"):
            decode_viewport_entity_update(
                {
                    "message_index": 0,
                    "timestamp_ms": 1000,
                    "viewport_left": 10,
                    "viewport_top": 20,
                    "entity_count": 1,
                    "equipment_cache_count": 0,
                    "positive_cache_count": 1,
                    "zero_cache_count": 0,
                    "entities": [1],
                }
            )


class TestAnalyzeViewportEntities:
    """Tests for viewport-entity extraction."""

    def test_skips_non_viewport_and_invalid_frames(self) -> None:
        """Skips sent, malformed, unsupported, and non-viewport frames cleanly."""
        magic = "viewport-entities-invalid"
        static_key = "N" * 64
        xor_table = build_xor_table(static_key, magic)

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        fake_fs._files[str(DEFAULT_STATIC_KEY_PATH)] = static_key
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            result = analyze_viewport_entities(
                _make_session(
                    [
                        CapturedMessage(
                            timestamp_ms=1000,
                            direction="sent",
                            payload=_make_viewport_payload(xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1001,
                            direction="received",
                            payload="not-base64!",
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1002,
                            direction="received",
                            payload=base64.b64encode(b"\x01").decode("ascii"),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1003,
                            direction="received",
                            payload=base64.b64encode(b"\x00\x00").decode("ascii"),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1004,
                            direction="received",
                            payload=encode_wire_frame(0x24, b"\x0e\x22\x12", xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1005,
                            direction="received",
                            payload=encode_wire_frame(
                                0x3D,
                                bytes([1, 1, 0, 5, 6, 0, 0, 1, 0, 0, 0, 0]),
                                xor_table,
                            ),
                            ws_url="wss://test/ws",
                        ),
                    ],
                    magic,
                )
            )
        finally:
            core_hooks.path_exists = old_exists
            core_hooks.read_text = old_read

        assert result == {"update_count": 0, "updates": []}

    def test_extracts_raw_viewport_entities(self) -> None:
        """Extracts raw rows and class counts from one viewport update."""
        magic = "viewport-entities-magic"
        static_key = "M" * 64
        xor_table = build_xor_table(static_key, magic)

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        fake_fs._files[str(DEFAULT_STATIC_KEY_PATH)] = static_key
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            result = analyze_viewport_entities(
                _make_session(
                    [
                        CapturedMessage(
                            timestamp_ms=1000,
                            direction="received",
                            payload=_make_viewport_payload(xor_table),
                            ws_url="wss://test/ws",
                        )
                    ],
                    magic,
                )
            )
        finally:
            core_hooks.path_exists = old_exists
            core_hooks.read_text = old_read

        assert result["update_count"] == 1
        update = result["updates"][0]
        assert update["viewport_left"] == 10
        assert update["viewport_top"] == 20
        assert update["entity_count"] == 3
        assert update["equipment_cache_count"] == 1
        assert update["positive_cache_count"] == 1
        assert update["zero_cache_count"] == 1
        assert update["entities"][0] == {
            "abs_x": 11,
            "abs_y": 20,
            "col": 1,
            "row": 0,
            "cache_value": -1,
            "overlay_value": 255,
            "terrain_type": 0,
        }
        assert update["entities"][1]["cache_value"] == 777
        assert update["entities"][2]["cache_value"] == 0
        assert update["entities"][2]["overlay_value"] == 2

    def test_formats_entity_dump(self) -> None:
        """Formats the raw dump for terminal inspection."""
        dump = ViewportEntityDumpDict(
            update_count=1,
            updates=[
                ViewportEntityUpdateDict(
                    message_index=3,
                    timestamp_ms=1100,
                    viewport_left=10,
                    viewport_top=20,
                    entity_count=1,
                    equipment_cache_count=1,
                    positive_cache_count=0,
                    zero_cache_count=0,
                    entities=[
                        ViewportEntityRowDict(
                            abs_x=11,
                            abs_y=20,
                            col=1,
                            row=0,
                            cache_value=-1,
                            overlay_value=255,
                            terrain_type=0,
                        )
                    ],
                )
            ],
        )

        formatted = format_viewport_entity_dump(dump)

        assert "viewport_updates=1" in formatted
        assert "viewport=(10,20)" in formatted
        assert "equipment_cache=1" in formatted
        assert "abs=(11,20)" in formatted
        assert "cache_value=-1" in formatted

    def test_raises_for_missing_magic_or_static_key(self) -> None:
        """Raises explicit errors for missing prerequisites."""
        with pytest.raises(ValueError, match="Capture session has no magic key"):
            analyze_viewport_entities(_make_session([], None))

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            reset_static_key_cache()
            with pytest.raises(XorStaticKeyUnavailableError, match="static XOR key unavailable"):
                analyze_viewport_entities(_make_session([], "magic"))
        finally:
            core_hooks.path_exists = old_exists
            core_hooks.read_text = old_read
