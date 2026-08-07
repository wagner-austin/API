"""Tests for shoot-to-viewport correlation."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture.shot_viewport_correlation import (
    ShotViewportCorrelationDict,
    ShotViewportCorrelationDumpDict,
    analyze_shot_viewport_correlation,
    decode_shot_viewport_correlation,
    decode_shot_viewport_correlation_dump,
    encode_shot_viewport_correlation,
    encode_shot_viewport_correlation_dump,
    format_shot_viewport_correlation,
)
from tankpit_bot.capture.viewport_entities import ViewportEntityRowDict
from tankpit_bot.capture.xor import (
    XorStaticKeyUnavailableError,
    reset_static_key_cache,
    xor_decode_body,
)
from tankpit_bot.protocol.codec import build_xor_table
from tankpit_bot.protocol.commands import CMD_SHOOT, TYPE_COMBAT
from tankpit_bot.types import CapturedMessage, CaptureSession
from tests.conftest import FakeFileSystem


def _encode_received_frame(msg_type: int, decoded_data: bytes, xor_table: bytes) -> str:
    """Encode one received frame for a capture session."""
    encoded_body = bytes([msg_type]) + xor_decode_body(decoded_data, xor_table)
    frame = bytes([len(encoded_body) & 0xFF, len(encoded_body) >> 8]) + encoded_body
    return base64.b64encode(frame).decode("ascii")


def _encode_sent_frame(decoded_body: bytes, xor_table: bytes) -> str:
    """Encode one XOR-protected sent frame for a capture session."""
    encoded_body = decoded_body[:1] + xor_decode_body(decoded_body[1:], xor_table)
    frame = bytes([len(encoded_body) & 0xFF, len(encoded_body) >> 8]) + encoded_body
    return base64.b64encode(frame).decode("ascii")


def _encode_entity_data(entity_id: int, value: int, terrain_type: int) -> bytes:
    """Encode one 0x5A entity payload word."""
    raw_id = 0xFFFF if entity_id == -1 else entity_id
    value_nibble = 8 if value == 255 else value
    z = (raw_id << 8) | (value_nibble << 4) | terrain_type
    return bytes([(z >> 16) & 0xFF, (z >> 8) & 0xFF, z & 0xFF])


def _make_viewport_payload(xor_table: bytes) -> str:
    """Create one viewport update with positive and anonymous rows."""
    decoded_data = (
        bytes([50, 60, 0])
        + _encode_entity_data(514, 255, 0)
        + bytes([20])
        + _encode_entity_data(-1, 255, 0)
    )
    return _encode_received_frame(0x5A, decoded_data, xor_table)


def _make_shoot_payload(target_x: int, target_y: int, target_id: int, xor_table: bytes) -> str:
    """Create one sent shoot command frame."""
    decoded_body = bytes(
        [
            ord("!"),
            TYPE_COMBAT,
            CMD_SHOOT,
            target_x & 0xFF,
            target_y & 0xFF,
            target_id & 0xFF,
            (target_id >> 8) & 0xFF,
        ]
    )
    return _encode_sent_frame(decoded_body, xor_table)


def _make_session(messages: list[CapturedMessage], magic: str | None) -> CaptureSession:
    """Create a capture session for shot correlation tests."""
    return CaptureSession(
        session_id="shot-viewport-correlation-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )


class TestShotViewportEncoding:
    """Tests for shot correlation encode/decode helpers."""

    def test_round_trips_row_and_dump(self) -> None:
        """Encodes and decodes nested shot correlation structures."""
        row = ViewportEntityRowDict(
            abs_x=50,
            abs_y=60,
            col=0,
            row=0,
            cache_value=514,
            overlay_value=255,
            terrain_type=0,
        )
        shot = ShotViewportCorrelationDict(
            shot_index=3,
            shot_timestamp_ms=1100,
            target_x=50,
            target_y=60,
            target_id=514,
            viewport_found=True,
            viewport_index=2,
            viewport_timestamp_ms=1050,
            viewport_left=50,
            viewport_top=60,
            positive_row_count=1,
            equipment_row_count=0,
            id_match_count=1,
            coord_match_count=1,
            positive_rows=[row],
            equipment_rows=[],
            id_matches=[row],
            coord_matches=[row],
        )
        dump = ShotViewportCorrelationDumpDict(shot_count=1, shots=[shot])

        assert decode_shot_viewport_correlation(encode_shot_viewport_correlation(shot)) == shot
        assert (
            decode_shot_viewport_correlation_dump(encode_shot_viewport_correlation_dump(dump))
            == dump
        )

    def test_rejects_non_object_nested_entries(self) -> None:
        """Rejects invalid nested JSON entries precisely."""
        with pytest.raises(JSONTypeError, match="shots\\[0\\] must be an object"):
            decode_shot_viewport_correlation_dump({"shot_count": 1, "shots": [1]})

        with pytest.raises(JSONTypeError, match="positive_rows\\[0\\] must be an object"):
            decode_shot_viewport_correlation(
                {
                    "shot_index": 0,
                    "shot_timestamp_ms": 1000,
                    "target_x": 1,
                    "target_y": 2,
                    "target_id": 3,
                    "viewport_found": False,
                    "viewport_index": -1,
                    "viewport_timestamp_ms": -1,
                    "viewport_left": -1,
                    "viewport_top": -1,
                    "positive_row_count": 0,
                    "equipment_row_count": 0,
                    "id_match_count": 0,
                    "coord_match_count": 0,
                    "positive_rows": [1],
                    "equipment_rows": [],
                    "id_matches": [],
                    "coord_matches": [],
                }
            )


class TestAnalyzeShotViewportCorrelation:
    """Tests for shoot-to-viewport correlation."""

    def test_skips_invalid_sent_payloads_and_commands(self) -> None:
        """Skips malformed sent payloads and non-shoot commands cleanly."""
        magic = "shot-viewport-invalid"
        static_key = "Q" * 64
        xor_table = build_xor_table(static_key, magic)

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        fake_fs._files[str(Path(__file__).resolve().parents[2] / "xor_static_key.txt")] = static_key
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            result = analyze_shot_viewport_correlation(
                _make_session(
                    [
                        CapturedMessage(
                            timestamp_ms=1000,
                            direction="sent",
                            payload="not-base64!",
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1001,
                            direction="sent",
                            payload=base64.b64encode(b"\x01").decode("ascii"),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1002,
                            direction="sent",
                            payload=_encode_sent_frame(bytes([ord("!"), 2]), xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1003,
                            direction="sent",
                            payload=_encode_sent_frame(bytes([ord("?"), 2, 102]), xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1004,
                            direction="sent",
                            payload=_encode_sent_frame(bytes([ord("!"), 2, 102]), xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1005,
                            direction="sent",
                            payload=_encode_sent_frame(
                                bytes([ord("!"), 4, 112, 10, 20]),
                                xor_table,
                            ),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1006,
                            direction="sent",
                            payload=_encode_sent_frame(
                                bytes([ord("!"), TYPE_COMBAT, CMD_SHOOT, 50, 60, 0]),
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

        assert result == {"shot_count": 0, "shots": []}

    def test_correlates_shot_with_latest_viewport_update(self) -> None:
        """Matches a sent shot against the latest raw 0x5A rows."""
        magic = "shot-viewport-correlation"
        static_key = "T" * 64
        xor_table = build_xor_table(static_key, magic)

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        fake_fs._files[str(Path(__file__).resolve().parents[2] / "xor_static_key.txt")] = static_key
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            result = analyze_shot_viewport_correlation(
                _make_session(
                    [
                        CapturedMessage(
                            timestamp_ms=1000,
                            direction="received",
                            payload=_make_viewport_payload(xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1100,
                            direction="sent",
                            payload=_make_shoot_payload(50, 60, 514, xor_table),
                            ws_url="wss://test/ws",
                        ),
                    ],
                    magic,
                )
            )
        finally:
            core_hooks.path_exists = old_exists
            core_hooks.read_text = old_read

        assert result["shot_count"] == 1
        shot = result["shots"][0]
        assert shot["viewport_found"] is True
        assert shot["viewport_index"] == 0
        assert shot["target_id"] == 514
        assert shot["positive_row_count"] == 1
        assert shot["equipment_row_count"] == 1
        assert shot["id_match_count"] == 1
        assert shot["coord_match_count"] == 1
        assert shot["id_matches"][0]["cache_value"] == 514
        assert shot["coord_matches"][0]["abs_x"] == 50
        assert shot["coord_matches"][0]["abs_y"] == 60

    def test_handles_shot_without_viewport_and_non_shoot_frames(self) -> None:
        """Handles missing prior viewport updates and ignores non-shoot sent frames."""
        magic = "shot-viewport-no-viewport"
        static_key = "R" * 64
        xor_table = build_xor_table(static_key, magic)

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        fake_fs._files[str(Path(__file__).resolve().parents[2] / "xor_static_key.txt")] = static_key
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            result = analyze_shot_viewport_correlation(
                _make_session(
                    [
                        CapturedMessage(
                            timestamp_ms=1000,
                            direction="sent",
                            payload=_encode_sent_frame(bytes([ord("!"), 2, 102]), xor_table),
                            ws_url="wss://test/ws",
                        ),
                        CapturedMessage(
                            timestamp_ms=1100,
                            direction="sent",
                            payload=_make_shoot_payload(12, 34, 777, xor_table),
                            ws_url="wss://test/ws",
                        ),
                    ],
                    magic,
                )
            )
        finally:
            core_hooks.path_exists = old_exists
            core_hooks.read_text = old_read

        assert result == {
            "shot_count": 1,
            "shots": [
                {
                    "shot_index": 1,
                    "shot_timestamp_ms": 1100,
                    "target_x": 12,
                    "target_y": 34,
                    "target_id": 777,
                    "viewport_found": False,
                    "viewport_index": -1,
                    "viewport_timestamp_ms": -1,
                    "viewport_left": -1,
                    "viewport_top": -1,
                    "positive_row_count": 0,
                    "equipment_row_count": 0,
                    "id_match_count": 0,
                    "coord_match_count": 0,
                    "positive_rows": [],
                    "equipment_rows": [],
                    "id_matches": [],
                    "coord_matches": [],
                }
            ],
        }

    def test_formats_correlation_dump(self) -> None:
        """Formats the correlation dump for terminal inspection."""
        row = ViewportEntityRowDict(
            abs_x=50,
            abs_y=60,
            col=0,
            row=0,
            cache_value=514,
            overlay_value=255,
            terrain_type=0,
        )
        dump = ShotViewportCorrelationDumpDict(
            shot_count=1,
            shots=[
                ShotViewportCorrelationDict(
                    shot_index=3,
                    shot_timestamp_ms=1100,
                    target_x=50,
                    target_y=60,
                    target_id=514,
                    viewport_found=True,
                    viewport_index=2,
                    viewport_timestamp_ms=1050,
                    viewport_left=50,
                    viewport_top=60,
                    positive_row_count=1,
                    equipment_row_count=0,
                    id_match_count=1,
                    coord_match_count=1,
                    positive_rows=[row],
                    equipment_rows=[],
                    id_matches=[row],
                    coord_matches=[row],
                )
            ],
        )

        formatted = format_shot_viewport_correlation(dump)

        assert "shot_count=1" in formatted
        assert "shot=(50,60)" in formatted
        assert "target_id=514" in formatted
        assert "id_matches=1" in formatted
        assert "coord_match abs=(50,60)" in formatted

    def test_raises_for_missing_magic_or_static_key(self) -> None:
        """Raises explicit errors for missing capture prerequisites."""
        with pytest.raises(ValueError, match="Capture session has no magic key"):
            analyze_shot_viewport_correlation(_make_session([], None))

        old_exists = core_hooks.path_exists
        old_read = core_hooks.read_text
        fake_fs = FakeFileSystem()
        core_hooks.path_exists = fake_fs.path_exists
        core_hooks.read_text = fake_fs.read_text
        try:
            reset_static_key_cache()
            with pytest.raises(XorStaticKeyUnavailableError, match="static XOR key unavailable"):
                analyze_shot_viewport_correlation(_make_session([], "magic"))
        finally:
            core_hooks.path_exists = old_exists
            core_hooks.read_text = old_read
