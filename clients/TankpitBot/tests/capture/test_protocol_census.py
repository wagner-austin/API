"""Tests for protocol-level capture census."""

from __future__ import annotations

import base64
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture import (
    analyze_protocol_census,
    build_xor_table,
    decode_protocol_census,
    encode_protocol_census,
    format_protocol_census,
)
from tankpit_bot.capture import protocol_census as pc
from tankpit_bot.capture.protocol_census import ProtocolCensusDict
from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.types.message import CapturedMessage
from tankpit_bot.types.session import CaptureSession
from tests.conftest import FakeFileSystem


def _encode_received_frame(msg_type: int, decoded_data: bytes, xor_table: bytes) -> str:
    """Encode a single received frame.

    Args:
        msg_type: Protocol type byte.
        decoded_data: XOR-decoded body bytes without the type byte.
        xor_table: XOR table for the session.

    Returns:
        Base64-encoded frame payload.
    """
    encoded_body = bytes([msg_type]) + xor_decode_body(decoded_data, xor_table)
    frame = bytes([len(encoded_body) & 0xFF, len(encoded_body) >> 8]) + encoded_body
    return base64.b64encode(frame).decode("ascii")


def _make_session(messages: list[CapturedMessage], magic: str) -> CaptureSession:
    """Create a typed capture session for tests."""
    return CaptureSession(
        session_id="protocol-census-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )


@pytest.fixture()
def _fake_fs() -> Generator[FakeFileSystem, None, None]:
    """Patch core file hooks with a fake filesystem."""
    old_exists = core_hooks.path_exists
    old_read = core_hooks.read_text
    fs = FakeFileSystem()
    core_hooks.path_exists = fs.path_exists
    core_hooks.read_text = fs.read_text
    try:
        yield fs
    finally:
        core_hooks.path_exists = old_exists
        core_hooks.read_text = old_read


class TestAnalyzeProtocolCensus:
    """Tests for analyze_protocol_census."""

    def test_separates_decoded_short_and_unsupported(
        self,
        _fake_fs: FakeFileSystem,
    ) -> None:
        """Counts decoded, short, and unsupported frames separately."""
        magic = "protocol-magic"
        static_key = "K" * 64
        xor_table = build_xor_table(static_key, magic)
        static_key_path = Path(__file__).resolve().parents[2] / "xor_static_key.txt"
        _fake_fs._files[str(static_key_path)] = static_key

        messages = [
            CapturedMessage(
                timestamp_ms=1000,
                direction="received",
                payload=_encode_received_frame(
                    0x2E,
                    bytes([0x24, 0x02, 0x7D, 0x04, 140, 137, 8, 3, 0, 0, 0, 0, 0]),
                    xor_table,
                ),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1100,
                direction="received",
                payload=_encode_received_frame(0x21, bytes([0x01, 0x02]), xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1200,
                direction="received",
                payload=_encode_received_frame(0x7B, bytes([0x11, 0x22, 0x33]), xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1300,
                direction="received",
                payload=base64.b64encode(b"\x02\x00+=").decode("ascii"),
                ws_url="wss://test/ws",
            ),
        ]

        result = analyze_protocol_census(_make_session(messages, magic))

        assert result["received_message_count"] == 4
        assert result["received_frame_count"] == 4
        assert result["text_frame_count"] == 1
        assert result["decoded_binary_frame_count"] == 1
        assert result["short_or_invalid_frame_count"] == 1
        assert result["unsupported_frame_count"] == 1
        assert result["framing_error_count"] == 0
        assert result["decoded"] == [{"label": "position_update", "count": 1}]

    def test_preserves_short_sample_hex(self, _fake_fs: FakeFileSystem) -> None:
        """Stores raw and decoded hex for short packets."""
        magic = "sample-magic"
        static_key = "S" * 64
        xor_table = build_xor_table(static_key, magic)
        static_key_path = Path(__file__).resolve().parents[2] / "xor_static_key.txt"
        _fake_fs._files[str(static_key_path)] = static_key
        decoded_data = bytes([0x99, 0x88])
        payload = _encode_received_frame(0x21, decoded_data, xor_table)

        result = analyze_protocol_census(
            _make_session(
                [
                    CapturedMessage(
                        timestamp_ms=1000,
                        direction="received",
                        payload=payload,
                        ws_url="wss://test/ws",
                    )
                ],
                magic,
            )
        )

        assert result["short_or_invalid"][0]["label"] == "0x21 len=3"
        assert result["short_or_invalid"][0]["sample_body_hex"].startswith("21")
        assert result["short_or_invalid"][0]["sample_decoded_hex"] == decoded_data.hex()


class TestProtocolCensusEncoding:
    """Tests for protocol census encode/decode helpers."""

    def test_round_trip(self) -> None:
        """Round-trips protocol census through JSON helpers."""
        original = ProtocolCensusDict(
            received_message_count=4,
            received_frame_count=5,
            text_frame_count=1,
            decoded_binary_frame_count=2,
            short_or_invalid_frame_count=1,
            unsupported_frame_count=1,
            framing_error_count=0,
            decoded=[{"label": "0x3D", "count": 2}],
            short_or_invalid=[
                {
                    "label": "0x21 len=3",
                    "count": 1,
                    "sample_body_hex": "210102",
                    "sample_decoded_hex": "0102",
                }
            ],
            unsupported=[
                {
                    "label": "0x7B len=4",
                    "count": 1,
                    "sample_body_hex": "7b001122",
                    "sample_decoded_hex": "001122",
                }
            ],
        )

        encoded = encode_protocol_census(original)
        decoded = decode_protocol_census(encoded)
        assert decoded == original

    def test_decode_rejects_non_object_entry(self) -> None:
        """Rejects non-object nested entries."""
        data: JSONObject = {
            "received_message_count": 1,
            "received_frame_count": 1,
            "text_frame_count": 0,
            "decoded_binary_frame_count": 0,
            "short_or_invalid_frame_count": 0,
            "unsupported_frame_count": 0,
            "framing_error_count": 0,
            "decoded": ["bad"],
            "short_or_invalid": [],
            "unsupported": [],
        }

        with pytest.raises(JSONTypeError, match="decoded\\[0\\] must be an object"):
            decode_protocol_census(data)

    def test_decode_rejects_non_object_short_entry(self) -> None:
        """Rejects non-object short-or-invalid entries."""
        data: JSONObject = {
            "received_message_count": 1,
            "received_frame_count": 1,
            "text_frame_count": 0,
            "decoded_binary_frame_count": 0,
            "short_or_invalid_frame_count": 0,
            "unsupported_frame_count": 0,
            "framing_error_count": 0,
            "decoded": [],
            "short_or_invalid": ["bad"],
            "unsupported": [],
        }

        with pytest.raises(JSONTypeError, match="short_or_invalid\\[0\\] must be an object"):
            decode_protocol_census(data)

    def test_decode_rejects_non_object_unsupported_entry(self) -> None:
        """Rejects non-object unsupported entries."""
        data: JSONObject = {
            "received_message_count": 1,
            "received_frame_count": 1,
            "text_frame_count": 0,
            "decoded_binary_frame_count": 0,
            "short_or_invalid_frame_count": 0,
            "unsupported_frame_count": 0,
            "framing_error_count": 0,
            "decoded": [],
            "short_or_invalid": [],
            "unsupported": ["bad"],
        }

        with pytest.raises(JSONTypeError, match="unsupported\\[0\\] must be an object"):
            decode_protocol_census(data)


def test_format_protocol_census_includes_sections() -> None:
    """Formats counts and packet sections."""
    result = ProtocolCensusDict(
        received_message_count=4,
        received_frame_count=5,
        text_frame_count=1,
        decoded_binary_frame_count=2,
        short_or_invalid_frame_count=1,
        unsupported_frame_count=1,
        framing_error_count=0,
        decoded=[{"label": "0x3D", "count": 2}],
        short_or_invalid=[
            {
                "label": "0x21 len=3",
                "count": 1,
                "sample_body_hex": "210102",
                "sample_decoded_hex": "0102",
            }
        ],
        unsupported=[
            {
                "label": "0x7B len=4",
                "count": 1,
                "sample_body_hex": "7b001122",
                "sample_decoded_hex": "001122",
            }
        ],
    )

    text = format_protocol_census(result)
    assert "received_messages=4" in text
    assert "decoded_binary_frames=2" in text
    assert "short_or_invalid:" in text
    assert "unsupported:" in text


def test_format_protocol_census_without_optional_sections() -> None:
    """Formats counts without decoded or undecoded sections."""
    result = ProtocolCensusDict(
        received_message_count=1,
        received_frame_count=1,
        text_frame_count=1,
        decoded_binary_frame_count=0,
        short_or_invalid_frame_count=0,
        unsupported_frame_count=0,
        framing_error_count=0,
        decoded=[],
        short_or_invalid=[],
        unsupported=[],
    )

    text = format_protocol_census(result)
    assert "decoded:" not in text
    assert "short_or_invalid:" not in text
    assert "unsupported:" not in text


def test_analyze_protocol_census_counts_framing_errors(
    _fake_fs: FakeFileSystem,
) -> None:
    """Counts payloads with invalid frame headers separately."""
    magic = "framing-magic"
    static_key = "F" * 64
    static_key_path = Path(__file__).resolve().parents[2] / "xor_static_key.txt"
    _fake_fs._files[str(static_key_path)] = static_key

    bad_payload = base64.b64encode(b"\x05\x00\x2e\x01").decode("ascii")
    result = analyze_protocol_census(
        _make_session(
            [
                CapturedMessage(
                    timestamp_ms=1000,
                    direction="received",
                    payload=bad_payload,
                    ws_url="wss://test/ws",
                )
            ],
            magic,
        )
    )

    assert result["received_message_count"] == 1
    assert result["received_frame_count"] == 0
    assert result["framing_error_count"] == 1


def test_analyze_protocol_census_rejects_missing_magic() -> None:
    """Raises when the capture session has no magic key."""
    session = _make_session([], "magic")
    session["magic"] = None

    with pytest.raises(ValueError, match="Capture session has no magic key"):
        analyze_protocol_census(session)


def test_analyze_protocol_census_skips_sent_messages(
    _fake_fs: FakeFileSystem,
) -> None:
    """Ignores sent messages in the census."""
    static_key = "G" * 64
    static_key_path = Path(__file__).resolve().parents[2] / "xor_static_key.txt"
    _fake_fs._files[str(static_key_path)] = static_key
    session = _make_session(
        [
            CapturedMessage(
                timestamp_ms=1000,
                direction="sent",
                payload=base64.b64encode(b"\x02\x00+=").decode("ascii"),
                ws_url="wss://test/ws",
            )
        ],
        "sent-magic",
    )

    result = analyze_protocol_census(session)
    assert result["received_message_count"] == 0
    assert result["received_frame_count"] == 0


def test_classify_frame_ignores_empty_body() -> None:
    """Ignores empty logical frames."""
    acc = pc._build_census_accumulator()
    pc._classify_frame(b"", b"\x00", acc)
    assert acc["received_frame_count"] == 0


def test_accumulate_message_ignores_invalid_base64() -> None:
    """Ignores invalid base64 payloads."""
    acc = pc._build_census_accumulator()
    pc._accumulate_message("not-base64!", b"\x00", acc)
    assert acc["received_frame_count"] == 0


def test_record_protocol_sample_preserves_first_sample_only() -> None:
    """Keeps the first sample for a packet label."""
    samples: dict[str, pc.ProtocolSampleDict] = {}
    pc._record_protocol_sample(samples, "label", b"\x21\xaa", b"\x10")
    pc._record_protocol_sample(samples, "label", b"\x21\xbb", b"\x11")
    assert samples["label"]["sample_body_hex"] == "21aa"
    assert samples["label"]["sample_decoded_hex"] == "10"


def test_message_label_covers_string_int_and_unknown() -> None:
    """Formats message labels for string, int, and unknown values."""
    assert pc._message_label("position_update") == "position_update"
    assert pc._message_label(0x3D) == "0x3D"
    assert pc._message_label(None) == "unknown"


def test_xor_decode_frame_body_handles_short_frame() -> None:
    """Returns empty bytes when frame body has no payload after type byte."""
    assert pc._xor_decode_frame_body(b"\x2e", b"\x00") == b""
