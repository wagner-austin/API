"""Tests for the wire decoder: framing and message dispatch.

``test_decoder.py`` was 755 lines; the per-message decoders are now a
sibling.
"""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
)

from tankpit_bot.decoder import (
    DecodedCommand,
    DecodedLobbyMessage,
    SessionDecoder,
    decode_decoded_command,
    decode_decoded_lobby_message,
    encode_decoded_command,
    encode_decoded_lobby_message,
)
from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.types import (
    CapturedMessage,
    CaptureSession,
)
from tests.wire_builders import frame_payload


def test_encode_decoded_command() -> None:
    """Test encoding DecodedCommand to JSON."""
    cmd = DecodedCommand(
        timestamp_ms=1000,
        direction="sent",
        raw_hex="21303132",
        decoded_hex="21414243",
        type_byte=0x41,
        cmd_byte=0x42,
        data_hex="43",
    )

    result = encode_decoded_command(cmd)

    assert result["timestamp_ms"] == 1000
    assert result["direction"] == "sent"
    assert result["raw_hex"] == "21303132"
    assert result["decoded_hex"] == "21414243"
    assert result["type_byte"] == 0x41
    assert result["cmd_byte"] == 0x42
    assert result["data_hex"] == "43"


def test_decode_decoded_command() -> None:
    """Test decoding DecodedCommand from JSON."""
    data: JSONObject = {
        "timestamp_ms": 2000,
        "direction": "received",
        "raw_hex": "aabbcc",
        "decoded_hex": "112233",
        "type_byte": 0x11,
        "cmd_byte": 0x22,
        "data_hex": "33",
    }

    result = decode_decoded_command(data)

    assert result["timestamp_ms"] == 2000
    assert result["direction"] == "received"
    assert result["type_byte"] == 0x11


def test_decode_decoded_command_invalid_direction_raises() -> None:
    """Test decode_decoded_command raises for invalid direction."""
    data: JSONObject = {
        "timestamp_ms": 1000,
        "direction": "invalid",
        "raw_hex": "00",
        "decoded_hex": "00",
        "type_byte": 0,
        "cmd_byte": 0,
        "data_hex": "",
    }

    with pytest.raises(JSONTypeError, match="Invalid direction"):
        decode_decoded_command(data)


def test_decode_decoded_command_missing_field_raises() -> None:
    """Test decode_decoded_command raises for missing field."""
    data: JSONObject = {"timestamp_ms": 1000}

    with pytest.raises(JSONTypeError):
        decode_decoded_command(data)


def test_encode_decoded_lobby_message() -> None:
    """Test encoding DecodedLobbyMessage to JSON."""
    msg = DecodedLobbyMessage(
        timestamp_ms=3000,
        direction="received",
        prefix="+",
        text="4|Room Name|5",
    )

    result = encode_decoded_lobby_message(msg)

    assert result["timestamp_ms"] == 3000
    assert result["direction"] == "received"
    assert result["prefix"] == "+"
    assert result["text"] == "4|Room Name|5"


def test_decode_decoded_lobby_message() -> None:
    """Test decoding DecodedLobbyMessage from JSON."""
    data: JSONObject = {
        "timestamp_ms": 4000,
        "direction": "sent",
        "prefix": "*",
        "text": "4",
    }

    result = decode_decoded_lobby_message(data)

    assert result["timestamp_ms"] == 4000
    assert result["direction"] == "sent"
    assert result["prefix"] == "*"
    assert result["text"] == "4"


def test_decode_decoded_lobby_message_invalid_direction_raises() -> None:
    """Test decode_decoded_lobby_message raises for invalid direction."""
    data: JSONObject = {
        "timestamp_ms": 1000,
        "direction": "bad",
        "prefix": "+",
        "text": "test",
    }

    with pytest.raises(JSONTypeError, match="Invalid direction"):
        decode_decoded_lobby_message(data)


def test_session_decoder_decodes_command() -> None:
    """Test SessionDecoder decodes a game command."""
    # Create a simple XOR table where XOR is identity (all zeros)
    codec = ProtocolCodec("AAA", "AAA")  # XOR with same = all zeros

    # Command body: ! + type + cmd
    body = bytes([ord("!"), 0x30, 0x42])
    payload = frame_payload(body)

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="sent",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="AAA",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 1
    assert decoder.commands[0]["type_byte"] == 0x30
    assert decoder.commands[0]["cmd_byte"] == 0x42
    assert decoder.commands[0]["direction"] == "sent"


def test_session_decoder_drops_a_frame_truncated_mid_body() -> None:
    """A body shorter than its header declares is not decoded.

    A capture can end mid-frame, and a websocket payload can arrive
    split. Slicing past the end does not raise in Python -- it silently
    yields the short bytes -- so without the length check the decoder
    reads a partial body as a whole message and records a lobby line
    the sender never completed. The header here declares ten bytes
    against seven present.
    """
    codec = ProtocolCodec("A", "A")

    body = b"+4|Room"
    raw = bytes([10, 0]) + body  # little-endian length 10, only 7 bytes follow
    payload = base64.b64encode(raw).decode("ascii")

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="received",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert decoder.lobby_messages == []


def test_session_decoder_decodes_lobby_message() -> None:
    """Test SessionDecoder decodes a lobby message."""
    codec = ProtocolCodec("A", "A")

    # Lobby message: +4|Room
    body = b"+4|Room"
    payload = frame_payload(body)

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="received",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.lobby_messages) == 1
    assert decoder.lobby_messages[0]["prefix"] == "+"
    assert decoder.lobby_messages[0]["text"] == "4|Room"


def test_session_decoder_skips_state_messages() -> None:
    """Test SessionDecoder skips binary state messages."""
    codec = ProtocolCodec("A", "A")

    # State message starts with '.'
    body = b"." + bytes([0x01, 0x02, 0x03])
    payload = frame_payload(body)

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="received",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 0
    assert len(decoder.lobby_messages) == 0


def test_session_decoder_skips_short_messages() -> None:
    """Test SessionDecoder skips messages too short for header."""
    codec = ProtocolCodec("A", "A")

    # Only 1 byte - too short for frame header
    payload = base64.b64encode(bytes([0x01])).decode("ascii")

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="sent",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 0


def test_session_decoder_skips_incomplete_frame() -> None:
    """Test SessionDecoder skips frames with incomplete body."""
    codec = ProtocolCodec("A", "A")

    # Header says 10 bytes but only 2 provided
    payload = base64.b64encode(bytes([0x0A, 0x00, 0x41, 0x42])).decode("ascii")

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="sent",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 0


def test_session_decoder_skips_empty_body() -> None:
    """Test SessionDecoder skips messages with empty body."""
    codec = ProtocolCodec("A", "A")

    # Header with zero length
    payload = base64.b64encode(bytes([0x00, 0x00])).decode("ascii")

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="sent",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 0


def test_session_decoder_skips_short_command() -> None:
    """Test SessionDecoder skips commands shorter than 3 bytes."""
    codec = ProtocolCodec("AA", "AA")

    # Command with only 2 bytes (need at least 3: ! + type + cmd)
    body = bytes([ord("!"), 0x30])
    payload = frame_payload(body)

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="sent",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="AA",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 0


def test_session_decoder_decodes_command_with_data() -> None:
    """Test SessionDecoder decodes command with data payload."""
    codec = ProtocolCodec("AAAAA", "AAAAA")

    # Command with data: ! + type + cmd + data
    body = bytes([ord("!"), 0x30, 0x42, 0xAB, 0xCD])
    payload = frame_payload(body)

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=[
            CapturedMessage(
                timestamp_ms=500,
                direction="sent",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        ],
        magic="AAAAA",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.commands) == 1
    assert decoder.commands[0]["data_hex"] == "abcd"
