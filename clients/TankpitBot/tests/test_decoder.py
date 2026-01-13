"""Tests for tankpit_bot.decoder module."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.decoder import (
    DecodedCommand,
    DecodedLobbyMessage,
    DecoderError,
    MissingMagicError,
    SessionDecoder,
    decode_decoded_command,
    decode_decoded_lobby_message,
    encode_decoded_command,
    encode_decoded_lobby_message,
    load_and_decode_session,
)
from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.protocol.framing import encode_frame
from tankpit_bot.types import CapturedMessage, CaptureSession
from tests.conftest import FakeEnv, FakeFileSystem

# =============================================================================
# Test Helpers
# =============================================================================


def make_payload(body: bytes) -> str:
    """Create base64 payload with frame header."""
    framed = encode_frame(body)
    return base64.b64encode(framed).decode("ascii")


# =============================================================================
# DecodedCommand Encode/Decode Tests
# =============================================================================


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


# =============================================================================
# DecodedLobbyMessage Encode/Decode Tests
# =============================================================================


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


# =============================================================================
# SessionDecoder Tests
# =============================================================================


def test_session_decoder_decodes_command() -> None:
    """Test SessionDecoder decodes a game command."""
    # Create a simple XOR table where XOR is identity (all zeros)
    codec = ProtocolCodec("AAA", "AAA")  # XOR with same = all zeros

    # Command body: ! + type + cmd
    body = bytes([ord("!"), 0x30, 0x42])
    payload = make_payload(body)

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


def test_session_decoder_decodes_lobby_message() -> None:
    """Test SessionDecoder decodes a lobby message."""
    codec = ProtocolCodec("A", "A")

    # Lobby message: +4|Room
    body = b"+4|Room"
    payload = make_payload(body)

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
    payload = make_payload(body)

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
    payload = make_payload(body)

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
    payload = make_payload(body)

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


def test_session_decoder_handles_all_lobby_prefixes() -> None:
    """Test SessionDecoder handles various lobby message prefixes."""
    codec = ProtocolCodec("A", "A")

    prefixes = ["%", "+", "*", "=", "$", "-"]
    messages = []

    for i, prefix in enumerate(prefixes):
        body = (prefix + "test").encode("utf-8")
        payload = make_payload(body)
        messages.append(
            CapturedMessage(
                timestamp_ms=i * 100,
                direction="received",
                payload=payload,
                ws_url="wss://test.com/ws",
            )
        )

    session = CaptureSession(
        session_id="test",
        start_timestamp_ms=0,
        end_timestamp_ms=1000,
        base_url="https://test.com",
        messages=messages,
        magic="A",
        game_log=[],
        tank_names={},
    )

    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    assert len(decoder.lobby_messages) == 6
    decoded_prefixes = [m["prefix"] for m in decoder.lobby_messages]
    assert decoded_prefixes == prefixes


# =============================================================================
# load_and_decode_session Tests
# =============================================================================


def test_load_and_decode_session_success(fake_fs: FakeFileSystem) -> None:
    """Test load_and_decode_session with valid files."""
    # Create static key
    fake_fs.write_text(Path("static.txt"), "ABCD")

    # Create session JSON
    body = bytes([ord("!"), 0x30, 0x42])
    payload = make_payload(body)

    session_json = f"""{{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": "ABCD",
        "messages": [{{
            "timestamp_ms": 500,
            "direction": "sent",
            "payload": "{payload}",
            "ws_url": "wss://test.com/ws"
        }}]
    }}"""

    fake_fs.write_text(Path("session.json"), session_json)

    decoder = load_and_decode_session(Path("session.json"), Path("static.txt"))

    assert len(decoder.commands) == 1


def test_load_and_decode_session_missing_magic_raises(fake_fs: FakeFileSystem) -> None:
    """Test load_and_decode_session raises for missing magic."""
    fake_fs.write_text(Path("static.txt"), "ABCD")

    session_json = """{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": null,
        "messages": []
    }"""

    fake_fs.write_text(Path("session.json"), session_json)

    with pytest.raises(MissingMagicError, match="no magic key"):
        load_and_decode_session(Path("session.json"), Path("static.txt"))


# =============================================================================
# Error Class Tests
# =============================================================================


def test_decoder_error_is_exception() -> None:
    """Test DecoderError is an Exception."""
    assert issubclass(DecoderError, Exception)


def test_missing_magic_error_is_decoder_error() -> None:
    """Test MissingMagicError is a DecoderError."""
    assert issubclass(MissingMagicError, DecoderError)


def test_session_decoder_skips_unknown_prefix() -> None:
    """Test SessionDecoder skips messages with unknown prefix."""
    codec = ProtocolCodec("A", "A")

    # Unknown prefix '@' - not '!', '.', or lobby prefixes
    body = b"@unknown"
    payload = make_payload(body)

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

    # Unknown prefix should be skipped
    assert len(decoder.commands) == 0
    assert len(decoder.lobby_messages) == 0


def test_load_and_decode_session_default_static_key(fake_fs: FakeFileSystem) -> None:
    """Test load_and_decode_session with default static_key_path."""
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    # Create static key at the default path
    fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCD")

    # Create session JSON
    body = bytes([ord("!"), 0x30, 0x42])
    payload = make_payload(body)

    session_json = f"""{{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": "ABCD",
        "messages": [{{
            "timestamp_ms": 500,
            "direction": "sent",
            "payload": "{payload}",
            "ws_url": "wss://test.com/ws"
        }}]
    }}"""

    fake_fs.write_text(Path("session.json"), session_json)

    # Call without static_key_path - should use default
    decoder = load_and_decode_session(Path("session.json"))

    assert len(decoder.commands) == 1


# =============================================================================
# main() Tests
# =============================================================================


def test_main_with_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints command summary."""
    from tankpit_bot.decoder import main
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    # Create static key
    fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCDE")

    # Create session with commands
    body = bytes([ord("!"), 0x30, 0x42, 0xAB])
    payload = make_payload(body)

    session_json = f"""{{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": "ABCDE",
        "messages": [{{
            "timestamp_ms": 500,
            "direction": "sent",
            "payload": "{payload}",
            "ws_url": "wss://test.com/ws"
        }}]
    }}"""

    fake_fs.write_text(Path("capture_session.json"), session_json)

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Decoded 1 commands" in output
    assert "type_byte=0x30" in output
    assert "cmd_byte=0x42" in output


def test_main_no_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() with empty session."""
    from tankpit_bot.decoder import main
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    # Create static key
    fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCD")

    # Create session with no messages
    session_json = """{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": "ABCD",
        "messages": []
    }"""

    fake_fs.write_text(Path("capture_session.json"), session_json)

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Decoded 0 commands" in output


def test_main_with_custom_output_path(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() reads TANKPIT_OUTPUT env var."""
    from tankpit_bot.decoder import main
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    fake_env.set("TANKPIT_OUTPUT", "custom_session.json")

    # Create static key
    fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCD")

    # Create session
    session_json = """{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": "ABCD",
        "messages": []
    }"""

    fake_fs.write_text(Path("custom_session.json"), session_json)

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Loading session from custom_session.json" in output


def test_main_multiple_command_types(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() groups commands by type_byte and cmd_byte."""
    from tankpit_bot.decoder import main
    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    # Create static key
    fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCDEFGH")

    # Create session with multiple command types
    body1 = bytes([ord("!"), 0x30, 0x42, 0xAB])
    body2 = bytes([ord("!"), 0x30, 0x43, 0xCD])  # Same type, different cmd
    body3 = bytes([ord("!"), 0x31, 0x42, 0xEF])  # Different type
    payload1 = make_payload(body1)
    payload2 = make_payload(body2)
    payload3 = make_payload(body3)

    session_json = f"""{{
        "session_id": "test",
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 1000,
        "base_url": "https://test.com",
        "magic": "ABCDEFGH",
        "messages": [
            {{"timestamp_ms": 100, "direction": "sent", "payload": "{payload1}", "ws_url": "wss://test.com/ws"}},
            {{"timestamp_ms": 200, "direction": "sent", "payload": "{payload2}", "ws_url": "wss://test.com/ws"}},
            {{"timestamp_ms": 300, "direction": "sent", "payload": "{payload3}", "ws_url": "wss://test.com/ws"}}
        ]
    }}"""

    fake_fs.write_text(Path("capture_session.json"), session_json)

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Decoded 3 commands" in output
    assert "type_byte=0x30" in output
    assert "type_byte=0x31" in output
    assert "cmd_byte=0x42" in output
    assert "cmd_byte=0x43" in output
