"""Tests for the per-message wire decoders."""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot.decoder import (
    DecoderError,
    MissingMagicError,
    SessionDecoder,
    load_and_decode_session,
)
from tankpit_bot.protocol.codec import ProtocolCodec
from tankpit_bot.types import (
    CapturedMessage,
    CaptureSession,
)
from tests.conftest import (
    FakeEnv,
    FakeFileSystem,
)
from tests.wire_builders import frame_payload


def test_session_decoder_handles_all_lobby_prefixes() -> None:
    """Test SessionDecoder handles various lobby message prefixes."""
    codec = ProtocolCodec("A", "A")

    prefixes = ["%", "+", "*", "=", "$", "-"]
    messages = []

    for i, prefix in enumerate(prefixes):
        body = (prefix + "test").encode("utf-8")
        payload = frame_payload(body)
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


def test_load_and_decode_session_success(fake_fs: FakeFileSystem) -> None:
    """Test load_and_decode_session with valid files."""
    # Create static key
    fake_fs.write_text(Path("static.txt"), "ABCD")

    # Create session JSON
    body = bytes([ord("!"), 0x30, 0x42])
    payload = frame_payload(body)

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
    payload = frame_payload(body)

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
    payload = frame_payload(body)

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
    payload1 = frame_payload(body1)
    payload2 = frame_payload(body2)
    payload3 = frame_payload(body3)

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
