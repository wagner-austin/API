"""Tests for scripts.decode module."""

from __future__ import annotations

import base64
import runpy
import sys
import types
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from scripts.decode import main

from tankpit_bot import _test_hooks
from tests.conftest import FakeEnv, FakeFileSystem


def _make_binary_payload(body: bytes) -> str:
    """Create a base64-encoded payload with 2-byte length header.

    Args:
        body: Raw body bytes.

    Returns:
        Base64-encoded payload string.
    """
    length = len(body)
    header = bytes([length >> 8, length & 0xFF])
    return base64.b64encode(header + body).decode("ascii")


def _make_short_payload() -> str:
    """Create a payload too short to decode (< 3 bytes total).

    Returns:
        Base64-encoded 2-byte payload (header only, no body).
    """
    return base64.b64encode(b"\x00\x00").decode("ascii")


def _make_text_payload(text: str) -> str:
    """Create a base64-encoded payload with 2-byte length header.

    Args:
        text: Text content for the payload body.

    Returns:
        Base64-encoded payload string.
    """
    body = text.encode("utf-8")
    length = len(body)
    header = bytes([length >> 8, length & 0xFF])
    return base64.b64encode(header + body).decode("ascii")


def _make_session_json(messages: list[dict[str, str]], magic: str | None = None) -> str:
    """Create a minimal capture session JSON string.

    Args:
        messages: List of message dicts with direction, payload, ws_url.
        magic: XOR magic key string, or None.

    Returns:
        JSON string of the session.
    """
    session = {
        "session_id": "test-session-001",
        "start_timestamp_ms": 1000000,
        "end_timestamp_ms": 1001000,
        "base_url": "https://tankpit.com/play",
        "messages": [
            {
                "timestamp_ms": 1000000 + i * 100,
                "direction": m["direction"],
                "payload": m["payload"],
                "ws_url": m.get("ws_url", "wss://test.tankpit.com/ws/"),
            }
            for i, m in enumerate(messages)
        ],
        "magic": magic,
    }
    return dump_json_str(session)


@pytest.fixture()
def _fake_env() -> Generator[FakeEnv, None, None]:
    """Provide a FakeEnv and restore hooks after test."""
    old_get_env = _test_hooks.get_env
    env = FakeEnv()
    _test_hooks.get_env = env
    yield env
    _test_hooks.get_env = old_get_env


@pytest.fixture()
def _fake_fs() -> Generator[FakeFileSystem, None, None]:
    """Provide a FakeFileSystem and restore hooks after test."""
    old_read = _test_hooks.read_text
    old_exists = _test_hooks.path_exists
    old_write = _test_hooks.write_text

    from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

    fake_static_key = "Y" + "A" * 999
    fs = FakeFileSystem()
    fs._files[str(DEFAULT_STATIC_KEY_PATH)] = fake_static_key

    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists
    _test_hooks.write_text = fs.write_text

    yield fs

    _test_hooks.read_text = old_read
    _test_hooks.path_exists = old_exists
    _test_hooks.write_text = old_write


class TestDecodeScript:
    """Tests for scripts.decode.main entry point."""

    def test_decode_text_messages(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode processes text messages from capture session."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        room_payload = _make_text_payload(
            "+2|World|24|flags|2|n|field24.gif|2026",
        )
        join_payload = _make_text_payload(
            "=2|Jan 1, 2020|TestTank|3|5|5|5|5",
        )
        messages = [
            {"direction": "received", "payload": room_payload},
            {"direction": "sent", "payload": _make_text_payload("*2")},
            {"direction": "received", "payload": join_payload},
        ]
        session_json = _make_session_json(messages, magic="testmagic123")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        # Patch sys.argv to use default path
        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "ROOM_LIST" in output
        assert "SELECT" in output
        assert "JOIN_CONFIRM" in output

    def test_decode_no_magic_exits(
        self,
        _fake_fs: FakeFileSystem,
    ) -> None:
        """Test decode exits with error when no magic key in session."""
        session_json = _make_session_json([], magic=None)
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_decode_missing_file_exits(
        self,
        _fake_fs: FakeFileSystem,
    ) -> None:
        """Test decode exits with error when file not found."""
        old_argv = sys.argv
        sys.argv = ["decode", "nonexistent.json"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_decode_custom_path(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode accepts custom session path as argument."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        room_payload = _make_text_payload(
            "+1|Practice|1|flags|2|p|field01.gif|2026",
        )
        messages = [
            {"direction": "received", "payload": room_payload},
        ]
        session_json = _make_session_json(messages, magic="custommagic")
        _fake_fs._files[str(Path("custom.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode", "custom.json"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "ROOM_LIST" in output
        assert "Practice" in output

    def test_decode_game_log_printed(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode prints game log entries when present."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        session = {
            "session_id": "test-log-session",
            "start_timestamp_ms": 1000000,
            "end_timestamp_ms": 1001000,
            "base_url": "https://tankpit.com/play",
            "messages": [],
            "magic": "logmagic",
            "game_log": [
                {"timestamp_ms": 1000100, "text": "LOCATION: 100,200", "category": "location"},
                {"timestamp_ms": 1000200, "text": "has been deactivated", "category": "combat"},
            ],
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "Game Log (2 entries)" in output
        assert "LOCATION: 100,200" in output
        assert "has been deactivated" in output

    def test_decode_short_payload_warns(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode warns on short sent/received payloads."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        short = _make_short_payload()
        messages = [
            {"direction": "sent", "payload": short},
            {"direction": "received", "payload": short},
        ]
        session_json = _make_session_json(messages, magic="m")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "decode failed" in output

    def test_decode_sent_xor_command(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode handles sent XOR command (0x21 prefix)."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        # 0x21 = '!', followed by enough bytes for XOR decode
        xor_body = bytes([0x21]) + b"\x00" * 10
        messages = [
            {"direction": "sent", "payload": _make_binary_payload(xor_body)},
        ]
        session_json = _make_session_json(messages, magic="xormagic")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "SENT" in output

    def test_decode_sent_raw_fallback(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode handles sent message with unknown type (RAW)."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        # 0x99 is not a text type and not 0x21
        raw_body = bytes([0x99, 0x01, 0x02, 0x03])
        messages = [
            {"direction": "sent", "payload": _make_binary_payload(raw_body)},
        ]
        session_json = _make_session_json(messages, magic="rawmagic")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "RAW" in output

    def test_decode_received_binary(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode handles received binary (non-text) messages."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        # 0x99 not in TEXT_MESSAGE_TYPES, with enough data for XOR
        binary_body = bytes([0x99]) + b"\x01" * 10
        messages = [
            {
                "direction": "received",
                "payload": _make_binary_payload(binary_body),
            },
        ]
        session_json = _make_session_json(messages, magic="binmagic")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "RECEIVED" in output

    def test_decode_received_empty_body(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode handles received binary with single-byte body."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        # Single byte body -> xor_decode returns empty
        binary_body = bytes([0x99])
        messages = [
            {
                "direction": "received",
                "payload": _make_binary_payload(binary_body),
            },
        ]
        session_json = _make_session_json(messages, magic="emptymagic")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "EMPTY" in output

    def test_decode_sent_xor_short_cmd(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test decode handles sent XOR with short decoded result."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        # 0x21 with only 1 extra byte -> decoded < 2 -> CMD fallback
        xor_body = bytes([0x21, 0x00])
        messages = [
            {"direction": "sent", "payload": _make_binary_payload(xor_body)},
        ]
        session_json = _make_session_json(messages, magic="shortcmd")
        _fake_fs._files[str(Path("capture_session.json"))] = session_json

        old_argv = sys.argv
        sys.argv = ["decode"]
        try:
            main()
        finally:
            sys.argv = old_argv
            reset_world_state()

        output = capsys.readouterr().out
        assert "CMD" in output


def test_decode_entrypoint_runs_as_main(
    _fake_fs: FakeFileSystem,
) -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
    path = str(Path("capture_session.json"))
    _fake_fs._files[path] = dump_json_str(
        {
            "session_id": "main-test",
            "start_timestamp_ms": 1000000,
            "end_timestamp_ms": 1001000,
            "base_url": "https://tankpit.com/play",
            "messages": [],
            "magic": "mainmagic",
        },
    )

    modules_to_clear = [k for k in sys.modules if k.startswith("scripts")]
    saved_modules: dict[str, types.ModuleType] = {}
    for mod in modules_to_clear:
        saved_modules[mod] = sys.modules.pop(mod)

    old_argv = sys.argv
    sys.argv = ["decode"]
    try:
        runpy.run_module("scripts.decode", run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.modules.update(saved_modules)
