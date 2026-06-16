"""Tests for scripts.analyze_protocol."""

from __future__ import annotations

import base64
import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from scripts import _test_hooks as script_hooks
from scripts.analyze_protocol import main

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture.xor import build_xor_table, xor_decode_body
from tests.conftest import FakeFileSystem


def _encode_received_frame(msg_type: int, decoded_data: bytes, xor_table: bytes) -> str:
    """Encode one received frame for a capture session.

    Args:
        msg_type: Protocol type byte.
        decoded_data: XOR-decoded data without the type byte.
        xor_table: Session XOR table.

    Returns:
        Base64-encoded frame payload.
    """
    encoded_body = bytes([msg_type]) + xor_decode_body(decoded_data, xor_table)
    frame = bytes([len(encoded_body) & 0xFF, len(encoded_body) >> 8]) + encoded_body
    return base64.b64encode(frame).decode("ascii")


@pytest.fixture()
def _fake_fs() -> Generator[FakeFileSystem, None, None]:
    """Patch script and core file hooks with a fake filesystem."""
    old_script_exists = script_hooks.path_exists
    old_script_read = script_hooks.read_text
    old_script_logging = script_hooks.setup_rich_logging
    old_core_exists = core_hooks.path_exists
    old_core_read = core_hooks.read_text

    fs = FakeFileSystem()
    script_hooks.path_exists = fs.path_exists
    script_hooks.read_text = fs.read_text
    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.path_exists = fs.path_exists
    core_hooks.read_text = fs.read_text

    try:
        yield fs
    finally:
        script_hooks.path_exists = old_script_exists
        script_hooks.read_text = old_script_read
        script_hooks.setup_rich_logging = old_script_logging
        core_hooks.path_exists = old_core_exists
        core_hooks.read_text = old_core_read


class TestAnalyzeProtocolScript:
    """Tests for the analyze_protocol script."""

    def test_prints_protocol_census(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Prints decoded, short, and unsupported sections."""
        magic = "script-magic"
        static_key = "Q" * 64
        xor_table = build_xor_table(static_key, magic)

        session = {
            "session_id": "protocol-script",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _encode_received_frame(
                        0x2E,
                        bytes([0x24, 0x02, 0x7D, 0x04, 140, 137, 8, 3, 0, 0, 0, 0, 0]),
                        xor_table,
                    ),
                    "ws_url": "wss://test/ws",
                },
                {
                    "timestamp_ms": 1100,
                    "direction": "received",
                    "payload": _encode_received_frame(0x21, bytes([0x01, 0x02]), xor_table),
                    "ws_url": "wss://test/ws",
                },
                {
                    "timestamp_ms": 1200,
                    "direction": "received",
                    "payload": _encode_received_frame(0x7B, bytes([0x11, 0x22, 0x33]), xor_table),
                    "ws_url": "wss://test/ws",
                },
            ],
            "magic": magic,
            "game_log": [],
            "tank_names": {},
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)
        _fake_fs._files[str(Path(__file__).resolve().parent.parent / "xor_static_key.txt")] = (
            static_key
        )

        old_argv = sys.argv
        sys.argv = ["analyze_protocol"]
        try:
            main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "decoded_binary_frames=1" in output
        assert "short_or_invalid_frames=1" in output
        assert "unsupported_frames=1" in output
        assert "position_update x1" in output
        assert "0x21 len=3" in output
        assert "0x7B len=4" in output

    def test_exits_when_file_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 when the capture session does not exist."""
        old_argv = sys.argv
        sys.argv = ["analyze_protocol", "missing.json"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_exits_when_magic_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 when the capture session has no magic key."""
        session = {
            "session_id": "no-magic",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [],
            "magic": None,
            "game_log": [],
            "tank_names": {},
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)

        old_argv = sys.argv
        sys.argv = ["analyze_protocol"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_exits_when_static_key_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 when xor_static_key.txt cannot be loaded."""
        session = {
            "session_id": "no-static-key",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [],
            "magic": "magic-value",
            "game_log": [],
            "tank_names": {},
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)

        old_argv = sys.argv
        sys.argv = ["analyze_protocol"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_module_entrypoint_runs_main(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Executes the module entrypoint and prints the census output."""
        magic = "script-main-magic"
        static_key = "Z" * 64
        xor_table = build_xor_table(static_key, magic)
        session = {
            "session_id": "script-main",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _encode_received_frame(
                        0x2E,
                        bytes([0x24, 0x02, 0x7D, 0x04, 140, 137, 8, 3, 0, 0, 0, 0, 0]),
                        xor_table,
                    ),
                    "ws_url": "wss://test/ws",
                }
            ],
            "magic": magic,
            "game_log": [],
            "tank_names": {},
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)
        _fake_fs._files[str(Path(__file__).resolve().parent.parent / "xor_static_key.txt")] = (
            static_key
        )

        old_argv = sys.argv
        sys.argv = ["scripts.analyze_protocol"]
        try:
            sys.modules.pop("scripts.analyze_protocol", None)
            runpy.run_module("scripts.analyze_protocol", run_name="__main__")
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "decoded_binary_frames=1" in output
