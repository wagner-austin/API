"""Tests for scripts.analyze_viewport.

The position_update analysis path was deleted 2026-06-20 along with
the container PositionUpdate decoder. The script now reports only
MovementResponse + ViewportUpdate + 13-byte shape census.
"""

from __future__ import annotations

import base64
import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from scripts.analyze_viewport import main

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture.xor import build_xor_table, xor_decode_body
from tests.conftest import FakeFileSystem


def _encode_received_frame(msg_type: int, decoded_data: bytes, xor_table: bytes) -> str:
    """Encode one received message frame for a capture session.

    Args:
        msg_type: Protocol message type byte.
        decoded_data: XOR-decoded message bytes without the type byte.
        xor_table: Session XOR table.

    Returns:
        Base64-encoded frame payload.
    """
    encoded_body = bytes([msg_type]) + xor_decode_body(decoded_data, xor_table)
    length = len(encoded_body)
    frame = bytes([length & 0xFF, length >> 8]) + encoded_body
    return base64.b64encode(frame).decode("ascii")


def _make_movement_response_payload(
    tank_id: int,
    x: int,
    y: int,
    xor_table: bytes,
) -> str:
    """Create a MovementResponse payload (12-byte body inc. carrying byte)."""
    decoded_data = bytes([1, tank_id & 0xFF, tank_id >> 8, x, y, 0, 0, 1, 0, 0, 5, 0])
    return _encode_received_frame(0x3D, decoded_data, xor_table)


def _make_viewport_update_payload(
    viewport_left: int,
    viewport_top: int,
    xor_table: bytes,
) -> str:
    """Create a ViewportUpdate payload."""
    decoded_data = bytes([viewport_left, viewport_top])
    return _encode_received_frame(0x5A, decoded_data, xor_table)


@pytest.fixture()
def _fake_fs() -> Generator[FakeFileSystem, None, None]:
    """Patch both script and core file hooks with a fake filesystem."""
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

    yield fs

    script_hooks.path_exists = old_script_exists
    script_hooks.read_text = old_script_read
    script_hooks.setup_rich_logging = old_script_logging
    core_hooks.path_exists = old_core_exists
    core_hooks.read_text = old_core_read


class TestAnalyzeViewportScript:
    """Tests for the analyze_viewport script."""

    def test_prints_viewport_evidence_summary(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Prints viewport inference summary from a capture session."""
        magic = "script-magic"
        static_key = "K" * 64
        xor_table = build_xor_table(static_key, magic)

        session = {
            "session_id": "script-analysis",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _make_movement_response_payload(638, 144, 137, xor_table),
                    "ws_url": "wss://test/ws",
                },
                {
                    "timestamp_ms": 1100,
                    "direction": "received",
                    "payload": _make_viewport_update_payload(136, 134, xor_table),
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
        sys.argv = ["analyze_viewport"]
        try:
            main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "self_tank_id=638" in output
        assert "capture_status=viewport_inferred" in output
        assert "viewport=(136,134)" in output

    def test_exits_when_file_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 when the target capture session does not exist."""
        old_argv = sys.argv
        sys.argv = ["analyze_viewport", "missing.json"]
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
        sys.argv = ["analyze_viewport"]
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
        sys.argv = ["analyze_viewport"]
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
        """Executes the module entrypoint and prints the viewport summary."""
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
                    "payload": _make_movement_response_payload(638, 144, 137, xor_table),
                    "ws_url": "wss://test/ws",
                },
                {
                    "timestamp_ms": 1100,
                    "direction": "received",
                    "payload": _make_viewport_update_payload(136, 134, xor_table),
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
        sys.argv = ["scripts.analyze_viewport"]
        try:
            sys.modules.pop("scripts.analyze_viewport", None)
            runpy.run_module("scripts.analyze_viewport", run_name="__main__")
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "self_tank_id=638" in output
