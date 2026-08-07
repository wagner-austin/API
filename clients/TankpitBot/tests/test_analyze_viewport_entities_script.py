"""Tests for scripts.analyze_viewport_entities."""

from __future__ import annotations

import base64
import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from scripts.analyze_viewport_entities import main

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture.xor import xor_decode_body
from tankpit_bot.protocol.codec import build_xor_table
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
    """Create one viewport update payload for the script tests.

    Args:
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    decoded_data = bytes([10, 20, 1]) + _encode_entity_data(-1, 255, 0)
    return _encode_received_frame(0x5A, decoded_data, xor_table)


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


class TestAnalyzeViewportEntitiesScript:
    """Tests for the raw viewport-entity dump script."""

    def test_prints_raw_viewport_entity_dump(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Prints one raw viewport update with entity rows."""
        magic = "viewport-entities-script"
        static_key = "R" * 64
        xor_table = build_xor_table(static_key, magic)
        session = {
            "session_id": "viewport-entities-script",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _make_viewport_payload(xor_table),
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
        sys.argv = ["analyze_viewport_entities"]
        try:
            main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "viewport_updates=1" in output
        assert "viewport=(10,20)" in output
        assert "equipment_cache=1" in output
        assert "cache_value=-1" in output

    def test_exits_when_file_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 when the capture session does not exist."""
        old_argv = sys.argv
        sys.argv = ["analyze_viewport_entities", "missing.json"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_exits_when_magic_or_static_key_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 for missing session magic or static key."""
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
        sys.argv = ["analyze_viewport_entities"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

        session["magic"] = "magic"
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)
        old_argv = sys.argv
        sys.argv = ["analyze_viewport_entities"]
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
        """Executes the module entrypoint and prints the raw dump."""
        magic = "viewport-entities-main"
        static_key = "S" * 64
        xor_table = build_xor_table(static_key, magic)
        session = {
            "session_id": "viewport-entities-main",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _make_viewport_payload(xor_table),
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
        sys.argv = ["analyze_viewport_entities"]
        try:
            sys.modules.pop("scripts.analyze_viewport_entities", None)
            runpy.run_module("scripts.analyze_viewport_entities", run_name="__main__")
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "viewport_updates=1" in output
