"""Tests for scripts.analyze_shot_viewport."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from scripts.analyze_shot_viewport import main

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.protocol.codec import build_xor_table, static_key_file_path
from tankpit_bot.protocol.commands import CMD_SHOOT, TYPE_COMBAT
from tests.conftest import FakeFileSystem
from tests.wire_builders import encode_wire_frame


def _encode_entity_data(entity_id: int, value: int, terrain_type: int) -> bytes:
    """Encode one 0x5A entity payload word."""
    raw_id = 0xFFFF if entity_id == -1 else entity_id
    value_nibble = 8 if value == 255 else value
    z = (raw_id << 8) | (value_nibble << 4) | terrain_type
    return bytes([(z >> 16) & 0xFF, (z >> 8) & 0xFF, z & 0xFF])


def _make_viewport_payload(xor_table: bytes) -> str:
    """Create one viewport update payload for the script tests."""
    decoded_data = bytes([50, 60, 0]) + _encode_entity_data(514, 255, 0)
    return encode_wire_frame(0x5A, decoded_data, xor_table)


def _make_shoot_payload(target_x: int, target_y: int, target_id: int, xor_table: bytes) -> str:
    """Create one sent shoot command payload."""
    return encode_wire_frame(
        ord("!"),
        bytes(
            [
                TYPE_COMBAT,
                CMD_SHOOT,
                target_x & 0xFF,
                target_y & 0xFF,
                target_id & 0xFF,
                (target_id >> 8) & 0xFF,
            ]
        ),
        xor_table,
    )


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


class TestAnalyzeShotViewportScript:
    """Tests for the shoot-to-viewport correlation script."""

    def test_prints_shot_viewport_correlation(
        self,
        _fake_fs: FakeFileSystem,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Prints one correlated shot row."""
        magic = "shot-viewport-script"
        static_key = "U" * 64
        xor_table = build_xor_table(static_key, magic)
        session = {
            "session_id": "shot-viewport-script",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _make_viewport_payload(xor_table),
                    "ws_url": "wss://test/ws",
                },
                {
                    "timestamp_ms": 1100,
                    "direction": "sent",
                    "payload": _make_shoot_payload(50, 60, 514, xor_table),
                    "ws_url": "wss://test/ws",
                },
            ],
            "magic": magic,
            "game_log": [],
            "tank_names": {},
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)
        _fake_fs._files[str(static_key_file_path())] = static_key

        old_argv = sys.argv
        sys.argv = ["analyze_shot_viewport"]
        try:
            main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "shot_count=1" in output
        assert "target_id=514" in output
        assert "id_matches=1" in output
        assert "coord_matches=1" in output

    def test_exits_when_file_missing(self, _fake_fs: FakeFileSystem) -> None:
        """Exits with code 1 when the capture session does not exist."""
        old_argv = sys.argv
        sys.argv = ["analyze_shot_viewport", "missing.json"]
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
        sys.argv = ["analyze_shot_viewport"]
        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

        session["magic"] = "magic"
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)
        old_argv = sys.argv
        sys.argv = ["analyze_shot_viewport"]
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
        """Executes the module entrypoint and prints the correlation dump."""
        magic = "shot-viewport-main"
        static_key = "V" * 64
        xor_table = build_xor_table(static_key, magic)
        session = {
            "session_id": "shot-viewport-main",
            "start_timestamp_ms": 1000,
            "end_timestamp_ms": 1500,
            "base_url": "https://tankpit.com/play",
            "messages": [
                {
                    "timestamp_ms": 1000,
                    "direction": "received",
                    "payload": _make_viewport_payload(xor_table),
                    "ws_url": "wss://test/ws",
                },
                {
                    "timestamp_ms": 1100,
                    "direction": "sent",
                    "payload": _make_shoot_payload(50, 60, 514, xor_table),
                    "ws_url": "wss://test/ws",
                },
            ],
            "magic": magic,
            "game_log": [],
            "tank_names": {},
        }
        _fake_fs._files[str(Path("capture_session.json"))] = dump_json_str(session)
        _fake_fs._files[str(static_key_file_path())] = static_key

        old_argv = sys.argv
        sys.argv = ["analyze_shot_viewport"]
        try:
            sys.modules.pop("scripts.analyze_shot_viewport", None)
            runpy.run_module("scripts.analyze_shot_viewport", run_name="__main__")
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert "shot_count=1" in output
