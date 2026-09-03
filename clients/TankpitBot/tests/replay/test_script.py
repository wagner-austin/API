"""Tests for the replay_bot CLI script."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.replay import _test_hooks as replay_hooks
from tankpit_bot.replay.types import ReplayTickTraceDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_self_state
from tankpit_bot.types import CaptureSession, encode_capture_session
from tests.conftest import FakeFileSystem


def _empty_session(magic: str | None = "testmagic") -> CaptureSession:
    """Create an empty capture session."""
    return CaptureSession(
        session_id="script-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=[],
        magic=magic,
        game_log=[],
        tank_names={},
    )


def _install_fake_fs(fs: FakeFileSystem) -> None:
    """Install fake FS hooks for testing.

    Seeds the static XOR key: a replay builds its session table from
    it and now RAISES when it is missing, where the module global used
    to stay ``None`` and hand the decoders undeciphered bytes
    ([[session-state-deglobalisation]]).

    Args:
        fs: The fake file system to install.
    """
    from tankpit_bot.capture.xor import reset_static_key_cache
    from tankpit_bot.resources import static_key_file_path

    fs.write_text(static_key_file_path(), "Y" + "A" * 999)
    reset_static_key_cache()
    _test_hooks.write_text = fs.write_text
    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists
    _test_hooks.append_text = fs.append_text


def _restore_hooks() -> None:
    """Restore real hooks after test."""
    _test_hooks.write_text = _test_hooks._real_write_text
    _test_hooks.read_text = _test_hooks._real_read_text
    _test_hooks.path_exists = _test_hooks._real_path_exists
    _test_hooks.append_text = _test_hooks._real_append_text
    _test_hooks.get_argv = _test_hooks._real_get_argv


class TestFormatTraceLine:
    """Tests for _format_trace_line."""

    def test_formats_all_fields(self) -> None:
        """_format_trace_line includes all trace fields."""
        from scripts.replay_bot import _format_trace_line

        from tankpit_bot.bot.ai.world_types import make_enemy_threat

        trace = ReplayTickTraceDict(
            tick_index=3,
            timestamp_ms=5000,
            self_x=100,
            self_y=120,
            fuel=500,
            behavior_mode="HUNT",
            behavior_score=900,
            behavior_reason="find_enemies",
            ai_mode="HUNT",
            ai_mode_state="ACQUIRE",
            command_type="map_open",
            target_x=0,
            target_y=0,
            combat_target_id=-1,
            resource_target_kind="",
            visible_threats=[
                make_enemy_threat(
                    tank_id=42,
                    x=110,
                    y=125,
                    distance=15,
                    damage_state=1,
                    rank=4,
                    team=1,
                    name="Artax",
                    is_bot=False,
                    timestamp_ms=900,
                ),
                make_enemy_threat(
                    tank_id=99,
                    x=130,
                    y=140,
                    distance=50,
                    damage_state=0,
                    rank=2,
                    team=2,
                    name="Yuppler",
                    is_bot=False,
                    timestamp_ms=800,
                ),
            ],
            container_count=5,
        )
        line = _format_trace_line(trace)
        assert "3" in line
        assert "pos=(100,120)" in line
        assert "fuel=  500" in line
        assert "HUNT" in line
        assert "score= 900" in line
        assert "cmd=map_open" in line
        assert "target=(0,0)" in line
        assert "Artax@d=15" in line
        assert "Yuppler@d=50" in line
        assert "containers=5" in line
        assert "ai=HUNT/ACQUIRE" in line
        assert "reason=find_enemies" in line

    def test_formats_durable_ai_state(self) -> None:
        """_format_trace_line shows durable AI mode state when active."""
        from scripts.replay_bot import _format_trace_line

        trace = ReplayTickTraceDict(
            tick_index=0,
            timestamp_ms=1000,
            self_x=50,
            self_y=60,
            fuel=150,
            behavior_mode="HUNT",
            behavior_score=950,
            behavior_reason="shoot_target",
            ai_mode="HUNT",
            ai_mode_state="ENGAGE",
            command_type="shoot",
            target_x=51,
            target_y=60,
            combat_target_id=42,
            resource_target_kind="",
            visible_threats=[],
            container_count=0,
        )
        line = _format_trace_line(trace)
        assert "ai=HUNT/ENGAGE" in line

    def test_formats_resource_lock(self) -> None:
        """_format_trace_line shows resource target kind when locked."""
        from scripts.replay_bot import _format_trace_line

        trace = ReplayTickTraceDict(
            tick_index=0,
            timestamp_ms=1000,
            self_x=50,
            self_y=60,
            fuel=200,
            behavior_mode="COLLECT",
            behavior_score=900,
            behavior_reason="fuel_collect",
            ai_mode="COLLECT",
            ai_mode_state="PICKUP",
            command_type="pickup_fuel",
            target_x=52,
            target_y=63,
            combat_target_id=-1,
            resource_target_kind="fuel",
            visible_threats=[],
            container_count=3,
        )
        line = _format_trace_line(trace)
        assert "resource=fuel" in line

    def test_formats_empty_threats(self) -> None:
        """_format_trace_line shows 0:[] for empty threat list."""
        from scripts.replay_bot import _format_trace_line

        trace = ReplayTickTraceDict(
            tick_index=0,
            timestamp_ms=1000,
            self_x=50,
            self_y=60,
            fuel=150,
            behavior_mode="HUNT",
            behavior_score=0,
            behavior_reason="find_enemies",
            ai_mode="UNSET",
            ai_mode_state="",
            command_type="map_open",
            target_x=0,
            target_y=0,
            combat_target_id=-1,
            resource_target_kind="",
            visible_threats=[],
            container_count=0,
        )
        line = _format_trace_line(trace)
        assert "threats=0:[]" in line


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    def test_missing_file_returns_1(self) -> None:
        """main() returns 1 when session file does not exist."""
        from scripts.replay_bot import main

        fs = FakeFileSystem()
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot", "nonexistent.json"]
        result = main()
        _restore_hooks()
        assert result == 1

    def test_no_magic_returns_1(self) -> None:
        """main() returns 1 when session has no magic key."""
        from scripts.replay_bot import main

        fs = FakeFileSystem()
        session = _empty_session(magic=None)
        encoded = encode_capture_session(session)
        session_json = dump_json_str(encoded)
        fs.write_text(Path("test_session.json"), session_json)
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot", "test_session.json"]
        result = main()
        _restore_hooks()
        assert result == 1

    def test_empty_session_returns_0(self) -> None:
        """main() returns 0 for valid session with no messages."""
        from scripts.replay_bot import main

        fs = FakeFileSystem()
        session = _empty_session()
        encoded = encode_capture_session(session)
        session_json = dump_json_str(encoded)
        fs.write_text(Path("test_session.json"), session_json)
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot", "test_session.json"]
        result = main()
        _restore_hooks()
        assert result == 0

    def test_default_path_used_when_no_args(self) -> None:
        """main() uses capture_session.json as default path."""
        from scripts.replay_bot import main

        fs = FakeFileSystem()
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot"]
        # File does not exist → returns 1
        result = main()
        _restore_hooks()
        assert result == 1

    def test_json_flag_parsed(self) -> None:
        """main() accepts --json flag without error for valid session."""
        from scripts.replay_bot import main

        fs = FakeFileSystem()
        session = _empty_session()
        encoded = encode_capture_session(session)
        session_json = dump_json_str(encoded)
        fs.write_text(Path("test_session.json"), session_json)
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot", "test_session.json", "--json"]
        result = main()
        _restore_hooks()
        assert result == 0

    def test_main_module_guard(self) -> None:
        """The __main__ guard calls main() via SystemExit."""
        import runpy
        import sys

        fs = FakeFileSystem()
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot", "nonexistent.json"]
        # Remove from sys.modules so runpy doesn't warn about prior import
        sys.modules.pop("scripts.replay_bot", None)
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.replay_bot", run_name="__main__")
        _restore_hooks()
        assert exc_info.value.code == 1

    def test_main_logs_traces(self) -> None:
        """main() logs each trace line when not in JSON mode.

        Uses process_received_message_hook to inject self_state during
        message processing so the planner produces traces.
        """
        from scripts.replay_bot import main

        from tankpit_bot.sniffer.decoders import process_received_message as real_prm
        from tankpit_bot.state.types import WorldStateDict

        call_count = 0

        def _injecting_hook(ws: WorldService, payload: str, xor_table: bytes) -> None:
            nonlocal call_count
            real_prm(ws, payload, xor_table)
            call_count += 1
            if call_count == 1:
                self_state = make_self_state(
                    x=50,
                    y=60,
                    fuel=150,
                    team=0,
                    tank_id=1,
                    rank=3,
                    leaderboard_position=0,
                )
                ws.world_state = WorldStateDict(
                    **{**ws.world_state, "self_state": self_state},
                )

        # Session with one message so final batch fires the planner
        from tankpit_bot.types import CapturedMessage

        msg = CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            # A well-formed 4-byte text frame. This read "BQBUZXN0"
            # until 2026-08-06 — a length prefix claiming FIVE bytes
            # over a four-byte body, i.e. a torn frame the old inline
            # walk silently truncated to nothing. The comment beside it
            # said "5-byte text frame", so the fixture had been wrong
            # since it was written and nothing could say so
            # ([[session-state-deglobalisation]]).
            payload="BABUZXN0",
            ws_url="wss://tankpit.com/ws",
        )
        session = CaptureSession(
            session_id="trace-test",
            start_timestamp_ms=1000,
            end_timestamp_ms=2000,
            base_url="https://tankpit.com/play",
            messages=[msg],
            magic="testmagic",
            game_log=[],
            tank_names={},
        )
        fs = FakeFileSystem()
        encoded_session = encode_capture_session(session)
        session_json = dump_json_str(encoded_session)
        fs.write_text(Path("test_session.json"), session_json)
        _install_fake_fs(fs)
        _test_hooks.get_argv = lambda: ["replay_bot", "test_session.json"]

        original_hook = replay_hooks.process_received_message_hook
        replay_hooks.process_received_message_hook = _injecting_hook

        result = main()

        replay_hooks.process_received_message_hook = original_hook
        _restore_hooks()
        assert result == 0
