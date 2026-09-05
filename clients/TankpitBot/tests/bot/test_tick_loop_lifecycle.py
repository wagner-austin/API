"""Tests for the tick loop's session lifecycle.

Stop-file detection, wind-down, archive-stamp parsing, interrupt exit
reasons, the runs-index row, and browser-closed teardown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.bot import tick_body as tick_body_module
from tankpit_bot.bot.base import Bot
from tankpit_bot.sniffer.world_service import WorldService
from tests.bot._tick_loop_fakes import (
    _BrowserClosedPage,
    _fail_tick_once_with_browser_closed,
    _fail_tick_once_with_session_exit,
    _FakePage,
)
from tests.conftest import (
    FakeEnv,
    FakeFileSystem,
)


class TestStopFileDetection:
    """Test stop-file sentinel terminates run_tick_loop."""

    def test_stop_file_ends_tick_loop(self, fake_env: FakeEnv) -> None:
        """run_tick_loop exits when the stop file exists.

        Uses fake hooks so no real file operations occur. The tick_once
        call exits early because self_state is None (no position), then
        the stop-file check triggers.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)

        stop_path = Path("C:/tmp/test_stop_file.sentinel")

        removed_files: list[Path] = []

        def fake_path_exists(path: Path) -> bool:
            return path == stop_path

        def fake_remove_file(path: Path) -> None:
            removed_files.append(path)

        _test_hooks.path_exists = fake_path_exists
        _test_hooks.remove_file = fake_remove_file

        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=stop_path,
        )

        assert removed_files == [stop_path]


class TestWindDown:
    """Tests for the session wind-down flag."""

    def test_bounded_session_raises_wind_down_flag_in_final_stretch(self) -> None:
        """A >2-window session flips ``wind_down`` 60 s before the budget.

        User request 2026-07-26: "run and then collect and exit
        cleanly, instead of the program killing it mid action."
        """
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=126,
            stop_file_path=Path("C:/tmp/absent.sentinel"),
        )
        assert bot._ai_state["wind_down"] is True

    def test_kill_target_triggers_wind_down(self) -> None:
        """Reaching the kill bound flips ``wind_down`` at the kill boundary.

        User request 2026-07-26: "maybe if we put it for kills instead
        of time based" — the kill boundary is the natural clean-exit
        point (no fight is ever interrupted).
        """
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["session_kill_count"] = 2
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=4,
            session_kills=2,
            stop_file_path=Path("C:/tmp/absent.sentinel"),
        )
        assert bot._ai_state["wind_down"] is True

    def test_short_diagnostic_session_never_winds_down(self) -> None:
        """Sessions of two windows or less keep the full loop active."""
        from tankpit_bot.bot.tick_loop import run_tick_loop

        bot = Bot("https://test.tankpit.com/", headless=True)
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=120,
            stop_file_path=Path("C:/tmp/absent.sentinel"),
        )
        assert bot._ai_state["wind_down"] is False


class TestExtractStampFromArchivePath:
    """Tests for ``_extract_stamp_from_archive_path``."""

    def test_returns_stamp_for_canonical_path(self) -> None:
        """The stamp segment between ``bot-`` and ``.events.jsonl`` is returned."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        result = _extract_stamp_from_archive_path("runs/bot/bot-20260620-150138.events.jsonl")
        assert result == "20260620-150138"

    def test_returns_stamp_for_windows_separator_path(self) -> None:
        """Windows separators in the input do not affect stamp extraction."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        result = _extract_stamp_from_archive_path(r"runs\bot\bot-20260620-150138.events.jsonl")
        assert result == "20260620-150138"

    def test_raises_when_prefix_missing(self) -> None:
        """Archive paths that don't start with ``bot-`` raise."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        with pytest.raises(ValueError, match="does not match bot-"):
            _extract_stamp_from_archive_path("runs/bot/probe-20260620-150138.events.jsonl")

    def test_raises_when_suffix_missing(self) -> None:
        """Archive paths that don't end with ``.events.jsonl`` raise."""
        from tankpit_bot.bot.tick_loop import _extract_stamp_from_archive_path

        with pytest.raises(ValueError, match="does not match bot-"):
            _extract_stamp_from_archive_path("runs/bot/bot-20260620-150138.log")


class TestInterruptedExitReason:
    """The interrupt flag should produce ``exit_reason=interrupted`` rows."""

    def setup_method(self) -> None:
        """Reset world state + interrupt flag before each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_interrupt_flag()

    def teardown_method(self) -> None:
        """Reset world state + interrupt flag after each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_interrupt_flag()

    def test_pre_set_interrupt_records_interrupted_row(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """Pre-setting the flag exits at tick 1 with ``interrupted``."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import (
            request_interrupt,
            run_tick_loop,
        )
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-150138")
        bot = Bot("https://test.tankpit.com/", headless=True)
        request_interrupt()

        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "interrupted"
        assert row["ticks"] == 1
        # The interrupted run also leaves the human-readable scorecard
        # ([[bot-behavior-contract]] §1.3: summary carries the exit
        # reason) — pinned byte-for-byte at the tick-1 empty-session
        # shape so a format drift is caught here, not in a live run.
        summary = fake_fs.get_written_files()["runs\\bot\\latest.summary.txt"]
        assert summary == (
            "TANKPIT SESSION SUMMARY\n"
            "========================================\n"
            "Ticks:    1\n"
            "Exit:     interrupted\n"
            "Kills:    0\n"
            "Shots:    0 (0 hits, 0 misses, 0 rejected)\n"
            "Hit rate: n/a\n"
            "Blocked:  0\n"
            "========================================\n"
            "Fuel:     0\n"
            "Duals:    0\n"
            "Homings:  0\n"
            "Radars:   0\n"
            "========================================\n"
            "Mode:     UNSET/\n"
        )


class TestAppendIndexRowEndToEnd:
    """End-to-end test of ``_emit_session_scorecard`` -> ``_append_index_row``.

    Configures bot runtime logging so :func:`get_bot_runtime_artifacts`
    returns a real artifact bundle, then asserts the index row landed
    in the fake filesystem with the expected stamp + exit reason.
    """

    def test_emit_session_scorecard_appends_index_row(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A bot session end writes a row matching the active artifacts."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import _emit_session_scorecard
        from tankpit_bot.diagnostics.runs_index import (
            DEFAULT_INDEX_PATH,
            HEADER_LINE,
            decode_row,
        )
        from tankpit_bot.ledger.decision import record_decision
        from tankpit_bot.ledger.outcome.map_open import emit_map_open_data_processed
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        ws = WorldService()
        configure_bot_runtime_logging("20260620-150138")
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)

        # One resolved decision (outcome-counts line) plus one still
        # pending at shutdown (unresolved-decisions line).
        record_decision(
            ws.ledger,
            action_kind="map_open",
            cmd_type="map_open",
            mode="HUNT",
            score=800,
            reason_kind="find_enemies",
            reason_context={},
            target_x=0,
            target_y=0,
            target_id=0,
        )
        emit_map_open_data_processed(ws.ledger, duration_ms=500)
        record_decision(
            ws.ledger,
            action_kind="scan",
            cmd_type="radar",
            mode="HUNT",
            score=700,
            reason_kind="scan_on_landing",
            reason_context={},
            target_x=1,
            target_y=2,
            target_id=0,
        )

        _emit_session_scorecard(bot, ticks=30, exit_reason="completed")

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        assert text.startswith(HEADER_LINE)
        rows = [
            decode_row(line)
            for line in text.splitlines()
            if line and not line.startswith("stamp\t")
        ]
        if len(rows) != 1:
            raise AssertionError(f"expected 1 index row, got {len(rows)}")
        row = rows[0]
        assert row["stamp"] == "20260620-150138"
        assert row["exit_reason"] == "completed"
        assert row["ticks"] == 30
        # 30 ticks * 2000ms / 1000 = 60 seconds (TICK_RATE_MS=2000).
        assert row["duration_s"] == 60


class TestBrowserClosedExit:
    """Browser closure mid-run records ``browser_closed`` and exits cleanly."""

    def setup_method(self) -> None:
        """Reset world state + interrupt flag before each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_interrupt_flag()

    def teardown_method(self) -> None:
        """Reset world state + interrupt flag after each test."""
        from tankpit_bot.bot.tick_loop import reset_interrupt_flag

        reset_interrupt_flag()

    def test_browser_closed_between_ticks_records_browser_closed(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """Closing the browser between ticks exits with ``browser_closed``."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-191000")
        bot = Bot("https://test.tankpit.com/", headless=True)

        run_tick_loop(
            bot,
            _BrowserClosedPage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "browser_closed"

    def test_browser_closed_during_tick_records_browser_closed(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A ``TargetClosedError`` raised from inside ``_tick_once`` exits cleanly."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260620-191000")
        bot = Bot("https://test.tankpit.com/", headless=True)

        saved_tick_once = tick_body_module._tick_once
        tick_body_module._tick_once = _fail_tick_once_with_browser_closed
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )
        tick_body_module._tick_once = saved_tick_once

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "browser_closed"

    def test_session_exit_request_records_its_reason(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """A ``SessionExitError`` from a decision owner ends the run with its reason.

        User contract (2026-07-02): when the bot cannot do its job
        (no viable targets, out of fuel) it exits cleanly with an
        analyzable ``exit_reason`` instead of crashing or looping.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop import run_tick_loop
        from tankpit_bot.diagnostics.runs_index import DEFAULT_INDEX_PATH, decode_row
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        configure_bot_runtime_logging("20260702-101500")
        bot = Bot("https://test.tankpit.com/", headless=True)

        saved_tick_once = tick_body_module._tick_once
        tick_body_module._tick_once = _fail_tick_once_with_session_exit
        run_tick_loop(
            bot,
            _FakePage(),
            session_seconds=0,
            stop_file_path=Path("C:/tmp/never_exists.sentinel"),
        )
        tick_body_module._tick_once = saved_tick_once

        text = fake_fs.get_written_files()[str(DEFAULT_INDEX_PATH)]
        data_lines = [line for line in text.splitlines() if line and not line.startswith("stamp\t")]
        if len(data_lines) != 1:
            raise AssertionError(f"expected 1 index row, got {len(data_lines)}")
        row = decode_row(data_lines[0])
        assert row["exit_reason"] == "no_viable_targets"
