"""End-to-end tests for the cross-session stats report.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
real producers (game-log kill registration, teleport attempt tracking,
wire emits, game-log feedback) -> timestamped JSONL artifacts via
:class:`tests.conftest.FakeFileSystem` ->
:func:`tankpit_bot.diagnostics.session_stats.build_session_stats`
sweeping the runs directory through the ``glob_paths`` hook. Nothing is
mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.browser.dom_scraper import GameLogEntry
from tankpit_bot.diagnostics.game_log_feedback import (
    record_pickup_dispatch,
    register_world_feedback_from_game_log,
)
from tankpit_bot.diagnostics.game_log_kills import register_kills_from_game_log
from tankpit_bot.diagnostics.session_stats import (
    build_session_stats,
    main,
    render_session_stats,
)
from tankpit_bot.diagnostics.teleport_attempts import (
    emit_teleport_attempt_outcome,
    record_teleport_dispatch,
)
from tankpit_bot.runtime_logging import (
    configure_bot_runtime_logging,
    emit_wire,
    emit_wire_complete,
)
from tankpit_bot.state import make_empty_world_state

_RUNS_DIR = Path("runs") / "bot"


def _emit_first_run_activity() -> None:
    """Produce one run's worth of events through the real producers."""
    register_kills_from_game_log(
        [GameLogEntry(text="purple-8 has been deactivated by you", category="combat")],
        make_empty_world_state(),
    )
    record_teleport_dispatch(target_x=10, target_y=20, message_index=0, sent_window="w")
    assert emit_teleport_attempt_outcome(status="landed_exact", messages=[])
    record_teleport_dispatch(target_x=30, target_y=40, message_index=0, sent_window="w")
    assert emit_teleport_attempt_outcome(status="stall_timeout", messages=[])
    emit_wire("shoot")
    emit_wire("pickup_fuel")
    emit_wire("pickup_equipment")
    emit_wire("radar")
    emit_wire_complete(
        action_kind="teleport",
        duration_ms=10000,
        signal="stall_timeout",
        target_x=30,
        target_y=40,
    )
    record_pickup_dispatch(13, 172)
    register_world_feedback_from_game_log(
        [GameLogEntry(text="Empty container", category="other")],
    )


def test_sweeps_runs_and_aggregates(fake_fs: FakeFileSystem) -> None:
    """Two runs produce two rows plus totals summing every column."""
    configure_bot_runtime_logging("20260610-100000")
    _emit_first_run_activity()
    configure_bot_runtime_logging("20260610-100500")
    emit_wire("shoot")

    report = build_session_stats(_RUNS_DIR)

    assert [row["run_id"] for row in report["rows"]] == [
        "bot-20260610-100000",
        "bot-20260610-100500",
    ]
    first = report["rows"][0]
    assert first["kills"] == 1
    assert first["teleports_ok"] == 1
    assert first["teleports_failed"] == 1
    assert first["shots"] == 1
    assert first["pickups"] == 2
    assert first["stalls"] == 1
    assert first["feedback_corrections"] == 1
    assert first["events"] > 0
    assert first["started"] != ""
    assert first["duration_s"] >= 0
    second = report["rows"][1]
    assert second["shots"] == 1
    assert second["kills"] == 0
    totals = report["totals"]
    assert totals["run_id"] == "TOTAL"
    assert totals["started"] == first["started"]
    assert totals["kills"] == 1
    assert totals["shots"] == 2
    assert totals["events"] == first["events"] + second["events"]


def test_render_lists_every_run_and_totals(fake_fs: FakeFileSystem) -> None:
    """The rendered table carries one line per run plus the totals row."""
    configure_bot_runtime_logging("20260610-100000")
    emit_wire("shoot")

    rendered = render_session_stats(build_session_stats(_RUNS_DIR))

    assert "TANKPIT CROSS-SESSION STATS" in rendered
    assert "bot-20260610-100000" in rendered
    assert "TOTAL" in rendered


def test_run_with_no_events_yields_zero_row(fake_fs: FakeFileSystem) -> None:
    """A configured run that never emitted an event reports all zeros."""
    configure_bot_runtime_logging("20260610-100000")

    report = build_session_stats(_RUNS_DIR)

    row = report["rows"][0]
    assert row["events"] == 0
    assert row["duration_s"] == 0
    assert row["started"] == ""
    assert report["totals"]["events"] == 0


def test_directory_without_artifacts_raises(fake_fs: FakeFileSystem) -> None:
    """A typo'd or empty directory fails fast instead of rendering empty."""
    with pytest.raises(FileNotFoundError):
        build_session_stats(Path("runs") / "nope")


def test_main_renders_default_runs_dir(fake_fs: FakeFileSystem) -> None:
    """The CLI defaults to ``runs/bot`` when no argument is supplied."""
    configure_bot_runtime_logging("20260610-100000")
    emit_wire("shoot")
    argv_value = ["tankpit-stats"]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original_get_argv


def test_main_accepts_runs_dir_argument(fake_fs: FakeFileSystem) -> None:
    """The CLI sweeps the directory passed as the first argument."""
    configure_bot_runtime_logging("20260610-100000")
    emit_wire("shoot")
    argv_value = ["tankpit-stats", str(_RUNS_DIR)]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original_get_argv
