"""End-to-end tests for the run-audit CLI and report builder.

Events flow through the REAL runtime-logging pipeline into the fake
file system, exactly as a live run writes them; the builder then reads
the same artifacts back. Nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.run_audit import build_run_audit, capture_path_for, main
from tankpit_bot.diagnostics.run_audit_types import make_finding
from tankpit_bot.runtime_logging import (
    configure_bot_runtime_logging,
    emit_diagnostic,
)

_LATEST_EVENTS = Path("runs") / "bot" / "latest.events.jsonl"


def test_capture_path_for_swaps_the_canonical_suffix() -> None:
    """*.events.jsonl maps to the sibling *.capture_session.json."""
    events = Path("runs") / "bot" / "bot-20260719-004608.events.jsonl"
    assert capture_path_for(events) == (
        Path("runs") / "bot" / "bot-20260719-004608.capture_session.json"
    )


def test_capture_path_for_appends_when_suffix_is_noncanonical() -> None:
    """A non-canonical events name still resolves to a sibling capture path."""
    assert capture_path_for(Path("odd.jsonl")) == Path("odd.jsonl.capture_session.json")


def test_build_run_audit_flags_missing_capture(fake_fs: FakeFileSystem) -> None:
    """Ledger findings plus the capture_missing warning when no capture exists."""
    configure_bot_runtime_logging("20260719-120000")
    emit_diagnostic(
        diagnostic_kind="session_scorecard",
        exit_reason="completed",
        ticks=10,
        kills=1,
    )

    report = build_run_audit(_LATEST_EVENTS)

    assert report["events_path"] == str(_LATEST_EVENTS)
    assert report["capture_path"] == str(Path("runs") / "bot" / "latest.capture_session.json")
    assert report["findings"] == [
        make_finding(
            "capture_missing",
            "warning",
            "no capture artifact beside the events file -- replay audit skipped",
        ),
        make_finding(
            "session_exit",
            "info",
            "session ended: completed",
            exit_reason="completed",
            ticks=10,
            kills=1,
        ),
    ]
    assert report["critical_count"] == 0
    assert report["warning_count"] == 1
    assert report["info_count"] == 1


def test_build_run_audit_reads_the_sibling_capture(fake_fs: FakeFileSystem) -> None:
    """A present capture artifact is decoded and replay-audited."""
    configure_bot_runtime_logging("20260719-120000")
    emit_diagnostic(
        diagnostic_kind="session_scorecard",
        exit_reason="completed",
        ticks=10,
        kills=0,
    )
    fake_fs.write_text(
        Path("runs") / "bot" / "latest.capture_session.json",
        '{"session_id": "s", "start_timestamp_ms": 0, "end_timestamp_ms": 1,'
        ' "base_url": "https://test", "messages": [], "magic": null,'
        ' "game_log": [], "tank_names": {}}',
    )

    report = build_run_audit(_LATEST_EVENTS)

    assert report["findings"] == [
        make_finding(
            "capture_unreadable",
            "warning",
            "capture carries no XOR magic -- replay audit skipped",
        ),
        make_finding(
            "session_exit",
            "info",
            "session ended: completed",
            exit_reason="completed",
            ticks=10,
            kills=0,
        ),
    ]


def test_build_run_audit_empty_artifact(fake_fs: FakeFileSystem) -> None:
    """A configured run that emitted nothing audits as an empty run."""
    configure_bot_runtime_logging("20260719-120000")

    report = build_run_audit(_LATEST_EVENTS)

    assert report["findings"] == [
        make_finding(
            "empty_run",
            "critical",
            "the events artifact contains no records -- the session "
            "died before the game loop produced anything",
        ),
        make_finding(
            "capture_missing",
            "warning",
            "no capture artifact beside the events file -- replay audit skipped",
        ),
    ]
    assert report["critical_count"] == 1


def test_main_audits_the_default_artifact(fake_fs: FakeFileSystem) -> None:
    """The CLI defaults to runs/bot/latest.events.jsonl and exits 0."""
    configure_bot_runtime_logging("20260719-120000")
    emit_diagnostic(
        diagnostic_kind="session_scorecard",
        exit_reason="completed",
        ticks=1,
        kills=0,
    )
    argv_value = ["tankpit-run-audit"]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        assert main() == 0
    finally:
        _test_hooks.get_argv = original_get_argv
