"""Tests for the wire-vs-ledger completion audit and dispatch-aware streaks.

The 2026-08-21 lesson: shoot ran 13 wire dispatches with ZERO recorded
completions for a whole session class, and every outcome-derived rule
confidently misread the silence (the liveness detector's first live
catch was a catch of itself). These tests pin the two instruments that
make that shape self-reporting: the completion-audit top-level rule
(dispatches with no completions = ledger modeling gap) and the streak
scan's dispatch awareness (a dispatched supersede is a re-aim, never a
zero-dispatch replan).
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem
from tests.diagnostics._issue_report_fixtures import _emit_session_room

from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.issue_report_renderer import render_issue_report
from tankpit_bot.runtime_logging import (
    configure_probe_runtime_logging,
    emit_diagnostic,
    emit_wire,
)


def test_dispatches_without_completions_flag_a_modeling_gap(
    fake_fs: FakeFileSystem,
) -> None:
    """A kind with wire traffic and zero completions is a top-level issue.

    Retro-shape of the soak false positive: shoot dispatches echo on
    the wire while every ledger row for the kind reads ``superseded``.
    The audit must name the kind and warn that its outcome labels are
    untrustworthy.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(5):
        emit_wire("shoot(%d,%d,id=0)", 220 + index, 170, action_kind="shoot")
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="superseded",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=0,
            dispatched=True,
        )

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    rendered = render_issue_report(report)

    assert report["wire_dispatches_by_kind"] == {"shoot": 5}
    assert "ledger modeling gap: 5 shoot commands reached the wire" in rendered
    assert "ZERO completions" in rendered
    # The dispatched supersedes must NOT read as a liveness stall.
    assert "liveness stall" not in rendered


def test_a_single_completion_clears_the_kind(fake_fs: FakeFileSystem) -> None:
    """One wire-confirmed resolution proves the completion path works."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(6):
        emit_wire("shoot(%d,%d,id=0)", 220 + index, 170, action_kind="shoot")
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind="shoot",
        outcome="fired",
        event_id=1,
        attempt_id=1,
        duration_ms=2100,
    )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "ledger modeling gap" not in rendered


def test_below_the_dispatch_floor_the_audit_stays_silent(
    fake_fs: FakeFileSystem,
) -> None:
    """A handful of dispatches with no completion is teardown, not a gap.

    The last in-flight action of a session legitimately never resolves
    (the soak's 178th pan); the floor keeps that truncation tail from
    impersonating a modeling gap.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for _ in range(4):
        emit_wire("scope(5)", action_kind="scope")

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "ledger modeling gap" not in rendered


def test_stall_timeouts_do_not_count_as_completions(fake_fs: FakeFileSystem) -> None:
    """A kind whose every dispatch stalls still has no working completion path."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(5):
        emit_wire("radar", action_kind="radar")
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="scan",
            outcome="stall_timeout",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=10000,
        )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "ledger modeling gap: 5 scan commands reached the wire" in rendered


def test_dispatched_supersedes_break_the_streak_scan(fake_fs: FakeFileSystem) -> None:
    """The streak scan skips supersedes whose decision reached the wire.

    Twelve dispatched supersedes are twelve re-aims over live commands
    — the exact shape of the soak's false alarm — and must not render
    a liveness-stall line. Twelve undispatched ones still must.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(12):
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="superseded",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=0,
            dispatched=True,
        )
    for index in range(12):
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="collect",
            outcome="superseded",
            event_id=50 + index,
            attempt_id=index + 1,
            duration_ms=0,
            dispatched=False,
        )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "consecutive shoot decisions" not in rendered
    assert "liveness stall: 12 consecutive collect decisions" in rendered


def test_clearance_shots_do_not_count_toward_combat_futility(
    fake_fs: FakeFileSystem,
) -> None:
    """A gatherer's ground shots are collect doctrine, not futile combat.

    The first live run after the echo receipt (2026-08-21): 53
    clearance shots, 0 kills, and the unsplit rule read it as "combat
    futility" — a working gatherer flagged as a broken fighter. With
    ``shoot:fired`` distinguishable, the rule counts only
    tank-targeted shots.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(25):
        emit_wire("shoot(%d,%d,id=0)", 200 + index, 170, action_kind="shoot")
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="fired",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=2040,
        )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "combat futility" not in rendered


def test_mixed_shots_report_only_the_tank_targeted_count(
    fake_fs: FakeFileSystem,
) -> None:
    """Futility names the tank-targeted count and the excluded clearance."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(25):
        emit_wire("shoot(%d,%d,id=11)", 200 + index, 170, action_kind="shoot")
    for index in range(5):
        emit_wire("shoot(%d,%d,id=0)", 100 + index, 90, action_kind="shoot")
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="fired",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=2040,
        )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert (
        "combat futility: 25 tank-targeted shots produced 0 observed kills "
        "(5 clearance shots excluded)" in rendered
    )


def test_outcome_section_renders_the_wire_tally_and_dispatch_marks(
    fake_fs: FakeFileSystem,
) -> None:
    """The outcomes section opens with the per-kind tally and marks supersedes."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_wire("shoot(220,170,id=0)", action_kind="shoot")
    emit_wire("radar", action_kind="radar")
    # Fire-and-forget sends have no ledger contract and stay untallied.
    emit_wire("chat(3)", action_kind="chat")
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind="scan",
        outcome="radar_complete",
        event_id=1,
        attempt_id=1,
        duration_ms=1900,
    )
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind="shoot",
        outcome="superseded",
        event_id=2,
        attempt_id=1,
        duration_ms=0,
        dispatched=True,
    )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "wire dispatches/ledger completions: scan=1/1 shoot=1/0" in rendered
    assert "chat" not in rendered
    assert "shoot#1 outcome=superseded duration_ms=0 dispatched=True" in rendered
    assert "scan#1 outcome=radar_complete duration_ms=1900" in rendered
    assert "radar_complete duration_ms=1900 dispatched" not in rendered
