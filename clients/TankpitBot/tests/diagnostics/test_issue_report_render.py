"""Tests for issue-report rendering and issue derivation."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FakeFileSystem
from tests.diagnostics._issue_report_fixtures import (
    _emit_fuel_target_selection,
    _emit_session_room,
)

from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.issue_report_renderer import (
    render_issue_report,
)
from tankpit_bot.runtime_logging import (
    configure_probe_runtime_logging,
    emit_diagnostic,
    emit_wire,
)


def test_render_issue_report_lists_map_open_skipped_origin(
    fake_fs: FakeFileSystem,
) -> None:
    """Already-open events are tallied per origin, not listed one per row.

    The already-open case is the normal path -- it fires on essentially
    every successful teleport -- so a row per event read as a list of
    failures and buried the rarer origin. Two events from one origin must
    collapse to a single counted line, and a second origin must still get
    its own.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for _ in range(2):
        emit_diagnostic(
            diagnostic_kind="map_open_skipped_already_open",
            origin="acquisition_phase",
            command_name="map_open",
        )
    emit_diagnostic(
        diagnostic_kind="map_open_skipped_already_open",
        origin="executor.dispatch_command.map_open",
        command_name="map_open",
    )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "already_open=3" in rendered
    assert "already open, no open sent: 2x from acquisition_phase" in rendered
    assert "already open, no open sent: 1x from executor.dispatch_command.map_open" in rendered
    assert "[SKIP]" not in rendered


def test_render_issue_report_lists_fuel_rejection_summary(
    fake_fs: FakeFileSystem,
) -> None:
    """A fuel-rejection majority shows up in the top-level summary."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    _emit_fuel_target_selection(cycle_id=1, target_present=False, summary="fuel: blocked")
    _emit_fuel_target_selection(cycle_id=2, target_present=False, summary="fuel: blocked")

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "2/2 fuel cycles had no actionable target (100%)" in rendered


def test_build_issue_report_ignores_unknown_diagnostic_kind(
    fake_fs: FakeFileSystem,
) -> None:
    """A DIAGNOSTIC event with a kind the report does not understand is dropped."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(
        diagnostic_kind="movement_probe_map_already_showing",
        target_x=120,
        target_y=130,
        map_open_delay_ms=150,
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["teleport_attempts"] == []
    assert report["map_open_skipped"] == []
    assert report["fuel_target_selections"] == []


def test_build_issue_report_passes_through_unrelated_channels(
    fake_fs: FakeFileSystem,
) -> None:
    """Records on non-routed channels (STATE/AI/SYNC/WORLD) do not affect counts."""
    from tankpit_bot.runtime_logging import (
        emit_ai,
        emit_state,
        emit_sync,
        emit_world,
    )

    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_state("IDLE")
    emit_ai("decision text")
    emit_sync("waiting on something")
    emit_world("fuel changed")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["teleport_attempts"] == []
    assert report["map_open_skipped"] == []
    assert report["fuel_target_selections"] == []
    assert report["map_open_dispatches"] == 0
    assert report["map_open_completions"] == 0


def test_build_issue_report_handles_empty_mode_record(fake_fs: FakeFileSystem) -> None:
    """A record whose ``mode`` is an empty string leaves the accumulator mode unchanged."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    # Append a hand-crafted line whose mode is the empty string.
    fake_fs.append_text(
        Path(artifacts["latest_events_path"]),
        '{"timestamp":"2026-06-07T22:12:30","level":"INFO","logger":"x",'
        '"mode":"","channel":"STATE","message":"IDLE"}\n',
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    # The empty-mode line did not overwrite the legitimate probe:fuel mode.
    assert report["mode"] == "probe:fuel"


def test_require_object_list_rejects_non_object_element() -> None:
    """``_require_object_list`` raises ``JSONTypeError`` for a non-object element."""
    from platform_core.json_utils import JSONObject, JSONTypeError

    from tankpit_bot.diagnostics.issue_report_codecs_records import _require_object_list

    payload: JSONObject = {"items": [{"k": "v"}, "not an object"]}

    with pytest.raises(JSONTypeError, match=r"items\[1\] must be object"):
        _require_object_list(payload, "items")


def test_scorecard_counts_tank_deactivated_kills(fake_fs: FakeFileSystem) -> None:
    """A ``tank_deactivated`` counts iff its killer is our own tank.

    Since the DOM game-log kill channel was retired (2026-07-19), the
    wire ``0x41 Deactivation`` is the single emitter -- but the stream
    carries EVERY deactivation in view, so the raw count stopped being
    the kill count the day two bots shared a room (2026-08-20 gatherer
    run: arterial's report read kills=2 with shots=0, both 0x41s
    naming its sibling). A repeated ``victim_id`` with our killer id
    is a respawned tank killed again and must count again (June
    capture bot-20260610-011333: victim 507 legitimately killed five
    times in one 45-minute session); a sibling's kill never counts.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(diagnostic_kind="tank_identity", tank_id=601)
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=42, killer_id=601)
    # Respawn re-kill of the same victim by us: counts again.
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=42, killer_id=601)
    # A fleet sibling's kill in the same room: never ours.
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=99, killer_id=1301)

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["scorecard"]["kills"] == 2


def test_scorecard_counts_combat_gate_diagnostics(fake_fs: FakeFileSystem) -> None:
    """Combat-gate DIAGNOSTIC events each feed their dedicated scorecard counter.

    Locks the wiring added 2026-06-19 alongside the freshness refactor.
    Without these counters the issue report ignores ``combat_miss``,
    ``combat_ghost_detected``, ``combat_stale_position``, and
    ``tank_damage_changed`` -- regressing this test would re-blind the
    report.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(
        diagnostic_kind="combat_miss",
        target_name="orange-8",
        target_id=534,
    )
    emit_diagnostic(
        diagnostic_kind="combat_miss",
        target_name="red-3",
        target_id=211,
    )
    emit_diagnostic(
        diagnostic_kind="combat_ghost_detected",
        target_name="purple-9",
        target_id=517,
        wire_age_ms=60000,
    )
    emit_diagnostic(
        diagnostic_kind="combat_stale_position",
        target_name="orange-8",
        target_id=534,
        position_age_ms=5000,
    )
    emit_diagnostic(
        diagnostic_kind="tank_damage_changed",
        tank_id=534,
        tank_name="orange-8",
        previous_damage_state=0,
        damage_state=1,
    )
    emit_diagnostic(
        diagnostic_kind="tank_damage_changed",
        tank_id=211,
        tank_name="red-3",
        previous_damage_state=1,
        damage_state=2,
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["scorecard"]["combat_misses"] == 2
    assert report["scorecard"]["combat_ghosts_blocked"] == 1
    assert report["scorecard"]["combat_stale_positions_blocked"] == 1
    assert report["scorecard"]["tank_damage_changes"] == 2


def test_scorecard_tracks_self_alignment_fuel_samples(fake_fs: FakeFileSystem) -> None:
    """``self_alignment_sample`` DIAGNOSTIC events feed fuel_min and fuel_last."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=150)
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=150)
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=200)

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["scorecard"]["fuel_min"] == 150
    assert report["scorecard"]["fuel_last"] == 200
    assert report["scorecard"]["fuel_sample_count"] == 3


def test_scorecard_counts_shoot_wire_events(fake_fs: FakeFileSystem) -> None:
    """WIRE events whose message starts with ``shoot(`` increment the shots counter."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_wire("shoot(dual)")
    emit_wire("shoot(missile)")
    emit_wire("shoot(homing)")
    # A non-shoot wire should NOT count.
    emit_wire("move(north)")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["scorecard"]["shots"] == 3


def test_scorecard_builds_state_budget_from_transitions(fake_fs: FakeFileSystem) -> None:
    """STATE channel transitions produce a sorted state budget in the scorecard.

    Hand-crafted JSONL lines with known timestamps are injected so the
    budget seconds are deterministic, not wall-clock-dependent.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")

    events_path = Path(artifacts["latest_events_path"])
    # Inject STATE transitions with known timestamps:
    #   T+0  IDLE -> COMBAT     (COMBAT for 5s)
    #   T+5  COMBAT -> MOVING   (MOVING for 3s)
    #   T+8  MOVING -> IDLE     (IDLE has no trailing interval)
    for ts, msg in [
        ("2026-06-07T22:12:00", "IDLE -> COMBAT"),
        ("2026-06-07T22:12:05", "COMBAT -> MOVING"),
        ("2026-06-07T22:12:08", "MOVING -> IDLE"),
    ]:
        fake_fs.append_text(
            events_path,
            '{"timestamp":"' + ts + '","level":"INFO","logger":"x",'
            '"mode":"probe:fuel","channel":"STATE","message":"' + msg + '"}\n',
        )

    report = build_issue_report(events_path)

    budget = report["scorecard"]["state_budget"]
    budget_map = {entry["state"]: entry["seconds"] for entry in budget}
    assert budget_map["COMBAT"] == 5
    assert budget_map["MOVING"] == 3
    # Sorted by descending seconds: COMBAT (5) before MOVING (3).
    assert budget[0]["state"] == "COMBAT"
    assert budget[1]["state"] == "MOVING"


def test_render_scorecard_section_shows_state_budget_lines(fake_fs: FakeFileSystem) -> None:
    """The rendered scorecard section includes per-state budget lines."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")

    events_path = Path(artifacts["latest_events_path"])
    for ts, msg in [
        ("2026-06-07T22:12:00", "IDLE -> COMBAT"),
        ("2026-06-07T22:12:10", "COMBAT -> IDLE"),
    ]:
        fake_fs.append_text(
            events_path,
            '{"timestamp":"' + ts + '","level":"INFO","logger":"x",'
            '"mode":"probe:fuel","channel":"STATE","message":"' + msg + '"}\n',
        )

    rendered = render_issue_report(build_issue_report(events_path))

    assert "COMBAT: 10s" in rendered


def test_scorecard_issue_fuel_floor_critical(fake_fs: FakeFileSystem) -> None:
    """Fuel dipping below 100 surfaces a fuel floor critical top-level issue."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=150)
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=50)

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    rendered = render_issue_report(report)

    assert report["scorecard"]["fuel_min"] == 50
    assert "fuel floor critical" in rendered
    assert "dipped to 50" in rendered


def test_scorecard_issue_combat_futility(fake_fs: FakeFileSystem) -> None:
    """20+ shots with 0 kills surfaces a combat futility top-level issue."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for _ in range(20):
        emit_wire("shoot(dual)")

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    rendered = render_issue_report(report)

    assert report["scorecard"]["shots"] == 20
    assert report["scorecard"]["kills"] == 0
    assert "combat futility" in rendered
    assert "20 shots produced 0 observed kills" in rendered


def test_suppressed_dispatch_streak_is_a_top_level_issue(fake_fs: FakeFileSystem) -> None:
    """Re-selecting a belief-refuted action surfaces as a top-level issue.

    Regression guard for the 2026-08-20 gatherer livelock's
    invisibility: arterial's report showed 93 suppressed pickups as 116
    zero-duration ``superseded`` collects and closed with "(no
    top-level issues detected)". Three same-target suppressions mean
    the planner was told "this cannot transfer" twice and selected the
    identical action anyway; a single suppression is the refusal
    prediction working and must NOT flag.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for _ in range(3):
        emit_diagnostic(
            diagnostic_kind="dispatch_suppressed",
            origin="executor.dispatch_command.refusal_prediction",
            command_name="pickup_equipment",
            target_x=133,
            target_y=129,
            predicted_error_code=7,
        )
    emit_diagnostic(
        diagnostic_kind="dispatch_suppressed",
        origin="executor.dispatch_command.refusal_prediction",
        command_name="pickup_fuel",
        target_x=90,
        target_y=91,
        predicted_error_code=5,
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    rendered = render_issue_report(report)

    assert report["suppressed_dispatches"][0]["count"] == 3
    assert report["suppressed_dispatches"][0]["command_name"] == "pickup_equipment"
    assert report["suppressed_dispatches"][1]["count"] == 1
    assert "SUPPRESSED DISPATCHES" in rendered
    assert "pickup_equipment to (133,129) x3 (predicted 0x52 code 7)" in rendered
    assert "planner re-selected a belief-refuted pickup_equipment to (133,129) 3x" in rendered
    # The single fuel suppression renders in the section but is not an issue.
    assert "pickup_fuel to (90,91) x1" in rendered
    assert "belief-refuted pickup_fuel" not in rendered


def test_zero_dispatch_streak_is_a_liveness_stall_issue(fake_fs: FakeFileSystem) -> None:
    """A long same-kind zero-dispatch replan streak flags, healthy churn doesn't.

    The veto-agnostic sibling of the suppressed-dispatch rule: catches
    the planner/veto gap class from the ``action_outcome`` stream
    alone, so artifacts from builds without the ``liveness_stall``
    diagnostic still surface it. Thresholds are empirical (459-run
    archive sweep 2026-08-20): healthy ceiling 7 (combat re-aiming),
    the one recorded livelock ran 93.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(12):
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="collect",
            outcome="superseded",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=0,
        )
    # A healthy-ceiling shoot streak must NOT flag.
    for index in range(7):
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="shoot",
            outcome="superseded",
            event_id=100 + index,
            attempt_id=index + 1,
            duration_ms=0,
        )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "liveness stall: 12 consecutive collect decisions" in rendered
    assert "consecutive shoot decisions" not in rendered


def test_zero_dispatch_streak_broken_by_a_real_outcome_does_not_flag(
    fake_fs: FakeFileSystem,
) -> None:
    """A genuine resolution mid-run splits the streak below the bar."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for index in range(11):
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="collect",
            outcome="superseded",
            event_id=index + 1,
            attempt_id=index + 1,
            duration_ms=0,
        )
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind="collect",
        outcome="container_consumed",
        event_id=50,
        attempt_id=12,
        duration_ms=1500,
    )
    for index in range(11):
        emit_diagnostic(
            diagnostic_kind="action_outcome",
            action_kind="collect",
            outcome="superseded",
            event_id=60 + index,
            attempt_id=13 + index,
            duration_ms=0,
        )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "liveness stall" not in rendered


def test_stall_timeouts_surface_as_a_top_level_issue(fake_fs: FakeFileSystem) -> None:
    """Self-healed stalls are visible, not buried in outcome counts.

    The 2026-08-20 lesson: the scope-pending radar drop hid in
    ``stall_timeout`` counts for 19 days because every stall
    recovered. Any stall now gets a top-level line with the per-kind
    breakdown — the report must surface what limps, not only what
    breaks.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind="scan",
        outcome="stall_timeout",
        event_id=1,
        attempt_id=1,
        duration_ms=12052,
    )
    emit_diagnostic(
        diagnostic_kind="action_outcome",
        action_kind="scope",
        outcome="stall_timeout",
        event_id=2,
        attempt_id=1,
        duration_ms=10322,
    )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "2 action(s) stalled to timeout and replanned" in rendered
    assert "scan=1 scope=1" in rendered


def test_displacement_orbit_is_a_top_level_issue(fake_fs: FakeFileSystem) -> None:
    """Three bounces at one destination flag; two do not.

    The third liveness flavor: a displaced teleport resolves as a
    SUCCESS, so destination repetition hides from every failure
    counter (the 08-05 ancestor ran 534 bounces at one tile with a
    clean report). Thresholds are empirical: the 11 pathological
    archive runs all repeat >= 3; healthy combat re-aims repeat at
    most twice.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    for _ in range(3):
        emit_diagnostic(
            diagnostic_kind="teleport_displacement",
            requested_x=128,
            requested_y=238,
            landed_x=121,
            landed_y=230,
            displacement=15,
        )
    for _ in range(2):
        emit_diagnostic(
            diagnostic_kind="teleport_displacement",
            requested_x=82,
            requested_y=192,
            landed_x=78,
            landed_y=188,
            displacement=8,
        )

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    rendered = render_issue_report(report)

    assert report["displaced_teleports"][0]["count"] == 3
    assert report["displaced_teleports"][0]["max_displacement"] == 15
    assert "displacement orbit: 3 teleports at (128,238) all refused" in rendered
    assert "(82,192)" not in rendered.split("TOP-LEVEL")[1]
