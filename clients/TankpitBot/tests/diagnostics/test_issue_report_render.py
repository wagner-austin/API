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
    """Each ``map_open_skipped_already_open`` entry appears in the rendered output."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(
        diagnostic_kind="map_open_skipped_already_open",
        origin="acquisition_phase",
        command_name="map_open",
    )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "[SKIP] origin=acquisition_phase" in rendered


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
    """Every ``tank_deactivated`` event counts as one kill.

    Since the DOM game-log kill channel was retired (2026-07-19), the
    wire ``0x41 Deactivation`` is the single emitter -- exactly one
    event per kill. A repeated ``victim_id`` is a respawned tank
    killed again and must count again (June capture
    bot-20260610-011333: victim 507 legitimately killed five times in
    one 45-minute session).
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=42)
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=42)  # respawn re-kill
    emit_diagnostic(diagnostic_kind="tank_deactivated", victim_id=99)

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["scorecard"]["kills"] == 3


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
