"""End-to-end tests for ``build_issue_report`` / ``main``.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_probe_runtime_logging` -> real
emit helpers (or :func:`tankpit_bot.runtime_logging.emit_diagnostic`
directly) -> real ``_HookEventArtifactHandler`` -> JSONL via
:class:`tests.conftest.FakeFileSystem` -> real
:func:`tankpit_bot.diagnostics.issue_report.build_issue_report` ->
assertions on the structured :class:`IssueReportDict`.

There is no monkeypatching of the emit helpers; nothing is mocked. The
JSONL the report consumes is the SAME bytes a production probe run
writes to ``runs/probe/latest.<name>.events.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks
from tankpit_bot.diagnostics.issue_report import build_issue_report
from tankpit_bot.diagnostics.issue_report_renderer import main, render_issue_report
from tankpit_bot.diagnostics.issue_report_types import SessionRoomRecordDict
from tankpit_bot.runtime_logging import (
    configure_probe_runtime_logging,
    emit_diagnostic,
    emit_wire,
    emit_wire_complete,
)


def _emit_session_room(room_id: str, field_image: str) -> None:
    """Emit a ``session_room_joined`` diagnostic through the real pipeline."""
    emit_diagnostic(
        diagnostic_kind="session_room_joined",
        room_id=room_id,
        field_image=field_image,
    )


def _emit_teleport_attempt(
    *,
    target_x: int,
    target_y: int,
    cycle_id: int,
    status: str,
    sent_window: str = "(none)",
    received_window: str = "(none)",
    page_snapshot_count: int = 0,
) -> None:
    """Emit one ``teleport_attempt`` diagnostic through the real pipeline."""
    emit_diagnostic(
        diagnostic_kind="teleport_attempt",
        target_x=target_x,
        target_y=target_y,
        teleport_cycle_id=cycle_id,
        status=status,
        sent_window=sent_window,
        received_window=received_window,
        page_snapshots="(none)",
        page_snapshot_count=page_snapshot_count,
    )


def _emit_fuel_target_selection(
    *,
    cycle_id: int,
    target_present: bool,
    target_x: int = -1,
    target_y: int = -1,
    summary: str = "fuel: total=0",
    decision_basis: str = "world_ts=0",
) -> None:
    """Emit one ``fuel_target_selection`` diagnostic through the real pipeline."""
    emit_diagnostic(
        diagnostic_kind="fuel_target_selection",
        radar_cycle_id=cycle_id,
        target_present=target_present,
        target_x=target_x,
        target_y=target_y,
        summary=summary,
        decision_basis=decision_basis,
        terrain_available=True,
        self_state_available=True,
    )


def test_build_issue_report_summarizes_clean_probe_run(fake_fs: FakeFileSystem) -> None:
    """A clean run with all teleports landing produces zero failure counts."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    _emit_teleport_attempt(target_x=131, target_y=110, cycle_id=1, status="landed_exact")
    _emit_teleport_attempt(target_x=147, target_y=110, cycle_id=2, status="landed_exact")
    _emit_fuel_target_selection(
        cycle_id=2, target_present=True, target_x=151, target_y=109, summary="fuel: ok"
    )
    emit_wire("map_open")
    emit_wire_complete(action_kind="map_open", duration_ms=850, signal="map_data_processed")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["mode"] == "probe:fuel"
    room = report["session_room"]
    assert room == SessionRoomRecordDict(
        room_id="1",
        field_image="field01.gif",
        timestamp=room["timestamp"] if room is not None else "",
    )
    assert report["teleport_success_count"] == 2
    assert report["teleport_failure_count"] == 0
    assert report["fuel_selected_count"] == 1
    assert report["fuel_rejected_count"] == 0
    assert report["map_open_dispatches"] == 1
    assert report["map_open_completions"] == 1
    assert len(report["wire_completes"]) == 1


def test_recovery_boxed_in_diagnostic_is_promoted_to_top_level_issue(
    fake_fs: FakeFileSystem,
) -> None:
    """A ``recovery_boxed_in`` diagnostic surfaces loudly in the summary.

    The boxed-in terminal action should be near-unreachable after the
    terrain-aware approach and capped search ring; any occurrence is a
    top-level issue so it cannot be silently absorbed by the fallback
    action.
    """
    from tankpit_bot.runtime_logging import configure_bot_runtime_logging

    artifacts = configure_bot_runtime_logging("20260610-120000")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(
        diagnostic_kind="recovery_boxed_in",
        behavior_mode="COLLECT_FUEL",
        fuel=140,
        self_x=100,
        self_y=100,
        patrol_waypoint_index=3,
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))
    rendered = render_issue_report(report)

    assert report["recovery_boxed_in_count"] == 1
    assert "recovery owner hit its boxed-in terminal action 1 time(s)" in rendered


def test_build_issue_report_counts_teleport_failures(fake_fs: FakeFileSystem) -> None:
    """Non-landed teleport statuses contribute to ``teleport_failure_count``."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    _emit_teleport_attempt(target_x=131, target_y=110, cycle_id=1, status="landed_exact")
    _emit_teleport_attempt(target_x=135, target_y=109, cycle_id=2, status="teleport_timeout")
    _emit_teleport_attempt(target_x=141, target_y=71, cycle_id=3, status="map_sync_timeout")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["teleport_success_count"] == 1
    assert report["teleport_failure_count"] == 2
    failure_statuses = [
        attempt["status"]
        for attempt in report["teleport_attempts"]
        if attempt["status"] not in ("landed_exact", "landed_inexact")
    ]
    assert failure_statuses == ["teleport_timeout", "map_sync_timeout"]


def test_build_issue_report_counts_fuel_rejections(fake_fs: FakeFileSystem) -> None:
    """Fuel target selections with ``target_present=False`` increment the rejection counter."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    _emit_fuel_target_selection(
        cycle_id=1, target_present=False, summary="fuel: nearest blocked_no_landing"
    )
    _emit_fuel_target_selection(
        cycle_id=2,
        target_present=True,
        target_x=151,
        target_y=109,
        summary="fuel: actionable",
    )
    _emit_fuel_target_selection(cycle_id=3, target_present=False, summary="fuel: none")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["fuel_selected_count"] == 1
    assert report["fuel_rejected_count"] == 2


def test_build_issue_report_records_map_open_skipped(fake_fs: FakeFileSystem) -> None:
    """``map_open_skipped_already_open`` events flow into the report's skipped list."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(
        diagnostic_kind="map_open_skipped_already_open",
        origin="acquisition_phase",
        command_name="map_open",
    )
    emit_diagnostic(
        diagnostic_kind="map_open_skipped_already_open",
        origin="executor.dispatch_command.teleport_precondition",
        command_name="map_open",
        teleport_target_x=131,
        teleport_target_y=110,
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert len(report["map_open_skipped"]) == 2
    origins = [s["origin"] for s in report["map_open_skipped"]]
    assert origins == [
        "acquisition_phase",
        "executor.dispatch_command.teleport_precondition",
    ]


def test_build_issue_report_handles_missing_session_room(fake_fs: FakeFileSystem) -> None:
    """A run with no ``session_room_joined`` diagnostic leaves ``session_room`` None."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_teleport_attempt(target_x=131, target_y=110, cycle_id=1, status="landed_exact")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    assert report["session_room"] is None


def test_build_issue_report_ignores_blank_lines_in_jsonl(fake_fs: FakeFileSystem) -> None:
    """Blank lines in the JSONL artifact are skipped without erroring."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    # Append a blank line; the loader must skip it without crashing.
    fake_fs.append_text(Path(artifacts["latest_events_path"]), "\n\n   \n")

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    room = report["session_room"]
    assert room == SessionRoomRecordDict(
        room_id="1",
        field_image="field01.gif",
        timestamp=room["timestamp"] if room is not None else "",
    )


def test_render_issue_report_lists_failure_target_and_summary_line(
    fake_fs: FakeFileSystem,
) -> None:
    """The rendered report names the failing teleport target and surfaces the issue."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    _emit_teleport_attempt(
        target_x=135,
        target_y=109,
        cycle_id=2,
        status="teleport_timeout",
        sent_window="55:[SENT] CMD map_open",
        received_window="57:[RECEIVED] MAP_DATA: len=867",
        page_snapshot_count=4,
    )

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "TANKPIT ISSUE REPORT" in rendered
    assert "id=1 field_image=field01.gif" in rendered
    assert "target=(135,109)" in rendered
    assert "teleport_timeout" in rendered
    assert "55:[SENT] CMD map_open" in rendered
    assert "1/1 teleports failed (100%)" in rendered


def test_render_issue_report_lists_top_level_no_issues(fake_fs: FakeFileSystem) -> None:
    """A perfectly clean run renders the ``no top-level issues detected`` sentence."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    _emit_teleport_attempt(target_x=131, target_y=110, cycle_id=1, status="landed_exact")
    _emit_fuel_target_selection(cycle_id=1, target_present=True, target_x=151, target_y=109)
    emit_wire("map_open")
    emit_wire_complete(action_kind="map_open", duration_ms=850, signal="map_data_processed")

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "(no top-level issues detected)" in rendered


def test_render_issue_report_calls_out_map_open_mismatch_only_for_bot_mode(
    fake_fs: FakeFileSystem,
) -> None:
    """A dispatch/completion gap raises a top-level issue only for ``bot`` mode.

    Action_lab probes do not flow through the HFSM completion gate, so
    every probe run has ``map_open_completions == 0``. The mismatch
    check is gated to mode == "bot" to avoid surfacing that gap as a
    false positive on probe artifacts; this test pins both the
    bot-mode (raises) and probe-mode (suppressed) behaviours.
    """
    from tankpit_bot.runtime_logging import configure_bot_runtime_logging

    bot_artifacts = configure_bot_runtime_logging("20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_wire("map_open")
    emit_wire("map_open")
    emit_wire_complete(action_kind="map_open", duration_ms=850, signal="map_data_processed")

    bot_rendered = render_issue_report(
        build_issue_report(Path(bot_artifacts["latest_events_path"]))
    )

    assert "map_open dispatch/completion mismatch" in bot_rendered
    assert "dispatched=2 vs completed=1" in bot_rendered

    probe_artifacts = configure_probe_runtime_logging("fuel", "20260331-230406")
    _emit_session_room("1", "field01.gif")
    emit_wire("map_open")
    emit_wire("map_open")
    emit_wire("map_open")

    probe_rendered = render_issue_report(
        build_issue_report(Path(probe_artifacts["latest_events_path"]))
    )

    assert "map_open dispatch/completion mismatch" not in probe_rendered


def test_render_issue_report_flags_unknown_session_room(fake_fs: FakeFileSystem) -> None:
    """A run without a session_room emit produces a top-level visibility issue."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_teleport_attempt(target_x=131, target_y=110, cycle_id=1, status="landed_exact")

    rendered = render_issue_report(build_issue_report(Path(artifacts["latest_events_path"])))

    assert "session room unknown" in rendered


def test_main_runs_report_against_provided_path(fake_fs: FakeFileSystem) -> None:
    """``main()`` reads from the path passed on argv and exits with 0.

    The injected argv mirrors the real ``sys.argv`` shape: ``argv[0]`` is
    the script name and the user-supplied path is at ``argv[1]``. ``main``
    strips ``argv[0]`` before resolving the source path so the helper
    receives only the user-facing arguments.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    argv_value = ["tankpit-issue-report", artifacts["latest_events_path"]]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: argv_value
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0


def test_main_defaults_to_bot_events_artifact_when_no_user_args(
    fake_fs: FakeFileSystem,
) -> None:
    """With only the script name on argv, ``main()`` reads the default bot path."""
    from tankpit_bot.runtime_logging import configure_bot_runtime_logging

    artifacts = configure_bot_runtime_logging("20260331-230405")
    _emit_session_room("1", "field01.gif")
    script_only: list[str] = ["tankpit-issue-report"]
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: script_only
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0
    assert artifacts["latest_events_path"].endswith("latest.events.jsonl")


def test_main_handles_empty_argv_via_test_hook(fake_fs: FakeFileSystem) -> None:
    """If the test hook returns ``[]``, ``main()`` still defaults to the bot path.

    The defensive ``full_argv[1:] if full_argv else []`` guard inside
    ``main()`` exists because the production ``sys.argv`` always has at
    least the script name, but the test hook lets us substitute any list.
    """
    from tankpit_bot.runtime_logging import configure_bot_runtime_logging

    configure_bot_runtime_logging("20260331-230405")
    _emit_session_room("1", "field01.gif")
    empty: list[str] = []
    original_get_argv = _test_hooks.get_argv
    _test_hooks.get_argv = lambda: empty
    try:
        exit_code = main()
    finally:
        _test_hooks.get_argv = original_get_argv

    assert exit_code == 0


def test_require_bool_field_raises_when_key_missing() -> None:
    """``_require_bool_field`` raises ``KeyError`` for an absent key.

    Exercises the missing-key branch directly so the strict accessor
    contract is asserted independently of the rest of the report
    pipeline.
    """
    from tankpit_bot.diagnostics.issue_report import _require_bool_field

    fields: dict[str, str | int | float | bool] = {}

    with pytest.raises(KeyError, match="target_present"):
        _require_bool_field(fields, "target_present")


def test_require_bool_field_rejects_non_bool_value() -> None:
    """``_require_bool_field`` raises ``TypeError`` when the value is not a bool."""
    from tankpit_bot.diagnostics.issue_report import _require_bool_field

    fields: dict[str, str | int | float | bool] = {"target_present": "true"}

    with pytest.raises(TypeError, match="must be bool"):
        _require_bool_field(fields, "target_present")


def test_classify_diagnostic_record_ignores_non_string_kind(
    fake_fs: FakeFileSystem,
) -> None:
    """A DIAGNOSTIC event whose ``diagnostic_kind`` is not a string is dropped silently.

    The emit pipeline always sets ``diagnostic_kind`` as a string -- the
    function signature requires it -- but a hand-edited or partially
    corrupt artifact could violate that contract. The router skips the
    record rather than raising so the rest of the report still renders,
    and this test pins the behaviour by writing the malformed JSONL
    line directly through :class:`FakeFileSystem`.
    """
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    # Append a hand-crafted DIAGNOSTIC event whose diagnostic_kind is a
    # number rather than a string. The decoder accepts the line (every
    # value is a primitive) but the classifier must skip it.
    fake_fs.append_text(
        Path(artifacts["latest_events_path"]),
        '{"timestamp":"2026-06-07T22:12:30","level":"INFO",'
        '"logger":"x","mode":"probe:fuel","channel":"DIAGNOSTIC",'
        '"message":"diagnostic_kind=42","diagnostic_kind":42}\n',
    )

    report = build_issue_report(Path(artifacts["latest_events_path"]))

    # Nothing was added to any DIAGNOSTIC bucket beyond the legitimate
    # session_room_joined emit above.
    assert report["teleport_attempts"] == []
    assert report["map_open_skipped"] == []
    assert report["fuel_target_selections"] == []


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
    from tankpit_bot.runtime_logging import emit_ai, emit_state, emit_sync, emit_world

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

    from tankpit_bot.diagnostics.issue_report_codecs import _require_object_list

    payload: JSONObject = {"items": [{"k": "v"}, "not an object"]}

    with pytest.raises(JSONTypeError, match=r"items\[1\] must be object"):
        _require_object_list(payload, "items")


def test_scorecard_counts_tank_deactivated_kills(fake_fs: FakeFileSystem) -> None:
    """A ``tank_deactivated`` DIAGNOSTIC event increments the scorecard ``kills`` counter."""
    artifacts = configure_probe_runtime_logging("fuel", "20260331-230405")
    _emit_session_room("1", "field01.gif")
    emit_diagnostic(diagnostic_kind="tank_deactivated", target_id=42)
    emit_diagnostic(diagnostic_kind="tank_deactivated", target_id=99)

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
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=300)
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
    emit_diagnostic(diagnostic_kind="self_alignment_sample", belief_fuel=300)
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
