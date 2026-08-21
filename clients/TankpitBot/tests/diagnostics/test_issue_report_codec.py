"""Round-trip tests for the whole-report issue codec.

Split from ``test_issue_report_types.py`` (2026-08-20, at the
file-size bar): record-level and scorecard-level codec tests stay
there; the four whole-``IssueReportDict`` tests live here.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
)
from tests.diagnostics._issue_report_fixtures import _round_trip

from tankpit_bot.diagnostics.issue_report_codecs import (
    decode_issue_report,
    encode_issue_report,
)
from tankpit_bot.diagnostics.issue_report_types import (
    ActionOutcomeRowDict,
    DisplacedTeleportRecordDict,
    FuelTargetSelectionRecordDict,
    IssueReportDict,
    MapOpenSkippedRecordDict,
    SessionRoomRecordDict,
    SuppressedDispatchRecordDict,
    TeleportAttemptRecordDict,
)


def test_issue_report_round_trip_with_session_room_present() -> None:
    """``IssueReportDict`` round-trips with a populated ``session_room`` field."""
    from tankpit_bot.diagnostics.issue_report_types import (
        SessionScorecardDict,
        make_unsampled_inventory_counts,
        make_zero_inventory_counts,
    )

    report = IssueReportDict(
        source_path="runs/probe/latest.fuel.events.jsonl",
        mode="probe:fuel",
        event_count=132,
        session_room=SessionRoomRecordDict(
            room_id="1",
            field_image="field01.gif",
            timestamp="2026-06-07T22:12:00",
        ),
        teleport_attempts=[
            TeleportAttemptRecordDict(
                target_x=131,
                target_y=110,
                teleport_cycle_id=1,
                status="landed_exact",
                timestamp="2026-06-07T22:12:30",
                sent_window="55:[SENT] CMD",
                received_window="57:[RECEIVED] MAP_DATA",
                page_snapshot_count=4,
            )
        ],
        map_open_skipped=[
            MapOpenSkippedRecordDict(origin="acquisition_phase", timestamp="2026-06-07T22:12:31")
        ],
        fuel_target_selections=[
            FuelTargetSelectionRecordDict(
                radar_cycle_id=2,
                target_present=True,
                target_x=151,
                target_y=109,
                summary="fuel: total=7 nearby=4 actionable=3",
                decision_basis="world_ts=1780821408807",
                timestamp="2026-06-07T22:12:32",
            )
        ],
        action_outcomes=[
            ActionOutcomeRowDict(
                action_kind="map_open",
                outcome="map_data_processed",
                event_id=7,
                attempt_id=3,
                duration_ms=850,
                timestamp="2026-06-07T22:12:33",
            )
        ],
        teleport_success_count=1,
        teleport_failure_count=0,
        fuel_selected_count=1,
        fuel_rejected_count=0,
        map_open_dispatches=1,
        map_open_completions=1,
        suppressed_dispatches=[
            SuppressedDispatchRecordDict(
                command_name="pickup_equipment",
                target_x=133,
                target_y=129,
                predicted_error_code=7,
                count=93,
            )
        ],
        displaced_teleports=[
            DisplacedTeleportRecordDict(
                requested_x=128,
                requested_y=238,
                count=4,
                max_displacement=15,
            )
        ],
        scorecard=SessionScorecardDict(
            duration_seconds=0,
            state_budget=[],
            kills=0,
            shots=0,
            combat_misses=0,
            combat_ghosts_blocked=0,
            combat_stale_positions_blocked=0,
            tank_damage_changes=0,
            fuel_min=-1,
            fuel_last=-1,
            fuel_sample_count=0,
            inventory_first=make_unsampled_inventory_counts(),
            inventory_last=make_unsampled_inventory_counts(),
            inventory_sample_count=0,
            equipment_gain_events=0,
            equipment_gained=make_zero_inventory_counts(),
            scans_extra=0,
            scans_builtin=0,
            physics_divergences=0,
            equipment_approaches=[],
            equipment_approach_distinct_targets=0,
            equipment_approach_max_repeats=0,
            action_outcome_counts={},
            fuel_low_water_threshold=100,
            fuel_low_water_episodes=[],
            teleport_spend=[],
            teleport_spend_total=0,
            ledger_teleport_spend_min=-1,
            ledger_teleport_spend_max=-1,
            ledger_shot_singles=-1,
            ledger_shot_duals=-1,
            ledger_shot_homings=-1,
            career_destroyed_last=-1,
            career_deactivated_last=-1,
            career_score_last=-1,
            career_playtime_seconds_last=-1,
            container_pickups_full=0,
            container_pickups_partial=0,
        ),
    )

    decoded = decode_issue_report(_round_trip(encode_issue_report(report)))

    assert decoded == report


def test_issue_report_round_trip_with_no_session_room() -> None:
    """``IssueReportDict`` round-trips with ``session_room`` set to None."""
    from tankpit_bot.diagnostics.issue_report_types import (
        SessionScorecardDict,
        make_unsampled_inventory_counts,
        make_zero_inventory_counts,
    )

    report = IssueReportDict(
        source_path="runs/probe/latest.fuel.events.jsonl",
        mode="probe:fuel",
        event_count=0,
        session_room=None,
        teleport_attempts=[],
        map_open_skipped=[],
        fuel_target_selections=[],
        action_outcomes=[],
        teleport_success_count=0,
        teleport_failure_count=0,
        fuel_selected_count=0,
        fuel_rejected_count=0,
        map_open_dispatches=0,
        map_open_completions=0,
        suppressed_dispatches=[],
        displaced_teleports=[],
        scorecard=SessionScorecardDict(
            duration_seconds=0,
            state_budget=[],
            kills=0,
            shots=0,
            combat_misses=0,
            combat_ghosts_blocked=0,
            combat_stale_positions_blocked=0,
            tank_damage_changes=0,
            fuel_min=-1,
            fuel_last=-1,
            fuel_sample_count=0,
            inventory_first=make_unsampled_inventory_counts(),
            inventory_last=make_unsampled_inventory_counts(),
            inventory_sample_count=0,
            equipment_gain_events=0,
            equipment_gained=make_zero_inventory_counts(),
            scans_extra=0,
            scans_builtin=0,
            physics_divergences=0,
            equipment_approaches=[],
            equipment_approach_distinct_targets=0,
            equipment_approach_max_repeats=0,
            action_outcome_counts={},
            fuel_low_water_threshold=100,
            fuel_low_water_episodes=[],
            teleport_spend=[],
            teleport_spend_total=0,
            ledger_teleport_spend_min=-1,
            ledger_teleport_spend_max=-1,
            ledger_shot_singles=-1,
            ledger_shot_duals=-1,
            ledger_shot_homings=-1,
            career_destroyed_last=-1,
            career_deactivated_last=-1,
            career_score_last=-1,
            career_playtime_seconds_last=-1,
            container_pickups_full=0,
            container_pickups_partial=0,
        ),
    )

    decoded = decode_issue_report(_round_trip(encode_issue_report(report)))

    assert decoded == report


def test_decode_issue_report_rejects_non_object_session_room() -> None:
    """A non-object ``session_room`` raises ``JSONTypeError`` at decode."""
    raw: JSONObject = {
        "source_path": "x",
        "mode": "bot",
        "event_count": 0,
        "session_room": "not an object",
        "teleport_attempts": [],
        "map_open_skipped": [],
        "fuel_target_selections": [],
        "action_outcomes": [],
        "teleport_success_count": 0,
        "teleport_failure_count": 0,
        "fuel_selected_count": 0,
        "fuel_rejected_count": 0,
        "map_open_dispatches": 0,
        "map_open_completions": 0,
    }

    with pytest.raises(JSONTypeError, match="session_room must be object or null"):
        decode_issue_report(raw)


def test_decode_issue_report_treats_absent_session_room_as_none() -> None:
    """A record without ``session_room`` decodes with the field set to None."""
    from tankpit_bot.diagnostics.issue_report_codecs_scorecard import encode_session_scorecard
    from tankpit_bot.diagnostics.issue_report_types import (
        SessionScorecardDict,
        make_unsampled_inventory_counts,
        make_zero_inventory_counts,
    )

    raw: JSONObject = {
        "source_path": "x",
        "mode": "bot",
        "event_count": 0,
        "teleport_attempts": [],
        "map_open_skipped": [],
        "fuel_target_selections": [],
        "action_outcomes": [],
        "teleport_success_count": 0,
        "teleport_failure_count": 0,
        "fuel_selected_count": 0,
        "fuel_rejected_count": 0,
        "map_open_dispatches": 0,
        "map_open_completions": 0,
        "suppressed_dispatches": [],
        "displaced_teleports": [],
        "scorecard": encode_session_scorecard(
            SessionScorecardDict(
                duration_seconds=0,
                state_budget=[],
                kills=0,
                shots=0,
                combat_misses=0,
                combat_ghosts_blocked=0,
                combat_stale_positions_blocked=0,
                tank_damage_changes=0,
                fuel_min=-1,
                fuel_last=-1,
                fuel_sample_count=0,
                inventory_first=make_unsampled_inventory_counts(),
                inventory_last=make_unsampled_inventory_counts(),
                inventory_sample_count=0,
                equipment_gain_events=0,
                equipment_gained=make_zero_inventory_counts(),
                scans_extra=0,
                scans_builtin=0,
                physics_divergences=0,
                equipment_approaches=[],
                equipment_approach_distinct_targets=0,
                equipment_approach_max_repeats=0,
                action_outcome_counts={},
                fuel_low_water_threshold=100,
                fuel_low_water_episodes=[],
                teleport_spend=[],
                teleport_spend_total=0,
                ledger_teleport_spend_min=-1,
                ledger_teleport_spend_max=-1,
                ledger_shot_singles=-1,
                ledger_shot_duals=-1,
                ledger_shot_homings=-1,
                career_destroyed_last=-1,
                career_deactivated_last=-1,
                career_score_last=-1,
                career_playtime_seconds_last=-1,
                container_pickups_full=0,
                container_pickups_partial=0,
            )
        ),
    }

    report = decode_issue_report(raw)

    assert report["session_room"] is None
