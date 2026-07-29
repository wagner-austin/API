"""Round-trip tests for every issue-report TypedDict."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.diagnostics.issue_report_codecs import (
    decode_action_outcome_row,
    decode_fuel_target_selection_record,
    decode_issue_report,
    decode_map_open_skipped_record,
    decode_session_room_record,
    decode_state_budget_record,
    decode_targeted_teleport_record,
    decode_teleport_attempt_record,
    encode_action_outcome_row,
    encode_fuel_target_selection_record,
    encode_issue_report,
    encode_map_open_skipped_record,
    encode_session_room_record,
    encode_state_budget_record,
    encode_targeted_teleport_record,
    encode_teleport_attempt_record,
)
from tankpit_bot.diagnostics.issue_report_types import (
    ActionOutcomeRowDict,
    FuelTargetSelectionRecordDict,
    IssueReportDict,
    MapOpenSkippedRecordDict,
    SessionRoomRecordDict,
    StateBudgetRecordDict,
    TargetedTeleportRecordDict,
    TeleportAttemptRecordDict,
)


def _round_trip(encoded: JSONObject) -> JSONObject:
    """Round-trip a dict through ``dump_json_str`` / ``load_json_str``."""
    return narrow_json_to_dict(load_json_str(dump_json_str(encoded)))


def test_teleport_attempt_record_round_trip() -> None:
    """``TeleportAttemptRecordDict`` round-trips through JSON encoding."""
    record = TeleportAttemptRecordDict(
        target_x=147,
        target_y=110,
        teleport_cycle_id=2,
        status="landed_exact",
        timestamp="2026-06-07T22:12:30",
        sent_window="68:[SENT] CMD: ! type=2 id=108 origin=bot_injected label=map_open",
        received_window="70:[RECEIVED] MAP_DATA: len=867",
        page_snapshot_count=4,
    )

    decoded = decode_teleport_attempt_record(_round_trip(encode_teleport_attempt_record(record)))

    assert decoded == record


def test_map_open_skipped_record_round_trip() -> None:
    """``MapOpenSkippedRecordDict`` round-trips through JSON encoding."""
    record = MapOpenSkippedRecordDict(
        origin="acquisition_phase",
        timestamp="2026-06-07T22:12:31",
    )

    decoded = decode_map_open_skipped_record(_round_trip(encode_map_open_skipped_record(record)))

    assert decoded == record


def test_fuel_target_selection_record_round_trip() -> None:
    """``FuelTargetSelectionRecordDict`` round-trips through JSON encoding."""
    record = FuelTargetSelectionRecordDict(
        radar_cycle_id=2,
        target_present=True,
        target_x=151,
        target_y=109,
        summary="fuel: total=7 nearby=4 actionable=3 blocked=1 no_landing=1 low_volume=0",
        decision_basis="world_ts=1780821408807 radar_cycle=2 viewport=(123,102) self=(131,110)",
        timestamp="2026-06-07T22:12:32",
    )

    decoded = decode_fuel_target_selection_record(
        _round_trip(encode_fuel_target_selection_record(record))
    )

    assert decoded == record


def test_fuel_target_selection_record_rejects_non_bool_target_present() -> None:
    """A non-boolean ``target_present`` is rejected at decode time."""
    raw: JSONObject = {
        "radar_cycle_id": 1,
        "target_present": "true",
        "target_x": -1,
        "target_y": -1,
        "summary": "fuel: total=0 nearby=0",
        "decision_basis": "world_ts=0 radar_cycle=1",
        "timestamp": "2026-06-07T22:12:32",
    }

    with pytest.raises(JSONTypeError, match="target_present must be bool"):
        decode_fuel_target_selection_record(raw)


def test_decode_scorecard_rejects_non_int_outcome_count() -> None:
    """The outcome-counts codec rejects non-int values."""
    from tankpit_bot.diagnostics.issue_report_codecs import (
        decode_session_scorecard,
        encode_session_scorecard,
    )
    from tankpit_bot.diagnostics.session_scorecard import (
        build_session_scorecard,
        new_scorecard_accumulator,
    )

    scorecard = build_session_scorecard(new_scorecard_accumulator())
    encoded = encode_session_scorecard(scorecard)
    assert decode_session_scorecard(encoded) == scorecard
    encoded["action_outcome_counts"] = {"shoot:hit": 4, "move:stall_timeout": 1}
    populated = decode_session_scorecard(encoded)
    assert populated["action_outcome_counts"] == {"shoot:hit": 4, "move:stall_timeout": 1}
    encoded["action_outcome_counts"] = {"move:stall_timeout": "many"}
    with pytest.raises(JSONTypeError, match="must be int"):
        decode_session_scorecard(encoded)


def test_action_outcome_row_round_trip() -> None:
    """``ActionOutcomeRowDict`` round-trips through JSON encoding."""
    record = ActionOutcomeRowDict(
        action_kind="map_open",
        outcome="map_data_processed",
        event_id=7,
        attempt_id=3,
        duration_ms=850,
        timestamp="2026-06-07T22:12:33",
    )

    decoded = decode_action_outcome_row(_round_trip(encode_action_outcome_row(record)))

    assert decoded == record


def test_session_room_record_round_trip() -> None:
    """``SessionRoomRecordDict`` round-trips through JSON encoding."""
    record = SessionRoomRecordDict(
        room_id="5",
        field_image="field05.gif",
        timestamp="2026-06-07T22:12:00",
    )

    decoded = decode_session_room_record(_round_trip(encode_session_room_record(record)))

    assert decoded == record


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
        recovery_boxed_in_count=0,
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
        recovery_boxed_in_count=0,
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
    from tankpit_bot.diagnostics.issue_report_codecs import encode_session_scorecard
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
        "recovery_boxed_in_count": 0,
    }

    report = decode_issue_report(raw)

    assert report["session_room"] is None


def test_state_budget_record_round_trip() -> None:
    """``StateBudgetRecordDict`` round-trips through JSON encoding."""
    record = StateBudgetRecordDict(state="COMBAT", seconds=42, stretches=7, max_seconds=12)

    decoded = decode_state_budget_record(_round_trip(encode_state_budget_record(record)))

    assert decoded == record


def test_state_budget_record_decodes_pre_stretch_artifacts() -> None:
    """State budget rows persisted before stretch stats decode as zeros."""
    decoded = decode_state_budget_record({"state": "COMBAT", "seconds": 42})

    assert decoded == StateBudgetRecordDict(state="COMBAT", seconds=42, stretches=0, max_seconds=0)


def test_fuel_low_water_episode_round_trip() -> None:
    """``FuelLowWaterEpisodeDict`` round-trips through JSON encoding."""
    from tankpit_bot.diagnostics.issue_report_codecs import (
        decode_fuel_low_water_episode,
        encode_fuel_low_water_episode,
    )
    from tankpit_bot.diagnostics.issue_report_types import FuelLowWaterEpisodeDict

    episode = FuelLowWaterEpisodeDict(
        start_timestamp="2026-07-29T11:06:35",
        end_timestamp="2026-07-29T11:06:43",
        duration_seconds=8,
        entry_fuel=372,
        min_fuel=140,
        cause_kind="teleport",
        cause_drop=158,
        cause_state="HUNT/CLOSE",
        recovery_fuel=1047,
        recovery_kind="collect",
    )

    decoded = decode_fuel_low_water_episode(_round_trip(encode_fuel_low_water_episode(episode)))

    assert decoded == episode


def test_teleport_spend_record_round_trip() -> None:
    """``TeleportSpendRecordDict`` round-trips through JSON encoding."""
    from tankpit_bot.diagnostics.issue_report_codecs import (
        decode_teleport_spend_record,
        encode_teleport_spend_record,
    )
    from tankpit_bot.diagnostics.issue_report_types import TeleportSpendRecordDict

    record = TeleportSpendRecordDict(bot_state="HUNT/CLOSE", drops=53, fuel_spent=7389)

    decoded = decode_teleport_spend_record(_round_trip(encode_teleport_spend_record(record)))

    assert decoded == record


def test_session_scorecard_decodes_pre_upgrade_artifacts() -> None:
    """Scorecards persisted before the 2026-07-29 analyzer upgrades decode
    with the sentinel/empty defaults for every new field."""
    from tankpit_bot.diagnostics.issue_report_codecs import (
        decode_session_scorecard,
        encode_session_scorecard,
    )
    from tankpit_bot.diagnostics.issue_report_types import (
        SessionScorecardDict,
        make_unsampled_inventory_counts,
        make_zero_inventory_counts,
    )

    scorecard = SessionScorecardDict(
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
    encoded = encode_session_scorecard(scorecard)
    for key in (
        "fuel_low_water_threshold",
        "fuel_low_water_episodes",
        "teleport_spend",
        "teleport_spend_total",
        "ledger_teleport_spend_min",
        "ledger_teleport_spend_max",
        "ledger_shot_singles",
        "ledger_shot_duals",
        "ledger_shot_homings",
    ):
        del encoded[key]

    decoded = decode_session_scorecard(encoded)

    assert decoded["fuel_low_water_threshold"] == 0
    assert decoded["fuel_low_water_episodes"] == []
    assert decoded["teleport_spend"] == []
    assert decoded["teleport_spend_total"] == 0
    assert decoded["ledger_teleport_spend_min"] == -1
    assert decoded["ledger_teleport_spend_max"] == -1
    assert decoded["ledger_shot_singles"] == -1
    assert decoded["ledger_shot_duals"] == -1
    assert decoded["ledger_shot_homings"] == -1


def test_targeted_teleport_record_round_trip() -> None:
    """``TargetedTeleportRecordDict`` round-trips through JSON encoding."""
    record = TargetedTeleportRecordDict(
        target_x=151,
        target_y=109,
        fuel=280,
        timestamp="2026-06-07T22:12:35",
    )

    decoded = decode_targeted_teleport_record(_round_trip(encode_targeted_teleport_record(record)))

    assert decoded == record


def test_require_object_rejects_non_dict() -> None:
    """``_require_object`` raises ``JSONTypeError`` when the value is not a dict."""
    from tankpit_bot.diagnostics.issue_report_codecs import _require_object

    payload: JSONObject = {"inventory_first": "not a dict"}

    with pytest.raises(JSONTypeError, match="inventory_first must be object"):
        _require_object(payload, "inventory_first")
