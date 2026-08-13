"""Tests for :mod:`tankpit_bot.diagnostics.session_scorecard`.

Covers distillation of a filled accumulator into a
:class:`SessionScorecardDict`, including the fuel low-water episodes.
"""

from __future__ import annotations

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    make_unsampled_inventory_counts,
    make_zero_inventory_counts,
)
from tankpit_bot.diagnostics.session_scorecard import build_session_scorecard
from tankpit_bot.diagnostics.session_scorecard_accumulator import (
    ScorecardAccumulatorDict,
    new_scorecard_accumulator,
    route_scorecard_record,
)
from tankpit_bot.runtime_records import RuntimeEventRecordDict


def _record(
    *,
    channel: str,
    message: str = "",
    timestamp: str = "2026-06-12T06:25:00",
    fields: dict[str, str | int | float | bool] | None = None,
) -> RuntimeEventRecordDict:
    """Build a runtime event record for routing tests.

    Args:
        channel: Event channel name.
        message: Event message text.
        timestamp: ISO timestamp.
        fields: Structured payload fields.

    Returns:
        Runtime event record.
    """
    return RuntimeEventRecordDict(
        timestamp=timestamp,
        level="INFO",
        logger="tankpit_bot.runtime.events",
        mode="bot",
        channel=channel,
        message=message,
        fields=fields if fields is not None else {},
    )


def _routed(records: list[RuntimeEventRecordDict]) -> ScorecardAccumulatorDict:
    """Route every record into a fresh accumulator.

    Args:
        records: Records in stream order.

    Returns:
        Routed accumulator.
    """
    accumulator = new_scorecard_accumulator()
    for record in records:
        route_scorecard_record(record, accumulator)
    return accumulator


def _fuel_sample_record(
    *,
    fuel: int,
    timestamp: str,
    bot_state: str = "HUNT/ENGAGE",
    in_flight: str = "shoot",
) -> RuntimeEventRecordDict:
    """Build a context-stamped ``self_alignment_sample`` record.

    Args:
        fuel: ``belief_fuel`` value.
        timestamp: ISO timestamp.
        bot_state: Ambient bot-state context.
        in_flight: Ambient in-flight action kind.

    Returns:
        Runtime event record.
    """
    return _record(
        channel="DIAGNOSTIC",
        timestamp=timestamp,
        fields={
            "diagnostic_kind": "self_alignment_sample",
            "belief_fuel": fuel,
            "bot_state": bot_state,
            "in_flight_action_kind": in_flight,
        },
    )


class TestBuildScorecard:
    """Tests for build_session_scorecard."""

    def test_empty_accumulator_builds_zeroed_scorecard(self) -> None:
        """No records at all produce a fully zeroed scorecard."""
        scorecard = build_session_scorecard(new_scorecard_accumulator())

        assert scorecard == SessionScorecardDict(
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

    def test_state_budget_credits_interval_to_earlier_destination(self) -> None:
        """Time between transitions belongs to the state the bot was in.

        The bare initial state announcement carries no transition arrow
        and is skipped.
        """
        accumulator = _routed(
            [
                _record(channel="STATE", message="INITIALIZING", timestamp="2026-06-12T06:25:00"),
                _record(
                    channel="STATE",
                    message="INITIALIZING -> SCANNING",
                    timestamp="2026-06-12T06:25:05",
                ),
                _record(
                    channel="STATE",
                    message="SCANNING -> COMBAT",
                    timestamp="2026-06-12T06:25:35",
                ),
                _record(
                    channel="STATE",
                    message="COMBAT -> IDLE",
                    timestamp="2026-06-12T06:25:45",
                ),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["state_budget"] == [
            StateBudgetRecordDict(state="SCANNING", seconds=30, stretches=1, max_seconds=30),
            StateBudgetRecordDict(state="COMBAT", seconds=10, stretches=1, max_seconds=10),
        ]
        assert scorecard["duration_seconds"] == 45

    def test_map_open_seconds_are_split_out_of_idle(self) -> None:
        """An IDLE stretch that was opening the map is not counted as idle.

        A map open is dispatched FROM idle and has no state of its own: a
        teleport needs the overlay open, the hop closes it again, and the
        open cannot share the hop's tick. Folding those seconds into IDLE
        overstated idleness by more than half in run 20260812-194435 --
        10 of 16 IDLE seconds were a map open in flight.

        The second IDLE stretch carries a ``shoot`` outcome and stays
        IDLE, which pins two things at once: the split needs a completion
        inside the window rather than relabelling every IDLE stretch, and
        it keys on the action KIND rather than on any outcome arriving.
        """
        accumulator = _routed(
            [
                _record(
                    channel="STATE",
                    message="COLLECTING -> IDLE",
                    timestamp="2026-08-12T19:45:12",
                ),
                _record(
                    channel="DIAGNOSTIC",
                    timestamp="2026-08-12T19:45:16",
                    fields={
                        "diagnostic_kind": "action_outcome",
                        "action_kind": "map_open",
                        "outcome": "map_data_processed",
                    },
                ),
                _record(
                    channel="STATE",
                    message="IDLE -> TELEPORTING",
                    timestamp="2026-08-12T19:45:16",
                ),
                _record(
                    channel="STATE",
                    message="TELEPORTING -> IDLE",
                    timestamp="2026-08-12T19:45:18",
                ),
                _record(
                    channel="DIAGNOSTIC",
                    timestamp="2026-08-12T19:45:20",
                    fields={
                        "diagnostic_kind": "action_outcome",
                        "action_kind": "shoot",
                        "outcome": "hit",
                    },
                ),
                _record(
                    channel="STATE",
                    message="IDLE -> SCANNING",
                    timestamp="2026-08-12T19:45:24",
                ),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        budget = {record["state"]: record["seconds"] for record in scorecard["state_budget"]}
        assert budget["IDLE/map_open"] == 4
        assert budget["IDLE"] == 6
        assert budget["TELEPORTING"] == 2
        assert accumulator["map_open_completions_at"] == ["2026-08-12T19:45:16"]

    def test_scope_shift_seconds_are_split_out_of_idle(self) -> None:
        """The tick that steers the scope is not idle either.

        A scope shift is a command, not a state, so the tick dispatching
        one transitions nowhere and its seconds land in IDLE -- the same
        reason map opens did. The quad sweep pays a shift to frame each
        quadrant before spending a radar on it, so this is the other half
        of what IDLE was hiding: with both split out, run
        20260812-194435 reports IDLE 0s across 5 zero-length stretches.

        The marker sits on the OPENING edge of the stretch, unlike the
        map open's completion, because the send happens in the tick that
        enters IDLE rather than the one that leaves it.
        """
        accumulator = _routed(
            [
                _record(
                    channel="STATE",
                    message="SCANNING -> IDLE",
                    timestamp="2026-08-12T19:45:34",
                ),
                _record(
                    channel="DIAGNOSTIC",
                    timestamp="2026-08-12T19:45:34",
                    fields={"diagnostic_kind": "scope_shift_sent"},
                ),
                _record(
                    channel="STATE",
                    message="IDLE -> SCANNING",
                    timestamp="2026-08-12T19:45:36",
                ),
                _record(
                    channel="STATE",
                    message="SCANNING -> IDLE",
                    timestamp="2026-08-12T19:45:38",
                ),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        budget = {record["state"]: record["seconds"] for record in scorecard["state_budget"]}
        assert budget["IDLE/scope_shift"] == 2
        assert "IDLE" not in budget
        assert accumulator["scope_shift_sends_at"] == ["2026-08-12T19:45:34"]

    def test_fuel_aggregates(self) -> None:
        """Fuel min/last/count come from the self_alignment_sample bucket."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "self_alignment_sample", "belief_fuel": 740},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "self_alignment_sample", "belief_fuel": 119},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "self_alignment_sample", "belief_fuel": 908},
                ),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["fuel_min"] == 119
        assert scorecard["fuel_last"] == 908
        assert scorecard["fuel_sample_count"] == 3

    def test_inventory_and_approach_aggregates(self) -> None:
        """Inventory first/last and approach repeat counts come from the buckets."""
        sample_fields: dict[str, str | int | float | bool] = {
            "diagnostic_kind": "inventory_sample",
            "armor": 0,
            "missile": 0,
            "radar_enabled": True,
        }
        approach_fields: dict[str, str | int | float | bool] = {
            "diagnostic_kind": "equipment_approach",
            "fuel": 838,
        }
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={**sample_fields, "dual": 12, "homing": 22, "radar": 0},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={**sample_fields, "dual": 19, "homing": 25, "radar": 3},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={**approach_fields, "target_x": 128, "target_y": 126},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={**approach_fields, "target_x": 128, "target_y": 126},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={**approach_fields, "target_x": 157, "target_y": 169},
                ),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["inventory_first"] == InventoryCountsDict(
            armor=0, dual=12, missile=0, homing=22, radar=0
        )
        assert scorecard["inventory_last"] == InventoryCountsDict(
            armor=0, dual=19, missile=0, homing=25, radar=3
        )
        assert scorecard["inventory_sample_count"] == 2
        assert scorecard["equipment_approach_distinct_targets"] == 2
        assert scorecard["equipment_approach_max_repeats"] == 2


class TestFuelLowWaterEpisodes:
    """Tests for the low-water episode narrative in the built scorecard."""

    def test_episode_captures_entry_min_cause_and_recovery(self) -> None:
        """A dip below the escape floor narrates its cause and recovery.

        Mirrors run 20260729-105325's chase dip: entry above threshold,
        a big teleport drop into the episode, a collect recovery.
        """
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "engagement_break",
                        "escape_floor": 354,
                        "fuel": 657,
                    },
                ),
                _fuel_sample_record(fuel=372, timestamp="2026-07-29T11:06:33"),
                _fuel_sample_record(
                    fuel=214,
                    timestamp="2026-07-29T11:06:35",
                    bot_state="HUNT/CLOSE",
                    in_flight="teleport",
                ),
                _fuel_sample_record(fuel=140, timestamp="2026-07-29T11:06:43"),
                _fuel_sample_record(
                    fuel=1047,
                    timestamp="2026-07-29T11:06:45",
                    in_flight="collect",
                ),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["fuel_low_water_threshold"] == 354
        assert scorecard["fuel_low_water_episodes"] == [
            {
                "start_timestamp": "2026-07-29T11:06:35",
                "end_timestamp": "2026-07-29T11:06:43",
                "duration_seconds": 8,
                "entry_fuel": 372,
                "min_fuel": 140,
                "cause_kind": "teleport",
                "cause_drop": 158,
                "cause_state": "HUNT/CLOSE",
                "recovery_fuel": 1047,
                "recovery_kind": "collect",
            }
        ]

    def test_session_bounds_use_sentinels_and_cause_falls_back(self) -> None:
        """A session that starts AND ends below threshold has no entry
        sample, no recovery sample, and no positive drop to blame --
        the entry sample's own context is the cause fallback."""
        accumulator = _routed(
            [
                _fuel_sample_record(
                    fuel=50,
                    timestamp="2026-07-29T10:00:00",
                    bot_state="COLLECT/SEARCH",
                    in_flight="map_open",
                ),
                _fuel_sample_record(fuel=60, timestamp="2026-07-29T10:00:02"),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        # No engagement_break -> static critical floor threshold.
        assert scorecard["fuel_low_water_threshold"] == 100
        assert scorecard["fuel_low_water_episodes"] == [
            {
                "start_timestamp": "2026-07-29T10:00:00",
                "end_timestamp": "2026-07-29T10:00:02",
                "duration_seconds": 2,
                "entry_fuel": -1,
                "min_fuel": 50,
                "cause_kind": "map_open",
                "cause_drop": 0,
                "cause_state": "COLLECT/SEARCH",
                "recovery_fuel": -1,
                "recovery_kind": "",
            }
        ]

    def test_separate_dips_become_separate_episodes(self) -> None:
        """Two dips separated by an above-threshold sample split cleanly."""
        accumulator = _routed(
            [
                _fuel_sample_record(fuel=500, timestamp="2026-07-29T10:00:00"),
                _fuel_sample_record(fuel=90, timestamp="2026-07-29T10:00:02"),
                _fuel_sample_record(fuel=600, timestamp="2026-07-29T10:00:04"),
                _fuel_sample_record(fuel=80, timestamp="2026-07-29T10:00:06"),
                _fuel_sample_record(fuel=700, timestamp="2026-07-29T10:00:08"),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert len(scorecard["fuel_low_water_episodes"]) == 2
        assert [e["min_fuel"] for e in scorecard["fuel_low_water_episodes"]] == [90, 80]

    def test_engagement_break_keeps_the_highest_floor(self) -> None:
        """Multiple engagement breaks keep the max escape floor."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "engagement_break", "escape_floor": 354},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "engagement_break", "escape_floor": 372},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "engagement_break", "escape_floor": 354},
                ),
            ]
        )

        assert accumulator["max_escape_floor"] == 372
