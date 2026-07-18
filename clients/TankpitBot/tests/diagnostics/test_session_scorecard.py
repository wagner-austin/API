"""Tests for the per-run session scorecard distillation."""

from __future__ import annotations

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TargetedTeleportRecordDict,
    make_unsampled_inventory_counts,
    make_zero_inventory_counts,
)
from tankpit_bot.diagnostics.session_scorecard import (
    ScorecardAccumulatorDict,
    build_session_scorecard,
    collect_scorecard_issues,
    new_scorecard_accumulator,
    render_scorecard_section,
    route_scorecard_record,
)
from tankpit_bot.runtime_logging import RuntimeEventRecordDict


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


class TestRouting:
    """Tests for route_scorecard_record."""

    def test_routes_state_shots_kills_and_fuel(self) -> None:
        """Every scorecard-relevant record lands in its bucket."""
        accumulator = _routed(
            [
                _record(channel="STATE", message="INITIALIZING -> IDLE"),
                _record(channel="WIRE", message="shoot(136,149,id=529)"),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "tank_deactivated", "victim_id": 529},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "self_alignment_sample", "belief_fuel": 740},
                ),
            ]
        )

        assert accumulator["state_transitions"] == [("2026-06-12T06:25:00", "INITIALIZING -> IDLE")]
        assert accumulator["shots"] == 1
        assert accumulator["kills"] == 1
        assert accumulator["fuel_samples"] == [740]

    def test_tracks_duration_bounds_from_every_record(self) -> None:
        """Even unrelated channels move the duration bounds."""
        accumulator = _routed(
            [
                _record(channel="AI", message="anything", timestamp="2026-06-12T06:25:00"),
                _record(channel="WIRE", message="teleport(1,2)", timestamp="2026-06-12T06:27:30"),
            ]
        )

        assert accumulator["first_timestamp"] == "2026-06-12T06:25:00"
        assert accumulator["last_timestamp"] == "2026-06-12T06:27:30"
        assert accumulator["shots"] == 0

    def test_routes_self_statistics_to_career_totals(self) -> None:
        """``self_statistics`` populates the career-* fields from the wire."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "self_statistics",
                        "playtime_hours": 12,
                        "playtime_minutes": 34,
                        "playtime_seconds": 56,
                        "playtime_seconds_total": 12 * 3600 + 34 * 60 + 56,
                        "destroyed": 4271,
                        "deactivated": 1893,
                        "score": 1003500,
                    },
                )
            ]
        )

        assert accumulator["career_destroyed_last"] == 4271
        assert accumulator["career_deactivated_last"] == 1893
        assert accumulator["career_score_last"] == 1003500
        assert accumulator["career_playtime_seconds_last"] == 12 * 3600 + 34 * 60 + 56

    def test_routes_container_pickup_dispatched_to_full_and_partial(self) -> None:
        """``container_pickup_dispatched`` splits records into full/partial tallies."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "container_pickup_dispatched",
                        "x": 80,
                        "y": 90,
                        "remaining_volume": 0,
                        "is_partial": False,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "container_pickup_dispatched",
                        "x": 81,
                        "y": 91,
                        "remaining_volume": 881,
                        "is_partial": True,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "container_pickup_dispatched",
                        "x": 82,
                        "y": 92,
                        "remaining_volume": 0,
                        "is_partial": False,
                    },
                ),
            ]
        )

        assert accumulator["container_pickups_full"] == 2
        assert accumulator["container_pickups_partial"] == 1

    def test_ignores_irrelevant_diagnostics(self) -> None:
        """Unrelated diagnostic kinds leave the buckets untouched."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "map_positions_parsed", "tank_count": 37},
                )
            ]
        )

        assert accumulator["kills"] == 0
        assert accumulator["fuel_samples"] == []

    def test_routes_inventory_gains_scans_and_approaches(self) -> None:
        """The four observability diagnostics land in their buckets."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "inventory_sample",
                        "armor": 0,
                        "dual": 12,
                        "missile": 0,
                        "homing": 22,
                        "radar": 0,
                        "radar_enabled": True,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "equipment_gain",
                        "armor": 0,
                        "dual": 7,
                        "missile": 0,
                        "homing": 2,
                        "radar": 3,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "equipment_gain",
                        "armor": 1,
                        "dual": 0,
                        "missile": 0,
                        "homing": 3,
                        "radar": 0,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "radar_dispatch", "uses_extra": True},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "radar_dispatch", "uses_extra": False},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "radar_dispatch", "uses_extra": False},
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "equipment_approach",
                        "target_x": 128,
                        "target_y": 126,
                        "fuel": 838,
                    },
                ),
            ]
        )

        assert accumulator["inventory_samples"] == [
            InventoryCountsDict(armor=0, dual=12, missile=0, homing=22, radar=0)
        ]
        assert accumulator["equipment_gain_events"] == 2
        assert accumulator["equipment_gained"] == InventoryCountsDict(
            armor=1, dual=7, missile=0, homing=5, radar=3
        )
        assert accumulator["scans_extra"] == 1
        assert accumulator["scans_builtin"] == 2
        assert accumulator["equipment_approaches"] == [
            TargetedTeleportRecordDict(
                target_x=128,
                target_y=126,
                fuel=838,
                timestamp="2026-06-12T06:25:00",
            )
        ]

    def test_routes_combat_gate_diagnostics(self) -> None:
        """Combat-gate diagnostics each increment their dedicated counter.

        Locks the wiring added 2026-06-19 alongside the freshness
        refactor. Without these counters the combat gates (ghost,
        stale-position) and the miss path emit DIAGNOSTIC events that
        the scorecard ignores -- regression of this test would
        re-blind the scorecard to those events.
        """
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "combat_miss",
                        "target_name": "orange-8",
                        "target_id": 534,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "combat_miss",
                        "target_name": "red-3",
                        "target_id": 211,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "combat_ghost_detected",
                        "target_name": "purple-9",
                        "target_id": 517,
                        "wire_age_ms": 60000,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "combat_stale_position",
                        "target_name": "orange-8",
                        "target_id": 534,
                        "position_age_ms": 5000,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "tank_damage_changed",
                        "tank_id": 534,
                        "tank_name": "orange-8",
                        "previous_damage_state": 0,
                        "damage_state": 1,
                    },
                ),
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "tank_damage_changed",
                        "tank_id": 211,
                        "tank_name": "red-3",
                        "previous_damage_state": 1,
                        "damage_state": 2,
                    },
                ),
            ]
        )

        assert accumulator["combat_misses"] == 2
        assert accumulator["combat_ghosts_blocked"] == 1
        assert accumulator["combat_stale_positions_blocked"] == 1
        assert accumulator["tank_damage_changes"] == 2


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
            equipment_approaches=[],
            equipment_approach_distinct_targets=0,
            equipment_approach_max_repeats=0,
            action_outcome_counts={},
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
            StateBudgetRecordDict(state="SCANNING", seconds=30),
            StateBudgetRecordDict(state="COMBAT", seconds=10),
        ]
        assert scorecard["duration_seconds"] == 45

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


class TestRenderAndIssues:
    """Tests for render_scorecard_section and collect_scorecard_issues."""

    @staticmethod
    def _scorecard(
        *,
        kills: int = 2,
        shots: int = 10,
        fuel_min: int = 405,
        fuel_sample_count: int = 5,
        state_budget: list[StateBudgetRecordDict] | None = None,
        equipment_approach_max_repeats: int = 1,
        inventory_sample_count: int = 3,
        radar_last: int = 11,
    ) -> SessionScorecardDict:
        """Build a scorecard with healthy defaults overridable per case."""
        return SessionScorecardDict(
            duration_seconds=240,
            state_budget=(
                state_budget
                if state_budget is not None
                else [StateBudgetRecordDict(state="COMBAT", seconds=113)]
            ),
            kills=kills,
            shots=shots,
            combat_misses=0,
            combat_ghosts_blocked=0,
            combat_stale_positions_blocked=0,
            tank_damage_changes=0,
            fuel_min=fuel_min,
            fuel_last=866,
            fuel_sample_count=fuel_sample_count,
            inventory_first=InventoryCountsDict(armor=0, dual=12, missile=0, homing=22, radar=10),
            inventory_last=InventoryCountsDict(
                armor=0, dual=19, missile=0, homing=25, radar=radar_last
            ),
            inventory_sample_count=inventory_sample_count,
            equipment_gain_events=5,
            equipment_gained=InventoryCountsDict(armor=0, dual=15, missile=0, homing=5, radar=3),
            scans_extra=3,
            scans_builtin=2,
            equipment_approaches=[],
            equipment_approach_distinct_targets=0,
            equipment_approach_max_repeats=equipment_approach_max_repeats,
            action_outcome_counts={},
            career_destroyed_last=-1,
            career_deactivated_last=-1,
            career_score_last=-1,
            career_playtime_seconds_last=-1,
            container_pickups_full=0,
            container_pickups_partial=0,
        )

    def test_render_includes_budget_and_aggregates(self) -> None:
        """The rendered section carries every aggregate line."""
        lines = render_scorecard_section(self._scorecard())

        assert lines[0] == "=== SESSION SCORECARD ==="
        assert "duration=240s kills=2 shots=10" in lines[1]
        assert "min=405 last=866 samples=5" in lines[2]
        assert any("COMBAT" in line and "113s" in line for line in lines)

    def test_render_handles_no_samples_and_no_transitions(self) -> None:
        """Empty fuel and state buckets render their explicit markers."""
        lines = render_scorecard_section(self._scorecard(fuel_sample_count=0, state_budget=[]))

        assert any("no samples" in line for line in lines)
        assert any("(no transitions)" in line for line in lines)

    def test_healthy_scorecard_raises_no_issues(self) -> None:
        """A healthy session contributes no top-level issues."""
        assert collect_scorecard_issues(self._scorecard()) == []

    def test_fuel_floor_issue(self) -> None:
        """A fuel dip below the critical band is surfaced."""
        issues = collect_scorecard_issues(self._scorecard(fuel_min=91))

        assert issues == ["fuel floor critical: belief fuel dipped to 91 (below 100)"]

    def test_combat_futility_issue(self) -> None:
        """Heavy shooting with zero kills is surfaced."""
        issues = collect_scorecard_issues(self._scorecard(kills=0, shots=43))

        assert issues == ["combat futility: 43 shots produced 0 observed kills"]

    def test_equipment_orbit_issue(self) -> None:
        """Three teleport approaches at one container is the orbit signature."""
        issues = collect_scorecard_issues(self._scorecard(equipment_approach_max_repeats=7))

        assert issues == [
            "equipment-approach orbit: one container teleport-approached 7 times "
            "without completing a pickup"
        ]

    def test_radars_exhausted_issue(self) -> None:
        """Ending the run with zero extra radars is surfaced."""
        issues = collect_scorecard_issues(self._scorecard(radar_last=0))

        assert issues == [
            "extra radars exhausted: run ended with 0 extra radars "
            "(scans degrade to the 5x5 built-in and equipment discovery stalls)"
        ]

    def test_no_radar_issue_without_inventory_samples(self) -> None:
        """A run with no inventory samples cannot claim radar exhaustion."""
        issues = collect_scorecard_issues(self._scorecard(radar_last=0, inventory_sample_count=0))

        assert issues == []
