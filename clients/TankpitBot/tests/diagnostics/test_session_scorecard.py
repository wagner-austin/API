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
    FuelSampleRecordDict,
    ScorecardAccumulatorDict,
    build_session_scorecard,
    collect_scorecard_issues,
    new_scorecard_accumulator,
    render_fuel_low_water_lines,
    render_scorecard_section,
    render_shot_billing_lines,
    render_teleport_spend_lines,
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
        assert accumulator["fuel_samples"] == [
            FuelSampleRecordDict(
                timestamp="2026-06-12T06:25:00",
                fuel=740,
                bot_state="",
                in_flight="none",
            )
        ]

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
                else [
                    StateBudgetRecordDict(state="COMBAT", seconds=113, stretches=9, max_seconds=29)
                ]
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
            physics_divergences=0,
            equipment_approaches=[],
            equipment_approach_distinct_targets=0,
            equipment_approach_max_repeats=equipment_approach_max_repeats,
            action_outcome_counts={},
            fuel_low_water_threshold=354,
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

    def test_render_includes_budget_and_aggregates(self) -> None:
        """The rendered section carries every aggregate line."""
        lines = render_scorecard_section(self._scorecard())

        assert lines[0] == "=== SESSION SCORECARD ==="
        assert "duration=240s kills=2 shots=10" in lines[1]
        assert "min=405 last=866 samples=5" in lines[2]
        assert any("COMBAT" in line and "113s" in line for line in lines)
        assert any("fuel low-water: none (never below 354)" in line for line in lines)
        assert any("teleport spend: none observed" in line for line in lines)
        # No damage_ledger event -> no billing reconciliation line.
        assert not any("shot billing" in line for line in lines)

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


def test_physics_divergence_routing_and_issue() -> None:
    """physics_divergence events count into the scorecard and raise an issue."""
    accumulator = new_scorecard_accumulator()
    route_scorecard_record(
        _record(
            channel="DIAGNOSTIC",
            fields={
                "diagnostic_kind": "physics_divergence",
                "residual": -45,
                "feasible_lo": -10,
                "feasible_hi": -10,
                "entry_kinds": "shot_dual",
                "fact_source": "wire_0x2E_tank_status_sync",
            },
        ),
        accumulator,
    )
    assert accumulator["physics_divergences"] == 1
    scorecard = build_session_scorecard(accumulator)
    assert scorecard["physics_divergences"] == 1
    assert "  physics divergences: 1" in render_scorecard_section(scorecard)
    issues = collect_scorecard_issues(scorecard)
    assert any("physics divergences: 1" in issue for issue in issues)


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


class TestTeleportSpendRouting:
    """Tests for the WORLD fuel-receipt teleport spend attribution."""

    @staticmethod
    def _world_fuel(
        message: str,
        *,
        in_flight: str = "teleport",
        bot_state: str = "HUNT/CLOSE",
    ) -> RuntimeEventRecordDict:
        """Build a WORLD-channel fuel transition record."""
        return _record(
            channel="WORLD",
            message=message,
            fields={"bot_state": bot_state, "in_flight_action_kind": in_flight},
        )

    def test_in_flight_debits_attribute_to_bot_state(self) -> None:
        """Teleport debits group by the ambient bot state."""
        accumulator = _routed(
            [
                self._world_fuel("Fuel: 1090 -> 823 (-267)", bot_state="COLLECT/SEARCH"),
                self._world_fuel("Fuel: 372 -> 214 (-158)"),
                self._world_fuel("Fuel: 214 -> 200 (-14)"),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["teleport_spend"] == [
            {"bot_state": "COLLECT/SEARCH", "drops": 1, "fuel_spent": 267},
            {"bot_state": "HUNT/CLOSE", "drops": 2, "fuel_spent": 172},
        ]
        assert scorecard["teleport_spend_total"] == 439

    def test_non_teleport_and_credit_receipts_are_ignored(self) -> None:
        """Only in-flight teleport DEBITS count."""
        accumulator = _routed(
            [
                self._world_fuel("Fuel: 500 -> 490 (-10)", in_flight="shoot"),
                self._world_fuel("Fuel: 490 -> 1100 (+610)"),
                self._world_fuel("Not a fuel line"),
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["teleport_spend"] == []
        assert scorecard["teleport_spend_total"] == 0

    def test_damage_ledger_routes_billing_and_spend_bounds(self) -> None:
        """The end-of-run ledger populates billing counts and bounds."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={
                        "diagnostic_kind": "damage_ledger",
                        "teleport_fuel_lo": -19290,
                        "teleport_fuel_hi": -11993,
                        "shot_single_count": 6,
                        "shot_dual_count": 170,
                        "shot_homing_count": 72,
                    },
                )
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["ledger_teleport_spend_max"] == 19290
        assert scorecard["ledger_teleport_spend_min"] == 11993
        assert scorecard["ledger_shot_singles"] == 6
        assert scorecard["ledger_shot_duals"] == 170
        assert scorecard["ledger_shot_homings"] == 72

    def test_damage_ledger_without_fuel_fields_keeps_sentinels(self) -> None:
        """A pre-fuel-book ledger event leaves the -1 sentinels."""
        accumulator = _routed(
            [
                _record(
                    channel="DIAGNOSTIC",
                    fields={"diagnostic_kind": "damage_ledger", "dealt": "", "taken": ""},
                )
            ]
        )

        scorecard = build_session_scorecard(accumulator)

        assert scorecard["ledger_teleport_spend_max"] == -1
        assert scorecard["ledger_teleport_spend_min"] == -1
        assert scorecard["ledger_shot_singles"] == -1


class TestNewRenderHelpers:
    """Tests for the section renderers added with the 2026-07-29 upgrades."""

    def test_shot_billing_line_renders_reconciliation(self) -> None:
        """With a ledger present the billing line explains the singles."""
        scorecard = TestRenderAndIssues._scorecard()
        scorecard["ledger_shot_singles"] = 6
        scorecard["ledger_shot_duals"] = 170
        scorecard["ledger_shot_homings"] = 72

        lines = render_shot_billing_lines(scorecard)

        assert len(lines) == 1
        assert "dual=170 homing=72 single=6" in lines[0]
        assert "server-billed non-connects" in lines[0]

    def test_low_water_lines_cap_the_episode_list(self) -> None:
        """More than the render cap of episodes summarizes the tail."""
        from tankpit_bot.diagnostics.issue_report_types import FuelLowWaterEpisodeDict

        scorecard = TestRenderAndIssues._scorecard()
        scorecard["fuel_low_water_episodes"] = [
            FuelLowWaterEpisodeDict(
                start_timestamp="2026-07-29T10:00:00",
                end_timestamp="2026-07-29T10:00:02",
                duration_seconds=2,
                entry_fuel=-1,
                min_fuel=50,
                cause_kind="teleport",
                cause_drop=158,
                cause_state="HUNT/CLOSE",
                recovery_fuel=-1,
                recovery_kind="",
            )
            for _ in range(12)
        ]

        lines = render_fuel_low_water_lines(scorecard)

        assert lines[0] == "  fuel low-water (below 354): 12 episode(s)"
        assert len(lines) == 12  # header + 10 rendered + tail summary
        assert lines[-1] == "    ... and 2 more episode(s)"
        # Sentinel bounds render as words, not -1.
        assert "entry=start" in lines[1]
        assert "recovery=session end" in lines[1]

    def test_teleport_spend_lines_render_bound_and_groups(self) -> None:
        """Spend rows render under the total with the ledger bound."""
        scorecard = TestRenderAndIssues._scorecard()
        scorecard["teleport_spend"] = [
            {"bot_state": "HUNT/CLOSE", "drops": 53, "fuel_spent": 7389},
            {"bot_state": "", "drops": 1, "fuel_spent": 10},
        ]
        scorecard["teleport_spend_total"] = 7399
        scorecard["ledger_teleport_spend_min"] = 11993
        scorecard["ledger_teleport_spend_max"] = 19290

        lines = render_teleport_spend_lines(scorecard)

        assert lines[0] == "  teleport spend: 7399 fuel (ledger bound 11993..19290)"
        assert lines[1] == "    HUNT/CLOSE: 7389 over 53 drop(s)"
        assert lines[2] == "    (no context): 10 over 1 drop(s)"
