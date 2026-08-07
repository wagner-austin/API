"""Tests for :mod:`tankpit_bot.diagnostics.session_scorecard_render`.

Covers the rendered report section, the per-block render helpers, and
the scorecard-derived issue list.
"""

from __future__ import annotations

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
)
from tankpit_bot.diagnostics.session_scorecard_accumulator import (
    ScorecardAccumulatorDict,
    new_scorecard_accumulator,
    route_scorecard_record,
)
from tankpit_bot.diagnostics.session_scorecard_render import (
    collect_scorecard_issues,
    render_fuel_low_water_lines,
    render_scorecard_section,
    render_shot_billing_lines,
    render_teleport_spend_lines,
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
