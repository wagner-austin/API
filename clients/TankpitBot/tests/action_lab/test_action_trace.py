"""Tests for shared action-lab phase tracing helpers."""

from __future__ import annotations

import logging

import pytest

from tankpit_bot.action_lab.action_trace import (
    ActionCycleTracker,
    build_fuel_decision_basis,
    format_fuel_decision_basis,
    format_fuel_decision_candidates,
    log_phase_overlaps,
)
from tankpit_bot.state import (
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)


class _FlatTerrain:
    """Minimal passable terrain fake."""

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def is_passable(self, x: int, y: int) -> bool:
        _ = (x, y)
        return True

    def get_terrain(self, x: int, y: int) -> str:
        _ = (x, y)
        return "."

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        _ = (center_x, center_y, width, height)
        return [[self.GROUND]]


def _world_with_viewport() -> WorldStateDict:
    """Build a sample world with self state and viewport."""
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1301,
            x=147,
            y=110,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=ViewportStateDict(left=139, top=102, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
        timestamp_ms=40_000,
    )


def test_action_cycle_tracker_begin_and_end_phase() -> None:
    """Cycle tracker starts and ends one phase cleanly."""
    tracker = ActionCycleTracker()

    cycle, overlaps = tracker.begin_phase("teleport", started_ms=1000)

    assert cycle["phase"] == "teleport"
    assert cycle["cycle_id"] == 1
    assert overlaps == []

    tracker.end_phase(cycle)


def test_action_cycle_tracker_reports_overlap_and_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    """Cycle tracker records and logs active-phase overlap violations."""
    tracker = ActionCycleTracker()
    tracker.begin_phase("radar", started_ms=1200)
    _, overlaps = tracker.begin_phase("move", started_ms=1300)

    with caplog.at_level(logging.INFO):
        log_phase_overlaps(overlaps, attempt_label="attempt-2")

    assert len(overlaps) == 1
    assert "ACTION_PHASE_OVERLAP attempt=attempt-2 active=radar#1" in caplog.text


def test_action_cycle_tracker_rejects_ending_inactive_phase() -> None:
    """Cycle tracker rejects ending a phase that is not active."""
    tracker = ActionCycleTracker()

    with pytest.raises(ValueError, match="is not active"):
        tracker.end_phase({"phase": "pickup", "cycle_id": 1, "started_ms": 1000})


def test_action_cycle_tracker_rejects_cycle_mismatch() -> None:
    """Cycle tracker rejects ending a stale cycle after the phase restarted."""
    tracker = ActionCycleTracker()
    first_cycle, _ = tracker.begin_phase("teleport", started_ms=1000)
    tracker.begin_phase("teleport", started_ms=1100)

    with pytest.raises(ValueError, match="active cycle mismatch"):
        tracker.end_phase(first_cycle)


def test_action_cycle_tracker_reset_restarts_cycle_numbers() -> None:
    """Cycle tracker reset clears active state and restarts counters."""
    tracker = ActionCycleTracker()
    tracker.begin_phase("move", started_ms=1000)

    tracker.reset()
    cycle, overlaps = tracker.begin_phase("move", started_ms=1200)

    assert cycle["cycle_id"] == 1
    assert overlaps == []


def test_build_fuel_decision_basis_and_formatters_capture_freshness_metadata() -> None:
    """Fuel decision basis includes freshness causality and formatting metadata."""
    world = _world_with_viewport()
    fresh = make_container_state(
        140,
        109,
        True,
        380,
        source="radar",
        refresh_kind="radar_response",
        timestamp_ms=39_000,
        failed_pickups=1,
    )
    stale = make_container_state(
        141,
        102,
        True,
        255,
        source="radar",
        refresh_kind="radar_cache_refresh",
        timestamp_ms=5_000,
        failed_pickups=0,
    )
    world["containers"]["140,109"] = fresh
    world["containers"]["141,102"] = stale

    basis = build_fuel_decision_basis(
        world,
        self_x=147,
        self_y=110,
        radar_cycle_id=4,
        terrain=_FlatTerrain(),
        fuel_target=fresh,
    )

    assert basis["radar_cycle_id"] == 4
    assert basis["selected_target_x"] == 140
    assert basis["selected_target_y"] == 109
    assert len(basis["candidates"]) == 2
    assert basis["candidates"][0]["x"] == 141
    assert basis["candidates"][0]["reason"] == "stale"
    assert basis["candidates"][0]["refresh_kind"] == "radar_cache_refresh"
    assert basis["candidates"][1]["selected"] is True

    formatted_basis = format_fuel_decision_basis(basis)
    formatted_candidates = format_fuel_decision_candidates(basis)

    assert "world_ts=40000 radar_cycle=4" in formatted_basis
    assert "refresh=radar_response@39000" in formatted_candidates
    assert "age=1000/3000" in formatted_candidates


def test_format_fuel_decision_basis_handles_no_candidates() -> None:
    """Fuel decision basis formatting handles empty candidate lists."""
    basis = build_fuel_decision_basis(
        _world_with_viewport(),
        self_x=147,
        self_y=110,
        radar_cycle_id=5,
        terrain=None,
        fuel_target=None,
    )

    assert format_fuel_decision_candidates(basis) == "none"
    assert "candidates=none" in format_fuel_decision_basis(basis)


def test_format_fuel_decision_candidates_rejects_non_positive_limit() -> None:
    """Fuel decision candidate formatting rejects invalid limits."""
    basis = build_fuel_decision_basis(
        _world_with_viewport(),
        self_x=147,
        self_y=110,
        radar_cycle_id=6,
        terrain=None,
        fuel_target=None,
    )

    with pytest.raises(ValueError, match="limit must be positive"):
        format_fuel_decision_candidates(basis, limit=0)
