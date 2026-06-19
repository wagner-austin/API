"""Tests for combat probe helpers and summary formatting."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab.combat_probe import (
    _manhattan,
    format_combat_probe_summary,
)
from tankpit_bot.action_lab.combat_probe_types import (
    CombatEngagementDict,
    CombatProbeSessionDict,
    CombatShotResultDict,
)


def _make_shot(
    *,
    shot_number: int = 1,
    result: Literal["hit", "miss", "timeout"] = "hit",
    distance: int = 1,
    weapon_byte: int | None = 1,
    self_x: int = 10,
    self_y: int = 20,
    target_x: int = 11,
    target_y: int = 20,
) -> CombatShotResultDict:
    return CombatShotResultDict(
        shot_number=shot_number,
        self_x=self_x,
        self_y=self_y,
        target_x=target_x,
        target_y=target_y,
        distance=distance,
        result=result,
        weapon_byte=weapon_byte,
        target_name="purple-1",
        target_id=500,
        timestamp_ms=100000,
    )


def _make_engagement(
    shots: list[CombatShotResultDict] | None = None,
    *,
    kill_confirmed: bool = False,
    target_fled: bool = False,
) -> CombatEngagementDict:
    default_shots = shots or [_make_shot()]
    hits = sum(1 for s in default_shots if s["result"] == "hit")
    misses = sum(1 for s in default_shots if s["result"] == "miss")
    timeouts = sum(1 for s in default_shots if s["result"] == "timeout")
    return CombatEngagementDict(
        target_id=500,
        target_name="purple-1",
        initial_target_x=11,
        initial_target_y=20,
        initial_distance=1,
        landed_x=10,
        landed_y=20,
        shots=default_shots,
        total_hits=hits,
        total_misses=misses,
        total_timeouts=timeouts,
        kill_confirmed=kill_confirmed,
        target_fled=target_fled,
        final_target_x=11,
        final_target_y=20,
        final_distance=1,
    )


def _make_session(
    engagements: list[CombatEngagementDict] | None = None,
) -> CombatProbeSessionDict:
    return CombatProbeSessionDict(
        session_id="test-session-1",
        start_timestamp_ms=90000,
        end_timestamp_ms=120000,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_engagements=3,
        max_shots_per_engagement=20,
        capture_session_path="",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 91000,
            "intel_ready_timestamp_ms": 92000,
            "initial_sync_started_ms": 93000,
            "initial_world_timestamp_ms": 94000,
            "command_ready_timestamp_ms": 95000,
            "first_attempt_started_ms": 96000,
            "game_ready_to_intel_ready_ms": 1000,
            "intel_ready_to_initial_world_ms": 2000,
            "initial_world_to_command_ready_ms": 1000,
            "command_ready_to_first_attempt_ms": 1000,
        },
        engagements=[_make_engagement()] if engagements is None else engagements,
    )


class TestManhattan:
    """Tests for Manhattan distance helper."""

    def test_adjacent(self) -> None:
        assert _manhattan(10, 20, 11, 20) == 1

    def test_same_point(self) -> None:
        assert _manhattan(5, 5, 5, 5) == 0

    def test_diagonal(self) -> None:
        assert _manhattan(0, 0, 3, 4) == 7

    def test_negative_coords(self) -> None:
        assert _manhattan(10, 10, 7, 6) == 7


class TestFormatCombatProbeSummary:
    """Tests for the summary formatter."""

    def test_single_engagement_all_hits(self) -> None:
        shots = [_make_shot(shot_number=i, distance=1) for i in range(1, 6)]
        eng = _make_engagement(shots=shots, kill_confirmed=True)
        session = _make_session(engagements=[eng])
        summary = format_combat_probe_summary(session)
        assert "engagements=1" in summary
        assert "hits=5" in summary
        assert "misses=0" in summary
        assert "kills=1" in summary
        assert "d=1:5h/0m" in summary

    def test_mixed_hits_misses_at_multiple_distances(self) -> None:
        shots = [
            _make_shot(shot_number=1, distance=1, result="hit"),
            _make_shot(shot_number=2, distance=1, result="hit"),
            _make_shot(shot_number=3, distance=3, result="miss", weapon_byte=None),
            _make_shot(shot_number=4, distance=5, result="miss", weapon_byte=None),
        ]
        eng = _make_engagement(shots=shots, target_fled=True)
        session = _make_session(engagements=[eng])
        summary = format_combat_probe_summary(session)
        assert "hits=2" in summary
        assert "misses=2" in summary
        assert "fled=1" in summary
        assert "d=1:2h/0m" in summary
        assert "d=3:0h/1m" in summary
        assert "d=5:0h/1m" in summary

    def test_empty_engagements(self) -> None:
        session = _make_session(engagements=[])
        summary = format_combat_probe_summary(session)
        assert "engagements=0" in summary
        assert "hits=0" in summary

    def test_timeout_not_counted_in_distance_table(self) -> None:
        shots = [
            _make_shot(shot_number=1, distance=1, result="hit"),
            _make_shot(shot_number=2, distance=1, result="timeout", weapon_byte=None),
        ]
        eng = _make_engagement(shots=shots)
        session = _make_session(engagements=[eng])
        summary = format_combat_probe_summary(session)
        assert "d=1:1h/0m" in summary
