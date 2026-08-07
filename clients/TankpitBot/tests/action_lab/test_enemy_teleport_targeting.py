"""Tests for enemy selection and result formatting."""

from __future__ import annotations

import pytest
from tests.action_lab._enemy_teleport_harness import (
    _enemy,
    _ProbeHarness,
    _snapshot,
    _target,
)
from tests.action_lab._replay_page import ReplayClock

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.enemy_teleport import (
    _enemy_by_id,
    _format_enemy_label,
    _make_terminal_result,
    _require_fresh_enemy_threat,
    format_enemy_teleport_probe_summary,
)
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
)
from tankpit_bot.state import (
    make_tank_state,
)


def test_require_fresh_enemy_threat_filters_old_entries() -> None:
    """Real ``analyze_threats`` against real tanks — closer fresh enemy wins.

    Probe self at (100, 100), team=2. Two enemy tanks on team=1 in the world:
    one far at distance 9 with old timestamp (filtered as stale at age >250ms
    relative to now=1000), one close at distance 2 with fresh timestamp.
    Real analyze_threats sorts by distance ascending; freshness filter in
    _require_fresh_enemy_threat then keeps the close, fresh one.
    """
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "1": make_tank_state(
            tank_id=1,
            x=109,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-1",
            is_bot=False,
            is_self=False,
            timestamp_ms=900,
        ),
        "2": make_tank_state(
            tank_id=2,
            x=102,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-2",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
    }

    result = _require_fresh_enemy_threat(probe, 1000, frozenset())

    if result is None:
        pytest.fail("expected real analyze_threats to return the fresh close enemy")
    assert result["tank_id"] == 2
    assert result["distance"] == 2
    assert result["timestamp_ms"] == 1500


def test_require_fresh_enemy_threat_returns_none_without_self_state() -> None:
    probe = _ProbeHarness()
    probe._self_state = None

    assert _require_fresh_enemy_threat(probe, 1000, frozenset()) is None


def test_require_fresh_enemy_threat_excludes_previously_targeted_enemy_ids() -> None:
    """Real ``analyze_threats`` over real tanks; previously-targeted IDs are skipped.

    Tank 1 at distance 1 (closest, but excluded), tank 2 at distance 2.
    Real analyze_threats returns both sorted by distance; the exclude set
    skips tank 1, so tank 2 is returned.
    """
    action_hooks.get_current_time_ms = ReplayClock(1500)
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "1": make_tank_state(
            tank_id=1,
            x=101,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-1",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
        "2": make_tank_state(
            tank_id=2,
            x=102,
            y=100,
            team=1,
            rank=1,
            damage_state=0,
            name="red-2",
            is_bot=False,
            is_self=False,
            timestamp_ms=1500,
        ),
    }

    result = _require_fresh_enemy_threat(probe, 1000, frozenset({1}))

    if result is None:
        pytest.fail("expected real analyze_threats to return tank 2 after excluding tank 1")
    assert result["tank_id"] == 2
    assert result["distance"] == 2


def test_enemy_by_id_returns_matching_enemy_and_none_when_missing() -> None:
    """Real ``analyze_threats`` produces threats from real tanks; lookup by ID works."""
    action_hooks.get_current_time_ms = ReplayClock(1000)
    probe = _ProbeHarness()
    probe._world_state["tanks"] = {
        "11": make_tank_state(
            tank_id=11,
            x=120,
            y=130,
            team=1,
            rank=1,
            damage_state=0,
            name="red-11",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
        "12": make_tank_state(
            tank_id=12,
            x=120,
            y=130,
            team=1,
            rank=1,
            damage_state=0,
            name="red-12",
            is_bot=False,
            is_self=False,
            timestamp_ms=1000,
        ),
    }

    match = _enemy_by_id(probe, 12)
    missing = _enemy_by_id(probe, 99)

    if match is None:
        pytest.fail("expected real analyze_threats to expose tank 12")
    assert match["tank_id"] == 12
    assert missing is None


def test_enemy_by_id_returns_none_without_self_state() -> None:
    probe = _ProbeHarness()
    probe._self_state = None

    assert _enemy_by_id(probe, 12) is None


def test_format_enemy_helpers_cover_terminal_result_and_summary() -> None:
    enemy = _enemy()
    target = _target()
    result = _make_terminal_result(
        acquisition_strategy="map_open",
        status="no_landing_tile",
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=None,
        fuel_before=900,
        world_timestamp_before=950,
        completion_timestamp_ms=1200,
        fuel_after=880,
        world_timestamp_after=1100,
        enemy=enemy,
        landing_target=target,
        landed_x=100,
        landed_y=101,
        message_start_index=5,
        message_end_index=9,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1100),
    )
    session = EnemyTeleportProbeSessionDict(
        session_id="enemy-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        acquisition_strategy="nearest_enemy",
        max_attempts=6,
        capture_session_path="enemy.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 100,
            "intel_ready_timestamp_ms": 150,
            "initial_sync_started_ms": 200,
            "initial_world_timestamp_ms": 250,
            "command_ready_timestamp_ms": 300,
            "first_attempt_started_ms": 325,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 50,
            "command_ready_to_first_attempt_ms": 25,
        },
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        heartbeat_interval_ms=0,
        attempts=[
            EnemyTeleportAttemptResultDict(**{**result, "status": "landed_adjacent"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "landed_not_adjacent"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "no_enemy"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "no_landing_tile"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "acquisition_timeout"}),
            EnemyTeleportAttemptResultDict(**{**result, "status": "teleport_timeout"}),
        ],
    )

    assert _format_enemy_label(enemy) == "enemy_50_120_130"
    assert result["acquisition_elapsed_ms"] is None
    assert format_enemy_teleport_probe_summary(session) == (
        "Enemy teleport probe complete: strategy=nearest_enemy attempts=6 "
        "landed_adjacent=1 landed_not_adjacent=1 no_enemy=1 no_landing_tile=1 "
        "acquisition_timeout=1 teleport_timeout=1 session_to_initial_sync_ms=199 "
        "initial_sync_to_command_ready_ms=100"
    )


def test_send_enemy_acquisition_dispatches_by_strategy() -> None:
    probe = _ProbeHarness()

    assert probe._send_enemy_acquisition("map_open") is True
    assert probe._send_enemy_acquisition("nearest_enemy") is True
    assert probe.open_map_calls == 1
    assert probe.request_enemy_calls == 1
