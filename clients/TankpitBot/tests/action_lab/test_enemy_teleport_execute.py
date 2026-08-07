"""Tests for ``execute_probe`` and the enemy-teleport entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._enemy_teleport_harness import (
    _FUEL_CAPTURE_PATH,
    _enemy,
    _ExecuteHarness,
    _FakeEnemyTeleportProbe,
    _ProbeHarness,
    _snapshot,
    _target,
    enemy_probe_module,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_page import ReplayClock
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_teleport import run_enemy_teleport_probe
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    decode_enemy_teleport_probe_session,
)
from tankpit_bot.action_lab.types import (
    TeleportTargetDict,
)
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    SelfStateDict,
    make_self_state,
)
from tankpit_bot.types import (
    decode_capture_session,
)


def test_execute_probe_raises_for_invalid_max_attempts() -> None:
    probe = _ProbeHarness()

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        probe.execute_probe(
            acquisition_strategy="nearest_enemy",
            max_attempts=0,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=500,
            heartbeat_interval_ms=0,
        )


def test_execute_probe_raises_when_playwright_is_missing() -> None:
    probe = _ProbeHarness()
    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                acquisition_strategy="nearest_enemy",
                max_attempts=1,
                initial_sync_timeout_ms=10000,
                acquisition_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=500,
                heartbeat_interval_ms=0,
            )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_collects_attempts() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    probe.results = [
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="nearest_enemy",
            status="landed_adjacent",
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            completion_timestamp_ms=1400,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=200,
            fuel_before=900,
            fuel_after=820,
            world_timestamp_before=950,
            world_timestamp_after=1450,
            enemy=_enemy(tank_id=50),
            landing_target=_target(),
            landed_signal_received=True,
            landed_x=119,
            landed_y=130,
            enemy_still_visible=True,
            enemy_distance_after=1,
            enemy_x_after=120,
            enemy_y_after=130,
            message_start_index=0,
            message_end_index=1,
            snapshot_before=_snapshot(1000),
            snapshot_after=_snapshot(1400),
        ),
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="nearest_enemy",
            status="landed_adjacent",
            acquisition_started_ms=1100,
            acquisition_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=200,
            fuel_before=820,
            fuel_after=760,
            world_timestamp_before=1450,
            world_timestamp_after=1550,
            enemy=_enemy(tank_id=51, x=121, y=130),
            landing_target=TeleportTargetDict(label="enemy_51_121_130", x=120, y=130),
            landed_signal_received=True,
            landed_x=120,
            landed_y=130,
            enemy_still_visible=True,
            enemy_distance_after=1,
            enemy_x_after=121,
            enemy_y_after=130,
            message_start_index=1,
            message_end_index=2,
            snapshot_before=_snapshot(1100),
            snapshot_after=_snapshot(1500),
        ),
    ]

    def _wait_initial(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page, provider, started_ms, timeout_ms)
        return (
            1200,
            make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_hooks.wait_for_initial_self_state = _wait_initial

    session = probe.execute_probe(
        acquisition_strategy="nearest_enemy",
        max_attempts=2,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        heartbeat_interval_ms=0,
    )

    assert len(session["attempts"]) == 2
    assert probe.acquisition_strategies == ["nearest_enemy", "nearest_enemy"]
    assert probe.excluded_tank_ids == [frozenset(), frozenset({50})]
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert session["startup_timing"]["first_attempt_started_ms"] == 1000
    assert recorded.browser_type.launches == [False]


def test_execute_probe_does_not_exclude_when_attempt_has_no_enemy() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    probe.results = [
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="map_open",
            status="no_enemy",
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1100,
            teleport_started_ms=None,
            completion_timestamp_ms=1200,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=None,
            fuel_before=900,
            fuel_after=900,
            world_timestamp_before=950,
            world_timestamp_after=1150,
            enemy=None,
            landing_target=None,
            landed_signal_received=False,
            landed_x=100,
            landed_y=100,
            enemy_still_visible=False,
            enemy_distance_after=None,
            enemy_x_after=None,
            enemy_y_after=None,
            message_start_index=0,
            message_end_index=0,
            snapshot_before=_snapshot(1000),
            snapshot_after=_snapshot(1200),
        ),
        EnemyTeleportAttemptResultDict(
            acquisition_strategy="map_open",
            status="landed_adjacent",
            acquisition_started_ms=1300,
            acquisition_sync_timestamp_ms=1400,
            teleport_started_ms=1500,
            completion_timestamp_ms=1700,
            acquisition_elapsed_ms=100,
            teleport_elapsed_ms=200,
            fuel_before=900,
            fuel_after=840,
            world_timestamp_before=1150,
            world_timestamp_after=1650,
            enemy=_enemy(tank_id=60, x=121, y=130),
            landing_target=TeleportTargetDict(label="enemy_60_121_130", x=120, y=130),
            landed_signal_received=True,
            landed_x=120,
            landed_y=130,
            enemy_still_visible=True,
            enemy_distance_after=1,
            enemy_x_after=121,
            enemy_y_after=130,
            message_start_index=0,
            message_end_index=1,
            snapshot_before=_snapshot(1300),
            snapshot_after=_snapshot(1700),
        ),
    ]

    def _wait_initial(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page, provider, started_ms, timeout_ms)
        return (
            1200,
            make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_hooks.wait_for_initial_self_state = _wait_initial

    session = probe.execute_probe(
        acquisition_strategy="map_open",
        max_attempts=2,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        heartbeat_interval_ms=0,
    )

    assert len(session["attempts"]) == 2
    assert probe.excluded_tank_ids == [frozenset(), frozenset()]


def test_run_enemy_teleport_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    enemy_probe_module.EnemyTeleportProbe = _FakeEnemyTeleportProbe
    session = run_enemy_teleport_probe(
        "https://tankpit.com/play",
        "enemy_teleport_probe.json",
        acquisition_strategy="map_open",
        max_attempts=3,
    )

    written = fake_fs.read_text(Path("enemy_teleport_probe.json"))
    decoded = decode_enemy_teleport_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("enemy_teleport_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session == decoded
    assert session["capture_session_path"] == "enemy_teleport_probe.capture_session.json"
    assert session["acquisition_strategy"] == "map_open"
    assert capture_decoded["session_id"] == "enemy-session"
