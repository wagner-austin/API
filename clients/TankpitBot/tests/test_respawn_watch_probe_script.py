"""Tests for scripts.respawn_watch_probe."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import respawn_watch_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab.enemy_teleport_types import EnemyTeleportProbeSessionDict
from tests.test_enemy_teleport_probe_script import _FakeSyncPlaywrightFactory


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = respawn_watch_probe.run_respawn_watch_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    respawn_watch_probe.run_respawn_watch_probe = original_run


def _session() -> EnemyTeleportProbeSessionDict:
    return EnemyTeleportProbeSessionDict(
        session_id="respawn-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        acquisition_strategy="map_open",
        max_attempts=4,
        capture_session_path="respawn_watch_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=0,
        heartbeat_interval_ms=0,
        attempts=[],
    )


def test_parse_bool_env() -> None:
    assert respawn_watch_probe._parse_bool_env("true") is True
    assert respawn_watch_probe._parse_bool_env(None) is False


def test_format_saved_path() -> None:
    assert (
        respawn_watch_probe._format_saved_path("respawn_watch_probe.json")
        == "Saved to: respawn_watch_probe.json"
    )


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    captured: list[str | bool | int] = []
    factory_calls: list[str] = []

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = None

    def _get_sync_playwright() -> _FakeSyncPlaywrightFactory:
        factory_calls.append("factory")
        return _FakeSyncPlaywrightFactory()

    core_hooks.get_sync_playwright = _get_sync_playwright

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        max_attempts: int = 4,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        engage_ms: int = 30000,
        shot_interval_ms: int = 2000,
        poll_ms: int = 60000,
        poll_interval_ms: int = 2000,
    ) -> EnemyTeleportProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_attempts,
                initial_sync_timeout_ms,
                acquisition_timeout_ms,
                teleport_timeout_ms,
                engage_ms,
                shot_interval_ms,
                poll_ms,
                poll_interval_ms,
            ]
        )
        return _session()

    respawn_watch_probe.run_respawn_watch_probe = _fake_run

    assert respawn_watch_probe.main() == 0
    assert factory_calls == ["factory"]
    assert captured == [
        "https://tankpit.com/play",
        "respawn_watch_probe.json",
        False,
        False,
        4,
        10000,
        3000,
        10000,
        30000,
        2000,
        60000,
        2000,
    ]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_RESPAWN_WATCH_PROBE_OUTPUT": "custom_respawn.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_RESPAWN_WATCH_MAX_ATTEMPTS": "6",
        "TANKPIT_RESPAWN_WATCH_INITIAL_SYNC_TIMEOUT_MS": "9000",
        "TANKPIT_RESPAWN_WATCH_ACQUISITION_TIMEOUT_MS": "3100",
        "TANKPIT_RESPAWN_WATCH_TELEPORT_TIMEOUT_MS": "11000",
        "TANKPIT_RESPAWN_WATCH_ENGAGE_MS": "20000",
        "TANKPIT_RESPAWN_WATCH_SHOT_INTERVAL_MS": "1500",
        "TANKPIT_RESPAWN_WATCH_POLL_MS": "50000",
        "TANKPIT_RESPAWN_WATCH_POLL_INTERVAL_MS": "2500",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        max_attempts: int = 4,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        engage_ms: int = 30000,
        shot_interval_ms: int = 2000,
        poll_ms: int = 60000,
        poll_interval_ms: int = 2000,
    ) -> EnemyTeleportProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_attempts,
                initial_sync_timeout_ms,
                acquisition_timeout_ms,
                teleport_timeout_ms,
                engage_ms,
                shot_interval_ms,
                poll_ms,
                poll_interval_ms,
            ]
        )
        return _session()

    respawn_watch_probe.run_respawn_watch_probe = _fake_run

    assert respawn_watch_probe.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_respawn.json",
        True,
        True,
        6,
        9000,
        3100,
        11000,
        20000,
        1500,
        50000,
        2500,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab.respawn_watch as respawn_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = respawn_module.run_respawn_watch_probe
    respawn_module.run_respawn_watch_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.respawn_watch_probe"]
    try:
        sys.modules.pop("scripts.respawn_watch_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.respawn_watch_probe", run_name="__main__")
    finally:
        respawn_module.run_respawn_watch_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
