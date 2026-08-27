"""Tests for scripts.radar_watch_probe."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import radar_watch_probe as radar_watch_script
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab.radar_watch import RadarWatchSessionDict
from tests.test_enemy_teleport_probe_script import _FakeSyncPlaywrightFactory


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = radar_watch_script.run_radar_watch_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    radar_watch_script.run_radar_watch_probe = original_run


def _session() -> RadarWatchSessionDict:
    return RadarWatchSessionDict(
        session_id="radar-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="radar_watch_probe.capture_session.json",
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
        duration_ms=1800000,
        scan_interval_ms=15000,
        map_poll_interval_ms=30000,
        walks_sent=118,
        extras_before=22,
        extras_enabled_at_start=True,
        toggles_sent=1,
        scans_sent=120,
        map_polls_sent=60,
        extras_after=22,
    )


def test_parse_bool_env() -> None:
    assert radar_watch_script._parse_bool_env("1") is True
    assert radar_watch_script._parse_bool_env(None) is False


def test_format_saved_path() -> None:
    assert (
        radar_watch_script._format_saved_path("radar_watch_probe.json")
        == "Saved to: radar_watch_probe.json"
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
        prefer_account: bool = True,
        duration_ms: int = 1800000,
        scan_interval_ms: int = 15000,
        map_poll_interval_ms: int = 30000,
        initial_sync_timeout_ms: int = 10000,
    ) -> RadarWatchSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                duration_ms,
                scan_interval_ms,
                map_poll_interval_ms,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    radar_watch_script.run_radar_watch_probe = _fake_run

    assert radar_watch_script.main() == 0
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/radar-watch-")
    assert output.endswith(".json")
    assert captured[2:] == [False, True, 1800000, 15000, 30000, 10000]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_RADAR_WATCH_OUTPUT": "custom_radar.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_RADAR_WATCH_DURATION_MS": "600000",
        "TANKPIT_RADAR_WATCH_SCAN_INTERVAL_MS": "10000",
        "TANKPIT_RADAR_WATCH_MAP_POLL_INTERVAL_MS": "20000",
        "TANKPIT_RADAR_WATCH_INITIAL_SYNC_TIMEOUT_MS": "9000",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = True,
        duration_ms: int = 1800000,
        scan_interval_ms: int = 15000,
        map_poll_interval_ms: int = 30000,
        initial_sync_timeout_ms: int = 10000,
    ) -> RadarWatchSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                duration_ms,
                scan_interval_ms,
                map_poll_interval_ms,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    radar_watch_script.run_radar_watch_probe = _fake_run

    assert radar_watch_script.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_radar.json",
        True,
        True,
        600000,
        10000,
        20000,
        9000,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab.radar_watch as radar_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = radar_module.run_radar_watch_probe
    radar_module.run_radar_watch_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.radar_watch_probe"]
    try:
        sys.modules.pop("scripts.radar_watch_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.radar_watch_probe", run_name="__main__")
    finally:
        radar_module.run_radar_watch_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
