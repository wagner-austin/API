"""Tests for scripts.mine_landing_probe."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import mine_landing_probe as mine_script
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab.mine_landing_probe import MineLandingProbeSessionDict
from tests.test_enemy_teleport_probe_script import _FakeSyncPlaywrightFactory


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = mine_script.run_mine_landing_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    mine_script.run_mine_landing_probe = original_run


def _session() -> MineLandingProbeSessionDict:
    return MineLandingProbeSessionDict(
        session_id="mine-landing-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="mine_landing_probe.capture_session.json",
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
        max_attempts=3,
        max_extras=6,
        search_scans=1,
        search_hops=1,
        attempts=[],
        detonations=0,
        coexists=0,
        displaced_off=3,
        extras_before=24,
        extras_enabled_at_start=True,
        toggles_sent=0,
        extras_after=23,
        fuel_before=1100,
        fuel_after=824,
    )


def test_parse_bool_env() -> None:
    assert mine_script._parse_bool_env("1") is True
    assert mine_script._parse_bool_env(None) is False


def test_format_saved_path() -> None:
    assert mine_script._format_saved_path("mine.json") == "Saved to: mine.json"


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
        max_attempts: int = 3,
        max_extras: int = 6,
        initial_sync_timeout_ms: int = 10000,
    ) -> MineLandingProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_attempts,
                max_extras,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    mine_script.run_mine_landing_probe = _fake_run

    assert mine_script.main() == 0
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/mine-landing-")
    assert output.endswith(".json")
    assert captured[2:] == [False, True, 3, 6, 10000]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_MINE_LANDING_OUTPUT": "custom_mine.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_MINE_LANDING_MAX_ATTEMPTS": "5",
        "TANKPIT_MINE_LANDING_MAX_EXTRAS": "4",
        "TANKPIT_MINE_LANDING_INITIAL_SYNC_TIMEOUT_MS": "9000",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = True,
        max_attempts: int = 3,
        max_extras: int = 6,
        initial_sync_timeout_ms: int = 10000,
    ) -> MineLandingProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_attempts,
                max_extras,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    mine_script.run_mine_landing_probe = _fake_run

    assert mine_script.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_mine.json",
        True,
        True,
        5,
        4,
        9000,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab.mine_landing_probe as mine_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = mine_module.run_mine_landing_probe
    mine_module.run_mine_landing_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.mine_landing_probe"]
    try:
        sys.modules.pop("scripts.mine_landing_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.mine_landing_probe", run_name="__main__")
    finally:
        mine_module.run_mine_landing_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
