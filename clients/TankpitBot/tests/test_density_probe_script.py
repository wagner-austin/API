"""Tests for scripts.density_probe."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import density_probe as density_script
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab.density_probe import DensityProbeSessionDict
from tests.test_enemy_teleport_probe_script import _FakeSyncPlaywrightFactory


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = density_script.run_density_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    density_script.run_density_probe = original_run


def _session() -> DensityProbeSessionDict:
    return DensityProbeSessionDict(
        session_id="density-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="density_probe.capture_session.json",
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
        max_extras=12,
        sites_planned=16,
        sites_scanned=12,
        sites_skipped=1,
        refuel_hops=4,
        bootstrap_pickups=2,
        extras_before=22,
        extras_enabled_at_start=False,
        toggles_sent=2,
        extras_after=10,
        fuel_before=1100,
        fuel_after=430,
    )


def test_parse_bool_env() -> None:
    assert density_script._parse_bool_env("1") is True
    assert density_script._parse_bool_env(None) is False


def test_format_saved_path() -> None:
    assert density_script._format_saved_path("density_probe.json") == "Saved to: density_probe.json"


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
        max_extras: int = 12,
        initial_sync_timeout_ms: int = 10000,
    ) -> DensityProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_extras,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    density_script.run_density_probe = _fake_run

    assert density_script.main() == 0
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/density-")
    assert output.endswith(".json")
    assert captured[2:] == [False, True, 12, 10000]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_DENSITY_OUTPUT": "custom_density.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_DENSITY_MAX_EXTRAS": "6",
        "TANKPIT_DENSITY_INITIAL_SYNC_TIMEOUT_MS": "9000",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = True,
        max_extras: int = 12,
        initial_sync_timeout_ms: int = 10000,
    ) -> DensityProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_extras,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    density_script.run_density_probe = _fake_run

    assert density_script.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_density.json",
        True,
        True,
        6,
        9000,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab.density_probe as density_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = density_module.run_density_probe
    density_module.run_density_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.density_probe"]
    try:
        sys.modules.pop("scripts.density_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.density_probe", run_name="__main__")
    finally:
        density_module.run_density_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
