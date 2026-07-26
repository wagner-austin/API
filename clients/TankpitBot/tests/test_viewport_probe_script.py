"""Tests for scripts.viewport_probe."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import viewport_probe as viewport_script
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab.viewport_probe import ViewportProbeSessionDict
from tests.test_enemy_teleport_probe_script import _FakeSyncPlaywrightFactory


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = viewport_script.run_viewport_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    viewport_script.run_viewport_probe = original_run


def _session() -> ViewportProbeSessionDict:
    return ViewportProbeSessionDict(
        session_id="viewport-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="viewport_probe.capture_session.json",
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
        walk_steps_per_phase=16,
        long_offsets=[6, 10, 14, 18, 24],
        walks_sent_off=16,
        longs_sent_off=5,
        walks_sent_on=16,
        longs_sent_on=5,
        toggles_sent=2,
        ack_states=[True, False],
        fuel_before=1000,
        fuel_after=520,
    )


def test_parse_bool_env() -> None:
    assert viewport_script._parse_bool_env("1") is True
    assert viewport_script._parse_bool_env(None) is False


def test_format_saved_path() -> None:
    assert (
        viewport_script._format_saved_path("viewport_probe.json") == "Saved to: viewport_probe.json"
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
        initial_sync_timeout_ms: int = 10000,
    ) -> ViewportProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    viewport_script.run_viewport_probe = _fake_run

    assert viewport_script.main() == 0
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/viewport-")
    assert output.endswith(".json")
    assert captured[2:] == [False, True, 10000]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_VIEWPORT_OUTPUT": "custom_viewport.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_VIEWPORT_INITIAL_SYNC_TIMEOUT_MS": "9000",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = True,
        initial_sync_timeout_ms: int = 10000,
    ) -> ViewportProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                initial_sync_timeout_ms,
            ]
        )
        return _session()

    viewport_script.run_viewport_probe = _fake_run

    assert viewport_script.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_viewport.json",
        True,
        True,
        9000,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab.viewport_probe as viewport_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = viewport_module.run_viewport_probe
    viewport_module.run_viewport_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.viewport_probe"]
    try:
        sys.modules.pop("scripts.viewport_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.viewport_probe", run_name="__main__")
    finally:
        viewport_module.run_viewport_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
