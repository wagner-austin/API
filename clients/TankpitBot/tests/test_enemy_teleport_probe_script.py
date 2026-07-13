"""Tests for scripts.enemy_teleport_probe."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import enemy_teleport_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.action_lab.enemy_teleport_types import EnemyTeleportProbeSessionDict


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_get_argv = core_hooks.get_argv
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = enemy_teleport_probe.run_enemy_teleport_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.get_argv = original_get_argv
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    enemy_teleport_probe.run_enemy_teleport_probe = original_run


def _session() -> EnemyTeleportProbeSessionDict:
    return EnemyTeleportProbeSessionDict(
        session_id="enemy-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        acquisition_strategy="map_open",
        max_attempts=3,
        capture_session_path="enemy_teleport_probe.capture_session.json",
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
        settle_delay_ms=500,
        attempts=[],
    )


class _FakeBrowserType:
    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
        args: list[str] | None = None,
    ) -> BrowserProtocol:
        _ = (headless, slow_mo, timeout, args)
        raise AssertionError("playwright factory should not be invoked in this script test")


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium: BrowserTypeProtocol = _FakeBrowserType()

    def stop(self) -> None:
        return None


class _FakeSyncPlaywrightManager:
    def __init__(self) -> None:
        self._playwright: PlaywrightProtocol = _FakePlaywright()

    def start(self) -> PlaywrightProtocol:
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        return self._playwright

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


class _FakeSyncPlaywrightFactory:
    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        return _FakeSyncPlaywrightManager()


def test_parse_bool_env() -> None:
    assert enemy_teleport_probe._parse_bool_env("true") is True
    assert enemy_teleport_probe._parse_bool_env("1") is True
    assert enemy_teleport_probe._parse_bool_env("yes") is True
    assert enemy_teleport_probe._parse_bool_env("false") is False
    assert enemy_teleport_probe._parse_bool_env(None) is False


def test_parse_optional_int_arg() -> None:
    assert enemy_teleport_probe._parse_optional_int_arg(["prog"], "--max-attempts") is None
    assert (
        enemy_teleport_probe._parse_optional_int_arg(
            ["prog", "--max-attempts", "9"],
            "--max-attempts",
        )
        == 9
    )
    with pytest.raises(ValueError, match="requires an integer value"):
        enemy_teleport_probe._parse_optional_int_arg(["prog", "--max-attempts"], "--max-attempts")


def test_parse_optional_strategy_arg() -> None:
    assert enemy_teleport_probe._parse_optional_strategy_arg(["prog"]) is None
    assert (
        enemy_teleport_probe._parse_optional_strategy_arg(
            ["prog", "--acquisition-strategy", "map_open"]
        )
        == "map_open"
    )
    with pytest.raises(ValueError, match="requires a value"):
        enemy_teleport_probe._parse_optional_strategy_arg(["prog", "--acquisition-strategy"])


def test_parse_acquisition_strategy_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="unsupported acquisition strategy"):
        enemy_teleport_probe._parse_acquisition_strategy("bad")


def test_parse_acquisition_strategy_accepts_nearest_enemy() -> None:
    assert enemy_teleport_probe._parse_acquisition_strategy("nearest_enemy") == "nearest_enemy"


def test_format_saved_path() -> None:
    assert (
        enemy_teleport_probe._format_saved_path("enemy_teleport_probe.json")
        == "Saved to: enemy_teleport_probe.json"
    )


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    captured: list[str | bool | int] = []
    factory_calls: list[str] = []
    logging_levels: list[str] = []

    script_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["enemy_teleport_probe"]
    core_hooks.sync_playwright = None

    def _get_sync_playwright() -> SyncPlaywrightFactoryProtocol:
        factory_calls.append("factory")
        return _FakeSyncPlaywrightFactory()

    core_hooks.get_sync_playwright = _get_sync_playwright

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        acquisition_strategy: str = "map_open",
        max_attempts: int = 3,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        settle_delay_ms: int = 500,
    ) -> EnemyTeleportProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                acquisition_strategy,
                max_attempts,
                initial_sync_timeout_ms,
                acquisition_timeout_ms,
                teleport_timeout_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    enemy_teleport_probe.run_enemy_teleport_probe = _fake_run

    assert enemy_teleport_probe.main() == 0
    assert logging_levels == ["INFO"]
    assert factory_calls == ["factory"]
    assert captured == [
        "https://tankpit.com/play",
        "enemy_teleport_probe.json",
        False,
        False,
        "map_open",
        3,
        10000,
        3000,
        10000,
        500,
    ]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_and_cli_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_ENEMY_TELEPORT_PROBE_OUTPUT": "custom.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_ENEMY_TELEPORT_ACQUISITION_STRATEGY": "map_open",
        "TANKPIT_ENEMY_TELEPORT_MAX_ATTEMPTS": "5",
        "TANKPIT_ENEMY_TELEPORT_INITIAL_SYNC_TIMEOUT_MS": "4000",
        "TANKPIT_ENEMY_TELEPORT_ACQUISITION_TIMEOUT_MS": "4100",
        "TANKPIT_ENEMY_TELEPORT_TIMEOUT_MS": "11000",
        "TANKPIT_ENEMY_TELEPORT_SETTLE_MS": "750",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: [
        "enemy_teleport_probe",
        "--max-attempts",
        "3",
        "--initial-sync-timeout-ms",
        "9000",
        "--acquisition-strategy",
        "map_open",
    ]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        acquisition_strategy: str = "map_open",
        max_attempts: int = 3,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        settle_delay_ms: int = 500,
    ) -> EnemyTeleportProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                acquisition_strategy,
                max_attempts,
                initial_sync_timeout_ms,
                acquisition_timeout_ms,
                teleport_timeout_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    enemy_teleport_probe.run_enemy_teleport_probe = _fake_run

    assert enemy_teleport_probe.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom.json",
        True,
        True,
        "map_open",
        3,
        9000,
        4100,
        11000,
        750,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab as action_lab

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["scripts.enemy_teleport_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = action_lab.run_enemy_teleport_probe
    action_lab.run_enemy_teleport_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.enemy_teleport_probe"]
    try:
        sys.modules.pop("scripts.enemy_teleport_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.enemy_teleport_probe", run_name="__main__")
    finally:
        action_lab.run_enemy_teleport_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
