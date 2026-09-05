"""Tests for scripts.teleport_probe."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import teleport_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.action_lab import DEFAULT_TELEPORT_STRATEGY
from tankpit_bot.action_lab.types import TeleportProbeSessionDict, TeleportTargetDict


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_get_argv = core_hooks.get_argv
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = teleport_probe.run_teleport_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.get_argv = original_get_argv
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    teleport_probe.run_teleport_probe = original_run


def _session() -> TeleportProbeSessionDict:
    return TeleportProbeSessionDict(
        session_id="script-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        teleport_strategy="sync_before_teleport",
        max_targets=3,
        capture_session_path="teleport_probe.capture_session.json",
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
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        targets=[],
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
        env: dict[str, str] | None = None,
    ) -> BrowserProtocol:
        _ = (headless, slow_mo, timeout, args, env)
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
    assert teleport_probe._parse_bool_env("true") is True
    assert teleport_probe._parse_bool_env("1") is True
    assert teleport_probe._parse_bool_env("yes") is True
    assert teleport_probe._parse_bool_env("false") is False
    assert teleport_probe._parse_bool_env(None) is False


def test_parse_optional_int_arg() -> None:
    assert teleport_probe._parse_optional_int_arg(["prog"], "--step-x") is None
    assert teleport_probe._parse_optional_int_arg(["prog", "--step-x", "9"], "--step-x") == 9
    with pytest.raises(ValueError, match="requires an integer value"):
        teleport_probe._parse_optional_int_arg(["prog", "--step-x"], "--step-x")


def test_parse_targets_cli() -> None:
    assert teleport_probe._parse_targets_cli(["prog"]) is None
    assert teleport_probe._parse_targets_cli(["prog", "--targets", "1:2"]) == [
        TeleportTargetDict(label="target_0", x=1, y=2)
    ]
    with pytest.raises(ValueError, match="requires a value"):
        teleport_probe._parse_targets_cli(["prog", "--targets"])


def test_parse_optional_strategy_arg() -> None:
    assert teleport_probe._parse_optional_strategy_arg(["prog"]) is None
    assert (
        teleport_probe._parse_optional_strategy_arg(
            ["prog", "--teleport-strategy", "immediate_after_map_open"]
        )
        == "immediate_after_map_open"
    )
    with pytest.raises(ValueError, match="requires a value"):
        teleport_probe._parse_optional_strategy_arg(["prog", "--teleport-strategy"])


def test_parse_teleport_strategy_rejects_unsupported_values() -> None:
    assert teleport_probe._parse_teleport_strategy("sync_before_teleport") == "sync_before_teleport"
    assert (
        teleport_probe._parse_teleport_strategy("immediate_after_map_open")
        == "immediate_after_map_open"
    )
    with pytest.raises(ValueError, match="unsupported teleport strategy"):
        teleport_probe._parse_teleport_strategy("bad")


def test_format_saved_path() -> None:
    assert (
        teleport_probe._format_saved_path("teleport_probe.json") == "Saved to: teleport_probe.json"
    )


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    captured: list[str | bool | int | list[TeleportTargetDict] | None] = []
    factory_calls: list[str] = []
    logging_levels: list[str] = []

    script_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["teleport_probe"]
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
        explicit_targets: list[TeleportTargetDict] | None = None,
        box_step_x: int = 8,
        box_step_y: int = 8,
        max_targets: int | None = None,
        teleport_strategy: str = DEFAULT_TELEPORT_STRATEGY,
        initial_sync_timeout_ms: int = 10000,
        map_sync_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        settle_delay_ms: int = 500,
    ) -> TeleportProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                explicit_targets,
                box_step_x,
                box_step_y,
                max_targets,
                teleport_strategy,
                initial_sync_timeout_ms,
                map_sync_timeout_ms,
                teleport_timeout_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    teleport_probe.run_teleport_probe = _fake_run

    assert teleport_probe.main() == 0
    assert logging_levels == ["INFO"]
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/teleport-")
    assert output.endswith(".json")
    assert captured[2:] == [
        False,
        False,
        None,
        8,
        8,
        None,
        "immediate_after_map_open",
        10000,
        3000,
        10000,
        500,
    ]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_and_cli_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int | list[TeleportTargetDict] | None] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_TELEPORT_PROBE_OUTPUT": "custom.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_TELEPORT_STRATEGY": "sync_before_teleport",
        "TANKPIT_TELEPORT_MAX_TARGETS": "5",
        "TANKPIT_TELEPORT_INITIAL_SYNC_TIMEOUT_MS": "4000",
        "TANKPIT_TELEPORT_MAP_SYNC_TIMEOUT_MS": "4000",
        "TANKPIT_TELEPORT_TIMEOUT_MS": "11000",
        "TANKPIT_TELEPORT_SETTLE_MS": "750",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: [
        "teleport_probe",
        "--targets",
        "5:6,7:8",
        "--step-x",
        "11",
        "--step-y",
        "12",
        "--max-targets",
        "3",
        "--teleport-strategy",
        "immediate_after_map_open",
        "--initial-sync-timeout-ms",
        "9000",
    ]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        explicit_targets: list[TeleportTargetDict] | None = None,
        box_step_x: int = 8,
        box_step_y: int = 8,
        max_targets: int | None = None,
        teleport_strategy: str = DEFAULT_TELEPORT_STRATEGY,
        initial_sync_timeout_ms: int = 10000,
        map_sync_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        settle_delay_ms: int = 500,
    ) -> TeleportProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                explicit_targets,
                box_step_x,
                box_step_y,
                max_targets,
                teleport_strategy,
                initial_sync_timeout_ms,
                map_sync_timeout_ms,
                teleport_timeout_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    teleport_probe.run_teleport_probe = _fake_run

    assert teleport_probe.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom.json",
        True,
        True,
        [
            TeleportTargetDict(label="target_0", x=5, y=6),
            TeleportTargetDict(label="target_1", x=7, y=8),
        ],
        11,
        12,
        3,
        "immediate_after_map_open",
        9000,
        4000,
        11000,
        750,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab as action_lab

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["scripts.teleport_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = action_lab.run_teleport_probe
    action_lab.run_teleport_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.teleport_probe"]
    try:
        sys.modules.pop("scripts.teleport_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.teleport_probe", run_name="__main__")
    finally:
        action_lab.run_teleport_probe = original_run
        sys.argv = old_argv
    assert exc.value.code == 0
