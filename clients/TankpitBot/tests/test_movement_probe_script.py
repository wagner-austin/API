"""Tests for scripts.movement_probe."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import movement_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.action_lab.movement_probe_types import MovementProbeSessionDict
from tankpit_bot.action_lab.types import TeleportTargetDict


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    """Restore script hooks after each test."""
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_get_argv = core_hooks.get_argv
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = movement_probe.run_movement_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.get_argv = original_get_argv
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    movement_probe.run_movement_probe = original_run


def _session() -> MovementProbeSessionDict:
    """Build a sample script session payload."""
    return MovementProbeSessionDict(
        session_id="movement-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_targets=3,
        capture_session_path="movement_probe.capture_session.json",
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
        move_timeout_ms=5000,
        settle_delay_ms=500,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        targets=[],
        attempts=[],
    )


class _FakeBrowserType:
    """Browser-type fake that should never actually launch."""

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> BrowserProtocol:
        """Raise if the script tries to open a browser in this test."""
        _ = (headless, slow_mo, timeout, args, env)
        raise AssertionError("playwright factory should not be invoked in this script test")


class _FakePlaywright:
    """Minimal Playwright fake for script tests."""

    def __init__(self) -> None:
        """Expose a fake chromium launcher."""
        self.chromium: BrowserTypeProtocol = _FakeBrowserType()

    def stop(self) -> None:
        """Ignore stop."""


class _FakeSyncPlaywrightManager:
    """Context manager returning a fake Playwright object."""

    def __init__(self) -> None:
        """Initialize the fake Playwright object."""
        self._playwright: PlaywrightProtocol = _FakePlaywright()

    def start(self) -> PlaywrightProtocol:
        """Return the wrapped Playwright fake."""
        return self._playwright

    def __enter__(self) -> PlaywrightProtocol:
        """Return the wrapped Playwright fake."""
        return self._playwright

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Ignore teardown."""
        _ = (exc_type, exc_val, exc_tb)


class _FakeSyncPlaywrightFactory:
    """Factory returning the fake Playwright context manager."""

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        """Return a new fake context manager."""
        return _FakeSyncPlaywrightManager()


def test_parse_bool_env() -> None:
    """Boolean env parsing accepts the supported truthy values."""
    assert movement_probe._parse_bool_env("true") is True
    assert movement_probe._parse_bool_env("1") is True
    assert movement_probe._parse_bool_env("yes") is True
    assert movement_probe._parse_bool_env("false") is False
    assert movement_probe._parse_bool_env(None) is False


def test_parse_optional_int_arg() -> None:
    """Optional integer arg parsing handles presence and absence."""
    assert movement_probe._parse_optional_int_arg(["prog"], "--max-targets") is None
    assert (
        movement_probe._parse_optional_int_arg(["prog", "--max-targets", "9"], "--max-targets") == 9
    )
    with pytest.raises(ValueError, match="requires an integer value"):
        movement_probe._parse_optional_int_arg(["prog", "--max-targets"], "--max-targets")


def test_parse_targets_cli() -> None:
    """Movement target parsing reuses the shared target parser."""
    assert movement_probe._parse_targets_cli(["prog"]) is None
    assert movement_probe._parse_targets_cli(["prog", "--targets", "1:2"]) == [
        TeleportTargetDict(label="target_0", x=1, y=2)
    ]
    with pytest.raises(ValueError, match="requires a value"):
        movement_probe._parse_targets_cli(["prog", "--targets"])


def test_has_flag() -> None:
    """Flag parsing is stable."""
    assert movement_probe._has_flag(["prog", "--queue-map-open"], "--queue-map-open") is True
    assert movement_probe._has_flag(["prog"], "--queue-map-open") is False


def test_format_saved_path() -> None:
    """Saved-path formatting is stable."""
    assert (
        movement_probe._format_saved_path("movement_probe.json") == "Saved to: movement_probe.json"
    )


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    """Movement probe script uses default values when no env or CLI overrides exist."""
    captured: list[str | bool | int | list[TeleportTargetDict] | None] = []
    factory_calls: list[str] = []
    logging_levels: list[str] = []

    script_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["movement_probe"]
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
        max_targets: int = 3,
        initial_sync_timeout_ms: int = 10000,
        move_timeout_ms: int = 5000,
        queue_map_open_during_move: bool = False,
        map_open_delay_ms: int = 0,
        settle_delay_ms: int = 500,
    ) -> MovementProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                explicit_targets,
                max_targets,
                initial_sync_timeout_ms,
                move_timeout_ms,
                queue_map_open_during_move,
                map_open_delay_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    movement_probe.run_movement_probe = _fake_run

    assert movement_probe.main() == 0
    assert logging_levels == ["INFO"]
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/movement-")
    assert output.endswith(".json")
    assert captured[2:] == [False, False, None, 3, 10000, 5000, False, 0, 500]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_and_cli_overrides(_restore_script_hooks: None) -> None:
    """Movement probe script applies CLI and env overrides."""
    recorded: list[str | bool | int | list[TeleportTargetDict] | None] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_MOVEMENT_PROBE_OUTPUT": "custom.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_MOVEMENT_MAX_TARGETS": "5",
        "TANKPIT_MOVEMENT_INITIAL_SYNC_TIMEOUT_MS": "4000",
        "TANKPIT_MOVEMENT_TIMEOUT_MS": "4100",
        "TANKPIT_MOVEMENT_QUEUE_MAP_OPEN": "yes",
        "TANKPIT_MOVEMENT_MAP_OPEN_DELAY_MS": "250",
        "TANKPIT_MOVEMENT_SETTLE_MS": "750",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: [
        "movement_probe",
        "--targets",
        "1:2,3:4",
        "--max-targets",
        "7",
        "--initial-sync-timeout-ms",
        "9000",
        "--move-timeout-ms",
        "9200",
        "--map-open-delay-ms",
        "400",
        "--queue-map-open",
    ]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        explicit_targets: list[TeleportTargetDict] | None = None,
        max_targets: int = 3,
        initial_sync_timeout_ms: int = 10000,
        move_timeout_ms: int = 5000,
        queue_map_open_during_move: bool = False,
        map_open_delay_ms: int = 0,
        settle_delay_ms: int = 500,
    ) -> MovementProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                explicit_targets,
                max_targets,
                initial_sync_timeout_ms,
                move_timeout_ms,
                queue_map_open_during_move,
                map_open_delay_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    movement_probe.run_movement_probe = _fake_run

    assert movement_probe.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom.json",
        True,
        True,
        [
            TeleportTargetDict(label="target_0", x=1, y=2),
            TeleportTargetDict(label="target_1", x=3, y=4),
        ],
        7,
        9000,
        9200,
        True,
        400,
        750,
    ]


def test_script_module_runs_main_via_runpy(_restore_script_hooks: None) -> None:
    """Executing the script module calls main and exits cleanly."""
    import tankpit_bot.action_lab as action_lab

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["movement_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = action_lab.run_movement_probe
    action_lab.run_movement_probe = lambda target_url, output_path, **kwargs: _session()
    original_argv = sys.argv[:]
    sys.argv = ["movement_probe"]
    try:
        sys.modules.pop("scripts.movement_probe", None)
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("scripts.movement_probe", run_name="__main__")
    finally:
        action_lab.run_movement_probe = original_run
        sys.argv = original_argv
    assert excinfo.value.code == 0
