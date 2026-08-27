"""Tests for scripts.fuel_probe."""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import fuel_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.action_lab.fuel_probe_types import FuelProbeSessionDict


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    """Restore script hooks after each test."""
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_get_argv = core_hooks.get_argv
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = fuel_probe.run_fuel_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.get_argv = original_get_argv
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    fuel_probe.run_fuel_probe = original_run


def _session() -> FuelProbeSessionDict:
    """Build a sample script session payload."""
    return FuelProbeSessionDict(
        session_id="fuel-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        target_pickups=3,
        max_attempts=3,
        capture_session_path="fuel_probe.capture_session.json",
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
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
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
    ) -> BrowserProtocol:
        """Raise if the script tries to open a browser in this test."""
        _ = (headless, slow_mo, timeout, args)
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
    assert fuel_probe._parse_bool_env("true") is True
    assert fuel_probe._parse_bool_env("1") is True
    assert fuel_probe._parse_bool_env("yes") is True
    assert fuel_probe._parse_bool_env("false") is False
    assert fuel_probe._parse_bool_env(None) is False


def test_parse_optional_int_arg() -> None:
    """Optional integer arg parsing handles presence and absence."""
    assert fuel_probe._parse_optional_int_arg(["prog"], "--max-attempts") is None
    assert (
        fuel_probe._parse_optional_int_arg(["prog", "--max-attempts", "9"], "--max-attempts") == 9
    )
    with pytest.raises(ValueError, match="requires an integer value"):
        fuel_probe._parse_optional_int_arg(["prog", "--max-attempts"], "--max-attempts")


def test_format_saved_path() -> None:
    """Saved-path formatting is stable."""
    assert fuel_probe._format_saved_path("fuel_probe.json") == "Saved to: fuel_probe.json"


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    """Fuel probe script uses default values when no env or CLI overrides exist."""
    captured: list[str | bool | int] = []
    factory_calls: list[str] = []
    logging_levels: list[str] = []

    script_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["fuel_probe"]
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
        target_pickups: int = 3,
        max_attempts: int = 3,
        initial_sync_timeout_ms: int = 10000,
        map_sync_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        radar_timeout_ms: int = 3000,
        pickup_timeout_ms: int = 3000,
        settle_delay_ms: int = 500,
    ) -> FuelProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                target_pickups,
                max_attempts,
                initial_sync_timeout_ms,
                map_sync_timeout_ms,
                teleport_timeout_ms,
                radar_timeout_ms,
                pickup_timeout_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    fuel_probe.run_fuel_probe = _fake_run

    assert fuel_probe.main() == 0
    assert logging_levels == ["INFO"]
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/fuel-")
    assert output.endswith(".json")
    assert captured[2:] == [False, False, 3, 9, 10000, 3000, 10000, 3000, 3000, 500]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_and_cli_overrides(_restore_script_hooks: None) -> None:
    """Fuel probe script applies CLI and env overrides."""
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_FUEL_PROBE_OUTPUT": "custom.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_FUEL_PROBE_TARGET_PICKUPS": "4",
        "TANKPIT_FUEL_PROBE_MAX_ATTEMPTS": "5",
        "TANKPIT_FUEL_PROBE_INITIAL_SYNC_TIMEOUT_MS": "4000",
        "TANKPIT_FUEL_PROBE_MAP_SYNC_TIMEOUT_MS": "4100",
        "TANKPIT_FUEL_PROBE_TELEPORT_TIMEOUT_MS": "11000",
        "TANKPIT_FUEL_PROBE_RADAR_TIMEOUT_MS": "4200",
        "TANKPIT_FUEL_PROBE_PICKUP_TIMEOUT_MS": "4300",
        "TANKPIT_FUEL_PROBE_SETTLE_MS": "750",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: [
        "fuel_probe",
        "--target-pickups",
        "3",
        "--max-attempts",
        "7",
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
        target_pickups: int = 3,
        max_attempts: int = 3,
        initial_sync_timeout_ms: int = 10000,
        map_sync_timeout_ms: int = 3000,
        teleport_timeout_ms: int = 10000,
        radar_timeout_ms: int = 3000,
        pickup_timeout_ms: int = 3000,
        settle_delay_ms: int = 500,
    ) -> FuelProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                target_pickups,
                max_attempts,
                initial_sync_timeout_ms,
                map_sync_timeout_ms,
                teleport_timeout_ms,
                radar_timeout_ms,
                pickup_timeout_ms,
                settle_delay_ms,
            ]
        )
        return _session()

    fuel_probe.run_fuel_probe = _fake_run

    assert fuel_probe.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom.json",
        True,
        True,
        3,
        7,
        9000,
        4100,
        11000,
        4200,
        4300,
        750,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    """Module entrypoint exits through main()."""
    import tankpit_bot.action_lab as action_lab

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["scripts.fuel_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = action_lab.run_fuel_probe
    action_lab.run_fuel_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.fuel_probe"]
    try:
        sys.modules.pop("scripts.fuel_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.fuel_probe", run_name="__main__")
    finally:
        action_lab.run_fuel_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
