"""Tests for scripts.enemy_tracking_probe.

The script resolves the target URL, output path and login flags from
env, and the two sampling bounds from either argv or env -- the only
probe script where a flag and an environment variable compete for the
same value, so both precedence directions are pinned here. The probe
is replaced throughout and the fake browser type raises if anything
tries to launch.
"""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator
from typing import Protocol

import pytest

from scripts import _test_hooks as script_hooks
from scripts import enemy_tracking_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.action_lab.enemy_tracking_types import EnemyTrackingProbeSessionDict
from tests.action_lab._enemy_tracking_harness import _make_session


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    """Restore script and core hooks after each test."""
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_get_argv = core_hooks.get_argv
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = enemy_tracking_probe.run_enemy_tracking_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.get_argv = original_get_argv
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    enemy_tracking_probe.run_enemy_tracking_probe = original_run


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


class _RunTrackingProbeFn(Protocol):
    """The probe run entry point, so the double is checked against it."""

    def __call__(
        self,
        target_url: str,
        output_path: str,
        *,
        headless: bool = ...,
        prefer_account: bool = ...,
        initial_sync_timeout_ms: int = ...,
        acquisition_timeout_ms: int = ...,
        teleport_timeout_ms: int = ...,
        shot_feedback_timeout_ms: int = ...,
        sample_interval_ms: int = ...,
        sample_duration_ms: int = ...,
    ) -> EnemyTrackingProbeSessionDict:
        """Run the probe and return its session."""


def _recording_run(captured: list[str | bool | int]) -> _RunTrackingProbeFn:
    """Build a run double that records every argument the script passed."""

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 5000,
        teleport_timeout_ms: int = 10000,
        shot_feedback_timeout_ms: int = 4000,
        sample_interval_ms: int = 1000,
        sample_duration_ms: int = 120000,
    ) -> EnemyTrackingProbeSessionDict:
        _ = (initial_sync_timeout_ms, acquisition_timeout_ms, teleport_timeout_ms)
        _ = shot_feedback_timeout_ms
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                sample_duration_ms,
                sample_interval_ms,
            ]
        )
        return _make_session()

    return _fake_run


def test_parse_bool_env() -> None:
    """Boolean env parsing accepts the supported truthy values."""
    assert enemy_tracking_probe._parse_bool_env("true") is True
    assert enemy_tracking_probe._parse_bool_env("1") is True
    assert enemy_tracking_probe._parse_bool_env("yes") is True
    assert enemy_tracking_probe._parse_bool_env("YES") is True
    assert enemy_tracking_probe._parse_bool_env("false") is False
    assert enemy_tracking_probe._parse_bool_env("") is False
    assert enemy_tracking_probe._parse_bool_env(None) is False


def test_parse_optional_int_arg() -> None:
    """Optional integer arg parsing handles presence, absence, and a bare flag."""
    flag = "--sample-interval-ms"
    assert enemy_tracking_probe._parse_optional_int_arg(["prog"], flag) is None
    assert enemy_tracking_probe._parse_optional_int_arg(["prog", flag, "250"], flag) == 250
    with pytest.raises(ValueError, match=f"{flag} requires an integer value"):
        enemy_tracking_probe._parse_optional_int_arg(["prog", flag], flag)


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    """No env and no flags leaves the documented sampling defaults."""
    captured: list[str | bool | int] = []
    factory_calls: list[str] = []
    logging_levels: list[str] = []

    script_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["enemy_tracking_probe"]
    core_hooks.sync_playwright = None

    def _get_sync_playwright() -> SyncPlaywrightFactoryProtocol:
        factory_calls.append("factory")
        return _FakeSyncPlaywrightFactory()

    core_hooks.get_sync_playwright = _get_sync_playwright
    enemy_tracking_probe.run_enemy_tracking_probe = _recording_run(captured)

    assert enemy_tracking_probe.main() == 0
    assert logging_levels == ["INFO"]
    assert factory_calls == ["factory"]
    assert captured == [
        "https://tankpit.com/play",
        "enemy_tracking_probe.json",
        False,
        False,
        120000,
        1000,
    ]
    assert callable(core_hooks.sync_playwright)


def test_main_takes_the_sampling_bounds_from_env(_restore_script_hooks: None) -> None:
    """With no flags, the two sampling bounds come from the environment."""
    captured: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_ENEMY_TRACKING_PROBE_OUTPUT": "custom_tracking.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_ENEMY_TRACKING_SAMPLE_DURATION_MS": "60000",
        "TANKPIT_ENEMY_TRACKING_SAMPLE_INTERVAL_MS": "500",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: ["enemy_tracking_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    enemy_tracking_probe.run_enemy_tracking_probe = _recording_run(captured)

    assert enemy_tracking_probe.main() == 0
    assert captured == [
        "https://tankpit.com/custom",
        "custom_tracking.json",
        True,
        True,
        60000,
        500,
    ]


def test_main_lets_a_cli_flag_beat_the_environment(_restore_script_hooks: None) -> None:
    """A flag wins over its env variable, and one flag does not drag the other.

    ``--sample-duration-ms`` is supplied and ``--sample-interval-ms``
    is not, so the duration takes the flag while the interval still
    falls back to the environment.
    """
    captured: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_ENEMY_TRACKING_SAMPLE_DURATION_MS": "60000",
        "TANKPIT_ENEMY_TRACKING_SAMPLE_INTERVAL_MS": "500",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: [
        "enemy_tracking_probe",
        "--sample-duration-ms",
        "9000",
    ]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    enemy_tracking_probe.run_enemy_tracking_probe = _recording_run(captured)

    assert enemy_tracking_probe.main() == 0
    assert captured[4:] == [9000, 500]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    """Module entrypoint exits through main().

    The swap goes on ``action_lab.enemy_tracking`` because that is the
    module the script binds ``run_enemy_tracking_probe`` from, and
    ``runpy`` re-imports the script fresh.
    """
    import tankpit_bot.action_lab.enemy_tracking as probe_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["scripts.enemy_tracking_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = probe_module.run_enemy_tracking_probe
    probe_module.run_enemy_tracking_probe = (
        lambda target_url, output_path, **kwargs: _make_session()
    )

    old_argv = sys.argv
    sys.argv = ["scripts.enemy_tracking_probe"]
    try:
        sys.modules.pop("scripts.enemy_tracking_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.enemy_tracking_probe", run_name="__main__")
    finally:
        probe_module.run_enemy_tracking_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
