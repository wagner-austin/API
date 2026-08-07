"""Tests for scripts.combat_probe.

The script is a thin CLI over :func:`run_combat_probe`: it resolves the
target URL, output path and the two engagement bounds from env and
argv, initializes the Playwright factory once, and reports. These
tests drive every one of those decisions with the probe itself
replaced, so no browser is launched -- the fake browser type asserts
if anything tries.
"""

from __future__ import annotations

import runpy
import sys
import types
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import combat_probe
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
    SyncPlaywrightFactoryProtocol,
)
from tankpit_bot.action_lab.combat_probe_types import CombatProbeSessionDict


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    """Restore script and core hooks after each test."""
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_get_argv = core_hooks.get_argv
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = combat_probe.run_combat_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.get_argv = original_get_argv
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    combat_probe.run_combat_probe = original_run


def _session() -> CombatProbeSessionDict:
    """Build a sample combat probe session payload."""
    return CombatProbeSessionDict(
        session_id="combat-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_engagements=3,
        max_shots_per_engagement=20,
        capture_session_path="combat_probe.capture_session.json",
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
        engagements=[],
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
    assert combat_probe._parse_bool_env("true") is True
    assert combat_probe._parse_bool_env("1") is True
    assert combat_probe._parse_bool_env("yes") is True
    assert combat_probe._parse_bool_env("TRUE") is True
    assert combat_probe._parse_bool_env("false") is False
    assert combat_probe._parse_bool_env("") is False
    assert combat_probe._parse_bool_env(None) is False


def test_parse_optional_int_arg() -> None:
    """Optional integer arg parsing handles presence, absence, and a bare flag."""
    assert combat_probe._parse_optional_int_arg(["prog"], "--max-shots") is None
    assert combat_probe._parse_optional_int_arg(["prog", "--max-shots", "12"], "--max-shots") == 12
    with pytest.raises(ValueError, match="--max-shots requires an integer value"):
        combat_probe._parse_optional_int_arg(["prog", "--max-shots"], "--max-shots")


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    """With no env or CLI overrides the script falls back to its documented defaults.

    The bounds are the interesting half: ``max_engagements or 3`` and
    ``max_shots or 20`` mean an absent flag takes the default, which is
    what this pins alongside the one-time factory initialization.
    """
    captured: list[str | bool | int] = []
    factory_calls: list[str] = []
    logging_levels: list[str] = []

    script_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["combat_probe"]
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
        max_engagements: int = 3,
        max_shots_per_engagement: int = 20,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 5000,
        teleport_timeout_ms: int = 10000,
    ) -> CombatProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_engagements,
                max_shots_per_engagement,
            ]
        )
        return _session()

    combat_probe.run_combat_probe = _fake_run

    assert combat_probe.main() == 0
    assert logging_levels == ["INFO"]
    assert factory_calls == ["factory"]
    assert captured == [
        "https://tankpit.com/play",
        "combat_probe.json",
        False,
        False,
        3,
        20,
    ]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_and_cli_overrides(_restore_script_hooks: None) -> None:
    """Env supplies the URL, output and flags; argv supplies the two bounds."""
    recorded: list[str | bool | int] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_COMBAT_PROBE_OUTPUT": "custom_combat.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.get_argv = lambda: [
        "combat_probe",
        "--max-engagements",
        "5",
        "--max-shots",
        "7",
    ]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        max_engagements: int = 3,
        max_shots_per_engagement: int = 20,
        initial_sync_timeout_ms: int = 10000,
        acquisition_timeout_ms: int = 5000,
        teleport_timeout_ms: int = 10000,
    ) -> CombatProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                max_engagements,
                max_shots_per_engagement,
            ]
        )
        return _session()

    combat_probe.run_combat_probe = _fake_run

    assert combat_probe.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_combat.json",
        True,
        True,
        5,
        7,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    """Module entrypoint exits through main().

    The re-import must see the fake, and the script binds
    ``run_combat_probe`` from the ``action_lab.combat_probe`` submodule
    rather than the package barrel -- so the swap goes on the submodule
    or ``runpy`` re-imports the real probe and tries to open a browser.
    """
    import tankpit_bot.action_lab.combat_probe as probe_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.get_argv = lambda: ["scripts.combat_probe"]
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = probe_module.run_combat_probe
    probe_module.run_combat_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.combat_probe"]
    try:
        sys.modules.pop("scripts.combat_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.combat_probe", run_name="__main__")
    finally:
        probe_module.run_combat_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
