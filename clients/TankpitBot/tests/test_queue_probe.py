"""Tests for scripts.queue_probe module."""

from __future__ import annotations

import types
from collections.abc import Generator

import pytest

from scripts import _test_hooks
from scripts import queue_probe as queue_probe_script
from tankpit_bot import _test_hooks as bot_hooks
from tankpit_bot._test_hooks.browser import (
    BrowserProtocol,
    BrowserTypeProtocol,
    PlaywrightProtocol,
    SyncPlaywrightContextManagerProtocol,
)
from tankpit_bot.action_lab.queue_probe_types import (
    QueueExperimentKind,
    QueueProbeSessionDict,
)


@pytest.fixture(autouse=True)
def _isolate_hooks() -> Generator[None, None, None]:
    """Save and restore hooks around each test."""
    orig_setup = _test_hooks.setup_rich_logging
    orig_playwright = bot_hooks.sync_playwright
    orig_get_playwright = bot_hooks.get_sync_playwright
    orig_argv = bot_hooks.get_argv
    orig_env = bot_hooks.get_env
    orig_run = queue_probe_script.run_queue_probe
    yield
    _test_hooks.setup_rich_logging = orig_setup
    bot_hooks.sync_playwright = orig_playwright
    bot_hooks.get_sync_playwright = orig_get_playwright
    bot_hooks.get_argv = orig_argv
    bot_hooks.get_env = orig_env
    queue_probe_script.run_queue_probe = orig_run


def _session() -> QueueProbeSessionDict:
    """Build a sample script session payload."""
    return QueueProbeSessionDict(
        session_id="queue-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="queue_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        experiment_timeout_ms=5000,
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
        experiments=[],
    )


def _noop_setup(level: _test_hooks.LogLevel) -> None:
    pass


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
        raise AssertionError("playwright factory should not be invoked")


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


class TestParseBoolEnv:
    def test_none_returns_false(self) -> None:
        assert queue_probe_script._parse_bool_env(None) is False

    def test_true_string(self) -> None:
        assert queue_probe_script._parse_bool_env("true") is True

    def test_one_string(self) -> None:
        assert queue_probe_script._parse_bool_env("1") is True

    def test_yes_string(self) -> None:
        assert queue_probe_script._parse_bool_env("yes") is True

    def test_false_string(self) -> None:
        assert queue_probe_script._parse_bool_env("false") is False

    def test_random_string(self) -> None:
        assert queue_probe_script._parse_bool_env("nope") is False


class TestParseOptionalIntArg:
    def test_present_flag(self) -> None:
        result = queue_probe_script._parse_optional_int_arg(
            ["--experiment-timeout-ms", "3000"], "--experiment-timeout-ms"
        )
        assert result == 3000

    def test_missing_flag(self) -> None:
        result = queue_probe_script._parse_optional_int_arg(
            ["--other", "5"], "--experiment-timeout-ms"
        )
        assert result is None

    def test_missing_value_raises(self) -> None:
        with pytest.raises(ValueError, match="requires an integer"):
            queue_probe_script._parse_optional_int_arg(
                ["--experiment-timeout-ms"], "--experiment-timeout-ms"
            )


class TestFormatSavedPath:
    def test_format(self) -> None:
        result = queue_probe_script._format_saved_path("output.json")
        assert result == "Saved to: output.json"


class TestMainPlaywrightAlreadySet:
    def test_skips_get_sync_playwright_when_already_set(self) -> None:
        """When sync_playwright is already set, main() skips get_sync_playwright."""
        captured: list[str | bool | int | None] = []

        _test_hooks.setup_rich_logging = _noop_setup
        bot_hooks.get_env = lambda key: None
        bot_hooks.get_argv = lambda: ["queue_probe"]
        bot_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

        def _fake_run(
            target_url: str,
            output_path: str,
            *,
            headless: bool = False,
            prefer_account: bool = False,
            initial_sync_timeout_ms: int = 10000,
            experiment_timeout_ms: int = 5000,
            experiment_kinds: list[QueueExperimentKind] | None = None,
        ) -> QueueProbeSessionDict:
            captured.append(target_url)
            return _session()

        queue_probe_script.run_queue_probe = _fake_run
        assert queue_probe_script.main() == 0
        assert captured == ["https://tankpit.com/play"]


class TestMainDefaultArgs:
    def test_main_with_defaults(self) -> None:
        """Queue probe main() passes default args to run_queue_probe."""
        captured: list[str | bool | int | None] = []
        logging_levels: list[str] = []

        _test_hooks.setup_rich_logging = lambda level: logging_levels.append(level)
        bot_hooks.get_env = lambda key: None
        bot_hooks.get_argv = lambda: ["queue_probe"]
        bot_hooks.sync_playwright = None

        def _fake_run(
            target_url: str,
            output_path: str,
            *,
            headless: bool = False,
            prefer_account: bool = False,
            initial_sync_timeout_ms: int = 10000,
            experiment_timeout_ms: int = 5000,
            experiment_kinds: list[QueueExperimentKind] | None = None,
        ) -> QueueProbeSessionDict:
            captured.extend(
                [
                    target_url,
                    output_path,
                    headless,
                    prefer_account,
                    initial_sync_timeout_ms,
                    experiment_timeout_ms,
                ]
            )
            return _session()

        queue_probe_script.run_queue_probe = _fake_run
        assert queue_probe_script.main() == 0
        assert logging_levels == ["INFO"]
        assert captured[0] == "https://tankpit.com/play"
        output = str(captured[1])
        assert output.startswith("runs/probe/queue-")
        assert output.endswith(".json")
        assert captured[2:] == [False, False, 10000, 5000]


class TestMainEnvOverrides:
    def test_main_with_env_overrides(self) -> None:
        """Queue probe main() applies env variable overrides."""
        captured: list[str | bool | int | None] = []

        _test_hooks.setup_rich_logging = _noop_setup
        env = {
            "TANKPIT_URL": "https://tankpit.com/custom",
            "TANKPIT_QUEUE_PROBE_OUTPUT": "custom.json",
            "TANKPIT_HEADLESS": "true",
            "TANKPIT_PREFER_ACCOUNT": "yes",
            "TANKPIT_QUEUE_INITIAL_SYNC_TIMEOUT_MS": "4000",
            "TANKPIT_QUEUE_EXPERIMENT_TIMEOUT_MS": "3000",
        }
        bot_hooks.get_env = lambda key: env.get(key)
        bot_hooks.get_argv = lambda: ["queue_probe"]
        bot_hooks.sync_playwright = None

        def _fake_run(
            target_url: str,
            output_path: str,
            *,
            headless: bool = False,
            prefer_account: bool = False,
            initial_sync_timeout_ms: int = 10000,
            experiment_timeout_ms: int = 5000,
            experiment_kinds: list[QueueExperimentKind] | None = None,
        ) -> QueueProbeSessionDict:
            captured.extend(
                [
                    target_url,
                    output_path,
                    headless,
                    prefer_account,
                    initial_sync_timeout_ms,
                    experiment_timeout_ms,
                ]
            )
            return _session()

        queue_probe_script.run_queue_probe = _fake_run
        assert queue_probe_script.main() == 0
        assert captured == [
            "https://tankpit.com/custom",
            "custom.json",
            True,
            True,
            4000,
            3000,
        ]


class TestMainCliOverrides:
    def test_main_with_cli_overrides(self) -> None:
        """CLI args override env variables."""
        captured: list[str | bool | int | None] = []

        _test_hooks.setup_rich_logging = _noop_setup
        bot_hooks.get_env = lambda key: None
        bot_hooks.get_argv = lambda: [
            "queue_probe",
            "--initial-sync-timeout-ms",
            "9000",
            "--experiment-timeout-ms",
            "7000",
        ]
        bot_hooks.sync_playwright = None

        def _fake_run(
            target_url: str,
            output_path: str,
            *,
            headless: bool = False,
            prefer_account: bool = False,
            initial_sync_timeout_ms: int = 10000,
            experiment_timeout_ms: int = 5000,
            experiment_kinds: list[QueueExperimentKind] | None = None,
        ) -> QueueProbeSessionDict:
            captured.extend(
                [
                    initial_sync_timeout_ms,
                    experiment_timeout_ms,
                ]
            )
            return _session()

        queue_probe_script.run_queue_probe = _fake_run
        assert queue_probe_script.main() == 0
        assert captured[0] == 9000
        assert captured[1] == 7000


class TestMainModuleEntry:
    def test_module_entry_invokes_main_and_exits_zero(self) -> None:
        """Running as __main__ invokes main() and raises SystemExit(0)."""
        import runpy
        import sys

        from tankpit_bot.action_lab import queue_probe as al_queue_probe

        _test_hooks.setup_rich_logging = _noop_setup
        bot_hooks.get_env = lambda key: None
        bot_hooks.get_argv = lambda: ["queue_probe"]
        bot_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
        original_run = al_queue_probe.run_queue_probe

        def _stub_run(
            target_url: str,
            output_path: str,
            *,
            headless: bool = False,
            prefer_account: bool = False,
            initial_sync_timeout_ms: int = 10000,
            experiment_timeout_ms: int = 5000,
            experiment_kinds: list[QueueExperimentKind] | None = None,
        ) -> QueueProbeSessionDict:
            _ = (target_url, output_path, headless, prefer_account)
            _ = (initial_sync_timeout_ms, experiment_timeout_ms, experiment_kinds)
            return _session()

        al_queue_probe.run_queue_probe = _stub_run
        original_argv = sys.argv[:]
        sys.argv = ["queue_probe"]
        try:
            sys.modules.pop("scripts.queue_probe", None)
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("scripts.queue_probe", run_name="__main__")
        finally:
            al_queue_probe.run_queue_probe = original_run
            sys.argv = original_argv
        assert excinfo.value.code == 0
