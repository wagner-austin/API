"""Tests for RQ worker entry point."""

from __future__ import annotations

import sys

from platform_core.config import config_test_hooks
from platform_core.job_events import default_events_channel
from platform_core.queues import TURKIC_QUEUE
from platform_core.testing import make_fake_env
from platform_workers.rq_harness import WorkerConfig

from turkic_api import _test_hooks
from turkic_api.worker_entry import (
    _build_config,
    _get_default_runner,
    _run_worker,
    main,
)


class _RecordingLogger:
    """Logger that records calls for testing."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, str]]] = []

    def info(self, message: str, *, extra: dict[str, str]) -> None:
        """Record the log message."""
        self.messages.append((message, extra))


class _RecordingRunner:
    """Worker runner that records calls for testing."""

    def __init__(self) -> None:
        self.configs: list[WorkerConfig] = []

    def __call__(self, config: WorkerConfig) -> None:
        """Record the config."""
        self.configs.append(config)


def test_build_config_reads_settings() -> None:
    """Test _build_config reads settings and uses TURKIC_QUEUE."""
    env = make_fake_env(
        {
            "TURKIC_REDIS_URL": "redis://test-host:6379/0",
            "TURKIC_DATA_DIR": "/data",
            "TURKIC_DATA_BANK_API_URL": "http://db",
            "TURKIC_DATA_BANK_API_KEY": "key",
        }
    )
    config_test_hooks.get_env = env

    cfg = _build_config()

    assert cfg["redis_url"] == "redis://test-host:6379/0"
    assert cfg["queue_name"] == TURKIC_QUEUE
    assert cfg["events_channel"] == default_events_channel("turkic")


def test_run_worker_logs_and_calls_runner() -> None:
    """Test _run_worker logs startup message and calls runner."""
    config: WorkerConfig = {
        "redis_url": "redis://test:6379/0",
        "queue_name": TURKIC_QUEUE,
        "events_channel": default_events_channel("turkic"),
    }
    logger = _RecordingLogger()
    runner = _RecordingRunner()

    _run_worker(config, logger, runner)

    # Verify logger was called
    assert len(logger.messages) == 1
    msg, extra = logger.messages[0]
    assert msg == "Starting RQ worker"
    assert extra["queue_name"] == TURKIC_QUEUE
    assert extra["events_channel"] == default_events_channel("turkic")

    # Verify runner was called with config
    assert len(runner.configs) == 1
    assert runner.configs[0] == config


def test_main_with_injected_dependencies() -> None:
    """Test main() with injected dependencies."""
    config: WorkerConfig = {
        "redis_url": "redis://injected:6379/0",
        "queue_name": TURKIC_QUEUE,
        "events_channel": default_events_channel("turkic"),
    }
    logger = _RecordingLogger()
    runner = _RecordingRunner()

    main(config=config, logger=logger, runner=runner)

    # Verify logger received startup message
    assert len(logger.messages) == 1
    assert logger.messages[0][0] == "Starting RQ worker"

    # Verify runner received config
    assert len(runner.configs) == 1
    assert runner.configs[0]["redis_url"] == "redis://injected:6379/0"


def test_main_builds_config_from_settings_when_not_provided() -> None:
    """Test main() builds config from settings when not provided."""
    env = make_fake_env(
        {
            "TURKIC_REDIS_URL": "redis://from-env:6379/0",
            "TURKIC_DATA_DIR": "/data",
            "TURKIC_DATA_BANK_API_URL": "http://db",
            "TURKIC_DATA_BANK_API_KEY": "key",
        }
    )
    config_test_hooks.get_env = env

    logger = _RecordingLogger()
    runner = _RecordingRunner()

    # Pass logger and runner but not config - should build from settings
    main(config=None, logger=logger, runner=runner)

    assert len(runner.configs) == 1
    assert runner.configs[0]["redis_url"] == "redis://from-env:6379/0"
    assert runner.configs[0]["queue_name"] == TURKIC_QUEUE


def test_get_default_runner_returns_test_runner_when_set() -> None:
    """Test _get_default_runner returns test_runner when set."""

    def _custom_runner(config: WorkerConfig) -> None:
        pass

    _test_hooks.test_runner = _custom_runner

    result = _get_default_runner()

    assert result is _custom_runner


def test_get_default_runner_returns_run_rq_worker_when_test_runner_none() -> None:
    """Test _get_default_runner returns run_rq_worker when test_runner is None."""
    from platform_workers.rq_harness import run_rq_worker

    _test_hooks.test_runner = None

    result = _get_default_runner()

    assert result is run_rq_worker


def test_main_uses_test_runner_when_set() -> None:
    """Test main() uses test_runner when set in _test_hooks."""
    env = make_fake_env(
        {
            "TURKIC_REDIS_URL": "redis://test-runner:6379/0",
            "TURKIC_DATA_DIR": "/data",
            "TURKIC_DATA_BANK_API_URL": "http://db",
            "TURKIC_DATA_BANK_API_KEY": "key",
        }
    )
    config_test_hooks.get_env = env

    received_configs: list[WorkerConfig] = []

    def _recording_runner(config: WorkerConfig) -> None:
        received_configs.append(config)

    # Set the test runner in _test_hooks
    _test_hooks.test_runner = _recording_runner

    # Call main() with no args - should use test_runner
    main()

    assert len(received_configs) == 1
    assert received_configs[0]["redis_url"] == "redis://test-runner:6379/0"
    assert received_configs[0]["queue_name"] == TURKIC_QUEUE


def test_main_guard_executes_main() -> None:
    """Test the if __name__ == '__main__' guard executes main().

    Uses runpy.run_module to actually execute the module as __main__.
    Because _test_hooks is a separate module, our test_runner persists.
    """
    import runpy

    env = make_fake_env(
        {
            "TURKIC_REDIS_URL": "redis://runpy-guard-test:6379/0",
            "TURKIC_DATA_DIR": "/data",
            "TURKIC_DATA_BANK_API_URL": "http://db",
            "TURKIC_DATA_BANK_API_KEY": "key",
        }
    )
    config_test_hooks.get_env = env

    received_configs: list[WorkerConfig] = []

    def _recording_runner(config: WorkerConfig) -> None:
        received_configs.append(config)

    # Set the test runner in _test_hooks BEFORE running as __main__
    _test_hooks.test_runner = _recording_runner

    # Remove the module from sys.modules to avoid the RuntimeWarning
    # about the module being found in sys.modules prior to execution
    module_name = "turkic_api.worker_entry"
    saved_module = sys.modules.pop(module_name, None)

    # Run the module as __main__ - this executes the guard
    runpy.run_module(
        module_name,
        run_name="__main__",
        alter_sys=False,
    )

    # Restore module to sys.modules if it was there before
    if saved_module is not None:
        sys.modules[module_name] = saved_module

    # The guard should have been triggered, calling main()
    assert len(received_configs) == 1
    assert received_configs[0]["redis_url"] == "redis://runpy-guard-test:6379/0"
    assert received_configs[0]["queue_name"] == TURKIC_QUEUE
