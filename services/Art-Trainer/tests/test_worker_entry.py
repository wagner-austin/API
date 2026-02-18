"""Tests for RQ worker entry point."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.config import _test_hooks as config_hooks
from platform_workers.rq_harness import WorkerConfig

from art_trainer import _test_hooks
from art_trainer.worker_entry import (
    _build_config,
    _get_default_runner,
    _run_worker,
    main,
)


def _make_env_getter(env_vars: dict[str, str]) -> Callable[[str], str | None]:
    """Create a fake get_env function that reads from a dict.

    Args:
        env_vars: Environment variables to return.

    Returns:
        Fake get_env function.
    """

    def _get_env(key: str) -> str | None:
        return env_vars.get(key)

    return _get_env


class _RecordingLogger:
    """Logger that records calls for testing."""

    messages: list[tuple[str, dict[str, str]]]

    def __init__(self) -> None:
        """Initialize the recording logger."""
        self.messages = []

    def info(self, message: str, *, extra: dict[str, str]) -> None:
        """Record the log message.

        Args:
            message: Log message.
            extra: Extra log fields.
        """
        self.messages.append((message, extra))


class _RecordingRunner:
    """Worker runner that records calls for testing."""

    configs: list[WorkerConfig]

    def __init__(self) -> None:
        """Initialize the recording runner."""
        self.configs = []

    def __call__(self, config: WorkerConfig) -> None:
        """Record the config.

        Args:
            config: Worker configuration.
        """
        self.configs.append(config)


def test_build_config_reads_env() -> None:
    """Test _build_config reads REDIS_URL."""
    config_hooks.get_env = _make_env_getter({"REDIS_URL": "redis://test-host:6379/0"})

    cfg = _build_config()

    assert cfg["redis_url"] == "redis://test-host:6379/0"
    assert cfg["queue_name"] == "art-trainer"
    assert cfg["events_channel"] == "art-trainer:events"


def test_build_config_requires_redis_url() -> None:
    """Test _build_config raises when REDIS_URL is missing."""
    config_hooks.get_env = _make_env_getter({})

    with pytest.raises(RuntimeError, match="REDIS_URL"):
        _build_config()


def test_run_worker_logs_and_calls_runner() -> None:
    """Test _run_worker logs startup message and calls runner."""
    config: WorkerConfig = {
        "redis_url": "redis://test:6379/0",
        "queue_name": "art-trainer",
        "events_channel": "art-trainer:events",
    }
    logger = _RecordingLogger()
    runner = _RecordingRunner()

    _run_worker(config, logger, runner)

    assert len(logger.messages) == 1
    msg, extra = logger.messages[0]
    assert msg == "Starting RQ worker"
    assert extra["queue"] == "art-trainer"
    assert extra["events_channel"] == "art-trainer:events"

    assert len(runner.configs) == 1
    assert runner.configs[0] == config


def test_main_with_injected_dependencies() -> None:
    """Test main() with injected dependencies."""
    config: WorkerConfig = {
        "redis_url": "redis://injected:6379/0",
        "queue_name": "art-trainer",
        "events_channel": "art-trainer:events",
    }
    logger = _RecordingLogger()
    runner = _RecordingRunner()

    main(config=config, logger=logger, runner=runner)

    assert len(logger.messages) == 1
    assert logger.messages[0][0] == "Starting RQ worker"

    assert len(runner.configs) == 1
    assert runner.configs[0]["redis_url"] == "redis://injected:6379/0"


def test_main_builds_config_from_env_when_not_provided() -> None:
    """Test main() builds config from environment when not provided."""
    config_hooks.get_env = _make_env_getter({"REDIS_URL": "redis://from-env:6379/0"})

    logger = _RecordingLogger()
    runner = _RecordingRunner()

    main(config=None, logger=logger, runner=runner)

    assert len(runner.configs) == 1
    assert runner.configs[0]["redis_url"] == "redis://from-env:6379/0"
    assert runner.configs[0]["queue_name"] == "art-trainer"


def test_get_default_runner_returns_test_runner_when_set() -> None:
    """Test _get_default_runner returns test_runner when set."""

    def _custom_runner(config: WorkerConfig) -> None:
        pass

    original = _test_hooks.test_runner
    _test_hooks.test_runner = _custom_runner

    result = _get_default_runner()

    _test_hooks.test_runner = original

    assert result is _custom_runner


def test_get_default_runner_returns_run_rq_worker_when_test_runner_none() -> None:
    """Test _get_default_runner returns run_rq_worker when test_runner is None."""
    from platform_workers.rq_harness import run_rq_worker

    original = _test_hooks.test_runner
    _test_hooks.test_runner = None

    result = _get_default_runner()

    _test_hooks.test_runner = original

    assert result is run_rq_worker


def test_main_uses_test_runner_when_set() -> None:
    """Test main() uses test_runner when set in _test_hooks."""
    config_hooks.get_env = _make_env_getter({"REDIS_URL": "redis://test-runner:6379/0"})

    received_configs: list[WorkerConfig] = []

    def _recording_runner(config: WorkerConfig) -> None:
        received_configs.append(config)

    original = _test_hooks.test_runner
    _test_hooks.test_runner = _recording_runner

    main()

    _test_hooks.test_runner = original

    assert len(received_configs) == 1
    assert received_configs[0]["redis_url"] == "redis://test-runner:6379/0"
    assert received_configs[0]["queue_name"] == "art-trainer"


def test_main_guard_executes_main() -> None:
    """Test the if __name__ == '__main__' guard executes main()."""
    import runpy
    import sys

    config_hooks.get_env = _make_env_getter({"REDIS_URL": "redis://runpy-guard-test:6379/0"})

    received_configs: list[WorkerConfig] = []

    def _recording_runner(config: WorkerConfig) -> None:
        received_configs.append(config)

    original = _test_hooks.test_runner
    _test_hooks.test_runner = _recording_runner

    module_name = "art_trainer.worker_entry"
    saved_module = sys.modules.pop(module_name, None)

    runpy.run_module(
        module_name,
        run_name="__main__",
        alter_sys=False,
    )

    if saved_module is not None:
        sys.modules[module_name] = saved_module

    _test_hooks.test_runner = original

    assert len(received_configs) == 1
    assert received_configs[0]["redis_url"] == "redis://runpy-guard-test:6379/0"
    assert received_configs[0]["queue_name"] == "art-trainer"
