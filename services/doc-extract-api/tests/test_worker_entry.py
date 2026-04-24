"""Tests for doc_extract_api.worker_entry."""

from __future__ import annotations

from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env
from platform_workers.rq_harness import WorkerConfig

from doc_extract_api import _test_hooks
from doc_extract_api.worker_entry import _build_config, _get_default_runner, _run_worker, main


class _FakeLogger:
    """Fake logger for testing."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *, extra: dict[str, str]) -> None:
        _ = extra
        self.messages.append(message)


class _FakeRunner:
    """Fake runner that records calls."""

    def __init__(self) -> None:
        self.called_with: WorkerConfig | None = None

    def __call__(self, config: WorkerConfig) -> None:
        self.called_with = config


class TestGetDefaultRunner:
    def test_returns_test_runner_when_set(self) -> None:
        runner = _FakeRunner()
        _test_hooks.test_runner = runner
        result = _get_default_runner()
        assert result is runner
        _test_hooks.test_runner = None

    def test_returns_rq_worker_when_none(self) -> None:
        _test_hooks.test_runner = None
        result = _get_default_runner()
        assert callable(result)


class TestBuildConfig:
    def test_builds_from_env(self) -> None:
        platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://test:6379/0"})
        config = _build_config()
        assert config["redis_url"] == "redis://test:6379/0"
        assert config["queue_name"] == "doc_extract"


class TestRunWorker:
    def test_calls_runner(self) -> None:
        config = WorkerConfig(
            redis_url="redis://test:6379/0",
            queue_name="doc_extract",
            events_channel="doc_extract:events",
        )
        logger = _FakeLogger()
        runner = _FakeRunner()
        _run_worker(config, logger, runner)
        assert runner.called_with is not None and runner.called_with["queue_name"] == "doc_extract"
        assert len(logger.messages) == 1


class TestMain:
    def test_main_with_explicit_deps(self) -> None:
        config = WorkerConfig(
            redis_url="redis://test:6379/0",
            queue_name="doc_extract",
            events_channel="doc_extract:events",
        )
        logger = _FakeLogger()
        runner = _FakeRunner()
        main(config=config, logger=logger, runner=runner)
        assert runner.called_with is not None and runner.called_with["queue_name"] == "doc_extract"

    def test_worker_entry_main_guard(self) -> None:
        """The if __name__ == '__main__' block calls main()."""
        import runpy
        import sys

        runner = _FakeRunner()
        _test_hooks.test_runner = runner

        module_name = "doc_extract_api.worker_entry"
        saved_module = sys.modules.pop(module_name, None)

        runpy.run_module(
            module_name,
            run_name="__main__",
            alter_sys=False,
        )

        if saved_module is not None:
            sys.modules[module_name] = saved_module

        assert runner.called_with is not None and runner.called_with["queue_name"] == "doc_extract"
        _test_hooks.test_runner = None
