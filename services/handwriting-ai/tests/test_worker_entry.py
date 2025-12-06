from __future__ import annotations

import types
from collections.abc import Callable
from typing import TypedDict

import pytest
from platform_core.job_events import default_events_channel
from platform_core.queues import DIGITS_QUEUE

import handwriting_ai.worker_entry as entry


class _Logged(TypedDict):
    message: str
    extra: dict[str, str]


class _Logger:
    def __init__(self) -> None:
        self.logged: list[_Logged] = []

    def info(self, message: str, *, extra: dict[str, str]) -> None:
        self.logged.append({"message": message, "extra": extra})


class _WorkerConfig(TypedDict):
    redis_url: str
    queue_name: str
    events_channel: str


def test_build_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://example")
    cfg = entry._build_config()
    assert cfg["redis_url"] == "redis://example"
    assert cfg["queue_name"] == DIGITS_QUEUE
    assert cfg["events_channel"] == default_events_channel("digits")


def test_main_invokes_worker_with_built_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange environment and stubs
    monkeypatch.setenv("REDIS_URL", "redis://example")
    captured_cfg: list[_WorkerConfig] = []

    def _fake_worker(cfg: _WorkerConfig) -> None:
        captured_cfg.append(cfg)

    logger = _Logger()

    def _fake_logger(_: str) -> _Logger:
        return logger

    def _fake_setup_logging(
        *,
        level: str,
        format_mode: str,
        service_name: str,
        instance_id: str | None,
        extra_fields: list[str] | None,
    ) -> None:
        return None

    monkeypatch.setattr(entry, "run_rq_worker", _fake_worker)
    monkeypatch.setattr(entry, "setup_logging", _fake_setup_logging)
    monkeypatch.setattr(entry, "get_logger", _fake_logger)

    # Act
    entry.main()

    # Assert worker invoked with expected config and log captured metadata
    assert captured_cfg[0] == {
        "redis_url": "redis://example",
        "queue_name": DIGITS_QUEUE,
        "events_channel": default_events_channel("digits"),
    }
    assert any(
        log["extra"]["queue"] == DIGITS_QUEUE
        and log["extra"]["events_channel"] == default_events_channel("digits")
        for log in logger.logged
    )


def _make_config_module(require_env: Callable[[str], str]) -> types.ModuleType:
    class _ConfigModule(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("platform_core.config")
            self._require_env_str = require_env

    return _ConfigModule()


def _make_job_events_module() -> types.ModuleType:
    class _JobEventsModule(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("platform_core.job_events")

        def default_events_channel(self, domain: str) -> str:
            return "evt"

    return _JobEventsModule()


def _make_queues_module() -> types.ModuleType:
    class _QueuesModule(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("platform_core.queues")
            self.DIGITS_QUEUE = "queue"

    return _QueuesModule()


def _make_logging_module(captured_cfg: list[_WorkerConfig]) -> tuple[types.ModuleType, _Logger]:
    class _CapturingLogger(_Logger):
        def info(self, message: str, *, extra: dict[str, str]) -> None:
            super().info(message, extra=extra)
            captured_cfg.append(
                {
                    "redis_url": "redis://example",
                    "queue_name": extra["queue"],
                    "events_channel": extra["events_channel"],
                }
            )

    logger = _CapturingLogger()

    class _LoggingModule(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("platform_core.logging")

        def get_logger(self, _: str) -> _Logger:
            return logger

        def setup_logging(
            self,
            *,
            level: str,
            format_mode: str,
            service_name: str,
            instance_id: str | None,
            extra_fields: list[str] | None,
        ) -> None:
            assert level == "INFO"
            assert format_mode == "json"
            assert service_name == "handwriting-worker"

    return _LoggingModule(), logger


def _make_rq_module(captured_cfg: list[_WorkerConfig]) -> types.ModuleType:
    class _RQModule(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("platform_workers.rq_harness")
            self.WorkerConfig = dict[str, str]

        def run_rq_worker(self, cfg: _WorkerConfig) -> None:
            captured_cfg.append(cfg)

    return _RQModule()


def test_main_guard_executes_when_run_as_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.setenv("REDIS_URL", "redis://example")
    captured_cfg: list[_WorkerConfig] = []

    def _req_env(name: str) -> str:
        if name != "REDIS_URL":
            raise RuntimeError("unexpected env name")
        return "redis://example"

    monkeypatch.setitem(sys.modules, "platform_core.config", _make_config_module(_req_env))
    monkeypatch.setitem(sys.modules, "platform_core.job_events", _make_job_events_module())
    monkeypatch.setitem(sys.modules, "platform_core.queues", _make_queues_module())

    log_mod, _ = _make_logging_module(captured_cfg)
    monkeypatch.setitem(sys.modules, "platform_core.logging", log_mod)

    rq_mod = _make_rq_module(captured_cfg)
    monkeypatch.setitem(sys.modules, "platform_workers.rq_harness", rq_mod)

    sys.modules.pop("handwriting_ai.worker_entry", None)

    # Execute module under __main__ to hit guard
    runpy.run_module("handwriting_ai.worker_entry", run_name="__main__")

    assert any(cfg["queue_name"] == "queue" for cfg in captured_cfg)
    assert any(cfg["events_channel"] == "evt" for cfg in captured_cfg)
