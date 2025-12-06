from __future__ import annotations

import runpy
import sys

import pytest
from platform_core.job_events import default_events_channel
from platform_core.queues import DATA_BANK_QUEUE
from platform_workers.rq_harness import WorkerConfig

import data_bank_api.worker_entry as entry


class _RunRecorder:
    def __init__(self) -> None:
        self.configs: list[WorkerConfig] = []

    def run(self, cfg: WorkerConfig) -> None:
        self.configs.append(cfg)


def test_worker_entry_builds_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def _require_redis(_name: str) -> str:
        return "redis://localhost:6379/0"

    recorder = _RunRecorder()

    monkeypatch.setattr(entry, "_require_env_str", _require_redis)
    monkeypatch.setattr(entry, "run_rq_worker", recorder.run)

    entry.main()

    assert len(recorder.configs) == 1
    cfg: WorkerConfig = recorder.configs[0]
    assert cfg["redis_url"] == "redis://localhost:6379/0"
    assert cfg["queue_name"] == DATA_BANK_QUEUE
    assert cfg["events_channel"] == default_events_channel("databank")


def test_worker_entry_runs_under_main(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RunRecorder()

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr("platform_workers.rq_harness.run_rq_worker", recorder.run)

    sys.modules.pop("data_bank_api.worker_entry", None)
    runpy.run_module("data_bank_api.worker_entry", run_name="__main__")

    assert len(recorder.configs) == 1
