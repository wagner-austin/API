from __future__ import annotations

import runpy
import sys

import pytest
from platform_core.job_events import default_events_channel
from platform_core.queues import TURKIC_QUEUE
from platform_workers.rq_harness import WorkerConfig

import turkic_api.api.worker_entry as entry
from turkic_api.api.config import Settings


class _RunRecorder:
    def __init__(self) -> None:
        self.configs: list[WorkerConfig] = []

    def run(self, cfg: WorkerConfig) -> None:
        self.configs.append(cfg)


def test_worker_entry_builds_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide deterministic settings
    def _settings() -> Settings:
        return Settings(
            redis_url="redis://localhost:6379/0",
            data_dir="/data",
            environment="test",
            data_bank_api_url="http://db",
            data_bank_api_key="key",
        )

    recorder = _RunRecorder()

    monkeypatch.setattr(entry, "settings_from_env", _settings)
    monkeypatch.setattr(entry, "run_rq_worker", recorder.run)

    entry.main()

    assert len(recorder.configs) == 1
    cfg: WorkerConfig = recorder.configs[0]
    assert cfg["redis_url"] == "redis://localhost:6379/0"
    assert cfg["queue_name"] == TURKIC_QUEUE
    assert cfg["events_channel"] == default_events_channel("turkic")


def test_worker_entry_runs_under_main(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RunRecorder()

    # Ensure settings resolution is deterministic without loading env defaults
    def _settings() -> Settings:
        return Settings(
            redis_url="redis://localhost:6379/0",
            data_dir="/data",
            environment="test",
            data_bank_api_url="http://db",
            data_bank_api_key="key",
        )

    monkeypatch.setattr(entry, "settings_from_env", _settings)
    monkeypatch.setattr("platform_workers.rq_harness.run_rq_worker", recorder.run)

    # Ensure clean import state to avoid runpy warning about preloaded module
    sys.modules.pop("turkic_api.api.worker_entry", None)
    runpy.run_module("turkic_api.api.worker_entry", run_name="__main__")

    assert len(recorder.configs) == 1
