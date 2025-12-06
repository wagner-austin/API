from __future__ import annotations

import pytest
from platform_core.digits_metrics_events import (
    DigitsArtifactV1,
    DigitsBestMetricsV1,
    DigitsPruneV1,
    DigitsUploadV1,
)
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_handles_non_notify_events() -> None:
    bot = FakeBot()
    sub = DigitsEventSubscriber(bot, redis_url="redis://")

    # Events that do not notify (state updates only)
    best: DigitsBestMetricsV1 = {
        "type": "digits.metrics.best.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "epoch": 1,
        "val_acc": 0.9,
    }
    artifact: DigitsArtifactV1 = {
        "type": "digits.metrics.artifact.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "path": "/m.pt",
    }
    upload: DigitsUploadV1 = {
        "type": "digits.metrics.upload.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "status": 200,
        "model_bytes": 1,
        "manifest_bytes": 1,
        "file_id": "fid",
        "file_sha256": "sha",
    }
    prune: DigitsPruneV1 = {
        "type": "digits.metrics.prune.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "deleted_count": 0,
    }

    await sub._handle_event(best)
    await sub._handle_event(artifact)
    await sub._handle_event(upload)
    await sub._handle_event(prune)


@pytest.mark.asyncio
async def test_digits_notifier_run_delegates_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = FakeBot()
    sub = DigitsEventSubscriber(bot, redis_url="redis://")
    called = {"n": 0}

    from platform_discord.task_runner import TaskRunner

    async def _once(self: TaskRunner) -> None:
        called["n"] += 1

    monkeypatch.setattr(TaskRunner, "run_once", _once, raising=True)
    await sub._run()
    assert called["n"] == 1
