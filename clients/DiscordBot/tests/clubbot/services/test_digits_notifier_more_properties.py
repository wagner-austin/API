from __future__ import annotations

import pytest
from platform_core.digits_metrics_events import (
    DigitsArtifactV1,
    DigitsBatchMetricsV1,
    DigitsBestMetricsV1,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsPruneV1,
    DigitsUploadV1,
)
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_properties_and_wrappers() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")

    # Access property getters to cover lines
    assert type(sub.configs) is dict
    assert type(sub.metrics) is dict
    # Internal aliases
    assert type(sub._configs) is dict
    assert type(sub._metrics_map) is dict
    assert type(sub._rt) is dict

    # Exercise wrapper helpers that forward to strict handlers
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
        "queue": "digits",
    }
    await sub._on_config(config)

    batch: DigitsBatchMetricsV1 = {
        "type": "digits.metrics.batch.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 1,
        "batch": 1,
        "total_batches": 1,
        "batch_loss": 0.1,
        "batch_acc": 0.9,
        "avg_loss": 0.1,
        "samples_per_sec": 10.0,
        "main_rss_mb": 1,
        "workers_rss_mb": 0,
        "worker_count": 0,
        "cgroup_usage_mb": 10,
        "cgroup_limit_mb": 100,
        "cgroup_pct": 10.0,
        "anon_mb": 1,
        "file_mb": 1,
    }
    await sub._on_batch(batch)

    epoch: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 1,
        "train_loss": 0.1,
        "val_acc": 0.9,
        "time_s": 10.0,
    }
    await sub._on_epoch(epoch)

    best: DigitsBestMetricsV1 = {
        "type": "digits.metrics.best.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "epoch": 1,
        "val_acc": 0.5,
    }
    sub._on_best(best)

    artifact: DigitsArtifactV1 = {
        "type": "digits.metrics.artifact.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "path": "/m.pt",
    }
    sub._on_artifact(artifact)

    upload: DigitsUploadV1 = {
        "type": "digits.metrics.upload.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "status": 200,
        "model_bytes": 1,
        "manifest_bytes": 1,
        "file_id": "fid",
        "file_sha256": "sha",
    }
    sub._on_upload(upload)

    prune: DigitsPruneV1 = {
        "type": "digits.metrics.prune.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "deleted_count": 0,
    }
    sub._on_prune(prune)

    # Completed flows through maybe_notify; cover path
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 1.0,
    }
    await sub._on_completed(completed)
