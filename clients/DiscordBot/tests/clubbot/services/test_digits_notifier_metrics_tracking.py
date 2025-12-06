"""Metrics tracking tests for DigitsEventSubscriber (ASCII-only assertions)."""

from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import (
    DigitsBatchMetricsV1,
    DigitsBestMetricsV1,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
)
from platform_core.job_events import JobFailedV1
from platform_discord.embed_helpers import (
    get_color_value,
    get_description,
    get_field,
    get_footer_text,
    get_title,
    has_field,
)
from platform_discord.handwriting.embeds import build_training_embed
from platform_discord.handwriting.types import TrainingConfig, TrainingMetrics
from tests.support.discord_fakes import TrackingBot, TrackingUser

import clubbot.services.jobs.digits_notifier as dn


@pytest.mark.asyncio
async def test_metrics_tracked_throughout_lifecycle() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    # Start training
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "req_metrics",
        "user_id": 100,
        "model_id": "mnist_full",
        "total_epochs": 5,
        "queue": "default",
        "batch_size": 64,
        "learning_rate": 0.001,
        "device": "cpu",
    }
    await sub._handle_event(config)

    # Batch
    batch: DigitsBatchMetricsV1 = {
        "type": "digits.metrics.batch.v1",
        "job_id": "req_metrics",
        "user_id": 100,
        "model_id": "mnist_full",
        "epoch": 1,
        "total_epochs": 5,
        "batch": 100,
        "total_batches": 200,
        "batch_loss": 0.5,
        "batch_acc": 0.8,
        "avg_loss": 0.6,
        "samples_per_sec": 1234.5,
        "main_rss_mb": 400,
        "workers_rss_mb": 200,
        "worker_count": 4,
        "cgroup_usage_mb": 700,
        "cgroup_limit_mb": 1024,
        "cgroup_pct": 68.4,
        "anon_mb": 500,
        "file_mb": 200,
    }
    await sub._handle_event(batch)

    # Epoch
    epoch: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "req_metrics",
        "user_id": 100,
        "model_id": "mnist_full",
        "epoch": 1,
        "total_epochs": 5,
        "train_loss": 0.45,
        "val_acc": 0.85,
        "time_s": 120.5,
    }
    await sub._handle_event(epoch)

    # Completed
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "req_metrics",
        "user_id": 100,
        "model_id": "mnist_full",
        "val_acc": 0.92,
    }
    await sub._handle_event(completed)

    # Final embed assertions
    assert len(user.embeds) == 4
    final_embed = user.embeds[-1]
    if final_embed is None:
        raise AssertionError("expected final embed")
    assert get_title(final_embed) == "Training Completed"
    assert get_color_value(final_embed) == 0x57F287
    assert "mnist_full" in (get_description(final_embed) or "")
    # Summary
    assert has_field(final_embed, "Training Summary")
    summary_field = get_field(final_embed, "Training Summary")
    if summary_field is None:
        raise AssertionError("expected Training Summary field")
    summary_value = str(summary_field["value"])
    assert "Final Avg Loss" in summary_value and "0.6000" in summary_value
    assert "Final Train Loss" in summary_value and "0.4500" in summary_value
    assert "Total Time" in summary_value and "2m 0s" in summary_value
    assert "Avg Speed" in summary_value and "1234.5 samples/sec" in summary_value
    assert "Best Epoch" in summary_value and "1" in summary_value
    assert "Peak Memory" in summary_value and "600 MB" in summary_value
    # Final performance
    assert has_field(final_embed, "Final Performance")
    perf_field = get_field(final_embed, "Final Performance")
    if perf_field is None:
        raise AssertionError("expected Final Performance field")
    perf_value = str(perf_field["value"])
    assert "92.00%" in perf_value or "0.92" in perf_value
    # Footer
    assert get_footer_text(final_embed) == "Request ID: req_metrics"


@pytest.mark.asyncio
async def test_completion_without_metrics_shows_empty_summary() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    # Call _on_completed with proper DigitsCompletedMetricsV1 TypedDict
    event: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "user_id": 101,
        "job_id": "req_no_metrics",
        "model_id": "mnist_no_prior",
        "val_acc": 0.88,
    }
    await sub._on_completed(event)
    assert user.embeds


@pytest.mark.asyncio
async def test_completion_with_partial_metrics() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "req_partial",
        "user_id": 102,
        "model_id": "mnist_partial",
        "total_epochs": 2,
        "queue": "default",
    }
    await sub._handle_event(config)
    best: DigitsBestMetricsV1 = {
        "type": "digits.metrics.best.v1",
        "job_id": "req_partial",
        "user_id": 102,
        "model_id": "mnist_partial",
        "epoch": 2,
        "val_acc": 0.95,
    }
    await sub._handle_event(best)
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "req_partial",
        "user_id": 102,
        "model_id": "mnist_partial",
        "val_acc": 0.95,
    }
    await sub._handle_event(completed)
    final_embed = user.embeds[-1]
    if final_embed is None:
        raise AssertionError("expected final embed")
    summary_field = get_field(final_embed, "Training Summary")
    if summary_field:
        summary_value = str(summary_field["value"])
        assert "Best Epoch" in summary_value and "2" in summary_value
        assert "Final Avg Loss" not in summary_value
        assert "Final Train Loss" not in summary_value
        assert "Total Time" not in summary_value
        assert "Avg Speed" not in summary_value
        assert "Peak Memory" not in summary_value


@pytest.mark.asyncio
async def test_batch_updates_metrics_correctly() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "req_batch_metrics",
        "user_id": 103,
        "model_id": "m",
        "total_epochs": 1,
        "queue": "default",
    }
    await sub._handle_event(config)
    batch1: DigitsBatchMetricsV1 = {
        "type": "digits.metrics.batch.v1",
        "job_id": "req_batch_metrics",
        "user_id": 103,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 1,
        "batch": 1,
        "total_batches": 3,
        "batch_loss": 1.0,
        "batch_acc": 0.5,
        "avg_loss": 1.0,
        "samples_per_sec": 100.0,
        "main_rss_mb": 100,
        "workers_rss_mb": 50,
        "worker_count": 2,
        "cgroup_usage_mb": 200,
        "cgroup_limit_mb": 1024,
        "cgroup_pct": 20.0,
        "anon_mb": 150,
        "file_mb": 50,
    }
    await sub._handle_event(batch1)
    batch2: DigitsBatchMetricsV1 = {
        "type": "digits.metrics.batch.v1",
        "job_id": "req_batch_metrics",
        "user_id": 103,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 1,
        "batch": 2,
        "total_batches": 3,
        "batch_loss": 0.5,
        "batch_acc": 0.8,
        "avg_loss": 0.75,
        "samples_per_sec": 200.0,
        "main_rss_mb": 300,
        "workers_rss_mb": 200,
        "worker_count": 2,
        "cgroup_usage_mb": 600,
        "cgroup_limit_mb": 1024,
        "cgroup_pct": 58.6,
        "anon_mb": 500,
        "file_mb": 100,
    }
    await sub._handle_event(batch2)
    batch3: DigitsBatchMetricsV1 = {
        "type": "digits.metrics.batch.v1",
        "job_id": "req_batch_metrics",
        "user_id": 103,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 1,
        "batch": 3,
        "total_batches": 3,
        "batch_loss": 0.3,
        "batch_acc": 0.9,
        "avg_loss": 0.6,
        "samples_per_sec": 150.0,
        "main_rss_mb": 150,
        "workers_rss_mb": 100,
        "worker_count": 2,
        "cgroup_usage_mb": 300,
        "cgroup_limit_mb": 1024,
        "cgroup_pct": 29.3,
        "anon_mb": 250,
        "file_mb": 50,
    }
    await sub._handle_event(batch3)
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "req_batch_metrics",
        "user_id": 103,
        "model_id": "m",
        "val_acc": 0.90,
    }
    await sub._handle_event(completed)
    final_embed = user.embeds[-1]
    if final_embed is None:
        raise AssertionError("expected final embed")
    summary_field = get_field(final_embed, "Training Summary")
    if summary_field is None:
        raise AssertionError("expected Training Summary field")
    summary_value = str(summary_field["value"])
    assert "0.6000" in summary_value
    assert "150.0 samples/sec" in summary_value
    assert "500 MB" in summary_value


@pytest.mark.asyncio
async def test_epoch_events_accumulate_time() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "req_time",
        "user_id": 104,
        "model_id": "m",
        "total_epochs": 3,
        "queue": "default",
    }
    await sub._handle_event(config)
    epoch1: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "req_time",
        "user_id": 104,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 3,
        "train_loss": 0.8,
        "val_acc": 0.7,
        "time_s": 60.0,
    }
    await sub._handle_event(epoch1)
    epoch2: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "req_time",
        "user_id": 104,
        "model_id": "m",
        "epoch": 2,
        "total_epochs": 3,
        "train_loss": 0.5,
        "val_acc": 0.85,
        "time_s": 75.0,
    }
    await sub._handle_event(epoch2)
    epoch3: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "req_time",
        "user_id": 104,
        "model_id": "m",
        "epoch": 3,
        "total_epochs": 3,
        "train_loss": 0.3,
        "val_acc": 0.92,
        "time_s": 90.0,
    }
    await sub._handle_event(epoch3)
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "req_time",
        "user_id": 104,
        "model_id": "m",
        "val_acc": 0.92,
    }
    await sub._handle_event(completed)
    final_embed = user.embeds[-1]
    if final_embed is None:
        raise AssertionError("expected final embed")
    summary_field = get_field(final_embed, "Training Summary")
    if summary_field is None:
        raise AssertionError("expected Training Summary field")
    summary_value = str(summary_field["value"])
    assert "Total Time" in summary_value and "3m 45s" in summary_value
    assert "0.3000" in summary_value


@pytest.mark.asyncio
async def test_failed_training_cleans_up_metrics() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "req_fail",
        "user_id": 105,
        "model_id": "m",
        "total_epochs": 5,
        "queue": "default",
    }
    await sub._handle_event(config)
    assert "req_fail" in sub.metrics
    failed: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "req_fail",
        "user_id": 105,
        "error_kind": "system",
        "message": "Out of memory",
        "domain": "digits",
    }
    await sub._handle_event(failed)
    assert "req_fail" not in sub.metrics
    assert "req_fail" not in sub._configs
    final_embed = user.embeds[-1]
    if final_embed is None:
        raise AssertionError("expected final embed")
    assert get_title(final_embed) == "Training Failed"
    assert has_field(final_embed, "System Error")


def test_completion_without_final_val_acc() -> None:
    cfg: TrainingConfig = {
        "model_id": "m",
        "total_epochs": 1,
        "queue": "default",
        "batch_size": None,
        "learning_rate": None,
        "device": None,
        "cpu_cores": None,
        "memory_mb": None,
        "optimal_threads": None,
        "optimal_workers": None,
        "augment": None,
        "aug_rotate": None,
        "aug_translate": None,
        "noise_prob": None,
        "dots_prob": None,
    }
    metrics: TrainingMetrics = {"best_epoch": 1}
    embed = build_training_embed(
        request_id="req_no_acc",
        config=cfg,
        status="completed",
        final_val_acc=None,
        final_metrics=metrics,
        run_id=None,
    )
    assert not has_field(embed, "Final Performance")
    assert has_field(embed, "Training Summary")


def test_completion_with_empty_run_id() -> None:
    cfg: TrainingConfig = {
        "model_id": "m",
        "total_epochs": 1,
        "queue": "default",
        "batch_size": None,
        "learning_rate": None,
        "device": None,
        "cpu_cores": None,
        "memory_mb": None,
        "optimal_threads": None,
        "optimal_workers": None,
        "augment": None,
        "aug_rotate": None,
        "aug_translate": None,
        "noise_prob": None,
        "dots_prob": None,
    }
    embed = build_training_embed(
        request_id="req_empty_run",
        config=cfg,
        status="completed",
        final_val_acc=0.9,
        final_metrics=None,
        run_id="",
    )
    assert not has_field(embed, "Run ID")


def test_completion_with_all_zero_metrics() -> None:
    cfg: TrainingConfig = {
        "model_id": "m",
        "total_epochs": 1,
        "queue": "default",
        "batch_size": None,
        "learning_rate": None,
        "device": None,
        "cpu_cores": None,
        "memory_mb": None,
        "optimal_threads": None,
        "optimal_workers": None,
        "augment": None,
        "aug_rotate": None,
        "aug_translate": None,
        "noise_prob": None,
        "dots_prob": None,
    }
    zero_metrics: TrainingMetrics = {}
    embed = build_training_embed(
        request_id="req_zero",
        config=cfg,
        status="completed",
        final_val_acc=0.9,
        final_metrics=zero_metrics,
        run_id="run123",
    )
    assert not has_field(embed, "Training Summary")
    assert has_field(embed, "Final Performance")


logger = logging.getLogger(__name__)
