from __future__ import annotations

from typing import TypedDict


class TrainingConfig(TypedDict):
    """Immutable training configuration for display in progress messages."""

    model_id: str
    total_epochs: int
    queue: str
    batch_size: int | None
    learning_rate: float | None
    device: str | None
    cpu_cores: int | None
    memory_mb: int | None
    optimal_threads: int | None
    optimal_workers: int | None
    augment: bool | None
    aug_rotate: float | None
    aug_translate: float | None
    noise_prob: float | None
    dots_prob: float | None


class BatchProgress(TypedDict):
    """Batch-level progress metrics for training updates."""

    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    batch_loss: float
    batch_acc: float
    avg_loss: float
    samples_per_sec: float
    main_rss_mb: int
    workers_rss_mb: int
    worker_count: int
    cgroup_usage_mb: int
    cgroup_limit_mb: int
    cgroup_pct: float
    anon_mb: int
    file_mb: int


class TrainingMetrics(TypedDict, total=False):
    """Cumulative training metrics tracked throughout the lifecycle for final summary."""

    final_avg_loss: float
    final_train_loss: float
    total_time_s: float
    avg_samples_per_sec: float
    best_epoch: int
    peak_memory_mb: int


__all__ = ["BatchProgress", "TrainingConfig", "TrainingMetrics"]
