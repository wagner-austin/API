from __future__ import annotations

from typing import TypedDict


class BatchMetrics(TypedDict):
    """Single source of truth for batch progress metrics.

    This TypedDict mirrors the metrics published in
    platform_core.digits_metrics_events.DigitsBatchMetricsV1 and is used
    internally to assemble those events.
    """

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


__all__ = ["BatchMetrics"]
