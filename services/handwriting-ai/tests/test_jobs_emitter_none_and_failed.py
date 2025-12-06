from __future__ import annotations

from platform_core.json_utils import JSONValue

import handwriting_ai.jobs.digits as dj


class _StubJobCtx:
    def __init__(self) -> None:
        self.progress_calls = 0
        self.started_calls = 0
        self.completed_calls = 0
        self.failed_calls = 0

    def publish_started(self) -> None:
        self.started_calls += 1

    def publish_progress(
        self, progress: int, message: str | None = None, payload: JSONValue | None = None
    ) -> None:
        self.progress_calls += 1

    def publish_completed(self, result_id: str, result_bytes: int) -> None:
        self.completed_calls += 1

    def publish_failed(self, error_kind: str, message: str) -> None:
        self.failed_calls += 1


def test_progress_emitter_no_publisher_noops() -> None:
    job_ctx = _StubJobCtx()
    payload: dj.DigitsTrainJobV1 = {
        "type": "digits.train.v1",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.01,
        "seed": 42,
        "augment": False,
        "notes": None,
    }
    em = dj._JobEmitter(job_ctx, payload)
    em.emit_batch(
        {
            "epoch": 1,
            "total_epochs": 1,
            "batch": 1,
            "total_batches": 1,
            "batch_loss": 0.1,
            "batch_acc": 0.9,
            "avg_loss": 0.1,
            "samples_per_sec": 100.0,
            "main_rss_mb": 100,
            "workers_rss_mb": 50,
            "worker_count": 2,
            "cgroup_usage_mb": 500,
            "cgroup_limit_mb": 1000,
            "cgroup_pct": 50.0,
            "anon_mb": 200,
            "file_mb": 150,
        }
    )
    em.emit_best(epoch=1, val_acc=0.5)
    em.emit_epoch(epoch=1, total_epochs=1, train_loss=0.1, val_acc=0.2, time_s=0.1)
    assert job_ctx.progress_calls == 3
