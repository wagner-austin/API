from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

import handwriting_ai.jobs.digits as dj


class _StubJobCtx:
    def __init__(self) -> None:
        self.progress_calls = 0
        self.failed_calls = 0

    def publish_started(self) -> None:
        pass

    def publish_progress(
        self, progress: int, message: str | None = None, payload: JSONValue | None = None
    ) -> None:
        self.progress_calls += 1

    def publish_completed(self, result_id: str, result_bytes: int) -> None:
        pass

    def publish_failed(self, error_kind: str, message: str) -> None:
        self.failed_calls += 1


def test_job_emitter_progress_zero_when_total_epochs_zero() -> None:
    job_ctx = _StubJobCtx()
    payload: dj.DigitsTrainJobV1 = {
        "type": "digits.train.v1",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "epochs": 0,
        "batch_size": 1,
        "lr": 0.01,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    em = dj._JobEmitter(job_ctx, payload)
    # total_epochs == 0 should clamp progress to 0
    em.emit_epoch(epoch=1, total_epochs=0, train_loss=0.0, val_acc=0.0, time_s=0.0)
    assert em.last_progress == 0
    assert job_ctx.progress_calls == 1


def test_process_train_job_missing_redis_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure REDIS_URL is not set so the guard branch is executed
    monkeypatch.delenv("REDIS_URL", raising=False)

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r-missing-redis",
        "user_id": 1,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    with pytest.raises(RuntimeError, match="REDIS_URL not configured"):
        dj._decode_and_process_train_job(payload)


def test_process_train_job_training_progress_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide a dummy REDIS_URL to pass the first guard
    monkeypatch.setenv("REDIS_URL", "redis://test")

    # Minimal JobContext stub that the code will use before hitting the branch
    job_ctx = _StubJobCtx()

    def _mk_ctx(*args: JSONValue, **kwargs: JSONValue) -> _StubJobCtx:
        return job_ctx

    monkeypatch.setattr(dj, "make_job_context", _mk_ctx, raising=True)

    # Force `from handwriting_ai.training import progress as training_progress` to bind None
    import handwriting_ai.training as training_pkg

    monkeypatch.setattr(training_pkg, "progress", None, raising=True)

    payload: dict[str, JSONValue] = {
        "type": "digits.train.v1",
        "request_id": "r-prog-none",
        "user_id": 7,
        "model_id": "m",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    with pytest.raises(RuntimeError, match="training progress module not available"):
        dj._decode_and_process_train_job(payload)
    assert job_ctx.failed_calls >= 1
