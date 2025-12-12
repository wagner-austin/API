from __future__ import annotations

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.job_events import JobDomain
from platform_core.json_utils import JSONValue
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto

import handwriting_ai.jobs.digits as dj
from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import JobContextProtocol


class _StubJobCtx:
    def __init__(self) -> None:
        self.progress_calls = 0
        self.failed_calls = 0

    def publish_started(self) -> None:
        pass

    def publish_progress(
        self, progress: int, message: str | None = None, payload: JSONValue | None = None
    ) -> None:
        _ = message
        _ = payload
        self.progress_calls += 1

    def publish_completed(self, result_id: str, result_bytes: int) -> None:
        _ = (result_id, result_bytes)

    def publish_failed(self, error_kind: str, message: str) -> None:
        _ = (error_kind, message)
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


def test_process_train_job_missing_redis_url_raises() -> None:
    # Configure fake env with no REDIS_URL
    env = make_fake_env({})  # Empty env
    config_test_hooks.get_env = env

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


def test_process_train_job_training_progress_none() -> None:
    # Provide REDIS_URL via fake env
    env = make_fake_env({"REDIS_URL": "redis://test"})
    config_test_hooks.get_env = env

    # Stub job context
    job_ctx = _StubJobCtx()

    def _mk_ctx(
        *,
        redis: RedisStrProto,
        domain: JobDomain,
        events_channel: str,
        job_id: str,
        user_id: int,
        queue_name: str,
    ) -> JobContextProtocol:
        _ = (redis, domain, events_channel, job_id, user_id, queue_name)
        return job_ctx

    _test_hooks.make_job_context = _mk_ctx

    # Return None from training progress module hook
    def _none_progress() -> None:
        return None

    _test_hooks.get_training_progress_module = _none_progress

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
