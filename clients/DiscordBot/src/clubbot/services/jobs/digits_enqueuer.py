from __future__ import annotations

from typing import Protocol, TypedDict

from platform_workers.rq_harness import (
    RQClientQueue,
    RQJobLike,
    RQRetryLike,
    _RedisBytesClient,
)
from platform_workers.rq_harness import (
    _JsonValue as _RQJsonValue,
)

from clubbot import _test_hooks


class DigitsEnqueuer(Protocol):
    def enqueue_train(
        self,
        *,
        request_id: str,
        user_id: int,
        model_id: str,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        augment: bool,
        notes: str | None = None,
    ) -> str: ...


class RQDigitsEnqueuerConfig(TypedDict):
    redis_url: str
    queue_name: str
    job_timeout_s: int
    result_ttl_s: int
    failure_ttl_s: int
    retry_max: int
    retry_intervals_s: tuple[int, int]


class RQDigitsEnqueuer:
    def __init__(
        self,
        *,
        redis_url: str,
        queue_name: str,
        job_timeout_s: int,
        result_ttl_s: int,
        failure_ttl_s: int,
        retry_max: int,
        retry_intervals_s: tuple[int, int],
    ) -> None:
        self._config: RQDigitsEnqueuerConfig = {
            "redis_url": redis_url,
            "queue_name": queue_name,
            "job_timeout_s": job_timeout_s,
            "result_ttl_s": result_ttl_s,
            "failure_ttl_s": failure_ttl_s,
            "retry_max": retry_max,
            "retry_intervals_s": retry_intervals_s,
        }

    def enqueue_train(
        self,
        *,
        request_id: str,
        user_id: int,
        model_id: str,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        augment: bool,
        notes: str | None = None,
    ) -> str:
        return _enqueue_train_impl(
            self._config,
            request_id=request_id,
            user_id=user_id,
            model_id=model_id,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            augment=augment,
            notes=notes,
        )


def _redis_from_url(redis_url: str) -> _RedisBytesClient:
    conn: _RedisBytesClient = _test_hooks.redis_raw_for_rq(redis_url)
    return conn


def _enqueue_train_impl(
    enqueuer: RQDigitsEnqueuerConfig,
    *,
    request_id: str,
    user_id: int,
    model_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    augment: bool,
    notes: str | None = None,
) -> str:
    conn: _RedisBytesClient = _redis_from_url(enqueuer["redis_url"])
    queue: RQClientQueue = _rq_queue(enqueuer["queue_name"], connection=conn)
    retry: RQRetryLike = _rq_retry(
        max_retries=enqueuer["retry_max"], intervals=list(enqueuer["retry_intervals_s"])
    )
    payload: _RQJsonValue = {
        "type": "digits.train.v1",
        "request_id": request_id,
        "user_id": int(user_id),
        "model_id": str(model_id),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "augment": bool(augment),
        "notes": (str(notes) if isinstance(notes, str) and notes else None),
    }
    job: RQJobLike = queue.enqueue(
        "handwriting_ai.jobs.digits.process_train_job",
        payload,
        job_timeout=enqueuer["job_timeout_s"],
        result_ttl=enqueuer["result_ttl_s"],
        failure_ttl=enqueuer["failure_ttl_s"],
        retry=retry,
        description=f"digits:{request_id}",
    )
    return job.get_id()


def _rq_queue(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
    return _test_hooks.rq_queue(name, connection=connection)


def _rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    return _test_hooks.rq_retry(max_retries=max_retries, intervals=intervals)
