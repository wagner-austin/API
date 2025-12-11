from __future__ import annotations

import logging
from typing import Final, NamedTuple

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
from clubbot._test_hooks import (
    RqBytesClientFactoryProtocol,
    RqQueueProtocol,
    RqRetryProtocol,
)
from clubbot.services.jobs.digits_enqueuer import RQDigitsEnqueuer


class EnqueueCall(NamedTuple):
    """Record of an enqueue call for assertions."""

    func_ref: str
    payload: dict[str, int | str | float | bool | None]
    job_timeout: int
    result_ttl: int
    failure_ttl: int
    retry_max: int
    retry_intervals: list[int]
    description: str


class _FakeJob:
    """Fake RQ job for testing."""

    __slots__ = ("_id",)

    def __init__(self, job_id: str) -> None:
        self._id = job_id

    def get_id(self) -> str:
        return self._id


class _FakeRetry:
    """Fake RQ Retry for testing."""

    __slots__ = ("interval", "max")

    def __init__(self, *, max: int, interval: list[int]) -> None:
        self.max = max
        self.interval = interval


class _FakeRedisBytesConnection:
    """Fake Redis bytes connection for testing."""

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        self._closed = False

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        """Ping implementation for protocol compliance."""
        return True

    def close(self) -> None:
        self._closed = True


class _FakeQueueWithRecording:
    """Fake RQ queue that records enqueue calls for verification."""

    __slots__ = ("_calls", "_connection", "_job_id", "_name")

    def __init__(
        self,
        name: str,
        connection: _RedisBytesClient,
        *,
        calls: list[EnqueueCall],
        job_id: str,
    ) -> None:
        self._name = name
        self._connection = connection
        self._calls = calls
        self._job_id = job_id

    def enqueue(
        self,
        func_ref: str,
        *args: _RQJsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        payload: dict[str, int | str | float | bool | None] = {}
        if len(args) > 0 and isinstance(args[0], dict):
            for k, v in args[0].items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    payload[str(k)] = v
        retry_max = 0
        retry_intervals: list[int] = []
        if isinstance(retry, _FakeRetry):
            retry_max = retry.max
            retry_intervals = list(retry.interval)
        self._calls.append(
            EnqueueCall(
                func_ref=func_ref,
                payload=payload,
                job_timeout=int(job_timeout or 0),
                result_ttl=int(result_ttl or 0),
                failure_ttl=int(failure_ttl or 0),
                retry_max=retry_max,
                retry_intervals=retry_intervals,
                description=str(description or ""),
            )
        )
        job: RQJobLike = _FakeJob(self._job_id)
        return job


class RecordingHooksState:
    """State holder for recording hooks during test."""

    __slots__ = ("calls", "connection", "job_id")

    def __init__(self, job_id: str = "jid1") -> None:
        self.calls: list[EnqueueCall] = []
        self.connection: _FakeRedisBytesConnection | None = None
        self.job_id = job_id


def _make_recording_hooks(
    state: RecordingHooksState,
) -> tuple[RqBytesClientFactoryProtocol, RqQueueProtocol, RqRetryProtocol]:
    """Create hooks that record calls for verification."""

    def _redis_hook(url: str) -> _RedisBytesClient:
        conn = _FakeRedisBytesConnection()
        state.connection = conn
        return conn

    def _queue_hook(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
        queue: RQClientQueue = _FakeQueueWithRecording(
            name, connection, calls=state.calls, job_id=state.job_id
        )
        return queue

    def _retry_hook(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
        retry: RQRetryLike = _FakeRetry(max=max_retries, interval=intervals)
        return retry

    return _redis_hook, _queue_hook, _retry_hook


def test_digits_enqueuer_builds_job_with_expected_args() -> None:
    """Test that RQDigitsEnqueuer passes correct arguments to RQ."""
    state = RecordingHooksState(job_id="jid1")
    redis_hook, queue_hook, retry_hook = _make_recording_hooks(state)

    # Set hooks
    _test_hooks.redis_raw_for_rq = redis_hook
    _test_hooks.rq_queue = queue_hook
    _test_hooks.rq_retry = retry_hook

    enq = RQDigitsEnqueuer(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        job_timeout_s=25200,
        result_ttl_s=86400,
        failure_ttl_s=604800,
        retry_max=2,
        retry_intervals_s=(60, 300),
    )
    job_id = enq.enqueue_train(
        request_id="r1",
        user_id=9,
        model_id="m",
        epochs=5,
        batch_size=32,
        lr=0.001,
        seed=42,
        augment=True,
        notes="hello",
    )

    # Verify job ID returned
    assert job_id == "jid1"

    # Verify connection was created - check it's a protocol-compliant connection
    # by verifying the close() method exists and returns without error
    conn = state.connection
    if conn is None:
        raise AssertionError("Expected connection to be created")
    conn.close()  # Should not raise

    # Verify enqueue call was recorded
    assert len(state.calls) == 1
    call = state.calls[0]

    # Verify function reference
    assert call.func_ref == "handwriting_ai.jobs.digits.process_train_job"

    # Verify payload
    payload = call.payload
    assert payload["type"] == "digits.train.v1"
    assert payload["request_id"] == "r1"
    assert payload["user_id"] == 9
    assert payload["model_id"] == "m"
    assert payload["epochs"] == 5
    assert payload["batch_size"] == 32
    assert payload["lr"] == 0.001
    assert payload["seed"] == 42
    assert payload["augment"] is True

    # Verify retry configuration
    assert call.retry_max == 2
    assert call.retry_intervals == [60, 300]

    # Verify job options
    assert call.job_timeout == 25200
    assert call.result_ttl == 86400
    assert call.failure_ttl == 604800
    assert call.description == "digits:r1"


_LOGGER: Final = logging.getLogger(__name__)
