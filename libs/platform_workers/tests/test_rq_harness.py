"""Tests for RQ harness functions."""

from __future__ import annotations

import pytest

from platform_workers.rq_harness import (
    WorkerConfig,
    get_current_job,
    rq_queue,
    rq_retry,
    run_rq_worker,
)
from platform_workers.testing import (
    FakeRedisBytesClient,
    FakeRQModule,
    _FakeCurrentJob,
    _FakeRQQueueInternal,
    _FakeRQWorkerInternal,
    hooks,
    make_fake_load_redis_bytes_module,
    make_fake_load_rq_module,
)


def test_run_rq_worker_invokes_worker() -> None:
    """Test run_rq_worker creates queue and worker then calls work()."""
    # Set up the redis bytes module hook
    redis_hook, _redis_module = make_fake_load_redis_bytes_module()
    hooks.load_redis_bytes_module = redis_hook

    # Set up the rq module hook - we need to track work() calls
    # Create a custom worker class that tracks calls
    work_calls: list[bool] = []

    class _TrackingWorker(_FakeRQWorkerInternal):
        def work(self, *, with_scheduler: bool) -> None:
            work_calls.append(with_scheduler)
            super().work(with_scheduler=with_scheduler)

    # Create a custom RQ module with our tracking worker
    class _TrackingRQModule(FakeRQModule):
        def __init__(self) -> None:
            super().__init__(current_job=None)
            self.SimpleWorker = _TrackingWorker

    tracking_module = _TrackingRQModule()

    def _hook() -> FakeRQModule:
        return tracking_module

    hooks.load_rq_module = _hook

    cfg: WorkerConfig = {
        "redis_url": "redis://x",
        "queue_name": "turkic",
        "events_channel": "turkic:events",
    }
    run_rq_worker(cfg)
    assert work_calls == [True]


def test_rq_queue_enqueue_wrapper() -> None:
    """Test rq_queue returns an adapter that wraps enqueue correctly."""

    rq_hook, _rq_module = make_fake_load_rq_module()
    hooks.load_rq_module = rq_hook

    conn = FakeRedisBytesClient()
    q_adapter = rq_queue("test", connection=conn)
    job = q_adapter.enqueue("my_func", "arg1", job_timeout=60, description="test job")
    assert job.get_id() == "job-my_func"


def test_rq_worker_work_wrapper() -> None:
    """Test _rq_simple_worker returns a worker that wraps work() correctly."""
    from platform_workers import rq_harness as rh

    rq_hook, _rq_module = make_fake_load_rq_module()
    hooks.load_rq_module = rq_hook

    conn = FakeRedisBytesClient()
    q: rh._RQQueueInternal = _FakeRQQueueInternal("test", connection=conn)
    worker = rh._rq_simple_worker([q], connection=conn)
    worker.work(with_scheduler=True)
    # If we got here without error, the worker.work() was called successfully


def test_get_current_job_returns_none_outside_worker() -> None:
    """Test get_current_job returns None when not in worker context."""
    rq_hook, _rq_module = make_fake_load_rq_module(current_job=None)
    hooks.load_rq_module = rq_hook

    result = get_current_job()
    assert result is None


def test_get_current_job_returns_job_inside_worker() -> None:
    """Test get_current_job returns job when in worker context."""
    fake_job = _FakeCurrentJob(job_id="job-123", origin="test-queue")
    rq_hook, _rq_module = make_fake_load_rq_module(current_job=fake_job)
    hooks.load_rq_module = rq_hook

    result = get_current_job()
    if result is None:
        pytest.fail("expected current job")
    assert result.get_id() == "job-123"
    assert result.origin == "test-queue"


def test_rq_retry_creates_retry_object() -> None:
    """Test rq_retry creates FakeRetry with correct values."""
    from platform_workers.testing import FakeRetry

    rq_hook, _rq_module = make_fake_load_rq_module()
    hooks.load_rq_module = rq_hook

    retry = rq_retry(max_retries=3, intervals=[10, 30, 60])
    # Verify we got a FakeRetry and check stored values
    assert type(retry) is FakeRetry
    assert retry.max_retries == 3
    assert retry.intervals == [10, 30, 60]


# =============================================================================
# Production Path Tests (hooks not set)
# =============================================================================


def test_rq_fetch_job_uses_hook() -> None:
    """Test rq_fetch_job uses the fetch_job hook when set."""
    from platform_workers.rq_harness import rq_fetch_job
    from platform_workers.testing import FakeFetchedJob, make_fake_fetch_job_found

    fake_job = FakeFetchedJob(job_id="job-abc", status="finished", result={"key": "value"})
    hooks.fetch_job = make_fake_fetch_job_found(fake_job)

    conn = FakeRedisBytesClient()
    result = rq_fetch_job("job-abc", conn)
    assert result.get_id() == "job-abc"
    assert result.get_status() == "finished"
    assert result.return_value() == {"key": "value"}


def test_rq_fetch_job_not_found_hook() -> None:
    """Test rq_fetch_job with not found hook raises NoSuchJobError."""
    from platform_workers.rq_harness import load_no_such_job_error, rq_fetch_job
    from platform_workers.testing import make_fake_fetch_job_not_found

    hooks.fetch_job = make_fake_fetch_job_not_found()
    exc_cls = load_no_such_job_error()

    conn = FakeRedisBytesClient()
    with pytest.raises(exc_cls):
        rq_fetch_job("nonexistent", conn)


def test_load_no_such_job_error_returns_exception_class() -> None:
    """Test load_no_such_job_error returns the NoSuchJobError class."""
    from platform_workers.rq_harness import load_no_such_job_error

    exc_cls = load_no_such_job_error()
    assert issubclass(exc_cls, Exception)
    # Verify we can instantiate and raise it
    err = exc_cls("test message")
    assert "test message" in str(err)


# =============================================================================
# Production Path Tests (hooks not set)
# =============================================================================


def test_load_rq_module_production_path() -> None:
    """Test _load_rq_module uses real rq when hook is None."""
    from platform_workers import rq_harness as rh

    # hooks are reset by conftest, so no hook is set
    result = rh._load_rq_module()
    # Verify it returns the real rq module with expected attributes
    assert callable(result.Queue)
    assert callable(result.SimpleWorker)
    assert callable(result.Retry)


def test_rq_fetch_job_production_path_imports_work() -> None:
    """Test rq_fetch_job production path can import Job class."""
    from platform_workers.rq_harness import _RQJobClassProto

    # We can't test the full production path without a real Redis,
    # but we can verify the import machinery works
    rq_job_mod = __import__("rq.job", fromlist=["Job"])
    job_cls: _RQJobClassProto = rq_job_mod.Job
    # Verify Job class has fetch attribute that is callable
    assert callable(job_cls.fetch)


def test_rq_fetch_job_production_path_not_found() -> None:
    """Test rq_fetch_job production path raises NoSuchJobError for missing jobs."""
    import fakeredis

    from platform_workers.rq_harness import load_no_such_job_error, rq_fetch_job

    exc_cls = load_no_such_job_error()
    conn = fakeredis.FakeRedis()

    with pytest.raises(exc_cls):
        rq_fetch_job("nonexistent-job-id", conn)


def test_rq_fetch_job_production_path_success() -> None:
    """Test rq_fetch_job production path successfully fetches an existing job."""
    import fakeredis
    from rq import Queue

    from platform_workers.rq_harness import rq_fetch_job

    # Use fakeredis which provides full Redis compatibility for rq
    conn = fakeredis.FakeRedis()

    # Create a job using rq directly so we have something to fetch
    queue = Queue(connection=conn)
    job = queue.enqueue(len, "test")  # Simple function call
    job_id = job.get_id()

    # Fetch the job using our production path (no hook set)
    fetched = rq_fetch_job(job_id, conn)
    assert fetched.get_id() == job_id
    assert fetched.get_status() in ("queued", "started", "finished", "failed")
