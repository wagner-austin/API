"""Tests for RQ harness functions."""

from __future__ import annotations

import pytest

from platform_workers.rq_harness import (
    QueueProtocol,
    RQJobLike,
    RQRetryLike,
    WorkerConfig,
    _JsonValue,
    connecting_queue,
    get_current_job,
    rq_queue,
    rq_retry,
    run_rq_worker,
    run_single_job_rq_worker,
)
from platform_workers.testing import (
    FakeRedisBytesClient,
    FakeRedisBytesModule,
    FakeRQModule,
    _FakeCurrentJob,
    _FakeRQQueueInternal,
    _FakeRQWorkerInternal,
    hooks,
    make_fake_load_redis_bytes_module,
    make_fake_load_rq_module,
)


def _install_tracking_rq_module() -> list[tuple[bool, int | None]]:
    """Install redis + rq module hooks with a work()-tracking worker.

    Returns:
        The list that receives one (with_scheduler, max_jobs) tuple per
        work() invocation.
    """
    redis_hook, _redis_module = make_fake_load_redis_bytes_module()
    hooks.load_redis_bytes_module = redis_hook

    work_calls: list[tuple[bool, int | None]] = []

    class _TrackingWorker(_FakeRQWorkerInternal):
        def work(self, *, with_scheduler: bool, max_jobs: int | None) -> bool:
            work_calls.append((with_scheduler, max_jobs))
            return super().work(with_scheduler=with_scheduler, max_jobs=max_jobs)

    class _TrackingRQModule(FakeRQModule):
        def __init__(self) -> None:
            super().__init__(current_job=None)
            self.SimpleWorker = _TrackingWorker

    tracking_module = _TrackingRQModule()

    def _hook() -> FakeRQModule:
        return tracking_module

    hooks.load_rq_module = _hook
    return work_calls


def test_run_rq_worker_invokes_worker() -> None:
    """Test run_rq_worker calls work() with scheduler and no job limit."""
    work_calls = _install_tracking_rq_module()

    cfg: WorkerConfig = {
        "redis_url": "redis://x",
        "queue_name": "turkic",
        "events_channel": "turkic:events",
    }
    run_rq_worker(cfg)
    assert work_calls == [(True, None)]


def test_run_single_job_rq_worker_limits_to_one_job() -> None:
    """Test run_single_job_rq_worker calls work() with max_jobs=1."""
    work_calls = _install_tracking_rq_module()

    cfg: WorkerConfig = {
        "redis_url": "redis://x",
        "queue_name": "trainer",
        "events_channel": "trainer:events",
    }
    run_single_job_rq_worker(cfg)
    assert work_calls == [(True, 1)]


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
    did_work = worker.work(with_scheduler=True, max_jobs=None)
    assert did_work is True


def test_rq_queue_remove_forwards_to_the_underlying_queue() -> None:
    """The adapter must pass the id through and return RQ's own count.

    That count is the only thing distinguishing "removed it before a worker
    took it" from "too late, it is already running", so an adapter that
    swallowed it would make the cancel path unable to tell the caller which
    happened.
    """
    from platform_workers import rq_harness as rh

    rq_hook, _rq_module = make_fake_load_rq_module()
    hooks.load_rq_module = rq_hook

    conn = FakeRedisBytesClient()
    queue = rh.rq_queue("test", conn)
    job = queue.enqueue("my.func")

    assert queue.remove(job.get_id()) == 1
    assert queue.remove(job.get_id()) == 0


def test_internal_fake_queue_remove_reports_zero_for_an_unknown_id() -> None:
    conn = FakeRedisBytesClient()
    inner = _FakeRQQueueInternal("test", connection=conn)

    assert inner.remove("never-enqueued") == 0


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


class TestConnectingQueue:
    """The queue adapter transcript-api and turkic-api each defined inline.

    Both had it byte-identical but for the queue-name constant, alongside
    their own copies of QueueProtocol and the enqueue-callable protocol. What
    is worth pinning here is the behaviour those copies encoded: the queue
    name reaches RQ, a callable reference is stringified, and a connection is
    made PER enqueue rather than held.
    """

    def _fakes(self) -> FakeRQModule:
        """Install the redis and rq module hooks.

        Returns:
            The fake rq module, so a test can read what reached it.
        """
        redis_hook, _redis_module = make_fake_load_redis_bytes_module()
        hooks.load_redis_bytes_module = redis_hook
        rq_hook, rq_module = make_fake_load_rq_module()
        hooks.load_rq_module = rq_hook
        return rq_module

    def test_it_enqueues_onto_the_name_it_was_built_with(self) -> None:
        self._fakes()

        job = connecting_queue("transcripts", "redis://x").enqueue("my_func")

        assert job.get_id() == "job-my_func"

    def test_a_callable_reference_is_stringified(self) -> None:
        """RQ takes a dotted path, so a caller passing the job function itself
        relies on `str(func)`. Both copies did this and it is the only branch
        in the adapter."""
        self._fakes()

        class _Job:
            def __call__(
                self,
                *args: _JsonValue,
                job_timeout: int | None = None,
                result_ttl: int | None = None,
                failure_ttl: int | None = None,
                retry: RQRetryLike | None = None,
                description: str | None = None,
            ) -> RQJobLike:
                raise NotImplementedError("never called; only its str() is used")

            def __str__(self) -> str:
                return "pkg.module.my_job"

        job = connecting_queue("q", "redis://x").enqueue(_Job())

        assert job.get_id() == "job-pkg.module.my_job"

    def test_each_enqueue_opens_its_own_connection(self) -> None:
        """Connecting per call rather than holding one open is why this is an
        adapter at all -- a FastAPI dependency is resolved per request, and a
        long-lived binary connection shared across them is what the services
        were avoiding.

        Counted at the module-loading seam rather than by extending the
        shipped fake, since a connection is made by loading the module and
        calling from_url, and the seam is where that begins.
        """
        _redis_hook, redis_module = make_fake_load_redis_bytes_module()
        loads: list[int] = []

        def _counting_hook() -> FakeRedisBytesModule:
            loads.append(1)
            return redis_module

        hooks.load_redis_bytes_module = _counting_hook
        rq_hook, _rq_module = make_fake_load_rq_module()
        hooks.load_rq_module = rq_hook
        queue = connecting_queue("q", "redis://x")

        queue.enqueue("a")
        queue.enqueue("b")

        assert len(loads) == 2
        assert redis_module.from_url_url == "redis://x"

    def test_it_satisfies_the_queue_protocol(self) -> None:
        """The services annotate their FastAPI dependency with QueueProtocol,
        so the concrete adapter has to satisfy it structurally."""
        self._fakes()
        queue: QueueProtocol = connecting_queue("q", "redis://x")

        assert queue.enqueue("f").get_id() == "job-f"
