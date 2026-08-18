"""Tests for RQ runtime import paths."""

from __future__ import annotations

from typing import Protocol

import platform_workers.rq_harness as rh
from platform_workers.testing import (
    FakeRedisBytesClient,
    _FakeRQQueueInternal,
    hooks,
    make_fake_load_rq_module,
)


def test_rq_runtime_imports_queue_and_worker() -> None:
    """Test _rq_queue_raw and _rq_simple_worker use the rq module hook."""
    rq_hook, _rq_module = make_fake_load_rq_module()
    hooks.load_rq_module = rq_hook

    conn = FakeRedisBytesClient()
    raw_queue = rh._rq_queue_raw("turkic", connection=conn)
    worker = rh._rq_simple_worker([raw_queue], connection=conn)

    # Verify the queue was created with correct name
    assert type(raw_queue) is _FakeRQQueueInternal
    assert raw_queue.name == "turkic"

    # Verify worker.work() can be called
    did_work = worker.work(with_scheduler=True, max_jobs=None)
    assert did_work is True


def test_public_rq_queue_factory() -> None:
    """Test rq_queue uses the rq module hook and returns adapter."""
    rq_hook, _rq_module = make_fake_load_rq_module()
    hooks.load_rq_module = rq_hook

    conn = FakeRedisBytesClient()
    q_adapter = rh.rq_queue("turkic", connection=conn)

    # Verify the adapter was created
    assert type(q_adapter) is rh._RQQueueAdapter

    # Verify the inner queue is our fake
    assert type(q_adapter._inner) is _FakeRQQueueInternal


def test_installed_rq_job_exposes_id_property() -> None:
    """Test the real rq Job still provides what the harness reads off it.

    Every other test in this suite drives `_FakeRQJob`, so the harness can
    diverge from rq without a single failure -- which is exactly what happened
    when rq dropped `Job.get_id()`: the fakes kept implementing it and the
    enqueue path raised AttributeError only in a running container. This test
    reads the installed library instead of a stand-in.
    """
    from redis import Redis
    from rq.job import Job

    class _JobIdReader(Protocol):
        """The single attribute the harness reads off an rq Job."""

        @property
        def id(self) -> str: ...

    # Redis() opens no socket until a command is issued, and rq only stores the
    # connection at construction, so nothing here touches a server.
    # `get_id()` is deliberately not asserted absent: it still exists in rq
    # 2.6.1 (this venv) and is gone in rq 2.10.0 (the container images), which
    # is exactly how the harness kept passing here while raising
    # AttributeError in production.
    real_job: _JobIdReader = Job(id="contract-job-id", connection=Redis())

    assert real_job.id == "contract-job-id"
