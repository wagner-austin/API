"""Tests for queue enqueue functionality in dependencies."""

from __future__ import annotations

from platform_workers.testing import (
    hooks,
    make_fake_load_redis_bytes_module,
    make_fake_load_rq_module,
)

from turkic_api.api.dependencies import get_queue, get_settings


def test_queue_enqueue_calls_underlying() -> None:
    """Test that get_queue properly enqueues jobs through platform_workers.

    Uses platform_workers.testing hooks to inject fake RQ module.
    The fake RQ module creates jobs with id "job-{func_ref}".
    """
    # Save original hooks
    orig_bytes_hook = hooks.load_redis_bytes_module
    orig_rq_hook = hooks.load_rq_module

    # Set up fake hooks
    bytes_hook, _bytes_mod = make_fake_load_redis_bytes_module()
    rq_hook, _rq_mod = make_fake_load_rq_module()
    hooks.load_redis_bytes_module = bytes_hook
    hooks.load_rq_module = rq_hook

    try:
        # Build queue using settings dependency
        q = get_queue(get_settings())

        # Enqueue using a string reference with valid RQ parameters
        job = q.enqueue("pkg.add", 1, 2, job_timeout=30, description="test job")

        # Verify job was created - the fake RQ module creates jobs with id "job-{func}"
        assert job.get_id() == "job-pkg.add"

        # Enqueue another job to verify pattern
        job2 = q.enqueue("pkg.subtract", 5, 3, result_ttl=300)
        assert job2.get_id() == "job-pkg.subtract"
    finally:
        # Restore original hooks
        hooks.load_redis_bytes_module = orig_bytes_hook
        hooks.load_rq_module = orig_rq_hook
