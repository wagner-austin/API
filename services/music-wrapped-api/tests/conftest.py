"""Shared test fixtures for music-wrapped-api tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.testing import make_fake_env
from platform_workers.testing import FakeQueue, FakeRedis, FakeRedisBytesClient

from music_wrapped_api import _test_hooks


@pytest.fixture(autouse=True)
def _reset_test_hooks() -> Generator[None, None, None]:
    """Reset all test hooks to their production defaults after each test.

    This ensures tests don't leak hook state to subsequent tests.
    """
    # Store original values - platform hooks
    original_platform_get_env = config_test_hooks.get_env

    # Store original values - music-wrapped hooks
    original_test_runner = _test_hooks.test_runner
    original_get_env = _test_hooks.get_env
    original_require_env = _test_hooks.require_env
    original_redis_factory = _test_hooks.redis_factory
    original_rq_conn = _test_hooks.rq_conn
    original_rq_queue_factory = _test_hooks.rq_queue_factory
    original_get_job = _test_hooks.get_job
    original_build_renderer = _test_hooks.build_renderer
    original_urlopen_get = _test_hooks.urlopen_get
    original_urlopen_post = _test_hooks.urlopen_post
    original_make_request = _test_hooks.make_request
    original_lfm_get_session_json = _test_hooks.lfm_get_session_json
    original_spotify_exchange_code = _test_hooks.spotify_exchange_code
    original_rand_state = _test_hooks.rand_state
    original_guard_find = _test_hooks.guard_find_monorepo_root
    original_guard_load = _test_hooks.guard_load_orchestrator

    yield

    # Restore original values - platform hooks
    config_test_hooks.get_env = original_platform_get_env

    # Restore original values - music-wrapped hooks
    _test_hooks.test_runner = original_test_runner
    _test_hooks.get_env = original_get_env
    _test_hooks.require_env = original_require_env
    _test_hooks.redis_factory = original_redis_factory
    _test_hooks.rq_conn = original_rq_conn
    _test_hooks.rq_queue_factory = original_rq_queue_factory
    _test_hooks.get_job = original_get_job
    _test_hooks.build_renderer = original_build_renderer
    _test_hooks.urlopen_get = original_urlopen_get
    _test_hooks.urlopen_post = original_urlopen_post
    _test_hooks.make_request = original_make_request
    _test_hooks.lfm_get_session_json = original_lfm_get_session_json
    _test_hooks.spotify_exchange_code = original_spotify_exchange_code
    _test_hooks.rand_state = original_rand_state
    _test_hooks.guard_find_monorepo_root = original_guard_find
    _test_hooks.guard_load_orchestrator = original_guard_load


@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Provide default test environment configuration via hooks."""
    env = make_fake_env(
        {
            "REDIS_URL": "redis://test-redis:6379/0",
            "SPOTIFY_CLIENT_ID": "test-spotify-id",
            "SPOTIFY_CLIENT_SECRET": "test-spotify-secret",
            "APPLE_DEVELOPER_TOKEN": "test-apple-dev-token",
            "LASTFM_API_KEY": "test-lastfm-key",
            "LASTFM_API_SECRET": "test-lastfm-secret",
        }
    )
    config_test_hooks.get_env = env

    # Also set hooks require_env to use our fake env
    def _fake_require_env(key: str) -> str:
        val = env.get(key)
        if val is None:
            raise RuntimeError(f"Missing required env var: {key}")
        return val

    _test_hooks.require_env = _fake_require_env

    def _fake_redis(url: str) -> FakeRedis:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    _test_hooks.redis_factory = _fake_redis

    def _fake_rq_conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    _test_hooks.rq_conn = _fake_rq_conn

    from platform_workers.rq_harness import _RedisBytesClient

    def _fake_rq_queue(name: str, connection: _RedisBytesClient) -> FakeQueue:
        _ = connection  # unused
        return FakeQueue(job_id="test-job-id")

    _test_hooks.rq_queue_factory = _fake_rq_queue

    # Provide deterministic state for tests
    _test_hooks.rand_state = lambda: "test-state-12345"
