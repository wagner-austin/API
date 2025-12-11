from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from data_bank_api import _test_hooks
from data_bank_api import health as health_mod
from data_bank_api import storage as storage_mod


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore all hooks after each test."""
    original_platform_get_env = platform_hooks.get_env
    original_get_env = _test_hooks.get_env
    original_redis_factory = _test_hooks.redis_factory
    original_test_runner = _test_hooks.test_runner
    original_storage_factory = _test_hooks.storage_factory
    original_ensure_corpus = _test_hooks.ensure_corpus_file
    original_local_corpus_factory = _test_hooks.local_corpus_service_factory
    original_data_bank_client_factory = _test_hooks.data_bank_client_factory
    original_is_writable = health_mod._is_writable
    original_free_gb = health_mod._free_gb
    original_mkstemp = health_mod._mkstemp
    original_os_replace = storage_mod._os_replace
    original_os_unlink = storage_mod._os_unlink
    original_path_stat = storage_mod._path_stat
    yield
    platform_hooks.get_env = original_platform_get_env
    _test_hooks.get_env = original_get_env
    _test_hooks.redis_factory = original_redis_factory
    _test_hooks.test_runner = original_test_runner
    _test_hooks.storage_factory = original_storage_factory
    _test_hooks.ensure_corpus_file = original_ensure_corpus
    _test_hooks.local_corpus_service_factory = original_local_corpus_factory
    _test_hooks.data_bank_client_factory = original_data_bank_client_factory
    health_mod._is_writable = original_is_writable
    health_mod._free_gb = original_free_gb
    health_mod._mkstemp = original_mkstemp
    storage_mod._os_replace = original_os_replace
    storage_mod._os_unlink = original_os_unlink
    storage_mod._path_stat = original_path_stat


@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Provide default test environment with REDIS_URL and FakeRedis."""
    env = make_fake_env({"REDIS_URL": "redis://ignored"})
    _test_hooks.get_env = env

    def _rf(url: str) -> RedisStrProto:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")  # Simulate one worker
        return r

    _test_hooks.redis_factory = _rf
