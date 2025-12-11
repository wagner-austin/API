"""Shared test fixtures for turkic-api tests."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.config import config_test_hooks
from platform_core.testing import make_fake_env
from platform_workers.testing import FakeRedis

from turkic_api import _test_hooks


def make_probs(*vals: float) -> NDArray[np.float64]:
    """Create a typed numpy array for test probability values.

    This helper ensures that numpy arrays in tests have explicit types
    to satisfy strict mypy checks.
    """
    float_vals: list[np.float64] = [np.float64(v) for v in vals]
    result: NDArray[np.float64] = np.array(float_vals, dtype=np.float64)
    return result


@pytest.fixture(autouse=True)
def _reset_test_hooks() -> Generator[None, None, None]:
    """Reset all test hooks to their production defaults after each test.

    This ensures tests don't leak hook state to subsequent tests.
    """
    # Store original values - platform hooks
    original_platform_get_env = config_test_hooks.get_env

    # Store original values - turkic hooks
    original_test_runner = _test_hooks.test_runner
    original_get_env = _test_hooks.get_env
    original_redis_factory = _test_hooks.redis_factory
    original_local_corpus_service_factory = _test_hooks.local_corpus_service_factory
    original_data_bank_client_factory = _test_hooks.data_bank_client_factory
    original_data_bank_downloader_factory = _test_hooks.data_bank_downloader_factory
    original_ensure_corpus_file = _test_hooks.ensure_corpus_file
    original_load_langid_model = _test_hooks.load_langid_model
    original_to_ipa = _test_hooks.to_ipa
    original_path_exists = _test_hooks.path_exists
    original_path_unlink = _test_hooks.path_unlink

    yield

    # Restore original values - platform hooks
    config_test_hooks.get_env = original_platform_get_env

    # Restore original values - turkic hooks
    _test_hooks.test_runner = original_test_runner
    _test_hooks.get_env = original_get_env
    _test_hooks.redis_factory = original_redis_factory
    _test_hooks.local_corpus_service_factory = original_local_corpus_service_factory
    _test_hooks.data_bank_client_factory = original_data_bank_client_factory
    _test_hooks.data_bank_downloader_factory = original_data_bank_downloader_factory
    _test_hooks.ensure_corpus_file = original_ensure_corpus_file
    _test_hooks.load_langid_model = original_load_langid_model
    _test_hooks.to_ipa = original_to_ipa
    _test_hooks.path_exists = original_path_exists
    _test_hooks.path_unlink = original_path_unlink


@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Provide default test environment configuration via hooks."""
    env = make_fake_env(
        {
            "TURKIC_REDIS_URL": "redis://test-redis:6379/0",
            "TURKIC_DATA_DIR": "/tmp/turkic-test-data",
            "TURKIC_DATA_BANK_API_KEY": "test-key",
            "TURKIC_DATA_BANK_API_URL": "http://db",
        }
    )
    config_test_hooks.get_env = env

    def _fake_redis(url: str) -> FakeRedis:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    _test_hooks.redis_factory = _fake_redis
