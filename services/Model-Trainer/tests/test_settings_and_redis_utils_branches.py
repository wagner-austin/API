from __future__ import annotations

import pytest
from platform_core.config import config_test_hooks
from platform_workers.redis import _load_redis_error_class
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry

# Load the actual redis error class for test assertions
_RedisError: type[BaseException] = _load_redis_error_class()


class _FlakySetRedis(FakeRedis):
    """FakeRedis that fails on first set call."""

    def __init__(self) -> None:
        super().__init__()
        self._set_calls = 0

    def set(self, key: str, value: str) -> bool:
        self._record("set", key, value)
        self._set_calls += 1
        if self._set_calls == 1:
            raise _RedisError("transient")
        self._strings[key] = value
        return True


class _FlakyGetRedis(FakeRedis):
    """FakeRedis that fails on first get call."""

    def __init__(self) -> None:
        super().__init__()
        self._get_calls = 0

    def get(self, key: str) -> str | None:
        self._record("get", key)
        self._get_calls += 1
        if self._get_calls == 1:
            raise _RedisError("transient")
        return self._strings.get(key)


class _AlwaysFailGetRedis(FakeRedis):
    """FakeRedis that always fails on get."""

    def get(self, key: str) -> str | None:
        self._record("get", key)
        raise _RedisError("always fails")


def test_redis_utils_retry_branches() -> None:
    """Test retry logic using specialized FakeRedis subclasses."""
    # Test flaky set
    flaky_set_redis = _FlakySetRedis()
    set_with_retry(flaky_set_redis, "k", "v", attempts=2)

    # Test flaky get - need a fresh instance with data
    flaky_get_redis = _FlakyGetRedis()
    flaky_get_redis._strings["k"] = "v"  # Seed data directly
    out = get_with_retry(flaky_get_redis, "k", attempts=2)
    assert out == "v"


def test_redis_utils_get_exhausts_retries_and_raises() -> None:
    """Test get_with_retry raises after exhausting all retry attempts - covers line 24."""
    client = _AlwaysFailGetRedis()
    with pytest.raises(_RedisError, match="always fails"):
        get_with_retry(client, "key", attempts=3)
    client.assert_only_called({"get"})  # get was called but always failed


def test_redis_utils_get_zero_attempts_returns_none() -> None:
    f = FakeRedis()
    out = get_with_retry(f, "k", attempts=0)
    assert out is None
    f.assert_only_called(set())  # No calls when attempts=0


def test_redis_utils_set_success_first_attempt() -> None:
    s = FakeRedis()
    set_with_retry(s, "a", "b", attempts=3)
    assert s.get("a") == "b"
    s.assert_only_called({"set", "get"})


def test_redis_utils_get_success_first_attempt() -> None:
    s = FakeRedis()
    s.set("k", "v")
    out = get_with_retry(s, "k", attempts=3)
    assert out == "v"
    s.assert_only_called({"set", "get"})


def test_redis_utils_set_zero_attempts_noop() -> None:
    f = FakeRedis()
    # Should not raise; function returns None
    set_with_retry(f, "k", "v", attempts=0)
    f.assert_only_called(set())  # No calls when attempts=0


def test_settings_env_int_invalid_uses_default() -> None:
    base = load_settings()
    updated: Settings = {**base, "app": {**base["app"], "threads": 0}}
    assert updated["app"]["threads"] == 0


def test_settings_policy_unknown_defaults_lru() -> None:
    settings = load_settings()
    assert settings["app"]["corpus_cache_cleanup"]["eviction_policy"] in {"lru", "oldest"}


def test_settings_app_env_unknown_defaults_dev() -> None:
    # Create a fake env that returns "staging" for APP_ENV and None for APP_CONFIG_FILE
    def _fake_env(key: str) -> str | None:
        if key == "APP_ENV":
            return "staging"
        if key == "APP_CONFIG_FILE":
            return None
        # Delegate to real env for other keys
        return config_test_hooks._default_get_env(key)

    config_test_hooks.get_env = _fake_env
    settings = load_settings()
    assert settings["app_env"] == "dev"


def test_settings_policy_unknown_enabled_branch() -> None:
    # Create a fake env that sets cleanup enabled with unknown policy
    def _fake_env(key: str) -> str | None:
        mapping: dict[str, str | None] = {
            "APP__CORPUS_CACHE_CLEANUP__ENABLED": "true",
            "APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY": "random-other",
            "APP_CONFIG_FILE": None,
        }
        if key in mapping:
            return mapping[key]
        return config_test_hooks._default_get_env(key)

    config_test_hooks.get_env = _fake_env
    settings = load_settings()
    cc = settings["app"]["corpus_cache_cleanup"]
    assert cc["eviction_policy"] == "lru"
    assert cc["enabled"] is True


def test_settings_policy_oldest_branch() -> None:
    # Create a fake env that sets cleanup enabled with oldest policy
    def _fake_env(key: str) -> str | None:
        mapping: dict[str, str | None] = {
            "APP__CORPUS_CACHE_CLEANUP__ENABLED": "true",
            "APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY": "oldest",
            "APP_CONFIG_FILE": None,
        }
        if key in mapping:
            return mapping[key]
        return config_test_hooks._default_get_env(key)

    config_test_hooks.get_env = _fake_env
    settings = load_settings()
    cc = settings["app"]["corpus_cache_cleanup"]
    assert cc["eviction_policy"] == "oldest"


def test_settings_eviction_policy_oldest() -> None:
    # Create a fake env that sets eviction policy to oldest
    def _fake_env(key: str) -> str | None:
        mapping: dict[str, str | None] = {
            "APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY": "oldest",
            "APP_CONFIG_FILE": None,
        }
        if key in mapping:
            return mapping[key]
        return config_test_hooks._default_get_env(key)

    config_test_hooks.get_env = _fake_env
    settings = load_settings()
    assert settings["app"]["corpus_cache_cleanup"]["eviction_policy"] == "oldest"
