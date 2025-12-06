from __future__ import annotations

import pytest
from platform_workers.redis import _load_redis_error_class
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry

# Load the actual redis error class for test assertions
_RedisError: type[BaseException] = _load_redis_error_class()


class _Flaky(FakeRedis):
    """FakeRedis that fails on first get/set call, succeeds on retry."""

    def __init__(self) -> None:
        super().__init__()
        self._set_calls = 0
        self._get_calls = 0

    def set(self, key: str, value: str) -> bool:
        self._set_calls += 1
        if self._set_calls == 1:
            raise _RedisError("transient")
        return super().set(key, value)

    def get(self, key: str) -> str | None:
        self._get_calls += 1
        if self._get_calls == 1:
            raise _RedisError("transient")
        return super().get(key)


class _AlwaysFails(FakeRedis):
    """FakeRedis that always raises RedisError on get/set."""

    def set(self, key: str, value: str) -> bool:
        raise _RedisError("always fails")

    def get(self, key: str) -> str | None:
        raise _RedisError("always fails")


class _NoopGetRaises(FakeRedis):
    """FakeRedis that raises if get is called - for testing zero-attempt paths."""

    def get(self, key: str) -> str | None:
        raise AssertionError("get should not be called when attempts=0")


class _NoopSetRaises(FakeRedis):
    """FakeRedis that raises if set is called - for testing zero-attempt paths."""

    def set(self, key: str, value: str) -> bool:
        raise AssertionError("set should not be called when attempts=0")


def test_redis_utils_retry_branches() -> None:
    f = _Flaky()
    set_with_retry(f, "k", "v", attempts=2)
    out = get_with_retry(f, "k", attempts=2)
    assert out == "v"


def test_redis_utils_get_exhausts_retries_and_raises() -> None:
    """Test get_with_retry raises after exhausting all retry attempts - covers line 24."""
    client = _AlwaysFails()
    with pytest.raises(_RedisError, match="always fails"):
        get_with_retry(client, "key", attempts=3)


def test_redis_utils_get_zero_attempts_returns_none() -> None:
    out = get_with_retry(_NoopGetRaises(), "k", attempts=0)
    assert out is None


def test_redis_utils_set_success_first_attempt() -> None:
    s = FakeRedis()
    set_with_retry(s, "a", "b", attempts=3)
    assert s.get("a") == "b"


def test_redis_utils_get_success_first_attempt() -> None:
    s = FakeRedis()
    s.set("k", "v")
    out = get_with_retry(s, "k", attempts=3)
    assert out == "v"


def test_redis_utils_set_zero_attempts_noop() -> None:
    # Should not raise; function returns None
    set_with_retry(_NoopSetRaises(), "k", "v", attempts=0)


def test_settings_env_int_invalid_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    base = load_settings()
    updated: Settings = {**base, "app": {**base["app"], "threads": 0}}
    assert updated["app"]["threads"] == 0


def test_settings_policy_unknown_defaults_lru(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings()
    assert settings["app"]["corpus_cache_cleanup"]["eviction_policy"] in {"lru", "oldest"}


def test_settings_app_env_unknown_defaults_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)
    settings = load_settings()
    assert settings["app_env"] == "dev"


def test_settings_policy_unknown_enabled_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__ENABLED", "true")
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY", "random-other")
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)
    settings = load_settings()
    cc = settings["app"]["corpus_cache_cleanup"]
    assert cc["eviction_policy"] == "lru"
    assert cc["enabled"] is True


def test_settings_policy_oldest_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__ENABLED", "true")
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY", "oldest")
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)
    settings = load_settings()
    cc = settings["app"]["corpus_cache_cleanup"]
    assert cc["eviction_policy"] == "oldest"


def test_settings_eviction_policy_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY", "oldest")
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)
    settings = load_settings()
    assert settings["app"]["corpus_cache_cleanup"]["eviction_policy"] == "oldest"
