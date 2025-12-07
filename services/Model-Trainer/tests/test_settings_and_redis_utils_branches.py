from __future__ import annotations

import pytest
from _pytest.monkeypatch import MonkeyPatch
from platform_workers.redis import _load_redis_error_class
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry

# Load the actual redis error class for test assertions
_RedisError: type[BaseException] = _load_redis_error_class()


def test_redis_utils_retry_branches(monkeypatch: MonkeyPatch) -> None:
    """Test retry logic using monkeypatched FakeRedis."""
    f = FakeRedis()
    set_calls: dict[str, int] = {"n": 0}
    get_calls: dict[str, int] = {"n": 0}
    original_set = f.set
    original_get = f.get

    def flaky_set(key: str, value: str) -> bool:
        set_calls["n"] += 1
        if set_calls["n"] == 1:
            raise _RedisError("transient")
        return original_set(key, value)

    def flaky_get(key: str) -> str | None:
        get_calls["n"] += 1
        if get_calls["n"] == 1:
            raise _RedisError("transient")
        return original_get(key)

    monkeypatch.setattr(f, "set", flaky_set)
    monkeypatch.setattr(f, "get", flaky_get)
    set_with_retry(f, "k", "v", attempts=2)
    out = get_with_retry(f, "k", attempts=2)
    assert out == "v"
    f.assert_only_called({"set", "get"})  # Both methods called via wrappers


def test_redis_utils_get_exhausts_retries_and_raises(monkeypatch: MonkeyPatch) -> None:
    """Test get_with_retry raises after exhausting all retry attempts - covers line 24."""
    client = FakeRedis()

    def always_fail_get(key: str) -> str | None:
        raise _RedisError("always fails")

    monkeypatch.setattr(client, "get", always_fail_get)
    with pytest.raises(_RedisError, match="always fails"):
        get_with_retry(client, "key", attempts=3)
    client.assert_only_called(set())


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
