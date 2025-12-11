"""Tests for redis factory functions and adapters."""

from __future__ import annotations

from platform_workers import redis as mod
from platform_workers.testing import (
    FakeRedisClient,
    hooks,
    make_fake_load_redis_asyncio_module,
    make_fake_load_redis_bytes_module,
    make_fake_load_redis_str_module,
)


class _FakeRedisBytes:
    """Minimal fake for bytes protocol testing."""

    def __init__(self) -> None:
        self._closed = False

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        self._closed = True


def test_redis_for_kv_factory() -> None:
    """Test redis_for_kv uses the str module loader hook."""
    fake = FakeRedisClient()
    hook_fn, _ = make_fake_load_redis_str_module(fake)
    hooks.load_redis_str_module = hook_fn

    r = mod.redis_for_kv("redis://x")
    assert r.ping()
    assert r.get("a") is None
    assert r.set("a", "1")
    assert r.get("a") == "1"
    assert r.hset("h", {"k": "v"}) == 1
    assert r.hgetall("h")["k"] == "v"
    assert r.publish("chan", "msg") == 1
    assert r.sadd("set1", "m1") == 1
    assert r.scard("set1") == 1
    r.close()
    fake.assert_only_called(
        {"ping", "get", "set", "hset", "hgetall", "publish", "sadd", "scard", "close"}
    )


def test_redis_for_rq_factory() -> None:
    """Test redis_for_rq uses the bytes module loader hook."""
    hook_fn, fake_module = make_fake_load_redis_bytes_module()
    hooks.load_redis_bytes_module = hook_fn

    r = mod.redis_for_rq("redis://x")
    assert r.ping()
    r.close()
    assert fake_module.from_url_called


def test_redis_runtime_import_kv() -> None:
    """Test redis_for_kv runtime import path."""
    fake = FakeRedisClient()
    hook_fn, _ = make_fake_load_redis_str_module(fake)
    hooks.load_redis_str_module = hook_fn

    kv = mod.redis_for_kv("redis://kv")
    assert kv.ping()
    fake.assert_only_called({"ping"})


def test_redis_runtime_import_rq() -> None:
    """Test redis_for_rq runtime import path."""
    hook_fn, fake_module = make_fake_load_redis_bytes_module()
    hooks.load_redis_bytes_module = hook_fn

    rb = mod.redis_for_rq("redis://rq")
    assert rb.ping()
    assert fake_module.from_url_called


def test_redis_str_adapter_comprehensive() -> None:
    """Test _RedisStrAdapter methods comprehensively."""
    fake = FakeRedisClient()
    adapter = mod._RedisStrAdapter(fake)
    assert adapter.ping()
    assert adapter.get("missing") is None
    assert adapter.set("key1", "val1")
    assert adapter.get("key1") == "val1"
    assert adapter.hset("hash1", {"f1": "v1", "f2": "v2"}) == 2
    assert adapter.hget("hash1", "f1") == "v1"
    assert adapter.hget("hash1", "missing") is None
    result = adapter.hgetall("hash1")
    assert result["f1"] == "v1"
    assert result["f2"] == "v2"
    assert adapter.publish("channel1", "message1") == 1
    assert adapter.sadd("set1", "member1") == 1
    assert adapter.sismember("set1", "member1")
    assert not adapter.sismember("set1", "notamember")
    adapter.close()
    fake.assert_only_called(
        {"ping", "get", "set", "hset", "hget", "hgetall", "publish", "sadd", "sismember", "close"}
    )


def test_redis_str_adapter_delete_and_expire() -> None:
    """Test _RedisStrAdapter delete and expire methods."""
    fake = FakeRedisClient()
    adapter = mod._RedisStrAdapter(fake)
    # Test delete
    adapter.set("key1", "val1")
    assert adapter.get("key1") == "val1"
    deleted = adapter.delete("key1")
    assert deleted >= 0  # Returns int count
    assert adapter.get("key1") is None
    # Test expire (returns bool)
    adapter.set("key2", "val2")
    result = adapter.expire("key2", 60)
    assert result is True or result is False  # Must be a bool
    fake.assert_only_called({"set", "get", "delete", "expire"})


def test_redis_bytes_adapter_comprehensive() -> None:
    """Test _RedisBytesAdapter methods."""
    fake = _FakeRedisBytes()
    adapter = mod._RedisBytesAdapter(fake)
    assert adapter.ping()
    adapter.close()


def test_redis_for_kv_runtime_module() -> None:
    """Test redis_for_kv with str module hook and URL verification."""
    fake = FakeRedisClient()
    hook_fn, _ = make_fake_load_redis_str_module(fake)
    hooks.load_redis_str_module = hook_fn

    kv = mod.redis_for_kv("redis://kv-runtime")
    assert kv.ping()
    assert kv.get("missing") is None
    assert kv.set("k1", "v1")
    assert kv.hset("h1", {"f": "v"}) == 1
    assert kv.sadd("s1", "m1") in (0, 1)
    assert kv.scard("s1") in (0, 1)
    fake.assert_only_called({"ping", "get", "set", "hset", "sadd", "scard"})


def test_redis_for_rq_runtime_module() -> None:
    """Test redis_for_rq with bytes module hook."""
    hook_fn, fake_module = make_fake_load_redis_bytes_module()
    hooks.load_redis_bytes_module = hook_fn

    rb = mod.redis_for_rq("redis://rq-runtime")
    assert rb.ping()
    assert fake_module.from_url_called
    assert fake_module.from_url_url == "redis://rq-runtime"


def test_redis_for_pubsub_factory() -> None:
    """Test redis_for_pubsub with asyncio module hook."""
    hook_fn, fake_module = make_fake_load_redis_asyncio_module()
    hooks.load_redis_asyncio_module = hook_fn

    client = mod.redis_for_pubsub("redis://pubsub")
    ps = client.pubsub()
    assert callable(ps.subscribe)
    assert fake_module.from_url_called


def test_redis_async_from_url_runtime_module() -> None:
    """Test _redis_async_from_url with asyncio module hook."""
    hook_fn, fake_module = make_fake_load_redis_asyncio_module()
    hooks.load_redis_asyncio_module = hook_fn

    client = mod.redis_for_pubsub("redis://async-test")
    ps = client.pubsub()
    assert callable(ps.subscribe)
    assert fake_module.from_url_url == "redis://async-test"


def test_load_redis_asyncio_module_imports_redis_asyncio() -> None:
    """Test _load_redis_asyncio_module with hook."""
    hook_fn, _fake_module = make_fake_load_redis_asyncio_module()
    hooks.load_redis_asyncio_module = hook_fn

    result = mod._load_redis_asyncio_module()
    assert callable(result.from_url)


def test_redis_error_class_exists() -> None:
    """Test that RedisError is a subclass of Exception."""
    assert issubclass(mod.RedisError, Exception)
    err = mod.RedisError("test error")
    assert str(err) == "test error"


def test_is_redis_error_returns_true_for_redis_exceptions() -> None:
    """Test is_redis_error with actual redis exceptions."""
    error_cls = mod._load_redis_error_class()
    redis_error = error_cls("test")
    assert mod.is_redis_error(redis_error)


def test_is_redis_error_returns_false_for_other_exceptions() -> None:
    """Test is_redis_error with non-redis exceptions."""
    assert not mod.is_redis_error(ValueError("test"))
    assert not mod.is_redis_error(RuntimeError("test"))
    assert not mod.is_redis_error(Exception("test"))


def test_load_redis_error_class_returns_redis_error() -> None:
    """Test that _load_redis_error_class returns the actual redis.exceptions.RedisError."""
    error_cls = mod._load_redis_error_class()
    # Verify that it's a subclass of BaseException
    assert issubclass(error_cls, BaseException)
    # Verify we can instantiate and raise it
    err = error_cls("test message")
    assert str(err) == "test message"


def test_redis_raw_for_rq_runtime_module() -> None:
    """Test that redis_raw_for_rq returns a raw Redis client."""
    hook_fn, fake_module = make_fake_load_redis_bytes_module()
    hooks.load_redis_bytes_module = hook_fn

    client = mod.redis_raw_for_rq("redis://raw-test")
    assert client.ping()
    assert fake_module.from_url_called
    assert fake_module.from_url_url == "redis://raw-test"


# =============================================================================
# Production Path Tests (hooks not set)
# =============================================================================


def test_load_redis_str_module_production_path() -> None:
    """Test _load_redis_str_module uses real redis when hook is None."""
    # hooks are reset by conftest, so no hook is set
    result = mod._load_redis_str_module()
    # Verify it returns something with from_url callable
    assert callable(result.from_url)


def test_load_redis_bytes_module_production_path() -> None:
    """Test _load_redis_bytes_module uses real redis when hook is None."""
    result = mod._load_redis_bytes_module()
    assert callable(result.from_url)


def test_load_redis_asyncio_module_production_path() -> None:
    """Test _load_redis_asyncio_module uses real redis.asyncio when hook is None."""
    result = mod._load_redis_asyncio_module()
    assert callable(result.from_url)
