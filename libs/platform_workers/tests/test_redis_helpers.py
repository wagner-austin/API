from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from platform_workers import redis as mod


class _KV:
    def ping(self, **kw: str | int | float | bool | None) -> bool:
        return True

    def set(self, name: str, value: str) -> bool | str:
        return True

    def get(self, name: str) -> str | None:
        return None

    def delete(self, name: str) -> int:
        return 0

    def expire(self, name: str, time: int) -> bool:
        return False

    def hset(self, name: str, mapping: dict[str, str]) -> int:
        return len(mapping)

    def hget(self, key: str, field: str) -> str | None:
        return None

    def hgetall(self, name: str) -> dict[str, str]:
        return {}

    def publish(self, channel: str, message: str) -> int:
        return 1

    def scard(self, name: str) -> int:
        return 0

    def sadd(self, name: str, value: str) -> int:
        return 1

    def sismember(self, key: str, member: str) -> bool:
        return False

    def close(self) -> None:
        pass


class _FakeRedisClient:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._hash: dict[str, dict[str, str]] = {}
        self._set: dict[str, set[str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, name: str, value: str) -> bool | str:
        self._kv[name] = value
        return True

    def get(self, name: str) -> str | None:
        return self._kv.get(name)

    def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            if name in self._kv:
                del self._kv[name]
                removed += 1
            if name in self._hash:
                del self._hash[name]
                removed += 1
            if name in self._set:
                del self._set[name]
                removed += 1
        return removed

    def expire(self, name: str, time: int) -> bool:
        return name in self._kv or name in self._hash or name in self._set

    def hset(self, name: str, mapping: dict[str, str]) -> int:
        d = self._hash.setdefault(name, {})
        for k, v in mapping.items():
            d[k] = v
        return len(mapping)

    def hget(self, name: str, key: str) -> str | None:
        return self._hash.get(name, {}).get(key)

    def hgetall(self, name: str) -> dict[str, str]:
        return self._hash.get(name, {}).copy()

    def publish(self, channel: str, message: str) -> int:
        return 1

    def scard(self, name: str) -> int:
        return len(self._set.get(name, set()))

    def sadd(self, name: str, value: str) -> int:
        s = self._set.setdefault(name, set())
        before = len(s)
        s.add(value)
        return 1 if len(s) > before else 0

    def sismember(self, name: str, value: str) -> bool | int:
        return value in self._set.get(name, set())

    def close(self) -> None:
        pass


class _FakeRedisStr:
    def __init__(self) -> None:
        self._pub: list[tuple[str, str]] = []
        self._kv: dict[str, str] = {}
        self._hash: dict[str, dict[str, str]] = {}
        self._set: dict[str, set[str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool | str:
        self._kv[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    def delete(self, key: str) -> int:
        removed = 0
        if key in self._kv:
            del self._kv[key]
            removed += 1
        if key in self._hash:
            del self._hash[key]
            removed += 1
        if key in self._set:
            del self._set[key]
            removed += 1
        return removed

    def expire(self, key: str, time: int) -> bool:
        return key in self._kv or key in self._hash or key in self._set

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        d = self._hash.setdefault(key, {})
        for k, v in mapping.items():
            d[k] = v
        return len(mapping)

    def hget(self, key: str, field: str) -> str | None:
        return self._hash.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        return self._hash.get(key, {}).copy()

    def publish(self, channel: str, message: str) -> int:
        self._pub.append((channel, message))
        return 1

    def scard(self, key: str) -> int:
        return len(self._set.get(key, set()))

    def sadd(self, key: str, member: str) -> int:
        s = self._set.setdefault(key, set())
        before = len(s)
        s.add(member)
        return 1 if len(s) > before else 0

    def sismember(self, key: str, member: str) -> bool:
        return member in self._set.get(key, set())

    def close(self) -> None:
        pass


class _FakeRedisBytes:
    def __init__(self) -> None:
        self._closed = False

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        self._closed = True


def test_redis_for_kv_factory(monkeypatch: MonkeyPatch) -> None:
    fake = _FakeRedisStr()

    def _factory(_url: str) -> mod.RedisStrProto:
        return fake

    monkeypatch.setattr(mod, "_redis_from_url_str", _factory, raising=True)
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


def test_redis_for_rq_factory(monkeypatch: MonkeyPatch) -> None:
    fake = _FakeRedisBytes()

    def _factory(_url: str) -> mod.RedisBytesProto:
        return fake

    monkeypatch.setattr(mod, "_redis_from_url_bytes", _factory, raising=True)
    r = mod.redis_for_rq("redis://x")
    assert r.ping()
    r.close()


def test_redis_runtime_import_kv(monkeypatch: MonkeyPatch) -> None:
    def _fake_from_url(
        url: str,
        **kwargs: str | int | float | bool | None,
    ) -> mod.RedisStrProto:
        return _KV()

    monkeypatch.setattr("platform_workers.redis.redis.from_url", _fake_from_url, raising=True)
    kv = mod.redis_for_kv("redis://kv")
    assert kv.ping()


def test_redis_runtime_import_rq(monkeypatch: MonkeyPatch) -> None:
    class _B:
        def ping(self, **kw: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            pass

    def _fake_from_url(
        url: str,
        **kwargs: str | int | float | bool | None,
    ) -> mod.RedisBytesProto:
        return _B()

    monkeypatch.setattr("platform_workers.redis.redis.from_url", _fake_from_url, raising=True)
    rb = mod.redis_for_rq("redis://rq")
    assert rb.ping()


def test_redis_str_adapter_comprehensive() -> None:
    fake = _FakeRedisClient()
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


def test_redis_str_adapter_delete_and_expire() -> None:
    fake = _FakeRedisClient()
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


def test_redis_bytes_adapter_comprehensive() -> None:
    fake = _FakeRedisBytes()
    adapter = mod._RedisBytesAdapter(fake)
    assert adapter.ping()
    adapter.close()


def test_redis_for_kv_runtime_module(monkeypatch: MonkeyPatch) -> None:
    class _Module:
        def from_url(
            self,
            url: str,
            *,
            encoding: str,
            decode_responses: bool,
            socket_connect_timeout: float,
            socket_timeout: float,
            retry_on_timeout: bool,
        ) -> _KV:
            assert url == "redis://kv-runtime"
            assert encoding == "utf-8"
            assert decode_responses
            assert socket_connect_timeout == 1.0
            assert socket_timeout == 1.0
            assert retry_on_timeout
            return _KV()

    monkeypatch.setattr(mod, "_load_redis_str_module", lambda: _Module(), raising=True)
    kv = mod.redis_for_kv("redis://kv-runtime")
    assert kv.ping()
    assert kv.get("missing") is None
    assert kv.set("k1", "v1")
    assert kv.hset("h1", {"f": "v"}) == 1
    assert kv.sadd("s1", "m1") in (0, 1)
    assert kv.scard("s1") in (0, 1)


def test_redis_for_rq_runtime_module(monkeypatch: MonkeyPatch) -> None:
    class _B:
        def __init__(self) -> None:
            self.closed = False

        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            self.closed = True

    class _Module:
        def from_url(
            self,
            url: str,
            *,
            decode_responses: bool,
            socket_connect_timeout: float,
            socket_timeout: float,
            retry_on_timeout: bool,
        ) -> _B:
            assert url == "redis://rq-runtime"
            assert not decode_responses
            assert socket_connect_timeout == 1.0
            assert socket_timeout == 1.0
            assert retry_on_timeout
            return _B()

    monkeypatch.setattr(mod, "_load_redis_bytes_module", lambda: _Module(), raising=True)
    rb = mod.redis_for_rq("redis://rq-runtime")
    assert rb.ping()


class _FakePubSub:
    async def subscribe(self, *channels: str) -> None:
        pass

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> mod.PubSubMessage | None:
        return None

    async def close(self) -> None:
        pass


class _FakeAsyncRedis:
    def pubsub(self) -> _FakePubSub:
        return _FakePubSub()


def test_redis_for_pubsub_factory(monkeypatch: MonkeyPatch) -> None:
    fake = _FakeAsyncRedis()

    def _fake_factory(_url: str) -> mod.RedisAsyncProto:
        return fake

    monkeypatch.setattr(mod, "_redis_async_from_url", _fake_factory, raising=True)
    client = mod.redis_for_pubsub("redis://pubsub")
    ps = client.pubsub()
    assert callable(ps.subscribe)


def test_redis_async_from_url_runtime_module(monkeypatch: MonkeyPatch) -> None:
    class _FakeAsyncioModule:
        def from_url(self, url: str, *, encoding: str, decode_responses: bool) -> _FakeAsyncRedis:
            assert url == "redis://async-test"
            assert encoding == "utf-8"
            assert decode_responses
            return _FakeAsyncRedis()

    monkeypatch.setattr(
        mod, "_load_redis_asyncio_module", lambda: _FakeAsyncioModule(), raising=True
    )
    client = mod.redis_for_pubsub("redis://async-test")
    ps = client.pubsub()
    assert callable(ps.subscribe)


def test_load_redis_asyncio_module_imports_redis_asyncio(monkeypatch: MonkeyPatch) -> None:
    import importlib

    class _FakeAsyncioMod:
        def from_url(self, url: str, *, encoding: str, decode_responses: bool) -> _FakeAsyncRedis:
            return _FakeAsyncRedis()

    def _fake_import(name: str) -> _FakeAsyncioMod:
        assert name == "redis.asyncio"
        return _FakeAsyncioMod()

    monkeypatch.setattr(importlib, "import_module", _fake_import, raising=True)
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


def test_redis_raw_for_rq_runtime_module(monkeypatch: MonkeyPatch) -> None:
    """Test that redis_raw_for_rq returns a raw Redis client."""

    class _RawClient:
        def __init__(self) -> None:
            self.closed = False

        def ping(self, **kwargs: str | int | float | bool | None) -> bool:
            return True

        def close(self) -> None:
            self.closed = True

    class _Module:
        def from_url(
            self,
            url: str,
            *,
            decode_responses: bool,
            socket_connect_timeout: float,
            socket_timeout: float,
            retry_on_timeout: bool,
        ) -> _RawClient:
            assert url == "redis://raw-test"
            assert not decode_responses
            assert socket_connect_timeout == 1.0
            assert socket_timeout == 1.0
            assert retry_on_timeout
            return _RawClient()

    monkeypatch.setattr(mod, "_load_redis_bytes_module", lambda: _Module(), raising=True)
    client = mod.redis_raw_for_rq("redis://raw-test")
    assert client.ping()
