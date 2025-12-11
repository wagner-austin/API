"""Tests for platform_workers.testing module."""

from __future__ import annotations

import pytest

from platform_workers.testing import (
    EnqueuedJob,
    FakeJob,
    FakeLogger,
    FakeQueue,
    FakeRedis,
    FakeRedisBytesClient,
    FakeRedisClient,
    FakeRedisHsetError,
    FakeRedisHsetRedisError,
    FakeRedisPublishError,
    FakeRetry,
    LogRecord,
    MethodCall,
    Published,
)


class TestFakeRedis:
    def test_ping_returns_true(self) -> None:
        redis = FakeRedis()
        assert redis.ping() is True

    def test_set_and_get_string(self) -> None:
        redis = FakeRedis()
        assert redis.set("key", "value") is True
        assert redis.get("key") == "value"
        assert redis.get("missing") is None

    def test_hset_and_hgetall(self) -> None:
        redis = FakeRedis()
        result = redis.hset("hash", {"a": "1", "b": "2"})
        assert result == 2
        assert redis.hgetall("hash") == {"a": "1", "b": "2"}
        assert redis.hgetall("missing") == {}

    def test_hget(self) -> None:
        redis = FakeRedis()
        redis.hset("hash", {"field": "value"})
        assert redis.hget("hash", "field") == "value"
        assert redis.hget("hash", "missing") is None
        assert redis.hget("missing", "field") is None

    def test_publish_tracks_messages(self) -> None:
        redis = FakeRedis()
        result = redis.publish("channel", "message")
        assert result == 1
        assert len(redis.published) == 1
        assert redis.published[0] == Published("channel", "message")

    def test_set_operations(self) -> None:
        redis = FakeRedis()
        assert redis.scard("myset") == 0
        assert redis.sadd("myset", "a") == 1
        assert redis.sadd("myset", "a") == 0  # Already exists
        assert redis.sadd("myset", "b") == 1
        assert redis.scard("myset") == 2
        assert redis.sismember("myset", "a") is True
        assert redis.sismember("myset", "c") is False
        assert redis.sismember("missing", "a") is False

    def test_delete_string(self) -> None:
        redis = FakeRedis()
        redis.set("key", "value")
        assert redis.delete("key") == 1
        assert redis.get("key") is None
        assert redis.delete("key") == 0  # Already deleted

    def test_delete_hash(self) -> None:
        redis = FakeRedis()
        redis.hset("hash", {"a": "1"})
        assert redis.delete("hash") == 1
        assert redis.hgetall("hash") == {}
        assert redis.delete("hash") == 0

    def test_delete_set(self) -> None:
        redis = FakeRedis()
        redis.sadd("myset", "member")
        assert redis.delete("myset") == 1
        assert redis.scard("myset") == 0
        assert redis.delete("myset") == 0

    def test_delete_missing_key(self) -> None:
        redis = FakeRedis()
        assert redis.delete("missing") == 0

    def test_expire_existing_key(self) -> None:
        redis = FakeRedis()
        redis.set("key", "value")
        assert redis.expire("key", 60) is True

    def test_expire_hash_key(self) -> None:
        redis = FakeRedis()
        redis.hset("hash", {"a": "1"})
        assert redis.expire("hash", 60) is True

    def test_expire_set_key(self) -> None:
        redis = FakeRedis()
        redis.sadd("myset", "member")
        assert redis.expire("myset", 60) is True

    def test_expire_missing_key(self) -> None:
        redis = FakeRedis()
        assert redis.expire("missing", 60) is False

    def test_close_clears_data(self) -> None:
        redis = FakeRedis()
        redis.set("key", "value")
        redis.hset("hash", {"a": "1"})
        redis.sadd("set", "member")
        redis.close()
        assert redis.get("key") is None
        assert redis.hgetall("hash") == {}
        assert redis.scard("set") == 0

    def test_calls_tracks_method_calls(self) -> None:
        redis = FakeRedis()
        redis.ping()
        redis.scard("key")
        assert len(redis.calls) == 2
        assert redis.calls[0] == MethodCall("ping", ())
        assert redis.calls[1] == MethodCall("scard", ("key",))

    def test_assert_only_called_passes(self) -> None:
        redis = FakeRedis()
        redis.ping()
        redis.scard("key")
        redis.assert_only_called({"ping", "scard"})  # Should not raise

    def test_assert_only_called_fails(self) -> None:
        redis = FakeRedis()
        redis.ping()
        redis.hset("key", {"a": "1"})
        with pytest.raises(AssertionError, match="Unexpected methods called"):
            redis.assert_only_called({"ping"})

    def test_get_calls_filters_by_method(self) -> None:
        redis = FakeRedis()
        redis.ping()
        redis.scard("key1")
        redis.scard("key2")
        scard_calls = redis.get_calls("scard")
        assert len(scard_calls) == 2
        assert scard_calls[0] == MethodCall("scard", ("key1",))
        assert scard_calls[1] == MethodCall("scard", ("key2",))


class TestFakeRedisBytesClient:
    def test_ping_returns_true(self) -> None:
        client = FakeRedisBytesClient()
        assert client.ping() is True

    def test_close_sets_closed_flag(self) -> None:
        client = FakeRedisBytesClient()
        assert client.closed is False
        client.close()
        assert client.closed is True


class TestFakeRedisPublishError:
    def test_publish_raises_os_error(self) -> None:
        redis = FakeRedisPublishError()
        with pytest.raises(OSError, match="simulated publish failure"):
            redis.publish("channel", "message")
        redis.assert_only_called({"publish"})


class TestFakeRedisHsetError:
    def test_hset_raises_runtime_error(self) -> None:
        redis = FakeRedisHsetError()
        with pytest.raises(RuntimeError, match="simulated hset failure"):
            redis.hset("key", {"field": "value"})
        redis.assert_only_called({"hset"})


class TestFakeRedisHsetRedisError:
    def test_hset_raises_redis_error(self) -> None:
        redis = FakeRedisHsetRedisError()
        from platform_workers.redis import _load_redis_error_class

        error_cls = _load_redis_error_class()
        with pytest.raises(error_cls, match="simulated Redis hset failure"):
            redis.hset("key", {"field": "value"})
        redis.assert_only_called({"hset"})


class TestFakeRedisConditionalHsetError:
    """Tests for FakeRedisConditionalHsetError (conditional RuntimeError on hset)."""

    def test_hset_succeeds_when_status_does_not_match(self) -> None:
        from platform_workers.testing import FakeRedisConditionalHsetError

        redis = FakeRedisConditionalHsetError(fail_on_status="failed")
        result = redis.hset("job:123", {"status": "running", "name": "test"})
        assert result == 2
        assert redis.hgetall("job:123") == {"status": "running", "name": "test"}

    def test_hset_raises_runtime_error_when_status_matches(self) -> None:
        from platform_workers.testing import FakeRedisConditionalHsetError

        redis = FakeRedisConditionalHsetError(fail_on_status="failed")
        with pytest.raises(RuntimeError, match="simulated hset failure on status=failed"):
            redis.hset("job:123", {"status": "failed"})


class TestFakeRedisConditionalHsetRedisError:
    """Tests for FakeRedisConditionalHsetRedisError (conditional RedisError on hset)."""

    def test_hset_succeeds_when_status_does_not_match(self) -> None:
        from platform_workers.testing import FakeRedisConditionalHsetRedisError

        redis = FakeRedisConditionalHsetRedisError(fail_on_status="failed")
        result = redis.hset("job:456", {"status": "running", "name": "test"})
        assert result == 2
        assert redis.hgetall("job:456") == {"status": "running", "name": "test"}

    def test_hset_raises_redis_error_when_status_matches(self) -> None:
        from platform_workers.redis import _load_redis_error_class
        from platform_workers.testing import FakeRedisConditionalHsetRedisError

        redis = FakeRedisConditionalHsetRedisError(fail_on_status="failed")
        error_cls = _load_redis_error_class()
        with pytest.raises(error_cls, match="simulated Redis hset failure on status=failed"):
            redis.hset("job:456", {"status": "failed"})


class TestFakeRedisClient:
    """Tests for FakeRedisClient (internal _RedisStrClient protocol)."""

    def test_ping_returns_true(self) -> None:
        client = FakeRedisClient()
        assert client.ping() is True

    def test_set_and_get(self) -> None:
        client = FakeRedisClient()
        assert client.set("name", "value") is True
        assert client.get("name") == "value"
        assert client.get("missing") is None

    def test_delete_variadic_from_string(self) -> None:
        client = FakeRedisClient()
        client.set("key1", "val1")
        client.set("key2", "val2")
        assert client.delete("key1", "key2") == 2
        assert client.get("key1") is None
        assert client.get("key2") is None

    def test_delete_from_hash(self) -> None:
        client = FakeRedisClient()
        client.hset("hash", {"a": "1"})
        assert client.delete("hash") == 1
        assert client.hgetall("hash") == {}

    def test_delete_from_set(self) -> None:
        client = FakeRedisClient()
        client.sadd("myset", "member")
        assert client.delete("myset") == 1
        assert client.scard("myset") == 0

    def test_delete_missing(self) -> None:
        client = FakeRedisClient()
        assert client.delete("missing") == 0

    def test_expire(self) -> None:
        client = FakeRedisClient()
        client.set("name", "value")
        assert client.expire("name", 60) is True
        assert client.expire("missing", 60) is False

    def test_hset_hget_hgetall(self) -> None:
        client = FakeRedisClient()
        assert client.hset("hash", {"a": "1", "b": "2"}) == 2
        assert client.hget("hash", "a") == "1"
        assert client.hget("hash", "missing") is None
        assert client.hgetall("hash") == {"a": "1", "b": "2"}
        assert client.hgetall("missing") == {}

    def test_publish(self) -> None:
        client = FakeRedisClient()
        assert client.publish("channel", "msg") == 1

    def test_set_operations(self) -> None:
        client = FakeRedisClient()
        assert client.scard("myset") == 0
        assert client.sadd("myset", "a") == 1
        assert client.sadd("myset", "a") == 0  # Already exists
        assert client.sismember("myset", "a") is True
        assert client.sismember("myset", "b") is False
        assert client.scard("myset") == 1

    def test_close(self) -> None:
        client = FakeRedisClient()
        client.close()  # Should not raise

    def test_assert_only_called_passes(self) -> None:
        client = FakeRedisClient()
        client.ping()
        client.scard("key")
        client.assert_only_called({"ping", "scard"})  # Should not raise

    def test_assert_only_called_fails(self) -> None:
        client = FakeRedisClient()
        client.ping()
        client.hset("key", {"a": "1"})
        with pytest.raises(AssertionError, match="Unexpected methods called"):
            client.assert_only_called({"ping"})


class TestFakeJob:
    def test_default_job_id(self) -> None:
        job = FakeJob()
        assert job.get_id() == "test-job-id"

    def test_custom_job_id(self) -> None:
        job = FakeJob("custom-id")
        assert job.get_id() == "custom-id"


class TestFakeFetchedJob:
    def test_default_values(self) -> None:
        from platform_workers.testing import FakeFetchedJob

        job = FakeFetchedJob()
        assert job.get_id() == "test-job-id"
        assert job.get_status() == "finished"
        assert job.return_value() is None

    def test_custom_values(self) -> None:
        from platform_workers.testing import FakeFetchedJob

        job = FakeFetchedJob(job_id="custom-123", status="queued", result={"key": "value"})
        assert job.get_id() == "custom-123"
        assert job.get_status() == "queued"
        assert job.return_value() == {"key": "value"}

    def test_all_status_types(self) -> None:
        from platform_workers.testing import FakeFetchedJob

        for status in ["queued", "started", "finished", "failed"]:
            job = FakeFetchedJob(status=status)
            assert job.get_status() == status


class TestFakeRetry:
    def test_stores_config(self) -> None:
        retry = FakeRetry(max=3, interval=[1, 2, 4])
        assert retry.max_retries == 3
        assert retry.intervals == [1, 2, 4]

    def test_empty_interval(self) -> None:
        retry = FakeRetry(max=0, interval=[])
        assert retry.max_retries == 0
        assert retry.intervals == []


class TestFakeQueue:
    def test_enqueue_returns_job(self) -> None:
        queue = FakeQueue()
        job = queue.enqueue("my.func", "arg1", "arg2")
        assert job.get_id() == "test-job-id"

    def test_enqueue_tracks_jobs(self) -> None:
        queue = FakeQueue()
        queue.enqueue(
            "my.func",
            "arg1",
            42,
            job_timeout=60,
            result_ttl=3600,
            failure_ttl=86400,
            description="test job",
        )
        assert len(queue.jobs) == 1
        job = queue.jobs[0]
        assert job == EnqueuedJob(
            func="my.func",
            args=("arg1", 42),
            job_timeout=60,
            result_ttl=3600,
            failure_ttl=86400,
            description="test job",
        )

    def test_enqueue_with_custom_job_id(self) -> None:
        queue = FakeQueue("custom-job")
        job = queue.enqueue("func")
        assert job.get_id() == "custom-job"


class TestFakeLogger:
    def test_debug_records_message(self) -> None:
        logger = FakeLogger()
        logger.debug("msg %s", "arg", extra={"key": "value"})
        assert len(logger.records) == 1
        assert logger.records[0] == LogRecord("debug", "msg %s", ("arg",), {"key": "value"})

    def test_info_records_message(self) -> None:
        logger = FakeLogger()
        logger.info("info message")
        assert len(logger.records) == 1
        assert logger.records[0].level == "info"
        assert logger.records[0].msg == "info message"

    def test_warning_records_message(self) -> None:
        logger = FakeLogger()
        logger.warning("warning message")
        assert len(logger.records) == 1
        assert logger.records[0].level == "warning"

    def test_error_records_message(self) -> None:
        logger = FakeLogger()
        logger.error("error message")
        assert len(logger.records) == 1
        assert logger.records[0].level == "error"

    def test_multiple_records(self) -> None:
        logger = FakeLogger()
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        assert len(logger.records) == 4
        levels = [r.level for r in logger.records]
        assert levels == ["debug", "info", "warning", "error"]


class TestFakePubSub:
    """Tests for FakePubSub async Redis client."""

    def test_subscribe_tracks_channels(self) -> None:
        import asyncio

        from platform_workers.testing import FakePubSub

        async def run() -> list[str]:
            pubsub = FakePubSub()
            await pubsub.subscribe("channel1", "channel2")
            return pubsub.subscriptions

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == ["channel1", "channel2"]

    def test_get_message_returns_none_when_empty(self) -> None:
        import asyncio

        from platform_workers.redis import PubSubMessage
        from platform_workers.testing import FakePubSub

        async def run() -> PubSubMessage | None:
            pubsub = FakePubSub()
            return await pubsub.get_message()

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is None

    def test_inject_message_and_get_message(self) -> None:
        import asyncio

        from platform_workers.redis import PubSubMessage
        from platform_workers.testing import FakePubSub

        async def run() -> PubSubMessage | None:
            pubsub = FakePubSub()
            pubsub.inject_message("test-channel", "test-data")
            return await pubsub.get_message()

        msg = asyncio.get_event_loop().run_until_complete(run())
        assert type(msg) is dict
        assert msg["channel"] == "test-channel"
        assert msg["data"] == "test-data"
        assert msg["type"] == "message"

    def test_close_sets_closed_flag(self) -> None:
        import asyncio

        from platform_workers.testing import FakePubSub

        async def run() -> bool:
            pubsub = FakePubSub()
            assert pubsub._closed is False
            await pubsub.close()
            return pubsub._closed

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is True


class TestFakeAsyncRedis:
    """Tests for FakeAsyncRedis client."""

    def test_pubsub_returns_fake_pubsub(self) -> None:
        from platform_workers.testing import FakeAsyncRedis, FakePubSub

        client = FakeAsyncRedis()
        pubsub = client.pubsub()
        assert type(pubsub) is FakePubSub


class TestFactoryFunctions:
    """Tests for factory helper functions."""

    def test_fake_kv_store_factory(self) -> None:
        from platform_workers.testing import FakeRedis, fake_kv_store_factory

        result = fake_kv_store_factory("redis://test")
        assert type(result) is FakeRedis

    def test_fake_rq_connection_factory(self) -> None:
        from platform_workers.testing import FakeRedisBytesClient, fake_rq_connection_factory

        result = fake_rq_connection_factory("redis://test")
        assert type(result) is FakeRedisBytesClient

    def test_fake_rq_queue_factory(self) -> None:
        from platform_workers.testing import FakeQueue, FakeRedisBytesClient, fake_rq_queue_factory

        conn = FakeRedisBytesClient()
        result = fake_rq_queue_factory("test-queue", conn)
        assert type(result) is FakeQueue

    def test_fake_rq_retry_factory(self) -> None:
        from platform_workers.testing import FakeRetry, fake_rq_retry_factory

        result = fake_rq_retry_factory(max_retries=3, intervals=[1, 2, 3])
        assert type(result) is FakeRetry
        assert result.max_retries == 3
        assert result.intervals == [1, 2, 3]


class TestNamedTuples:
    def test_published_fields(self) -> None:
        p = Published("ch", "msg")
        assert p.channel == "ch"
        assert p.payload == "msg"

    def test_method_call_fields(self) -> None:
        m = MethodCall("ping", ())
        assert m.method == "ping"
        assert m.args == ()
        m2 = MethodCall("scard", ("key",))
        assert m2.method == "scard"
        assert m2.args == ("key",)

    def test_enqueued_job_fields(self) -> None:
        e = EnqueuedJob("func", ("a",), 60, 3600, 86400, "desc")
        assert e.func == "func"
        assert e.args == ("a",)
        assert e.job_timeout == 60
        assert e.result_ttl == 3600
        assert e.failure_ttl == 86400
        assert e.description == "desc"

    def test_log_record_fields(self) -> None:
        r = LogRecord("info", "msg", ("a",), {"k": "v"})
        assert r.level == "info"
        assert r.msg == "msg"
        assert r.args == ("a",)
        assert r.extra == {"k": "v"}
