"""Tests for platform_workers.testing module."""

from __future__ import annotations

from platform_workers.testing import (
    EnqueuedJob,
    FakeJob,
    FakeLogger,
    FakeQueue,
    FakeRedis,
    FakeRedisBytesClient,
    FakeRetry,
    LogRecord,
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


class TestFakeRedisBytesClient:
    def test_ping_returns_true(self) -> None:
        client = FakeRedisBytesClient()
        assert client.ping() is True

    def test_close_is_noop(self) -> None:
        client = FakeRedisBytesClient()
        client.close()  # Should not raise


class TestFakeJob:
    def test_default_job_id(self) -> None:
        job = FakeJob()
        assert job.get_id() == "test-job-id"

    def test_custom_job_id(self) -> None:
        job = FakeJob("custom-id")
        assert job.get_id() == "custom-id"


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


class TestNamedTuples:
    def test_published_fields(self) -> None:
        p = Published("ch", "msg")
        assert p.channel == "ch"
        assert p.payload == "msg"

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
