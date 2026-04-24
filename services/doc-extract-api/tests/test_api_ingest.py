"""Tests for doc_extract_api.api.ingest."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Response
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from doc_extract_api import _test_hooks
from doc_extract_api.api.ingest import _create_job, _get_job


def _setup_redis(redis: FakeRedis) -> None:
    """Wire a FakeRedis into the hook."""

    def _factory(url: str) -> RedisStrProto:
        _ = url
        return redis

    factory: Callable[[str], RedisStrProto] = _factory
    _test_hooks.redis_factory = factory


class TestCreateJob:
    def test_creates_queued_job(self) -> None:
        redis = FakeRedis()
        _setup_redis(redis)

        result = _create_job(
            title="Test Doc",
            file_path="/tmp/test.pdf",
            category="general",
            source="https://example.com",
        )
        assert result["status"] == "queued"
        assert result["title"] == "Test Doc"
        assert result["category"] == "general"
        assert len(str(result["job_id"])) == 36  # UUID length
        redis.assert_only_called({"hset"})

    def test_invalid_category(self) -> None:
        redis = FakeRedis()
        _setup_redis(redis)

        import pytest

        with pytest.raises(ValueError, match="Invalid category"):
            _create_job(
                title="Doc",
                file_path="/tmp/test.pdf",
                category="nonexistent",
            )
        redis.assert_only_called(set())

    def test_empty_title(self) -> None:
        redis = FakeRedis()
        _setup_redis(redis)

        import pytest

        with pytest.raises(ValueError, match="title"):
            _create_job(title="", file_path="/tmp/test.pdf", category="general")
        redis.assert_only_called(set())


class TestGetJob:
    def test_get_existing_job(self) -> None:
        redis = FakeRedis()
        _setup_redis(redis)

        created = _create_job(
            title="Test Doc",
            file_path="/tmp/test.pdf",
            category="budget",
        )
        job_id = str(created["job_id"])

        response = Response()
        result = _get_job(job_id, response)
        assert result["title"] == "Test Doc"
        assert result["status"] == "queued"
        redis.assert_only_called({"hset", "hgetall"})

    def test_get_missing_job(self) -> None:
        redis = FakeRedis()
        _setup_redis(redis)

        response = Response()
        result = _get_job("nonexistent-id", response)
        assert "error" in result
        assert response.status_code == 404
        redis.assert_only_called({"hgetall"})
