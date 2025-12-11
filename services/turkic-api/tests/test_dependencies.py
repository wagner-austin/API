"""Tests for FastAPI dependencies in turkic-api."""

from __future__ import annotations

from typing import ClassVar

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from turkic_api import _test_hooks
from turkic_api.api.dependencies import get_redis, get_settings


class TrackingFakeRedis(FakeRedis):
    """FakeRedis that tracks whether it was created."""

    instances: ClassVar[list[FakeRedis]] = []

    def __init__(self) -> None:
        super().__init__()
        TrackingFakeRedis.instances.append(self)


def test_get_redis_closes_client() -> None:
    """Test that get_redis generator closes the client on teardown."""
    # Reset tracking
    TrackingFakeRedis.instances = []

    def _fake_redis_factory(url: str) -> RedisStrProto:
        return TrackingFakeRedis()

    _test_hooks.redis_factory = _fake_redis_factory

    gen = get_redis(get_settings())
    client = next(gen)
    # Verify the client is the one we created via our tracking class
    assert len(TrackingFakeRedis.instances) == 1
    assert client is TrackingFakeRedis.instances[0]

    # Trigger generator cleanup
    with pytest.raises(StopIteration):
        gen.send(None)

    assert TrackingFakeRedis.instances
    assert TrackingFakeRedis.instances[0].closed is True
    TrackingFakeRedis.instances[0].assert_only_called({"close"})


def test_get_redis_uses_settings_url() -> None:
    """Test that get_redis passes the settings URL to the factory."""
    captured_urls: list[str] = []

    def _capturing_redis_factory(url: str) -> RedisStrProto:
        captured_urls.append(url)
        return FakeRedis()

    _test_hooks.redis_factory = _capturing_redis_factory

    settings = get_settings()
    gen = get_redis(settings)
    next(gen)

    assert len(captured_urls) == 1
    assert captured_urls[0] == settings["redis_url"]

    # Trigger cleanup
    with pytest.raises(StopIteration):
        gen.send(None)
    FakeRedis().assert_only_called(set())
