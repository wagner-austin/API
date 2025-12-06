from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.logging import get_logger
from platform_core.turkic_jobs import turkic_job_key
from platform_workers.testing import FakeQueue, FakeRedis

from turkic_api.api.config import Settings
from turkic_api.api.main import create_app
from turkic_api.api.routes.jobs import _to_hash_redis
from turkic_api.api.types import LoggerProtocol, RQJobLike, RQRetryLike, _EnqCallable
from turkic_api.core.models import UnknownJson


def test_to_hash_redis_wraps_real_redis() -> None:
    r = FakeRedis()
    adapter = _to_hash_redis(r)
    key = turkic_job_key("x")
    adapter.hset(key, {"a": "1"})
    assert adapter.hgetall(key)["a"] == "1"


def test_create_app_overrides_settings_and_logger() -> None:
    def _settings() -> Settings:
        return Settings(
            redis_url="redis://localhost:6379/0",
            data_dir="/tmp",
            environment="test",
            data_bank_api_url="",
            data_bank_api_key="",
        )

    def _logger() -> LoggerProtocol:
        return get_logger("override")

    app = create_app(settings_provider=_settings, logger_provider=_logger)
    if app is None:
        pytest.fail("expected app")


def test_create_app_applies_all_overrides() -> None:
    def _settings() -> Settings:
        return Settings(
            redis_url="redis://localhost:6379/0",
            data_dir="/tmp",
            environment="test",
            data_bank_api_url="",
            data_bank_api_key="",
        )

    def _logger() -> LoggerProtocol:
        return get_logger("override")

    def _redis_provider(settings: Settings) -> FakeRedis:
        return FakeRedis()

    def _queue_provider() -> FakeQueue:
        return FakeQueue()

    app = create_app(
        redis_provider=_redis_provider,
        queue_provider=_queue_provider,
        settings_provider=_settings,
        logger_provider=_logger,
    )
    if app is None:
        pytest.fail("expected app")


def test_create_app_defaults_paths() -> None:
    app = create_app()
    if app is None:
        pytest.fail("expected app")


def test_create_app_type_checking_else_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import turkic_api.api.main as m

    monkeypatch.setattr(m, "TYPE_CHECKING", True, raising=False)
    app = m.create_app()
    if app is None:
        pytest.fail("expected app")
    monkeypatch.setattr(m, "TYPE_CHECKING", False, raising=False)


def test_to_hash_redis_identity() -> None:
    r = FakeRedis()
    r.hset(turkic_job_key("x"), {"x": "1"})
    adapter = _to_hash_redis(r)
    assert adapter.hgetall(turkic_job_key("x"))["x"] == "1"


class _ProviderQueue:
    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: UnknownJson,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        class _Job(RQJobLike):
            def get_id(self) -> str:
                return "test-job-id"

        return _Job()


def test_provider_context_generator_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import turkic_api.api.main as m
    import turkic_api.api.provider_context as pc

    ctx = pc.provider_context
    prev = (
        ctx.settings_provider,
        ctx.redis_provider,
        ctx.queue_provider,
        ctx.logger_provider,
    )

    settings_obj = Settings(
        redis_url="redis://localhost:6379/0",
        data_dir="/tmp/data",
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="k",
    )

    def settings_provider() -> Settings:
        return settings_obj

    def redis_provider(_settings: Settings) -> Generator[m.RedisCombinedProtocol, None, None]:
        yield FakeRedis()

    provided_logger = get_logger("override-provider")

    def logger_provider() -> LoggerProtocol:
        return provided_logger

    ctx.settings_provider = settings_provider
    ctx.redis_provider = redis_provider
    ctx.queue_provider = lambda: _ProviderQueue()
    ctx.logger_provider = logger_provider

    settings = pc.get_settings_from_context()
    redis_gen = pc.get_redis_from_context(settings)
    redis_client = next(redis_gen)
    with pytest.raises(StopIteration):
        next(redis_gen)
    queue_obj = pc.get_queue_from_context(settings)
    logger_obj = pc.get_logger_from_context()

    ctx.settings_provider, ctx.redis_provider, ctx.queue_provider, ctx.logger_provider = prev

    assert type(redis_client).__name__ == "FakeRedis"
    assert type(queue_obj).__name__ == "_ProviderQueue"
    assert logger_obj is provided_logger


def test_get_redis_context_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import turkic_api.api.provider_context as pc

    ctx = pc.provider_context
    prev = (
        ctx.settings_provider,
        ctx.redis_provider,
        ctx.queue_provider,
        ctx.logger_provider,
    )

    pc.provider_context.settings_provider = None
    pc.provider_context.redis_provider = None
    pc.provider_context.queue_provider = None
    pc.provider_context.logger_provider = None

    try:
        st = pc.get_settings_from_context()
        assert st["redis_url"].startswith("redis://") or st["redis_url"] == ""
    finally:
        (
            ctx.settings_provider,
            ctx.redis_provider,
            ctx.queue_provider,
            ctx.logger_provider,
        ) = prev
