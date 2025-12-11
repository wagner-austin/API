"""Pytest configuration and fixtures for transcript-api tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from transcript_api import _test_hooks
from transcript_api.dependencies import provider_context


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore all hooks after each test."""
    # Save original hooks
    original_platform_get_env = platform_hooks.get_env
    original_redis_factory = _test_hooks.redis_factory
    original_os_stat = _test_hooks.os_stat
    original_os_path_getsize = _test_hooks.os_path_getsize
    original_os_remove = _test_hooks.os_remove
    original_mkdtemp = _test_hooks.mkdtemp
    original_subprocess_run = _test_hooks.subprocess_run
    original_test_runner = _test_hooks.test_runner
    original_openai_client_factory = _test_hooks.openai_client_factory
    original_yt_api_factory = _test_hooks.yt_api_factory
    original_yt_exceptions_factory = _test_hooks.yt_exceptions_factory
    original_yt_dlp_factory = _test_hooks.yt_dlp_factory
    original_audio_chunker_factory = _test_hooks.audio_chunker_factory
    original_ffmpeg_available = _test_hooks.ffmpeg_available
    original_stt_client_builder = _test_hooks.stt_client_builder
    original_probe_client_builder = _test_hooks.probe_client_builder
    original_stt_provider_factory = _test_hooks.stt_provider_factory

    # Save provider context
    original_redis_provider = provider_context.redis_provider
    original_queue_provider = provider_context.queue_provider
    original_logger_provider = provider_context.logger_provider

    yield

    # Restore all hooks
    platform_hooks.get_env = original_platform_get_env
    _test_hooks.redis_factory = original_redis_factory
    _test_hooks.os_stat = original_os_stat
    _test_hooks.os_path_getsize = original_os_path_getsize
    _test_hooks.os_remove = original_os_remove
    _test_hooks.mkdtemp = original_mkdtemp
    _test_hooks.subprocess_run = original_subprocess_run
    _test_hooks.test_runner = original_test_runner
    _test_hooks.openai_client_factory = original_openai_client_factory
    _test_hooks.yt_api_factory = original_yt_api_factory
    _test_hooks.yt_exceptions_factory = original_yt_exceptions_factory
    _test_hooks.yt_dlp_factory = original_yt_dlp_factory
    _test_hooks.audio_chunker_factory = original_audio_chunker_factory
    _test_hooks.ffmpeg_available = original_ffmpeg_available
    _test_hooks.stt_client_builder = original_stt_client_builder
    _test_hooks.probe_client_builder = original_probe_client_builder
    _test_hooks.stt_provider_factory = original_stt_provider_factory

    # Restore provider context
    provider_context.redis_provider = original_redis_provider
    provider_context.queue_provider = original_queue_provider
    provider_context.logger_provider = original_logger_provider


@pytest.fixture(autouse=True)
def _default_test_env() -> None:
    """Provide default test environment with REDIS_URL and FakeRedis."""
    env = make_fake_env({"REDIS_URL": "redis://test-redis"})
    platform_hooks.get_env = env

    def _fake_redis(url: str) -> RedisStrProto:
        r = FakeRedis()
        r.sadd("rq:workers", "worker-1")
        return r

    _test_hooks.redis_factory = _fake_redis
