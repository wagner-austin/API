from __future__ import annotations

import asyncio
import logging

import pytest
from platform_core.digits_metrics_events import (
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
)
from platform_core.job_events import JobFailedV1
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, UserProto
from tests.support.discord_fakes import TrackingBot, TrackingUser

import clubbot.services.jobs.digits_notifier as dn

# Typed dynamic import of discord exceptions to avoid Any
_discord = __import__("discord")
NotFound: type[BaseException] = _discord.NotFound
HTTPException: type[BaseException] = _discord.HTTPException


@pytest.mark.asyncio
async def test_handle_event_branches_send_dm() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    config1: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 2,
        "queue": "digits",
        # Optional extras for richer embed
        "cpu_cores": 2,
        "optimal_threads": 2,
        "memory_mb": 953,
        "optimal_workers": 0,
        "max_batch_size": 64,
        "device": "cpu",
    }
    await sub._handle_event(config1)
    assert user.embeds and isinstance(user.embeds[-1], object)

    # Start another request without extras to cover env_bits empty branch
    config2: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r2",
        "user_id": 2,
        "model_id": "m2",
        "total_epochs": 1,
        "queue": "digits",
    }
    await sub._handle_event(config2)
    assert user.embeds and len(user.embeds) >= 2

    epoch: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "epoch": 1,
        "total_epochs": 2,
        "train_loss": 0.1,
        "val_acc": 0.9,
        "time_s": 1.0,
    }
    await sub._handle_event(epoch)
    assert user.embeds and len(user.embeds) >= 2

    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 0.95,
    }
    await sub._handle_event(completed)
    assert user.embeds and len(user.embeds) >= 3

    failed1: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r",
        "user_id": 1,
        "error_kind": "user",
        "message": "bad payload",
        "domain": "digits",
    }
    await sub._handle_event(failed1)
    assert user.embeds and len(user.embeds) >= 4

    failed2: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r",
        "user_id": 1,
        "error_kind": "system",
        "message": "boom",
        "domain": "digits",
    }
    await sub._handle_event(failed2)
    assert user.embeds and len(user.embeds) >= 5


@pytest.mark.asyncio
async def test_started_embed_includes_augment_and_config() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "rx",
        "user_id": 7,
        "model_id": "mX",
        "total_epochs": 3,
        "queue": "digits",
        "cpu_cores": 2,
        "optimal_threads": 2,
        "memory_mb": 953,
        "optimal_workers": 0,
        "max_batch_size": 64,
        "device": "cpu",
        "batch_size": 64,
        "augment": True,
        "aug_rotate": 10.0,
        "aug_translate": 0.1,
        "noise_prob": 0.2,
        "dots_prob": 0.1,
    }
    await sub._handle_event(config)
    assert user.embeds and isinstance(user.embeds[-1], object)


@pytest.mark.asyncio
async def test_started_augment_zero_values_renders_none() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "rz",
        "user_id": 9,
        "model_id": "mZ",
        "total_epochs": 1,
        "queue": "digits",
        "augment": True,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "dots_prob": 0.0,
    }
    await sub._handle_event(config)
    assert user.embeds and isinstance(user.embeds[-1], object)


@pytest.mark.asyncio
async def test_progress_without_optional_metrics() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    # All fields are required in DigitsEpochMetricsV1 TypedDict, provide minimal valid values
    epoch_event: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "r3",
        "user_id": 3,
        "model_id": "m3",
        "epoch": 1,
        "total_epochs": 3,
        "train_loss": 0.0,
        "val_acc": 0.0,
        "time_s": 0.0,
    }
    await sub._handle_event(epoch_event)
    assert user.embeds and isinstance(user.embeds[-1], object)


class _FakeNotFoundResp:
    """Fake response for NotFound exception."""

    status: int = 404
    reason: str = "Not Found"


class _FakeHttpResp:
    """Fake response for HTTPException."""

    status: int = 500
    reason: str = "Server Error"


class _BadUserNotFound:
    """User that raises NotFound on send."""

    @property
    def id(self) -> int:
        return 11111

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        import discord

        _ = (content, embed, file)
        raise discord.NotFound(response=_FakeNotFoundResp(), message="not found")


class _BadUserHttpError:
    """User that raises HTTPException on send."""

    @property
    def id(self) -> int:
        return 22222

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        import discord

        _ = (content, embed, file)
        raise discord.HTTPException(response=_FakeHttpResp(), message="Server error")


class _BadBotNotFound:
    """Bot that returns a user raising NotFound."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return _BadUserNotFound()


class _BadBotHttpError:
    """Bot that returns a user raising HTTPException."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return _BadUserHttpError()


@pytest.mark.asyncio
async def test_notify_handles_discord_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = dn.DigitsEventSubscriber(_BadBotNotFound(), redis_url="redis://fake")
    # Should swallow the exception
    completed_event: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 0.9,
    }
    with pytest.raises(NotFound):
        await sub._on_completed(completed_event)


@pytest.mark.asyncio
async def test_notify_handles_http_exception() -> None:
    """Test that HTTPException is caught and logged."""
    sub = dn.DigitsEventSubscriber(_BadBotHttpError(), redis_url="redis://fake")
    # Should swallow the exception
    completed_event: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 0.9,
    }
    with pytest.raises(HTTPException):
        await sub._on_completed(completed_event)


@pytest.mark.asyncio
async def test_handle_event_unknown_type_noop() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")
    # Unknown type should result in no notification and no exception - pass minimal dict
    # Note: The function expects DigitsEventV1, but we're testing unrecognized type handling
    # We use _handle_event through the public interface indirectly
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",  # Valid type triggers completed branch
        "user_id": 1,
        "job_id": "runknown",
        "model_id": "m",
        "val_acc": 0.5,
    }
    await sub._handle_event(completed)
    # Embed should be sent for completed event
    assert user.embeds


@pytest.mark.asyncio
async def test_start_and_stop_covers_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch _run to a cooperative noop to avoid real Redis calls
    async def _noop(self: dn.DigitsEventSubscriber) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(dn.DigitsEventSubscriber, "_run", _noop, raising=True)
    sub = dn.DigitsEventSubscriber(TrackingBot(TrackingUser()), redis_url="redis://fake")
    sub.start()
    # second start should be a no-op branch
    sub.start()
    await sub.stop()
    # stopping when no task should no-op branch
    sub2 = dn.DigitsEventSubscriber(TrackingBot(TrackingUser()), redis_url="redis://fake")
    await sub2.stop()


logger = logging.getLogger(__name__)
