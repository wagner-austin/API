"""Tests for DigitsEventSubscriber thin wrapper.

Event routing and embed generation are tested in platform_discord.
These tests verify the subscriber wrapper behavior:
- Hook mechanism for dependency injection
- Discord exception propagation
- Start/stop lifecycle
"""

from __future__ import annotations

import asyncio
from typing import NoReturn

import discord
import pytest
from platform_core.digits_metrics_events import DigitsCompletedMetricsV1, DigitsConfigV1
from platform_core.job_events import JobFailedV1
from platform_discord.embed_helpers import EmbedProto
from platform_discord.handwriting import DigitsEventV1, DigitsRuntime, RequestAction
from platform_discord.protocols import FileProto
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import TrackingBot, TrackingUser

import clubbot.services.jobs.digits_notifier as dn

# Typed dynamic import to avoid Any in exception types
_discord = __import__("discord")
Forbidden: type[BaseException] = _discord.Forbidden
HTTPException: type[BaseException] = _discord.HTTPException
NotFound: type[BaseException] = _discord.NotFound


class _Response:
    """Fake HTTP response for discord exceptions."""

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason


class _ForbiddenUser:
    """User that raises Forbidden on send."""

    @property
    def id(self) -> int:
        return 99999

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> NoReturn:
        _ = (content, embed, file)
        raise discord.Forbidden(
            response=_Response(status=403, reason="Forbidden"),
            message="Cannot send DM to user",
        )


class _ForbiddenBot:
    """Bot that returns a user raising Forbidden."""

    async def fetch_user(self, user_id: int, /) -> _ForbiddenUser:
        _ = user_id
        return _ForbiddenUser()


class _HTTPExceptionUser:
    """User that raises HTTPException on send."""

    @property
    def id(self) -> int:
        return 99998

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> NoReturn:
        _ = (content, embed, file)
        raise discord.HTTPException(
            response=_Response(status=429, reason="Too Many Requests"),
            message="Rate limited",
        )


class _HTTPExceptionBot:
    """Bot that returns a user raising HTTPException."""

    async def fetch_user(self, user_id: int, /) -> _HTTPExceptionUser:
        _ = user_id
        return _HTTPExceptionUser()


class _NotFoundUser:
    """User that raises NotFound on send."""

    @property
    def id(self) -> int:
        return 99997

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> NoReturn:
        _ = (content, embed, file)
        raise discord.NotFound(
            response=_Response(status=404, reason="Not Found"),
            message="User not found",
        )


class _NotFoundBot:
    """Bot that returns a user raising NotFound."""

    async def fetch_user(self, user_id: int, /) -> _NotFoundUser:
        _ = user_id
        return _NotFoundUser()


class _FakeMessageSource:
    """Fake message source for lifecycle tests."""

    __slots__ = ("closed",)

    def __init__(self) -> None:
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        _ = channel

    async def get(self) -> str | None:
        await asyncio.sleep(0)
        return None

    async def close(self) -> None:
        self.closed = True


def _make_source(url: str) -> MessageSource:
    """Factory for fake message sources."""
    _ = url
    return _FakeMessageSource()


@pytest.mark.asyncio
async def test_handle_event_sends_dm() -> None:
    """Verify subscriber sends DM when handler returns action."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot=bot, redis_url="redis://fake")

    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "mnist",
        "total_epochs": 5,
        "queue": "digits",
    }
    await sub._handle_event(config)
    assert len(user.embeds) == 1


@pytest.mark.asyncio
async def test_hook_override_returns_none() -> None:
    """Verify hook can be overridden to suppress notifications."""
    user = TrackingUser()
    bot = TrackingBot(user)

    # Save original and override
    original = dn.handle_digits_event

    def _fake_handler(runtime: DigitsRuntime, event: DigitsEventV1) -> RequestAction | None:
        _ = (runtime, event)
        return None

    dn.handle_digits_event = _fake_handler
    try:
        sub = dn.DigitsEventSubscriber(bot=bot, redis_url="redis://fake")
        config: DigitsConfigV1 = {
            "type": "digits.metrics.config.v1",
            "job_id": "r_hook",
            "user_id": 2,
            "model_id": "m",
            "total_epochs": 1,
            "queue": "digits",
        }
        await sub._handle_event(config)
        # No embed sent because hook returned None
        assert len(user.embeds) == 0
    finally:
        dn.handle_digits_event = original


@pytest.mark.asyncio
async def test_maybe_notify_skips_none_embed() -> None:
    """Verify _maybe_notify skips when embed is None."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(bot=bot, redis_url="redis://fake")

    action: RequestAction = {"user_id": 1, "request_id": "r", "embed": None}
    await sub._maybe_notify(action)
    assert len(user.embeds) == 0


@pytest.mark.asyncio
async def test_notify_propagates_forbidden() -> None:
    """Verify Forbidden exception propagates (no best-effort)."""
    sub = dn.DigitsEventSubscriber(bot=_ForbiddenBot(), redis_url="redis://fake")
    event: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r_forbidden",
        "user_id": 999,
        "error_kind": "system",
        "message": "Test error",
        "domain": "digits",
    }
    with pytest.raises(Forbidden):
        await sub._handle_event(event)


@pytest.mark.asyncio
async def test_notify_propagates_http_exception() -> None:
    """Verify HTTPException propagates (no best-effort)."""
    sub = dn.DigitsEventSubscriber(bot=_HTTPExceptionBot(), redis_url="redis://fake")
    event: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r_http",
        "user_id": 998,
        "model_id": "mnist",
        "total_epochs": 1,
        "queue": "digits",
    }
    with pytest.raises(HTTPException):
        await sub._handle_event(event)


@pytest.mark.asyncio
async def test_notify_propagates_not_found() -> None:
    """Verify NotFound exception propagates (no best-effort)."""
    sub = dn.DigitsEventSubscriber(bot=_NotFoundBot(), redis_url="redis://fake")
    event: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r_notfound",
        "user_id": 997,
        "model_id": "mnist",
        "val_acc": 0.95,
    }
    with pytest.raises(NotFound):
        await sub._handle_event(event)


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    """Verify start/stop lifecycle branches."""
    sub = dn.DigitsEventSubscriber(
        bot=TrackingBot(TrackingUser()),
        redis_url="redis://fake",
        source_factory=_make_source,
    )
    # Start twice (second is no-op)
    sub.start()
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()

    # Stop without start (no-op)
    sub2 = dn.DigitsEventSubscriber(
        bot=TrackingBot(TrackingUser()),
        redis_url="redis://fake",
        source_factory=_make_source,
    )
    await sub2.stop()


@pytest.mark.asyncio
async def test_custom_events_channel() -> None:
    """Verify custom events channel is accepted."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = dn.DigitsEventSubscriber(
        bot=bot,
        redis_url="redis://fake",
        events_channel="custom:channel",
    )
    # Verify construction works with custom channel by sending an event
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r_custom",
        "user_id": 3,
        "model_id": "m",
        "total_epochs": 1,
        "queue": "digits",
    }
    await sub._handle_event(config)
    assert len(user.embeds) == 1
