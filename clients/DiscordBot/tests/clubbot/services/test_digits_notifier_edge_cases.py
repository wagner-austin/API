"""Tests for edge cases and error handling in digits_notifier."""

from __future__ import annotations

import logging
from typing import NoReturn

import discord
import pytest
from platform_core.digits_metrics_events import DigitsCompletedMetricsV1, DigitsConfigV1
from platform_core.job_events import JobFailedV1
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto

import clubbot.services.jobs.digits_notifier as dn

# Typed dynamic import to avoid Any in exception types
_discord = __import__("discord")
Forbidden: type[BaseException] = _discord.Forbidden
HTTPException: type[BaseException] = _discord.HTTPException
NotFound: type[BaseException] = _discord.NotFound


class _Response:
    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason


class _FailingUser:
    """Mock user that raises Discord API exceptions."""

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


class _FailingBot:
    def __init__(self) -> None:
        self.user = _FailingUser()

    async def fetch_user(self, user_id: int, /) -> _FailingUser:
        _ = user_id
        return self.user


@pytest.mark.asyncio
async def test_notify_raises_discord_forbidden() -> None:
    """_notify propagates Discord.Forbidden (no best-effort)."""
    bot = _FailingBot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

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


class _HTTPExceptionUser:
    """Mock user that raises HTTP exceptions."""

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
    def __init__(self) -> None:
        self.user = _HTTPExceptionUser()

    async def fetch_user(self, user_id: int, /) -> _HTTPExceptionUser:
        _ = user_id
        return self.user


@pytest.mark.asyncio
async def test_notify_raises_http_exception() -> None:
    """_notify propagates Discord.HTTPException (no best-effort)."""
    bot = _HTTPExceptionBot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    event: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r_http_error",
        "user_id": 998,
        "model_id": "mnist",
        "total_epochs": 1,
        "queue": "digits",
    }
    with pytest.raises(HTTPException):
        await sub._handle_event(event)


class _NotFoundUser:
    """Mock user that raises NotFound exceptions."""

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
    def __init__(self) -> None:
        self.user = _NotFoundUser()

    async def fetch_user(self, user_id: int, /) -> _NotFoundUser:
        _ = user_id
        return self.user


@pytest.mark.asyncio
async def test_notify_raises_not_found() -> None:
    """_notify propagates Discord.NotFound (no best-effort)."""
    bot = _NotFoundBot()
    sub = dn.DigitsEventSubscriber(bot, redis_url="redis://fake")

    event: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r_not_found",
        "user_id": 997,
        "model_id": "mnist",
        "val_acc": 0.95,
    }
    with pytest.raises(NotFound):
        await sub._handle_event(event)


logger = logging.getLogger(__name__)
