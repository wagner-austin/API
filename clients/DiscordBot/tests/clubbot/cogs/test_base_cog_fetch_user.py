"""Tests for BaseCog.notify_user and dm_file defensive isinstance checks.

These tests use the bot_fetch_user hook to inject an object that satisfies
FetchedUserLike but fails isinstance(obj, UserProto), covering the raise
paths in base.py lines 166 and 180.
"""

from __future__ import annotations

import io

import discord
import pytest
from platform_discord.protocols import BotProto
from tests.support.discord_fakes import FakeBot

from clubbot import _test_hooks
from clubbot._test_hooks import FetchedUserLike
from clubbot.cogs.base import BaseCog


class _NotAUserProto:
    """Object that has id property but not send() - fails isinstance(obj, UserProto).

    This satisfies FetchedUserLike but NOT UserProto since UserProto requires send().
    """

    @property
    def id(self) -> int:
        return 999


async def _fake_fetch_non_userproto(bot: BotProto, user_id: int) -> FetchedUserLike:
    """Return an object that satisfies FetchedUserLike but not UserProto."""
    _ = bot, user_id
    return _NotAUserProto()


class _TestableCog(BaseCog):
    """Testable cog that can have its bot set externally."""

    def __init__(self, bot: BotProto) -> None:
        super().__init__()
        self.bot = bot


@pytest.mark.asyncio
async def test_notify_user_isinstance_fail_catches_exception() -> None:
    """Test notify_user catches DHTTPExceptionError when isinstance fails.

    When bot_fetch_user returns an object that doesn't satisfy isinstance(obj, UserProto),
    the code raises DHTTPExceptionError which is caught by the exception handler.
    """
    _test_hooks.bot_fetch_user = _fake_fetch_non_userproto
    bot = FakeBot()
    cog = _TestableCog(bot)

    # Should not raise - the exception is caught and logged
    await cog.notify_user(1, "test message")


@pytest.mark.asyncio
async def test_dm_file_isinstance_fail_catches_exception() -> None:
    """Test dm_file catches DHTTPExceptionError when isinstance fails.

    When bot_fetch_user returns an object that doesn't satisfy isinstance(obj, UserProto),
    the code raises DHTTPExceptionError which is caught by the exception handler.
    """
    _test_hooks.bot_fetch_user = _fake_fetch_non_userproto
    bot = FakeBot()
    cog = _TestableCog(bot)

    # Create a test file
    file = discord.File(io.BytesIO(b"test data"), filename="test.txt")

    # Should not raise - the exception is caught and logged
    await cog.dm_file(1, "test content", file)
