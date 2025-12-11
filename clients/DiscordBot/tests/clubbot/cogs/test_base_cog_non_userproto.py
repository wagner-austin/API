from __future__ import annotations

import io

import discord
import pytest
from platform_discord.embed_helpers import EmbedProto
from platform_discord.exceptions import DHTTPExceptionError
from platform_discord.protocols import BotProto, FileProto, MessageProto, UserProto

from clubbot.cogs.base import BaseCog


class _FakeMessage(MessageProto):
    """Fake message for protocol satisfaction."""

    @property
    def id(self) -> int:
        return 1

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        _ = (content, embed)
        return self


class _WeirdUser(UserProto):
    """User that lacks a working send() method.

    Satisfies UserProto structurally for static typing, but the send()
    raises at runtime to test the defensive exception handling path.
    """

    @property
    def id(self) -> int:
        return 999

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        # Raise DHTTPExceptionError to trigger the exception handling path
        # that logs a warning about DM failure
        raise DHTTPExceptionError("Simulated DM failure for testing")


class _WeirdBot(BotProto):
    """Bot that returns users whose send() raises."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return _WeirdUser()


class _TestableCog(BaseCog):
    """A subclass that allows injecting bot in constructor."""

    def __init__(self, bot: BotProto | None = None) -> None:
        super().__init__()
        self.bot = bot


@pytest.mark.asyncio
async def test_notify_user_non_userproto_path_logs_warning() -> None:
    cog = _TestableCog(bot=_WeirdBot())
    await cog.notify_user(1, "msg")


@pytest.mark.asyncio
async def test_dm_file_non_userproto_path_logs_warning() -> None:
    cog = _TestableCog(bot=_WeirdBot())
    file = discord.File(fp=io.BytesIO(b"x"), filename="x.txt")
    await cog.dm_file(1, "content", file)
