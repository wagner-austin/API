from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NoReturn

import discord
import pytest
from platform_discord.discord_types import Embed, File
from platform_discord.protocols import (
    FileProto,
    FollowupProto,
    MessageProto,
    ResponseProto,
    UserProto,
)
from tests.support.discord_fakes import FakeMessage, FakeUser

from clubbot.cogs.base import BaseCog


class _ResponseForDefer:
    """Response that supports defer for testing."""

    def __init__(self, done: bool) -> None:
        self._done = done

    def is_done(self) -> bool:
        return self._done

    async def defer(self, *, ephemeral: bool = False) -> None:
        # Sleep a tick so that the task remains not-done after awaiting sleep(0)
        await asyncio.sleep(0.01)

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: Embed | None = None,
        ephemeral: bool = False,
    ) -> None:
        return None


class _FollowupStub:
    """Followup stub for testing."""

    async def send(
        self,
        content: str | None = None,
        *,
        embed: Embed | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        _ = (content, embed, file, ephemeral)
        return FakeMessage()


class _InteractionForDefer:
    """Interaction for testing _safe_defer."""

    def __init__(self, done: bool) -> None:
        self._response_impl = _ResponseForDefer(done)
        self._followup_impl = _FollowupStub()
        self._user_impl = FakeUser()

    @property
    def response(self) -> ResponseProto:
        return self._response_impl

    @property
    def followup(self) -> FollowupProto:
        return self._followup_impl

    @property
    def user(self) -> UserProto:
        return self._user_impl


@pytest.mark.asyncio
async def test_safe_defer_not_done_path() -> None:
    cog = BaseCog()
    inter = _InteractionForDefer(done=False)
    ok = await cog._safe_defer(inter, ephemeral=True)
    assert ok is True


class _UserNoSend:
    """User that returns object without send method."""

    @property
    def id(self) -> int:
        return 99999

    async def send(
        self,
        content: str | None = None,
        *,
        embed: Embed | None = None,
        file: File | None = None,
    ) -> NoReturn:
        raise RuntimeError("User has no send")


class _BotReturnsInvalidUser:
    """Bot that returns a user without proper send method."""

    async def fetch_user(self, user_id: int, /) -> _UserNoSend:
        return _UserNoSend()


@pytest.mark.asyncio
async def test_notify_user_and_dm_file_error_path(tmp_path: Path) -> None:
    cog = BaseCog()
    cog.bot = _BotReturnsInvalidUser()
    await cog.notify_user(1, "hello")
    # Create a real discord.File for the dm_file test
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"content")
    await cog.dm_file(1, "content", discord.File(test_file))
