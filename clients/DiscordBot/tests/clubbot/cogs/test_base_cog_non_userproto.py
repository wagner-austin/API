from __future__ import annotations

import io

import discord
import pytest

from clubbot.cogs.base import BaseCog


class _WeirdUser:
    # No send() method; deliberately not UserProto
    @property
    def id(self) -> int:
        return 999


class _WeirdBot:
    async def fetch_user(self, user_id: int, /) -> _WeirdUser:
        _ = user_id
        return _WeirdUser()


@pytest.mark.asyncio
async def test_notify_user_non_userproto_path_logs_warning() -> None:
    cog = BaseCog()
    # Bypass type checker for attribute assignment
    object.__setattr__(cog, "bot", _WeirdBot())
    await cog.notify_user(1, "msg")


@pytest.mark.asyncio
async def test_dm_file_non_userproto_path_logs_warning() -> None:
    cog = BaseCog()
    object.__setattr__(cog, "bot", _WeirdBot())
    file = discord.File(fp=io.BytesIO(b"x"), filename="x.txt")
    await cog.dm_file(1, "content", file)
