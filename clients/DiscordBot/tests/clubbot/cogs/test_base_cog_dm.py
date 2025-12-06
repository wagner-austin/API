from __future__ import annotations

import logging
from typing import NoReturn

import pytest
from platform_discord.discord_types import Embed, File
from platform_discord.protocols import MessageProto, UserProto
from tests.support.discord_fakes import FakeMessage

from clubbot.cogs.base import BaseCog


class _OkUser:
    def __init__(self) -> None:
        self.messages: list[str] = []

    @property
    def id(self) -> int:
        return 99999

    async def send(
        self,
        content: str | None = None,
        *,
        embed: Embed | None = None,
        file: File | None = None,
    ) -> MessageProto:
        if content is not None:
            self.messages.append(content)
        return FakeMessage()


class _OkBot:
    def __init__(self) -> None:
        self._u = _OkUser()

    async def fetch_user(self, user_id: int, /) -> UserProto:
        return self._u


class _FailBot:
    async def fetch_user(self, user_id: int, /) -> NoReturn:
        raise RuntimeError("nope")


@pytest.mark.asyncio
async def test_notify_user_success_and_failure_are_safe() -> None:
    cog = BaseCog()
    # Success path
    ok_bot = _OkBot()
    cog.bot = ok_bot
    await cog.notify_user(1, "hello")
    assert ok_bot._u.messages == ["hello"]

    # Failure path: should not raise
    fail_bot = _FailBot()
    cog.bot = fail_bot
    await cog.notify_user(2, "ignored")


logger = logging.getLogger(__name__)
