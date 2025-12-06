from __future__ import annotations

import logging

import pytest
from platform_discord.discord_types import Embed
from platform_discord.protocols import (
    FileProto,
    FollowupProto,
    MessageProto,
    ResponseProto,
    UserProto,
)
from tests.support.discord_fakes import FakeMessage, FakeUser

from clubbot.cogs.base import BaseCog


class _ResponseWithTracking:
    """Response that tracks sent messages."""

    def __init__(self, done: bool = False) -> None:
        self._done = done
        self.sent: list[tuple[str | None, bool]] = []

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: Embed | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.sent.append((content, ephemeral))

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True


class _FollowupWithTracking:
    """Followup that tracks sent messages."""

    def __init__(self) -> None:
        self.sent: list[tuple[str | None, bool]] = []

    async def send(
        self,
        content: str | None = None,
        *,
        embed: Embed | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        _ = (embed, file)
        self.sent.append((content, ephemeral))
        return FakeMessage()


class _InteractionWithTracking:
    """Interaction that tracks sends for verification."""

    def __init__(self, done: bool = False) -> None:
        self._response_impl = _ResponseWithTracking(done=done)
        self._followup_impl = _FollowupWithTracking()
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
async def test_handle_exception_without_request_id_branch() -> None:
    cog = BaseCog()
    inter = _InteractionWithTracking(done=False)
    # Build a logger adapter with non-string request_id to exercise branch
    logger = logging.LoggerAdapter(logging.getLogger(__name__), {"request_id": 123})
    await cog.handle_exception(inter, logger, RuntimeError("boom"))
    assert inter._response_impl.sent or inter._followup_impl.sent
