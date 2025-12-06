from __future__ import annotations

import io
import logging

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


class _ResponseWithSent:
    """Response that tracks sent messages."""

    def __init__(self, done: bool = False, raise_on_send: bool = False) -> None:
        self._done = done
        self._raise = raise_on_send
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
        if self._raise:
            raise RuntimeError("boom")
        self.sent.append((content, ephemeral))

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True


class _FollowupWithSent:
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

    def __init__(self, done: bool = False, raise_on_send: bool = False) -> None:
        self._response_impl = _ResponseWithSent(done=done, raise_on_send=raise_on_send)
        self._followup_impl = _FollowupWithSent()
        self._user = FakeUser()

    @property
    def response(self) -> ResponseProto:
        return self._response_impl

    @property
    def followup(self) -> FollowupProto:
        return self._followup_impl

    @property
    def user(self) -> UserProto:
        return self._user


@pytest.mark.asyncio
async def test_handle_user_error_sends_on_response_or_followup() -> None:
    cog = BaseCog()
    log = cog.request_logger("r1")

    # response path
    inter = _InteractionWithTracking(done=False)
    await cog.handle_user_error(inter, log, "msg")
    assert inter._response_impl.sent and not inter._followup_impl.sent

    # followup fallback path when response raises
    inter2 = _InteractionWithTracking(done=False, raise_on_send=True)
    await cog.handle_user_error(inter2, log, "msg2")
    assert inter2._followup_impl.sent


@pytest.mark.asyncio
async def test_handle_exception_includes_req_and_branches() -> None:
    cog = BaseCog()
    log = cog.request_logger("r99")

    # done=True path goes through followup
    inter = _InteractionWithTracking(done=True)
    await cog.handle_exception(inter, log, RuntimeError("x"))
    assert inter._followup_impl.sent
    sent_content = inter._followup_impl.sent[0][0]
    assert sent_content is not None and "req=r99" in sent_content

    # done=False path goes through response
    inter2 = _InteractionWithTracking(done=False)
    await cog.handle_exception(inter2, log, RuntimeError("y"))
    assert inter2._response_impl.sent


@pytest.mark.asyncio
async def test_handle_exception_ignores_nondict_extra() -> None:
    """Ensure branch where logger.extra is not a dict is covered."""
    cog = BaseCog()

    class _LogNonDictExtra:
        """Logger with non-dict extra attribute."""

        extra: str = "nondict"

        def debug(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
            return None

        def info(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
            return None

        def warning(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
            return None

        def exception(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
            return None

    inter = _InteractionWithTracking(done=True)
    await cog.handle_exception(inter, _LogNonDictExtra(), RuntimeError("x"))
    # Message should not include req= suffix since extra is non-dict
    assert inter._followup_impl.sent
    sent_content = inter._followup_impl.sent[0][0]
    assert sent_content is not None and "req=" not in sent_content


@pytest.mark.asyncio
async def test_dm_file_and_notify_user_bot_none_and_failure_paths() -> None:
    cog = BaseCog()
    # bot is None path
    cog.bot = None
    await cog.notify_user(1, "x")
    await cog.dm_file(1, "x", discord.File(fp=io.BytesIO(b"x"), filename="x.txt"))


@pytest.mark.asyncio
async def test_dm_file_success_path() -> None:
    from typing import NoReturn

    cog = BaseCog()

    class _UserWithTracking:
        """User that tracks sent messages."""

        def __init__(self) -> None:
            self.sent: list[tuple[str | None, File | None]] = []

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
            self.sent.append((content, file))
            return FakeMessage()

    class _BotWithUser:
        """Bot that returns a tracking user."""

        def __init__(self) -> None:
            self._user = _UserWithTracking()

        async def fetch_user(self, user_id: int, /) -> UserProto:
            return self._user

    bot = _BotWithUser()
    cog.bot = bot
    dummy_file = discord.File(fp=io.BytesIO(b"x"), filename="x.txt")
    await cog.dm_file(7, "hello", dummy_file)
    assert bot._user.sent
    assert bot._user.sent[-1][0] == "hello"
    # File was passed to send method
    if bot._user.sent[-1][1] is None:
        raise AssertionError("expected file in send")

    # failure path: fetch_user raises
    class _FailBot:
        async def fetch_user(self, user_id: int, /) -> NoReturn:
            raise RuntimeError("nope")

    cog.bot = _FailBot()
    await cog.notify_user(1, "x")
    await cog.dm_file(1, "x", discord.File(fp=io.BytesIO(b"x"), filename="x.txt"))


logger = logging.getLogger(__name__)
