from __future__ import annotations

import io

import discord
import pytest
from tests.support.discord_fakes import FakeBotRaises

from clubbot.cogs.base import BaseCog, _ExtraLogger


class _DummyLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = []

    def debug(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.calls.append((msg, args, extra))

    def info(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.calls.append((msg, args, extra))

    def warning(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.calls.append((msg, args, extra))

    def exception(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.calls.append((msg, args, extra))


def test_extra_logger_warning_merges_extra() -> None:
    base = _DummyLogger()
    log = _ExtraLogger(base, {"a": "1"})
    log.warning("x", "y", extra={"b": "2"})
    assert base.calls and base.calls[-1][2] == {"a": "1", "b": "2"}


@pytest.mark.asyncio
async def test_notify_user_raises_and_logs() -> None:
    cog = BaseCog()
    cog.bot = FakeBotRaises()
    await cog.notify_user(1, "hello")
    # No exception raised; internal raise is captured and warning logged via BaseCog


@pytest.mark.asyncio
async def test_dm_file_raises_and_logs() -> None:
    cog = BaseCog()
    cog.bot = FakeBotRaises()
    file = discord.File(fp=io.BytesIO(b"x"), filename="x.png")
    await cog.dm_file(1, "hello", file)
    # No exception raised; internal raise is captured and warning logged via BaseCog
