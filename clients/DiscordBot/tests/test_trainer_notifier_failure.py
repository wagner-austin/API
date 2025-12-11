from __future__ import annotations

import logging
from typing import NoReturn

import discord
import pytest
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, UserProto
from platform_discord.trainer.handler import decode_trainer_event

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber
from tests.support.discord_fakes import FakeEmbed

# Typed dynamic import for discord exception type
_discord = __import__("discord")
Forbidden: type[BaseException] = _discord.Forbidden


class _FakeResp:
    """Fake response for discord.Forbidden."""

    def __init__(self) -> None:
        self.status = 403
        self.reason = "Forbidden"


class _FakeUserForbidden:
    """Fake user that raises Forbidden on send."""

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
        raise discord.Forbidden(_FakeResp(), "no")


class _FakeBotForbidden:
    """Fake bot that returns user that raises Forbidden."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        return _FakeUserForbidden()


@pytest.mark.asyncio
async def test_notifier_notify_dm_failure_propagates() -> None:
    sub = TrainerEventSubscriber(bot=_FakeBotForbidden(), redis_url="redis://example")

    with pytest.raises(Forbidden):
        await sub.notify(1, "r", FakeEmbed(title="t"))


def test_decode_unknown_event_is_none() -> None:
    assert decode_trainer_event('{"type":"trainer.train.unknown"}') is None


logger = logging.getLogger(__name__)
