from __future__ import annotations

import pytest
from platform_discord.embed_helpers import EmbedProto, create_embed
from platform_discord.protocols import FileProto, MessageProto, UserProto

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


class _Msg(MessageProto):
    @property
    def id(self) -> int:
        return 1

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        _ = content
        return self


class _User(UserProto):
    def __init__(self) -> None:
        self.sent: list[EmbedProto | None] = []

    @property
    def id(self) -> int:
        return 42

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = (content, file)
        self.sent.append(embed)
        return _Msg()


class _Bot:
    def __init__(self, user: UserProto) -> None:
        self._user = user

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return self._user


@pytest.mark.asyncio
async def test_maybe_notify_noop_when_no_embed() -> None:
    sub = DigitsEventSubscriber(_Bot(_User()), redis_url="redis://")
    await sub._maybe_notify({"user_id": 1, "request_id": "r", "embed": None})


@pytest.mark.asyncio
async def test_notify_edits_existing_message() -> None:
    u = _User()
    sub = DigitsEventSubscriber(_Bot(u), redis_url="redis://")
    req_id = "r"
    sub._messages[req_id] = _Msg()
    await sub.notify(1, req_id, create_embed(title="t"))
