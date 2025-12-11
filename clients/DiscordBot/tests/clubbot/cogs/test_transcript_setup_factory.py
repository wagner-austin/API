from __future__ import annotations

import pytest
from discord.ext import commands
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, UserProto
from tests.support.settings import build_settings

import clubbot.cogs.transcript as mod
from clubbot import _test_hooks


class _Msg(MessageProto):
    @property
    def id(self) -> int:
        return 1

    async def edit(
        self,
        *,
        content: str | None = None,
        embed: EmbedProto | None = None,
    ) -> MessageProto:
        _ = content, embed
        return self


class _User(UserProto):
    @property
    def id(self) -> int:
        return 1

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = content, embed, file
        return _Msg()


class _Bot(commands.Cog):
    def __init__(self) -> None:
        self.added: list[commands.Cog] = []

    async def add_cog(self, cog: commands.Cog) -> None:
        self.added.append(cog)

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id
        return _User()


@pytest.mark.asyncio
async def test_setup_adds_cog_with_env() -> None:
    # Use hook to inject test settings instead of monkeypatch.setenv
    _test_hooks.load_settings = lambda: build_settings(
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
    )
    bot = _Bot()
    await mod.setup(bot)
    assert bot.added and bot.added[-1].__class__.__name__ == "TranscriptCog"
