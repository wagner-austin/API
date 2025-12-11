from __future__ import annotations

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_wires_trainer_cog() -> None:
    cfg = build_settings(
        discord_token="x",
        model_trainer_api_url="https://example",
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer.from_env()
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)
    assert "TrainerCog" in bot.cogs
