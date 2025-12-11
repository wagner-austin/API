from __future__ import annotations

import logging

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_digits_wiring_without_redis() -> None:
    # Enable digits service but do not set redis_url to exercise the no-enqueuer branch
    cfg = build_settings(
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
        handwriting_api_url="http://localhost:1234",
        redis_url=None,
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer.from_env()
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)
    assert "DigitsCog" in bot.cogs


logger = logging.getLogger(__name__)
