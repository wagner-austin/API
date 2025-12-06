from __future__ import annotations

import logging

import discord
import pytest
from discord.ext import commands

from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_digits_wiring_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enable digits service but do not set REDIS_URL to exercise the no-enqueuer branch
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("TRANSCRIPT_PROVIDER", "api")
    monkeypatch.setenv("TRANSCRIPT_API_URL", "http://localhost:8000")
    monkeypatch.setenv("HANDWRITING_API_URL", "http://localhost:1234")

    cont = ServiceContainer.from_env()
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)
    assert "DigitsCog" in bot.cogs


logger = logging.getLogger(__name__)
