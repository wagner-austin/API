from __future__ import annotations

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_wires_real_cogs() -> None:
    # Configure settings with transcript API enabled
    cfg = build_settings(
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    # Build container from env (no Digits to keep light)
    cont = ServiceContainer.from_env()
    # Use real Bot instance (no login)
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)

    # Validate expected cogs attached
    names = set(bot.cogs.keys())
    assert {"QRCog", "InviteCog", "TranscriptCog"}.issubset(names)
