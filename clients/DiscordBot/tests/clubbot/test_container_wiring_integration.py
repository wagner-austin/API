from __future__ import annotations

import discord
import pytest
from discord.ext import commands

from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_wires_real_cogs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure minimal environment
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    # Build container from env (no Digits to keep light)
    # Ensure transcript provider is API-only per refactor
    monkeypatch.setenv("TRANSCRIPT_PROVIDER", "api")
    monkeypatch.setenv("TRANSCRIPT_API_URL", "http://localhost:8000")
    cont = ServiceContainer.from_env()
    # Use real Bot instance (no login)
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)

    # Validate expected cogs attached
    names = set(bot.cogs.keys())
    assert {"QRCog", "InviteCog", "TranscriptCog"}.issubset(names)
