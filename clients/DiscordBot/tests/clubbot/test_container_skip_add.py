from __future__ import annotations

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_wire_bot_skips_when_cogs_present() -> None:
    # Build settings for this test
    cfg = build_settings(
        discord_token="x",
        transcript_provider="api",
        transcript_api_url="http://localhost:8000",
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer.from_env()

    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)

    # Pre-register placeholder cogs with the expected names
    class QRCog(commands.Cog):
        def __init__(self) -> None:
            super().__init__()

    class InviteCog(commands.Cog):
        def __init__(self) -> None:
            super().__init__()

    class TranscriptCog(commands.Cog):
        def __init__(self) -> None:
            super().__init__()

    await bot.add_cog(QRCog())
    await bot.add_cog(InviteCog())
    await bot.add_cog(TranscriptCog())

    # Now wiring should detect existing cogs and not add duplicates
    await cont.wire_bot_async(bot)
    names = list(bot.cogs.keys())
    assert names.count("QRCog") == 1 and names.count("InviteCog") == 1
    assert names.count("TranscriptCog") == 1
