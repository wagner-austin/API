from __future__ import annotations

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_from_env_stt_no_digits_no_trainer() -> None:
    # Provider not API, no handwriting and no trainer URLs
    cfg = build_settings(
        transcript_provider="stt",
        handwriting_api_url="",
        model_trainer_api_url="",
    )

    def _test_load_settings() -> DiscordbotSettings:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer.from_env()
    assert cont.transcript_service is None
    assert cont.digits_service is None

    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)

    names = set(bot.cogs.keys())
    assert "QRCog" in names and "InviteCog" in names
    assert "TranscriptCog" not in names and "DigitsCog" not in names and "TrainerCog" not in names
