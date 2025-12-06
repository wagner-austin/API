from __future__ import annotations

import discord
import pytest
from discord.ext import commands
from tests.support.settings import build_settings

from clubbot.container import ServiceContainer


@pytest.mark.asyncio
async def test_container_from_env_stt_no_digits_no_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provider not API, no handwriting and no trainer URLs
    monkeypatch.setenv("DISCORD_TOKEN", "x")
    monkeypatch.setenv("TRANSCRIPT_PROVIDER", "stt")
    # Intentionally omit HANDWRITING_API_URL and MODEL_TRAINER_API_URL

    cfg = build_settings(
        transcript_provider="stt",
        handwriting_api_url="",
        model_trainer_api_url="",
    )
    # Override autouse guard to use our custom settings for this test
    monkeypatch.setattr("clubbot.container.load_discordbot_settings", lambda: cfg, raising=True)
    cont = ServiceContainer.from_env()
    assert cont.transcript_service is None
    assert cont.digits_service is None

    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    await cont.wire_bot_async(bot)

    names = set(bot.cogs.keys())
    assert "QRCog" in names and "InviteCog" in names
    assert "TranscriptCog" not in names and "DigitsCog" not in names and "TrainerCog" not in names
