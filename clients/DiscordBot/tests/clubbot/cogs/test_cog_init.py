from __future__ import annotations

import logging

import discord
import pytest
from discord.ext import commands
from platform_discord.protocols import wrap_bot
from tests.conftest import _build_settings

from clubbot.cogs.qr import QRCog
from clubbot.config import DiscordbotSettings
from clubbot.services.qr.client import QRService


def _make_cfg() -> DiscordbotSettings:
    return _build_settings(qr_default_border=2, qr_api_url="http://localhost:8080")


@pytest.mark.asyncio
async def test_qr_cog_can_be_added_to_bot() -> None:
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)
    cfg = _make_cfg()
    service = QRService(cfg)

    cog = QRCog(wrap_bot(bot), cfg, service)
    await bot.add_cog(cog)

    assert bot.get_cog("QRCog") is cog


logger = logging.getLogger(__name__)
