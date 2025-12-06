from __future__ import annotations

import time

import discord
from discord import app_commands

from ..config import DiscordbotSettings, load_discordbot_settings
from .base import BaseCog, BotForSetup, _BotProto


class ExampleCog(BaseCog):
    """Minimal example cog using BaseCog with request-scoped logging.

    Not auto-loaded. To try it, call:
      bot.load_extension("clubbot.cogs.example")
    """

    def __init__(self, bot: _BotProto, config: DiscordbotSettings) -> None:
        super().__init__()
        self.bot = bot
        self.config = config

    @app_commands.command(name="ping", description="Simple ping to verify the bot is responsive")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def ping(self, interaction: discord.Interaction) -> None:
        req_id = self.new_request_id()
        log = self.request_logger(req_id)
        started = time.time()
        if interaction.response.is_done():
            await interaction.followup.send(f"Pong! req={req_id}")
        else:
            await interaction.response.send_message(f"Pong! req={req_id}")
        elapsed = int((time.time() - started) * 1000)
        log.info("Handled /ping in %sms", str(elapsed))


async def setup(bot: BotForSetup) -> None:
    cfg = load_discordbot_settings()
    await bot.add_cog(ExampleCog(bot, cfg))
