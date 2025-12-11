from __future__ import annotations

import discord
from discord import app_commands
from monorepo_guards._types import UnknownJson
from platform_discord.embed_helpers import add_field, create_embed
from platform_discord.protocols import InteractionProto

from .. import _test_hooks
from ..config import DiscordbotSettings
from .base import BaseCog, _BotProto


def _resolve_app_id(client: _BotProto) -> int:
    app_id_obj: UnknownJson = getattr(client, "application_id", None)
    if isinstance(app_id_obj, int):
        return app_id_obj
    raise RuntimeError("application_id is required for invite generation")


class InviteCog(BaseCog):
    def __init__(self, bot: _BotProto, config: DiscordbotSettings) -> None:
        super().__init__()
        self.bot = bot
        self.config = config

    @app_commands.command(name="invite", description="Get the server invite link for this bot")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def invite(self, interaction: discord.Interaction) -> None:
        wrapped: InteractionProto = _test_hooks.wrap_interaction(interaction)
        await self._invite_impl(wrapped)

    async def _invite_impl(self, interaction: InteractionProto) -> None:
        """Internal implementation for testability."""
        bot = self.bot
        if bot is None:
            raise RuntimeError("Bot instance is required for invite generation")
        app_id = _resolve_app_id(bot)
        perms = "2147601408"
        guild_url = (
            f"https://discord.com/api/oauth2/authorize?client_id={app_id}"
            f"&permissions={perms}&scope=bot%20applications.commands"
        )
        embed = create_embed(title="Invite Link", color=0x5865F2)
        add_field(embed, name="Guild Install (server admins)", value=guild_url, inline=False)

        if interaction.response.is_done():
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)
