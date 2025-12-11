from __future__ import annotations

import asyncio
import time
from io import BytesIO

import discord
from discord import app_commands
from platform_core.errors import AppError
from platform_core.logging import get_logger
from platform_discord.protocols import (
    FileProto,
    InteractionProto,
)
from platform_discord.rate_limiter import RateLimiter

from .. import _test_hooks
from ..config import DiscordbotSettings
from .base import BaseCog, BotForSetup, _BotProto, _Logger


class QRCog(BaseCog):
    def __init__(
        self, bot: _BotProto, config: DiscordbotSettings, qr_service: _test_hooks.QRServiceLike
    ) -> None:
        super().__init__()
        self.bot = bot
        self.config = config
        self.qr_service = qr_service
        # Per-user rate limiting to prevent rapid-fire requests
        rate_limit = config["qr"]["rate_limit"]
        rate_window = config["qr"]["rate_window_seconds"]
        self.rate_limiter = RateLimiter(rate_limit, rate_window)

    @app_commands.command(name="qrcode", description="Create a QR code from a URL")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(url="URL to encode as a QR code")
    async def qrcode(self, interaction: discord.Interaction, url: str) -> None:
        # Wrap discord.Interaction for Protocol-based methods
        wrapped: InteractionProto = _test_hooks.wrap_interaction(interaction)
        await self._qrcode_impl(wrapped, url)

    async def _qrcode_impl(self, interaction: InteractionProto, url: str) -> None:
        """Internal implementation that accepts Protocol types for testability."""
        ephemeral = not self.config["qr"]["public_responses"]
        if not await self._safe_defer(interaction, ephemeral=ephemeral):
            return

        # Request-scoped logging
        req_id = self.new_request_id()
        log = self.request_logger(req_id)
        user_obj = interaction.user

        user_id = self.decode_int_attr(user_obj, "id")
        if user_id is None:
            await self.handle_user_error(
                interaction, log, "Could not determine your user id for rate limiting"
            )
            return
        log.debug("QR command invoked by user=%s for url=%s", str(user_id), url[:50])

        await self._process_qr(interaction, url, user_id, log)

    async def _process_qr(
        self,
        interaction: InteractionProto,
        url: str,
        user_id: int,
        log: _Logger,
    ) -> None:
        if not await self.check_rate_limit(
            interaction,
            rate_limiter=self.rate_limiter,
            user_id=user_id,
            command="qrcode",
            log=log,
            public_responses=self.config["qr"]["public_responses"],
        ):
            return

        try:
            # Generate image in a worker thread to avoid blocking the event loop
            result = await self._generate_qr_image(url)
        except AppError as e:
            log.info("User input error: %s", str(e))
            await self.handle_user_error(interaction, log, str(e))
            return
        except RuntimeError as e:
            get_logger(__name__).warning("QR generation failed: %s", e)
            await self.handle_exception(interaction, log, e)
            return
        # Unexpected exceptions propagate to global handler

        # Build embed and filename qrcode_{timestamp}.png and send
        ts = int(time.time())
        filename = f"qrcode_{ts}.png"
        # discord.File satisfies FileProto structurally
        file: FileProto = discord.File(fp=BytesIO(result.image_png), filename=filename)
        content = f"QR for <{result.url}>"
        from platform_discord.qr.embeds import build_qr_embed as _build_qr_embed

        embed = _build_qr_embed(url=result.url)
        await interaction.followup.send(
            content=content,
            embed=embed,
            file=file,
            ephemeral=not self.config["qr"]["public_responses"],
        )
        log.info("QR code sent successfully for url=%s", result.url[:50])

    async def _generate_qr_image(self, url: str) -> _test_hooks.QRResultLike:
        return await asyncio.to_thread(self.qr_service.generate_qr, url)


async def setup(bot: BotForSetup) -> None:
    cfg = _test_hooks.load_settings()
    service = _test_hooks.qr_service_factory(cfg)
    await bot.add_cog(QRCog(bot, cfg, service))
