from __future__ import annotations

import asyncio
import io
from typing import Protocol

import discord
from discord import app_commands
from platform_core.errors import AppError
from platform_core.logging import get_logger
from platform_discord.protocols import InteractionProto
from platform_discord.rate_limiter import RateLimiter

from .. import _test_hooks
from ..config import DiscordbotSettings, load_discordbot_settings
from ..services.transcript.client import TranscriptService
from .base import BaseCog, BotForSetup, _BotProto


class _HasId(Protocol):
    @property
    def id(self) -> int | None: ...


class TranscriptCog(BaseCog):
    def __init__(
        self,
        bot: _BotProto,
        config: DiscordbotSettings,
        transcript_service: TranscriptService,
        *,
        autostart_subscriber: bool = True,
    ) -> None:
        super().__init__()
        self.bot = bot
        self.config = config
        self.transcript_service = transcript_service
        rl = int(self.config["transcript"]["rate_limit"])
        win = int(self.config["transcript"]["rate_window_seconds"])
        self.rate_limiter = RateLimiter(rl, win)
        # After refactor, all transcript work runs in the API; no background jobs here
        self._autostart_subscriber = autostart_subscriber
        get_logger(__name__).info("Background jobs disabled (API provider)")

    # Lifecycle helpers for orchestrator/bot events
    def ensure_subscriber_started(self) -> None:
        return None

    async def cog_unload(self) -> None:
        return None

    @app_commands.command(
        name="transcript",
        description="Download and clean a YouTube video transcript",
    )
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(url="YouTube video URL")
    async def transcript(self, interaction: discord.Interaction, url: str) -> None:
        wrapped: InteractionProto = _test_hooks.wrap_interaction(interaction)
        guild_obj: _HasId | None = getattr(interaction, "guild", None)
        await self._transcript_impl(
            wrapped,
            interaction.user,
            guild_obj,
            url,
        )

    async def _transcript_impl(
        self,
        wrapped: InteractionProto,
        user_obj: _HasId | None,
        guild_obj: _HasId | None,
        url: str,
    ) -> None:
        """Internal implementation for testability."""
        public = self.config["transcript"]["public_responses"] is True
        if not await self._safe_defer(wrapped, ephemeral=not public):
            return

        req_id = self.new_request_id()
        log = self.request_logger(req_id)

        user_id = self.decode_int_attr(user_obj, "id")
        if user_id is None:
            await self.handle_user_error(
                wrapped, log, "Could not determine your user id for rate limiting"
            )
            return
        guild_id = self.decode_int_attr(guild_obj, "id")
        guild_id_str = str(guild_id) if guild_id is not None else "None"
        log.debug("Transcript command invoked by user=%s guild=%s", str(user_id), guild_id_str)

        # Validate early that it's a YouTube URL for nice errors
        _ = _test_hooks.validate_youtube_url(url)

        if not await self.check_rate_limit(
            wrapped,
            rate_limiter=self.rate_limiter,
            user_id=user_id,
            command="transcript",
            log=log,
            public_responses=public,
        ):
            return

        # Fetch via API provider without try/except by inspecting task outcome
        from ..services.transcript.client import TranscriptResult

        coro = _test_hooks.asyncio_to_thread(self.transcript_service.fetch_cleaned, url)
        task = asyncio.create_task(coro)
        await asyncio.wait({task})
        exc = task.exception()
        if isinstance(exc, AppError):
            log.info("User input error: %s", str(exc))
            await self.handle_user_error(wrapped, log, str(exc))
            return
        if isinstance(exc, Exception):
            get_logger(__name__).warning("Transcript processing failed: %s", exc)
            await self.handle_exception(wrapped, log, exc)
            return
        result_obj = task.result()
        if not isinstance(result_obj, TranscriptResult):
            raise RuntimeError("Unexpected result type from task")
        result = result_obj

        public = self.config["transcript"]["public_responses"] is True
        header = f"Transcript for <{result.url}>"
        # Always send as attachment to avoid spammy inline posts
        fname = f"transcript_{result.video_id}.txt"
        data = result.text.encode("utf-8")
        # Enforce attachment size limit
        if self._is_attachment_too_large(data):
            await self.handle_user_error(
                wrapped,
                log,
                (
                    f"Transcript is too large to attach (> {self._get_attachment_limit_mb()} MB). "
                    "Please try a shorter video."
                ),
            )
            return
        file = discord.File(fp=io.BytesIO(data), filename=fname)
        from platform_discord.transcript.embeds import (
            build_transcript_embed as _build_transcript_embed,
        )
        from platform_discord.transcript.types import TranscriptInfo as _TranscriptInfo

        info: _TranscriptInfo = {
            "url": result.url,
            "video_id": result.video_id,
            "chars": len(result.text),
        }
        embed = _build_transcript_embed(info=info)
        await wrapped.followup.send(content=header, embed=embed, file=file, ephemeral=not public)
        log.info("Transcript sent successfully for vid=%s", result["video_id"])

    def _get_attachment_limit_mb(self) -> int:
        """Return the configured attachment size limit in MB (default 25)."""
        return int(self.config["transcript"]["max_attachment_mb"])

    def _is_attachment_too_large(self, data: bytes) -> bool:
        """True if payload exceeds the configured attachment limit."""
        limit_mb = self._get_attachment_limit_mb()
        return limit_mb > 0 and len(data) > limit_mb * 1024 * 1024

    # STT-specific validation helpers removed; handled server-side in API

    # STT request handling removed; handled server-side in transcript-api

    # Worker-side job handling moved into clubbot.workers.transcript

    # Failure handling and retry policy provided via helpers; no per-cog duplication

    # User notification helpers moved to BaseCog


async def setup(bot: BotForSetup) -> None:
    cfg = load_discordbot_settings()
    service = TranscriptService(cfg)
    await bot.add_cog(TranscriptCog(bot, cfg, service))
