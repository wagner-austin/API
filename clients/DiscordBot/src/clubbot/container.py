from __future__ import annotations

from typing import ClassVar

from discord.ext import commands
from platform_core.logging import get_logger
from platform_core.queues import DIGITS_QUEUE
from platform_discord.protocols import wrap_bot

from .config import DiscordbotSettings, load_discordbot_settings, require_discord_token
from .services.digits.app import DigitService
from .services.jobs.digits_enqueuer import RQDigitsEnqueuer
from .services.qr.client import QRService
from .services.transcript.client import TranscriptService


class ServiceContainer:
    """Strict application container (no dict fallbacks)."""

    __slots__ = ("__dict__", "cfg", "digits_service", "qr_service", "transcript_service")

    _logger: ClassVar = get_logger(__name__)

    def __init__(
        self,
        *,
        cfg: DiscordbotSettings,
        qr_service: QRService,
        transcript_service: TranscriptService | None = None,
        digits_service: DigitService | None = None,
    ) -> None:
        self.cfg = cfg
        self.qr_service = qr_service
        self.transcript_service = transcript_service
        self.digits_service = digits_service

    @classmethod
    def from_env(cls) -> ServiceContainer:
        cfg = load_discordbot_settings()
        require_discord_token(cfg)
        qr_service = QRService(cfg)
        transcript_service: TranscriptService | None = None
        provider = cfg["transcript"]["provider"]
        if provider == "api":
            transcript_service = TranscriptService(cfg)
        digits_service = None
        if cfg["handwriting"]["api_url"]:
            digits_service = DigitService(cfg)
        return cls(
            cfg=cfg,
            qr_service=qr_service,
            transcript_service=transcript_service,
            digits_service=digits_service,
        )

    async def wire_bot_async(self, bot: commands.Bot) -> None:
        """Attach all cogs to the bot (idempotent)."""
        from .cogs.digits import DigitsCog
        from .cogs.invite import InviteCog
        from .cogs.qr import QRCog
        from .cogs.trainer import TrainerCog
        from .cogs.transcript import TranscriptCog

        cfg = self.cfg
        logger = self._logger
        wrapped = wrap_bot(bot)

        if bot.get_cog("QRCog") is None:
            await bot.add_cog(QRCog(wrapped, cfg, self.qr_service))
            logger.info("Loaded cog: QRCog")

        if bot.get_cog("InviteCog") is None:
            await bot.add_cog(InviteCog(wrapped, cfg))
            logger.info("Loaded cog: InviteCog")

        provider = cfg["transcript"]["provider"]
        if provider == "api" and bot.get_cog("TranscriptCog") is None:
            svc = self.transcript_service or TranscriptService(cfg)
            await bot.add_cog(TranscriptCog(wrapped, cfg, svc, autostart_subscriber=False))
            logger.info("Loaded cog: TranscriptCog")

        if self.digits_service is not None and bot.get_cog("DigitsCog") is None:
            redis_url = cfg["redis"]["redis_url"] or ""
            enqueuer = _build_digits_enqueuer(redis_url)
            await bot.add_cog(
                DigitsCog(
                    wrapped,
                    cfg,
                    self.digits_service,
                    enqueuer=enqueuer,
                    autostart_subscriber=False,
                )
            )
            logger.info("Loaded cog: DigitsCog")

        trainer_url = cfg["model_trainer"]["api_url"]
        if trainer_url and bot.get_cog("TrainerCog") is None:
            await bot.add_cog(TrainerCog(wrapped, cfg, autostart_subscriber=False))
            logger.info("Loaded cog: TrainerCog")


__all__ = ["ServiceContainer"]


def _build_digits_enqueuer(redis_url: str) -> RQDigitsEnqueuer | None:
    url = (redis_url or "").strip()
    if not url:
        return None
    return RQDigitsEnqueuer(
        redis_url=url,
        queue_name=DIGITS_QUEUE,
        job_timeout_s=25200,
        result_ttl_s=86400,
        failure_ttl_s=604800,
        retry_max=2,
        retry_intervals_s=(60, 300),
    )
