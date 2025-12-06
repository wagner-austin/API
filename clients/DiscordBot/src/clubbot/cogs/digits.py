from __future__ import annotations

from typing import Final, Protocol

import discord
from discord import Member, User, app_commands
from platform_core.digits_metrics_events import DEFAULT_DIGITS_EVENTS_CHANNEL
from platform_core.errors import AppError, ErrorCode
from platform_core.logging import get_logger
from platform_discord.embed_helpers import add_field, create_embed, set_footer
from platform_discord.protocols import InteractionProto, UserProto, wrap_interaction
from platform_discord.rate_limiter import RateLimiter

from ..config import DiscordbotSettings
from ..services.digits.app import DigitService
from ..services.handai.client import HandwritingAPIError, PredictResult
from ..services.jobs.digits_enqueuer import DigitsEnqueuer
from ..services.jobs.digits_notifier import DigitsEventSubscriber
from .base import BaseCog, _BotProto

_PNG: Final[str] = "image/png"
_JPEG: Final[str] = "image/jpeg"
_JPG: Final[str] = "image/jpg"
_ALLOWED: Final[tuple[str, ...]] = (_PNG, _JPEG, _JPG)


class _HasId(Protocol):
    @property
    def id(self) -> int | None: ...


class DigitsCog(BaseCog):
    def __init__(
        self,
        bot: _BotProto,
        config: DiscordbotSettings,
        service: DigitService,
        enqueuer: DigitsEnqueuer | None = None,
        *,
        autostart_subscriber: bool = True,
    ) -> None:
        super().__init__()
        self.bot = bot
        self.config = config
        self.service = service
        self.rate_limiter = RateLimiter(
            config["digits"]["rate_limit"], config["digits"]["rate_window_seconds"]
        )
        self._enqueuer: DigitsEnqueuer | None = enqueuer
        self._subscriber: DigitsEventSubscriber | None = None
        self._autostart_subscriber = autostart_subscriber
        redis_url = self.config["redis"]["redis_url"] or ""
        if redis_url:
            self._subscriber = DigitsEventSubscriber(
                bot=self.bot, redis_url=redis_url, events_channel=DEFAULT_DIGITS_EVENTS_CHANNEL
            )
            if self._autostart_subscriber:
                self._subscriber.start()
                get_logger(__name__).info(
                    "Digits events subscriber started (channel=%s)", DEFAULT_DIGITS_EVENTS_CHANNEL
                )

    def ensure_subscriber_started(self) -> None:
        sub = self._subscriber
        if sub is not None:
            sub.start()

    async def cog_unload(self) -> None:
        sub = self._subscriber
        if sub is not None:
            await sub.stop()
        self._subscriber = None

    @app_commands.command(name="read", description="Recognize a handwritten digit from an image")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(image="PNG or JPEG image of a single digit")
    async def read(self, interaction: discord.Interaction, image: discord.Attachment) -> None:
        wrapped: InteractionProto = wrap_interaction(interaction)
        await self._read_impl(wrapped, interaction.user, image)

    async def _read_impl(
        self,
        interaction: InteractionProto,
        user_obj: _HasId | None,
        image: discord.Attachment,
    ) -> None:
        public = self.config["digits"]["public_responses"] is True
        if not await self._safe_defer(interaction, ephemeral=not public):
            return

        request_id = self.new_request_id()
        log = self.request_logger(request_id)

        user_id = self.decode_int_attr(user_obj, "id")
        if user_id is None:
            await self.handle_user_error(interaction, log, "Could not determine your user id")
            return

        if not await self.check_rate_limit(
            interaction,
            rate_limiter=self.rate_limiter,
            user_id=user_id,
            command="read",
            log=log,
            public_responses=public,
        ):
            return

        try:
            self._validate_attachment(image)
            data = await image.read()
            result = await self.service.read_image(
                data=data,
                filename=image.filename or "image",
                content_type=image.content_type or "",
                request_id=request_id,
            )
        except AppError as e:
            log.info("User input error: %s", str(e))
            await self.handle_user_error(interaction, log, str(e))
            return
        except HandwritingAPIError as e:
            log.info("Handwriting API error: %s", str(e))
            msg = _user_message_from_api_error(e)
            await self.handle_user_error(interaction, log, msg)
            return
        except RuntimeError as e:
            get_logger(__name__).warning("Digit read failed: %s", e)
            await self.handle_exception(interaction, log, e)
            return

        content = _format_result(result)
        await interaction.followup.send(
            content=content, ephemeral=self.config["digits"]["public_responses"] is not True
        )
        log.info("Digit read sent successfully")

    @app_commands.command(name="train", description="Queue a background training job (digits)")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    async def train(self, interaction: discord.Interaction) -> None:
        wrapped: InteractionProto = wrap_interaction(interaction)
        await self._train_impl(wrapped, interaction.user)

    async def _train_impl(
        self, wrapped: InteractionProto, user_obj: User | Member | UserProto | None
    ) -> None:
        public = self.config["digits"]["public_responses"] is True
        if not await self._safe_defer(wrapped, ephemeral=not public):
            return

        request_id = self.new_request_id()
        log = self.request_logger(request_id)

        user_id = self.decode_int_attr(user_obj, "id")
        if user_id is None:
            await self.handle_user_error(wrapped, log, "Could not determine your user id")
            return

        if self._enqueuer is None:
            await wrapped.followup.send("Training is not configured.", ephemeral=True)
            log.info("Train requested but enqueuer is not configured")
            return

        model_id = "mnist_resnet18_v1"
        epochs = 15
        batch_size = 256
        lr = 0.0015
        seed = 42
        augment = True
        notes = "requested via /train"

        try:
            job_id = self._enqueuer.enqueue_train(
                request_id=request_id,
                user_id=user_id,
                model_id=model_id,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                augment=augment,
                notes=notes,
            )
        except RuntimeError as e:
            get_logger(__name__).warning("Train enqueue failed: %s", e)
            await self.handle_exception(wrapped, log, e)
            return

        embed = create_embed(
            title="Training Job Queued",
            description=(
                "Your training job has been queued successfully!\n"
                "You'll receive **DM updates** with progress and results."
            ),
            color=0x5865F2,
        )
        add_field(
            embed,
            name="Job Info",
            value=f"**Model:** `{model_id}`\n**Job ID:** `{job_id}`",
            inline=True,
        )
        add_field(
            embed,
            name="Configuration",
            value=(
                f"**Epochs:** `{epochs}`\n**Batch Size:** `{batch_size}`\n**Learning Rate:** `{lr}`"
            ),
            inline=True,
        )
        augment_status = "Augmentations: Enabled" if augment else "Augmentations: Disabled"
        add_field(embed, name="Augmentations", value=augment_status, inline=False)
        set_footer(embed, text=f"Request ID: {request_id}")

        await wrapped.followup.send(
            embed=embed, ephemeral=self.config["digits"]["public_responses"] is not True
        )
        log.info("Queued training req=%s job=%s", request_id, job_id)

    async def _obsolete_cog_unload(self) -> None:
        sub = self._subscriber
        if sub is not None:
            await sub.stop()

    def _validate_attachment(self, att: discord.Attachment) -> None:
        ctype = (att.content_type or "").lower()
        if ctype not in _ALLOWED:
            raise AppError(
                ErrorCode.INVALID_INPUT,
                "Unsupported file type; please upload a PNG or JPEG image",
                http_status=400,
            )
        max_bytes = int(self.service.max_image_bytes)
        size_raw = int(att.size if isinstance(att.size, int) else 0)
        size = size_raw * 1024 * 1024 if 0 < max_bytes <= 2048 and 0 < size_raw < 4096 else size_raw
        if size > 0 and max_bytes > 0 and size > max_bytes:
            mb = max_bytes // (1024 * 1024)
            raise AppError(
                ErrorCode.INVALID_INPUT, f"Image is too large (max {mb} MB)", http_status=400
            )


def _top_k_indices(probs: list[float] | tuple[float, ...], k: int = 3) -> list[int]:
    items: list[tuple[int, float]] = [(i, float(p)) for i, p in enumerate(probs)]

    def _second(pair: tuple[int, float]) -> float:
        return pair[1]

    items.sort(key=_second, reverse=True)
    return [items[i][0] for i in range(min(k, len(items)))]


def _format_result(res: PredictResult) -> str:
    top3 = _top_k_indices(res.probs, 3)
    parts = [f"Digit: {res.digit} ({res.confidence * 100:.1f}% confidence)."]
    top_parts = [f"{i}={res.probs[i]:.3f}" for i in top3]
    parts.append(f"Top-3: {', '.join(top_parts)}.")
    parts.append(f"Model: {res.model_id}.")
    if res.uncertain:
        parts.append("Low confidence; try larger digits or darker ink.")
    return " ".join(parts)


def _user_message_from_api_error(e: HandwritingAPIError) -> str:
    if e.status == 401:
        return "Service is not authorized. Please contact an admin."
    if e.status == 413 or (e.code == "too_large"):
        return "Image is too large."
    if e.status == 415 or (e.code == "unsupported_media_type"):
        return "Unsupported file type; please upload a PNG or JPEG image."
    if e.status == 400 and (e.code in {"invalid_image", "bad_dimensions", "preprocessing_failed"}):
        return "Could not process image. Please try another image."
    if e.status == 504 or e.code == "timeout":
        base = str(e) or "Request timed out"
        return f"timeout: {base}" + (f" (req {e.request_id})" if e.request_id else "")
    code = e.code or "internal_error"
    base = str(e) or f"HTTP {e.status}"
    return f"{code}: {base}" + (f" (req {e.request_id})" if e.request_id else "")


__all__ = [
    "DEFAULT_DIGITS_EVENTS_CHANNEL",
    "DigitsCog",
    "_format_result",
    "_top_k_indices",
    "_user_message_from_api_error",
]
