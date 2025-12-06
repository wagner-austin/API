"""Test settings factory for DiscordBot tests."""

from __future__ import annotations

from typing import Final, Protocol

from clubbot.config import (
    DigitsConfig,
    DiscordbotSettings,
    DiscordConfig,
    HandwritingConfig,
    ModelTrainerConfig,
    QRConfig,
    RedisConfig,
    TranscriptConfig,
)

_BASE_DISCORD_TOKEN: Final[str] = "test_token"
_BASE_QR_API_URL: Final[str] = "http://localhost:9000/qr"
_BASE_TRANSCRIPT_API_URL: Final[str] = "http://localhost:8000"
_BASE_HANDWRITING_API_URL: Final[str] = "http://localhost:7000"
_BASE_MODEL_TRAINER_API_URL: Final[str] = "http://localhost:9001"


class SettingsFactory(Protocol):
    """Protocol for settings factory callable."""

    def __call__(
        self,
        *,
        discord_token: str | None = None,
        guild_id: str | None = None,
        guild_ids: list[int] | None = None,
        commands_sync_global: bool | None = None,
        qr_api_url: str | None = None,
        qr_rate_limit: int | None = None,
        qr_rate_window_seconds: int | None = None,
        qr_default_border: int | None = None,
        qr_default_box_size: int | None = None,
        qr_default_fill_color: str | None = None,
        qr_default_back_color: str | None = None,
        qr_public_responses: bool | None = None,
        transcript_api_url: str | None = None,
        transcript_provider: str | None = None,
        transcript_preferred_langs: str | None = None,
        transcript_public_responses: bool | None = None,
        transcript_max_file_mb: int | None = None,
        transcript_enable_chunking: bool | None = None,
        transcript_max_video_seconds: int | None = None,
        transcript_max_attachment_mb: int | None = None,
        redis_url: str | None = None,
        handwriting_api_url: str | None = None,
        handwriting_api_timeout_seconds: int | None = None,
        handwriting_api_max_retries: int | None = None,
        digits_max_image_mb: int | None = None,
        digits_public_responses: bool | None = None,
        digits_rate_limit: int | None = None,
        digits_rate_window_seconds: int | None = None,
        model_trainer_api_url: str | None = None,
        model_trainer_api_key: str | None = None,
        model_trainer_api_timeout_seconds: int | None = None,
        model_trainer_api_max_retries: int | None = None,
        model_trainer_rate_limit: int | None = None,
        model_trainer_rate_window_seconds: int | None = None,
    ) -> DiscordbotSettings: ...


def _resolve_optional_url(value: str | None, default: str) -> str | None:
    """Resolve optional URL: None=default, empty string=explicitly None."""
    if value == "":
        return None
    if value is not None:
        return value
    return default


def _resolve_required_url(value: str | None, default: str) -> str:
    """Resolve required URL: None=default, empty string=empty (for testing)."""
    if value is not None:
        return value
    return default


def build_settings(
    *,
    discord_token: str | None = None,
    guild_id: str | None = None,
    guild_ids: list[int] | None = None,
    commands_sync_global: bool | None = None,
    qr_api_url: str | None = None,
    qr_rate_limit: int | None = None,
    qr_rate_window_seconds: int | None = None,
    qr_default_border: int | None = None,
    qr_default_box_size: int | None = None,
    qr_default_fill_color: str | None = None,
    qr_default_back_color: str | None = None,
    qr_public_responses: bool | None = None,
    transcript_api_url: str | None = None,
    transcript_provider: str | None = None,
    transcript_preferred_langs: str | None = None,
    transcript_public_responses: bool | None = None,
    transcript_max_file_mb: int | None = None,
    transcript_enable_chunking: bool | None = None,
    transcript_max_video_seconds: int | None = None,
    transcript_max_attachment_mb: int | None = None,
    redis_url: str | None = None,
    handwriting_api_url: str | None = None,
    handwriting_api_timeout_seconds: int | None = None,
    handwriting_api_max_retries: int | None = None,
    digits_max_image_mb: int | None = None,
    digits_public_responses: bool | None = None,
    digits_rate_limit: int | None = None,
    digits_rate_window_seconds: int | None = None,
    model_trainer_api_url: str | None = None,
    model_trainer_api_key: str | None = None,
    model_trainer_api_timeout_seconds: int | None = None,
    model_trainer_api_max_retries: int | None = None,
    model_trainer_rate_limit: int | None = None,
    model_trainer_rate_window_seconds: int | None = None,
) -> DiscordbotSettings:
    """Build DiscordbotSettings with test defaults."""
    token = discord_token if discord_token is not None else _BASE_DISCORD_TOKEN

    sync_global = commands_sync_global if commands_sync_global is not None else False

    discord_cfg: DiscordConfig = {
        "token": token,
        "guild_id": guild_id,
        "guild_ids": guild_ids if guild_ids is not None else [],
        "log_level": "INFO",
        "commands_sync_global": sync_global,
    }

    qr_url = qr_api_url if qr_api_url is not None else _BASE_QR_API_URL
    qr_rate = qr_rate_limit if qr_rate_limit is not None else 1
    qr_window = qr_rate_window_seconds if qr_rate_window_seconds is not None else 60
    qr_box = qr_default_box_size if qr_default_box_size is not None else 10
    qr_border = qr_default_border if qr_default_border is not None else 1
    qr_fill = qr_default_fill_color or "#000000"
    qr_back = qr_default_back_color or "#FFFFFF"
    qr_public = qr_public_responses if qr_public_responses is not None else False

    qr_cfg: QRConfig = {
        "api_url": qr_url,
        "rate_limit": qr_rate,
        "rate_window_seconds": qr_window,
        "default_error_correction": "M",
        "default_box_size": qr_box,
        "default_border": qr_border,
        "default_fill_color": qr_fill,
        "default_back_color": qr_back,
        "public_responses": qr_public,
    }

    t_url = _resolve_required_url(transcript_api_url, _BASE_TRANSCRIPT_API_URL)
    t_prov = transcript_provider if transcript_provider is not None else "api"
    t_langs = transcript_preferred_langs or "en,en-US"
    t_public = transcript_public_responses if transcript_public_responses is not None else False
    t_max_file_mb = transcript_max_file_mb if transcript_max_file_mb is not None else 25
    t_enable_chunking = (
        transcript_enable_chunking if transcript_enable_chunking is not None else True
    )
    t_max_video_seconds = (
        transcript_max_video_seconds if transcript_max_video_seconds is not None else 5400
    )
    t_max_attachment_mb = (
        transcript_max_attachment_mb if transcript_max_attachment_mb is not None else 25
    )

    transcript_cfg: TranscriptConfig = {
        "api_url": t_url,
        "provider": t_prov,
        "public_responses": t_public,
        "rate_limit": 2,
        "rate_window_seconds": 60,
        "preferred_langs": t_langs,
        "max_message_chars": 1800,
        "max_video_seconds": t_max_video_seconds,
        "max_file_mb": t_max_file_mb,
        "stt_rtf": 0.5,
        "dl_mib_per_sec": 4.0,
        "stt_api_timeout_seconds": 900,
        "stt_api_max_retries": 2,
        "max_attachment_mb": t_max_attachment_mb,
        "estimated_text_kb_per_min": 1.0,
        "enable_chunking": t_enable_chunking,
        "chunk_threshold_mb": 20.0,
        "target_chunk_mb": 20.0,
        "max_chunk_duration_seconds": 600.0,
        "max_concurrent_chunks": 3,
        "silence_threshold_db": -40.0,
        "silence_duration_seconds": 0.5,
        "cookies_text": None,
        "cookies_path": None,
        "youtube_api_key": None,
        "openai_api_key": None,
    }

    redis_cfg: RedisConfig = {
        "redis_url": redis_url,
        "job_queue_brpop_timeout_seconds": 0,
        "rq_transcript_job_timeout_sec": 600,
        "rq_transcript_result_ttl_sec": 86400,
        "rq_transcript_failure_ttl_sec": 604800,
        "rq_transcript_retry_max": 2,
        "rq_transcript_retry_intervals_sec": (60, 300),
    }

    hw_url = _resolve_optional_url(handwriting_api_url, _BASE_HANDWRITING_API_URL)
    hw_timeout = handwriting_api_timeout_seconds or 5
    hw_retries = handwriting_api_max_retries or 1

    handwriting_cfg: HandwritingConfig = {
        "api_url": hw_url,
        "api_key": None,
        "api_timeout_seconds": hw_timeout,
        "api_max_retries": hw_retries,
    }

    digits_mb = digits_max_image_mb if digits_max_image_mb is not None else 2
    digits_public = digits_public_responses if digits_public_responses is not None else False
    digits_rate = digits_rate_limit if digits_rate_limit is not None else 2
    digits_window = digits_rate_window_seconds if digits_rate_window_seconds is not None else 60

    digits_cfg: DigitsConfig = {
        "public_responses": digits_public,
        "rate_limit": digits_rate,
        "rate_window_seconds": digits_window,
        "max_image_mb": digits_mb,
    }

    trainer_url = _resolve_optional_url(model_trainer_api_url, _BASE_MODEL_TRAINER_API_URL)
    trainer_timeout = (
        model_trainer_api_timeout_seconds if model_trainer_api_timeout_seconds is not None else 10
    )
    trainer_retries = (
        model_trainer_api_max_retries if model_trainer_api_max_retries is not None else 1
    )
    trainer_rate = model_trainer_rate_limit if model_trainer_rate_limit is not None else 1
    trainer_window = (
        model_trainer_rate_window_seconds if model_trainer_rate_window_seconds is not None else 10
    )

    model_trainer_cfg: ModelTrainerConfig = {
        "api_url": trainer_url,
        "api_key": model_trainer_api_key,
        "api_timeout_seconds": trainer_timeout,
        "api_max_retries": trainer_retries,
        "rate_limit": trainer_rate,
        "rate_window_seconds": trainer_window,
    }

    return {
        "discord": discord_cfg,
        "qr": qr_cfg,
        "transcript": transcript_cfg,
        "redis": redis_cfg,
        "handwriting": handwriting_cfg,
        "digits": digits_cfg,
        "model_trainer": model_trainer_cfg,
    }


def make_settings_factory() -> SettingsFactory:
    """Create a settings factory function."""

    def factory(
        *,
        discord_token: str | None = None,
        guild_id: str | None = None,
        guild_ids: list[int] | None = None,
        commands_sync_global: bool | None = None,
        qr_api_url: str | None = None,
        qr_rate_limit: int | None = None,
        qr_rate_window_seconds: int | None = None,
        qr_default_border: int | None = None,
        qr_default_box_size: int | None = None,
        qr_default_fill_color: str | None = None,
        qr_default_back_color: str | None = None,
        qr_public_responses: bool | None = None,
        transcript_api_url: str | None = None,
        transcript_provider: str | None = None,
        transcript_preferred_langs: str | None = None,
        transcript_public_responses: bool | None = None,
        transcript_max_file_mb: int | None = None,
        transcript_enable_chunking: bool | None = None,
        transcript_max_video_seconds: int | None = None,
        transcript_max_attachment_mb: int | None = None,
        redis_url: str | None = None,
        handwriting_api_url: str | None = None,
        handwriting_api_timeout_seconds: int | None = None,
        handwriting_api_max_retries: int | None = None,
        digits_max_image_mb: int | None = None,
        digits_public_responses: bool | None = None,
        digits_rate_limit: int | None = None,
        digits_rate_window_seconds: int | None = None,
        model_trainer_api_url: str | None = None,
        model_trainer_api_key: str | None = None,
        model_trainer_api_timeout_seconds: int | None = None,
        model_trainer_api_max_retries: int | None = None,
        model_trainer_rate_limit: int | None = None,
        model_trainer_rate_window_seconds: int | None = None,
    ) -> DiscordbotSettings:
        return build_settings(
            discord_token=discord_token,
            guild_id=guild_id,
            guild_ids=guild_ids,
            commands_sync_global=commands_sync_global,
            qr_api_url=qr_api_url,
            qr_rate_limit=qr_rate_limit,
            qr_rate_window_seconds=qr_rate_window_seconds,
            qr_default_border=qr_default_border,
            qr_default_box_size=qr_default_box_size,
            qr_default_fill_color=qr_default_fill_color,
            qr_default_back_color=qr_default_back_color,
            qr_public_responses=qr_public_responses,
            transcript_api_url=transcript_api_url,
            transcript_provider=transcript_provider,
            transcript_preferred_langs=transcript_preferred_langs,
            transcript_public_responses=transcript_public_responses,
            transcript_max_file_mb=transcript_max_file_mb,
            transcript_enable_chunking=transcript_enable_chunking,
            transcript_max_video_seconds=transcript_max_video_seconds,
            transcript_max_attachment_mb=transcript_max_attachment_mb,
            redis_url=redis_url,
            handwriting_api_url=handwriting_api_url,
            handwriting_api_timeout_seconds=handwriting_api_timeout_seconds,
            handwriting_api_max_retries=handwriting_api_max_retries,
            digits_max_image_mb=digits_max_image_mb,
            digits_public_responses=digits_public_responses,
            digits_rate_limit=digits_rate_limit,
            digits_rate_window_seconds=digits_rate_window_seconds,
            model_trainer_api_url=model_trainer_api_url,
            model_trainer_api_key=model_trainer_api_key,
            model_trainer_api_timeout_seconds=model_trainer_api_timeout_seconds,
            model_trainer_api_max_retries=model_trainer_api_max_retries,
            model_trainer_rate_limit=model_trainer_rate_limit,
            model_trainer_rate_window_seconds=model_trainer_rate_window_seconds,
        )

    return factory
