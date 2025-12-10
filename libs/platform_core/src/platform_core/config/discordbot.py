from __future__ import annotations

from typing import TypedDict

from ._utils import (
    LogLevel,
    _optional_env_str,
    _parse_bool,
    _parse_float,
    _parse_int,
    _parse_log_level,
    _parse_str,
)


class DiscordConfig(TypedDict, total=True):
    token: str
    guild_id: str | None
    guild_ids: list[int]
    log_level: LogLevel
    commands_sync_global: bool


class QRConfig(TypedDict, total=True):
    api_url: str
    rate_limit: int
    rate_window_seconds: int
    default_error_correction: str
    default_box_size: int
    default_border: int
    default_fill_color: str
    default_back_color: str
    public_responses: bool


class TranscriptConfig(TypedDict, total=True):
    api_url: str
    provider: str
    public_responses: bool
    rate_limit: int
    rate_window_seconds: int
    preferred_langs: str
    max_message_chars: int
    max_video_seconds: int
    max_file_mb: int
    stt_rtf: float
    dl_mib_per_sec: float
    stt_api_timeout_seconds: int
    stt_api_max_retries: int
    max_attachment_mb: int
    estimated_text_kb_per_min: float
    enable_chunking: bool
    chunk_threshold_mb: float
    target_chunk_mb: float
    max_chunk_duration_seconds: float
    max_concurrent_chunks: int
    silence_threshold_db: float
    silence_duration_seconds: float
    cookies_text: str | None
    cookies_path: str | None
    youtube_api_key: str | None
    openai_api_key: str | None


class RedisConfig(TypedDict, total=True):
    redis_url: str | None
    job_queue_brpop_timeout_seconds: int
    rq_transcript_job_timeout_sec: int
    rq_transcript_result_ttl_sec: int
    rq_transcript_failure_ttl_sec: int
    rq_transcript_retry_max: int
    rq_transcript_retry_intervals_sec: tuple[int, int]


class HandwritingConfig(TypedDict, total=True):
    api_url: str | None
    api_key: str | None
    api_timeout_seconds: int
    api_max_retries: int


class DigitsConfig(TypedDict, total=True):
    public_responses: bool
    rate_limit: int
    rate_window_seconds: int
    max_image_mb: int


class ModelTrainerConfig(TypedDict, total=True):
    api_url: str | None
    api_key: str | None
    api_timeout_seconds: int
    api_max_retries: int
    rate_limit: int
    rate_window_seconds: int


class GatewayConfig(TypedDict, total=True):
    api_url: str


class DiscordbotSettings(TypedDict, total=True):
    discord: DiscordConfig
    qr: QRConfig
    transcript: TranscriptConfig
    redis: RedisConfig
    handwriting: HandwritingConfig
    digits: DigitsConfig
    model_trainer: ModelTrainerConfig
    gateway: GatewayConfig


def load_discordbot_settings() -> DiscordbotSettings:
    single_guild = _optional_env_str("DISCORD_GUILD_ID")
    multi_raw = _optional_env_str("DISCORD_GUILD_IDS")
    guild_ids: list[int] = []
    if multi_raw is not None:
        guild_ids = [int(p) for p in multi_raw.replace(",", " ").split() if p.isdigit()]
    elif single_guild is not None and single_guild.isdigit():
        guild_ids = [int(single_guild)]

    retry_raw = _parse_str("RQ_TRANSCRIPT_RETRY_INTERVALS_SEC", "60,300")
    retry_parts = [p for p in retry_raw.replace(",", " ").split() if p]
    retry_intervals: tuple[int, int] = (
        (int(retry_parts[0]), int(retry_parts[1]))
        if len(retry_parts) >= 2 and retry_parts[0].isdigit() and retry_parts[1].isdigit()
        else (60, 300)
    )

    openai_key_primary = _optional_env_str("OPENAI_API_KEY")
    openai_key_alt = _optional_env_str("OPEN_AI_API_KEY")
    openai_key = openai_key_primary if openai_key_primary is not None else openai_key_alt

    discord_cfg: DiscordConfig = {
        "token": _parse_str("DISCORD_TOKEN", ""),
        "guild_id": single_guild,
        "guild_ids": guild_ids,
        "log_level": _parse_log_level("LOG_LEVEL", "INFO"),
        "commands_sync_global": _parse_bool("COMMANDS_SYNC_GLOBAL", False),
    }

    # --- Gateway URL ---
    gateway_base_url = _parse_str("API_GATEWAY_URL", "")

    qr_cfg: QRConfig = {
        "api_url": f"{gateway_base_url}/qr" if gateway_base_url else _parse_str("QR_API_URL", ""),
        "rate_limit": _parse_int("QRCODE_RATE_LIMIT", 1),
        "rate_window_seconds": _parse_int("QRCODE_RATE_WINDOW_SECONDS", 1),
        "default_error_correction": _parse_str("QR_DEFAULT_ERROR_CORRECTION", "M").upper(),
        "default_box_size": _parse_int("QR_DEFAULT_BOX_SIZE", 10),
        "default_border": _parse_int("QR_DEFAULT_BORDER", 1),
        "default_fill_color": _parse_str("QR_DEFAULT_FILL_COLOR", "#000000"),
        "default_back_color": _parse_str("QR_DEFAULT_BACK_COLOR", "#FFFFFF"),
        "public_responses": _parse_bool("QR_PUBLIC_RESPONSES", True),
    }

    transcript_direct_url = _parse_str("TRANSCRIPT_API_URL", "http://localhost:8000")
    transcript_cfg: TranscriptConfig = {
        "api_url": f"{gateway_base_url}/transcript" if gateway_base_url else transcript_direct_url,
        "provider": _parse_str("TRANSCRIPT_PROVIDER", "api").strip().lower(),
        "public_responses": _parse_bool("TRANSCRIPT_PUBLIC_RESPONSES", False),
        "rate_limit": _parse_int("TRANSCRIPT_RATE_LIMIT", 2),
        "rate_window_seconds": _parse_int("TRANSCRIPT_RATE_WINDOW_SECONDS", 60),
        "preferred_langs": _parse_str("TRANSCRIPT_PREFERRED_LANGS", "en,en-US,en-GB"),
        "max_message_chars": _parse_int("TRANSCRIPT_MAX_MESSAGE_CHARS", 1800),
        "max_video_seconds": 5400,
        "max_file_mb": _parse_int("TRANSCRIPT_MAX_FILE_MB", 25),
        "stt_rtf": _parse_float("TRANSCRIPT_STT_RTF", 0.5),
        "dl_mib_per_sec": _parse_float("TRANSCRIPT_DL_MIB_PER_SEC", 4.0),
        "stt_api_timeout_seconds": _parse_int("TRANSCRIPT_STT_API_TIMEOUT_SECONDS", 900),
        "stt_api_max_retries": _parse_int("TRANSCRIPT_STT_API_MAX_RETRIES", 2),
        "max_attachment_mb": _parse_int("TRANSCRIPT_MAX_ATTACHMENT_MB", 25),
        "estimated_text_kb_per_min": _parse_float("TRANSCRIPT_ESTIMATED_TEXT_KB_PER_MIN", 1.0),
        "enable_chunking": _parse_bool("TRANSCRIPT_ENABLE_CHUNKING", True),
        "chunk_threshold_mb": _parse_float("TRANSCRIPT_CHUNK_THRESHOLD_MB", 20.0),
        "target_chunk_mb": _parse_float("TRANSCRIPT_TARGET_CHUNK_MB", 20.0),
        "max_chunk_duration_seconds": _parse_float("TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS", 600.0),
        "max_concurrent_chunks": _parse_int("TRANSCRIPT_MAX_CONCURRENT_CHUNKS", 3),
        "silence_threshold_db": _parse_float("TRANSCRIPT_SILENCE_THRESHOLD_DB", -40.0),
        "silence_duration_seconds": _parse_float("TRANSCRIPT_SILENCE_DURATION_SECONDS", 0.5),
        "cookies_text": _optional_env_str("TRANSCRIPT_COOKIES_TEXT"),
        "cookies_path": _optional_env_str("TRANSCRIPT_COOKIES_PATH"),
        "youtube_api_key": _optional_env_str("YOUTUBE_API_KEY"),
        "openai_api_key": openai_key,
    }

    redis_cfg: RedisConfig = {
        "redis_url": _optional_env_str("REDIS_URL"),
        "job_queue_brpop_timeout_seconds": _parse_int("JOB_QUEUE_BRPOP_TIMEOUT_SECONDS", 0),
        "rq_transcript_job_timeout_sec": _parse_int("RQ_TRANSCRIPT_JOB_TIMEOUT_SEC", 600),
        "rq_transcript_result_ttl_sec": _parse_int("RQ_TRANSCRIPT_RESULT_TTL_SEC", 86400),
        "rq_transcript_failure_ttl_sec": _parse_int("RQ_TRANSCRIPT_FAILURE_TTL_SEC", 604800),
        "rq_transcript_retry_max": _parse_int("RQ_TRANSCRIPT_RETRY_MAX", 2),
        "rq_transcript_retry_intervals_sec": retry_intervals,
    }

    hw_direct_url = _optional_env_str("HANDWRITING_API_URL")
    handwriting_cfg: HandwritingConfig = {
        "api_url": f"{gateway_base_url}/handwriting" if gateway_base_url else hw_direct_url,
        "api_key": _optional_env_str("HANDWRITING_API_KEY"),
        "api_timeout_seconds": _parse_int("HANDWRITING_API_TIMEOUT_SECONDS", 5),
        "api_max_retries": _parse_int("HANDWRITING_API_MAX_RETRIES", 1),
    }

    digits_cfg: DigitsConfig = {
        "public_responses": _parse_bool("DIGITS_PUBLIC_RESPONSES", False),
        "rate_limit": _parse_int("DIGITS_RATE_LIMIT", 2),
        "rate_window_seconds": _parse_int("DIGITS_RATE_WINDOW_SECONDS", 60),
        "max_image_mb": _parse_int("DIGITS_MAX_IMAGE_MB", 2),
    }

    model_trainer_direct_url = _optional_env_str("MODEL_TRAINER_API_URL")
    model_trainer_cfg: ModelTrainerConfig = {
        "api_url": f"{gateway_base_url}/trainer" if gateway_base_url else model_trainer_direct_url,
        "api_key": _optional_env_str("MODEL_TRAINER_API_KEY"),
        "api_timeout_seconds": _parse_int("MODEL_TRAINER_API_TIMEOUT_SECONDS", 10),
        "api_max_retries": _parse_int("MODEL_TRAINER_API_MAX_RETRIES", 1),
        "rate_limit": _parse_int("MODEL_TRAINER_RATE_LIMIT", 1),
        "rate_window_seconds": _parse_int("MODEL_TRAINER_RATE_WINDOW_SECONDS", 10),
    }

    gateway_cfg: GatewayConfig = {
        "api_url": gateway_base_url,
    }

    return {
        "discord": discord_cfg,
        "qr": qr_cfg,
        "transcript": transcript_cfg,
        "redis": redis_cfg,
        "handwriting": handwriting_cfg,
        "digits": digits_cfg,
        "model_trainer": model_trainer_cfg,
        "gateway": gateway_cfg,
    }


def require_discord_token(settings: DiscordbotSettings) -> None:
    if not settings["discord"]["token"]:
        raise RuntimeError("DISCORD_TOKEN is required. Set it in your .env file.")


__all__ = [
    "DigitsConfig",
    "DiscordConfig",
    "DiscordbotSettings",
    "GatewayConfig",
    "HandwritingConfig",
    "LogLevel",
    "ModelTrainerConfig",
    "QRConfig",
    "RedisConfig",
    "TranscriptConfig",
    "load_discordbot_settings",
    "require_discord_token",
]
