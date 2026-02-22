"""Art-Trainer configuration types and loader.

This module defines the TypedDicts for Art-Trainer service configuration
and provides a settings loader from environment variables.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.logging import LogLevel

from ._utils import (
    _parse_bool,
    _parse_int,
    _parse_str,
)


class ArtTrainerLoggingConfig(TypedDict, total=True):
    """Logging configuration for Art-Trainer.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    level: LogLevel


class ArtTrainerRedisConfig(TypedDict, total=True):
    """Redis configuration for Art-Trainer.

    Attributes:
        enabled: Whether Redis is enabled.
        url: Redis connection URL.
    """

    enabled: bool
    url: str


class ArtTrainerRQConfig(TypedDict, total=True):
    """RQ (Redis Queue) configuration for Art-Trainer.

    Attributes:
        queue_name: Name of the RQ queue.
        job_timeout_sec: Job timeout in seconds.
        result_ttl_sec: Result TTL in seconds.
        failure_ttl_sec: Failure record TTL in seconds.
        retry_max: Maximum retry attempts.
        retry_intervals_sec: Comma-separated retry intervals.
    """

    queue_name: str
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int
    retry_max: int
    retry_intervals_sec: str


class ArtTrainerAppConfig(TypedDict, total=True):
    """Application configuration for Art-Trainer.

    Attributes:
        data_root: Root directory for data files.
        output_root: Root directory for training outputs.
        logs_root: Root directory for log files.
        data_bank_api_url: URL for data-bank API.
        data_bank_api_key: API key for data-bank.
        kohya_ss_path: Path to Kohya_ss installation.
        comfyui_lora_path: Path to ComfyUI LoRA models directory.
        blip_model_name: BLIP model name for auto-captioning.
        caption_trigger_word: Default trigger word for captions.
        gemini_api_key: API key for Google Gemini.
        openai_api_key: API key for OpenAI.
    """

    data_root: str
    output_root: str
    logs_root: str
    data_bank_api_url: str
    data_bank_api_key: str
    kohya_ss_path: str
    comfyui_lora_path: str
    blip_model_name: str
    caption_trigger_word: str
    gemini_api_key: str
    openai_api_key: str


class ArtTrainerSecurityConfig(TypedDict, total=True):
    """Security configuration for Art-Trainer.

    Attributes:
        api_key: API key for authentication.
    """

    api_key: str


class ArtTrainerSettings(TypedDict, total=True):
    """Complete settings for Art-Trainer service.

    Attributes:
        app_env: Application environment (dev or prod).
        logging: Logging configuration.
        redis: Redis configuration.
        rq: RQ configuration.
        app: Application configuration.
        security: Security configuration.
    """

    app_env: Literal["dev", "prod"]
    logging: ArtTrainerLoggingConfig
    redis: ArtTrainerRedisConfig
    rq: ArtTrainerRQConfig
    app: ArtTrainerAppConfig
    security: ArtTrainerSecurityConfig


def load_art_trainer_settings() -> ArtTrainerSettings:
    """Load Art-Trainer settings from environment variables.

    Returns:
        Complete ArtTrainerSettings TypedDict.
    """
    level_str = _parse_str("LOGGING__LEVEL", "INFO")
    level: LogLevel = "INFO"
    if level_str == "DEBUG":
        level = "DEBUG"
    elif level_str == "INFO":
        level = "INFO"
    elif level_str == "WARNING":
        level = "WARNING"
    elif level_str == "ERROR":
        level = "ERROR"
    elif level_str == "CRITICAL":
        level = "CRITICAL"

    logging_cfg: ArtTrainerLoggingConfig = {
        "level": level,
    }

    redis_cfg: ArtTrainerRedisConfig = {
        "enabled": _parse_bool("REDIS__ENABLED", True),
        "url": _parse_str("REDIS__URL", "redis://redis:6379/0"),
    }

    rq_cfg: ArtTrainerRQConfig = {
        "queue_name": _parse_str("RQ__QUEUE_NAME", "art-trainer"),
        "job_timeout_sec": _parse_int("RQ__JOB_TIMEOUT_SEC", 86_400),
        "result_ttl_sec": _parse_int("RQ__RESULT_TTL_SEC", 86_400),
        "failure_ttl_sec": _parse_int("RQ__FAILURE_TTL_SEC", 7 * 86_400),
        "retry_max": _parse_int("RQ__RETRY_MAX", 1),
        "retry_intervals_sec": _parse_str("RQ__RETRY_INTERVALS_SEC", "300"),
    }

    gateway_url = _parse_str("API_GATEWAY_URL", "")
    direct_url = _parse_str("APP__DATA_BANK_API_URL", "")
    data_bank_url = f"{gateway_url}/data-bank" if gateway_url else direct_url

    app_cfg: ArtTrainerAppConfig = {
        "data_root": _parse_str("APP__DATA_ROOT", "/data"),
        "output_root": _parse_str("APP__OUTPUT_ROOT", "/data/output"),
        "logs_root": _parse_str("APP__LOGS_ROOT", "/data/logs"),
        "data_bank_api_url": data_bank_url,
        "data_bank_api_key": _parse_str("APP__DATA_BANK_API_KEY", ""),
        "kohya_ss_path": _parse_str("APP__KOHYA_SS_PATH", "/opt/kohya_ss"),
        "comfyui_lora_path": _parse_str("APP__COMFYUI_LORA_PATH", "/opt/ComfyUI/models/loras"),
        "blip_model_name": _parse_str(
            "APP__BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-large"
        ),
        "caption_trigger_word": _parse_str("APP__CAPTION_TRIGGER_WORD", "sks person"),
        "gemini_api_key": _parse_str("GEMINI_API_KEY", ""),
        "openai_api_key": _parse_str("OPENAI_API_KEY", ""),
    }

    security_cfg: ArtTrainerSecurityConfig = {
        "api_key": _parse_str("SECURITY__API_KEY", ""),
    }

    app_env_str = _parse_str("APP_ENV", "dev")
    app_env: Literal["dev", "prod"] = "prod" if app_env_str == "prod" else "dev"

    return {
        "app_env": app_env,
        "logging": logging_cfg,
        "redis": redis_cfg,
        "rq": rq_cfg,
        "app": app_cfg,
        "security": security_cfg,
    }


__all__ = [
    "ArtTrainerAppConfig",
    "ArtTrainerLoggingConfig",
    "ArtTrainerRQConfig",
    "ArtTrainerRedisConfig",
    "ArtTrainerSecurityConfig",
    "ArtTrainerSettings",
    "load_art_trainer_settings",
]
