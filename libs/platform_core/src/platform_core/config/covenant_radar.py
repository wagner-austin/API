from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.logging import LogLevel

from ._utils import _parse_bool, _parse_int, _parse_str


class CovenantRadarLoggingConfig(TypedDict, total=True):
    """Logging configuration."""

    level: LogLevel


class CovenantRadarRedisConfig(TypedDict, total=True):
    """Redis connection configuration."""

    enabled: bool
    url: str


class CovenantRadarRQConfig(TypedDict, total=True):
    """RQ job queue configuration."""

    queue_name: str
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int


class CovenantRadarAppConfig(TypedDict, total=True):
    """Application configuration."""

    data_root: str
    models_root: str
    logs_root: str
    active_model_path: str


class CovenantRadarSettings(TypedDict, total=True):
    """Configuration for covenant-radar-api service."""

    app_env: Literal["dev", "prod"]
    logging: CovenantRadarLoggingConfig
    redis: CovenantRadarRedisConfig
    rq: CovenantRadarRQConfig
    app: CovenantRadarAppConfig
    database_url: str


def load_covenant_radar_settings() -> CovenantRadarSettings:
    """Load covenant-radar settings from environment variables.

    Environment variables:
        APP_ENV: Application environment (dev/prod, default: dev)
        LOGGING__LEVEL: Log level (default: INFO)
        REDIS__ENABLED: Enable Redis (default: true)
        REDIS__URL or REDIS_URL: Redis connection URL (default: redis://redis:6379/0)
        RQ__QUEUE_NAME: RQ queue name (default: covenant)
        RQ__JOB_TIMEOUT_SEC: Job timeout in seconds (default: 3600)
        RQ__RESULT_TTL_SEC: Result TTL in seconds (default: 86400)
        RQ__FAILURE_TTL_SEC: Failure TTL in seconds (default: 604800)
        APP__DATA_ROOT: Data root directory (default: /data)
        APP__MODELS_ROOT: Models directory (default: /data/models)
        APP__LOGS_ROOT: Logs directory (default: /data/logs)
        DATABASE_URL: PostgreSQL connection URL (required)
    """
    level_str = _parse_str("LOGGING__LEVEL", "INFO")
    level: LogLevel = "INFO"
    if level_str == "DEBUG":
        level = "DEBUG"
    elif level_str == "WARNING":
        level = "WARNING"
    elif level_str == "ERROR":
        level = "ERROR"
    elif level_str == "CRITICAL":
        level = "CRITICAL"

    logging_cfg: CovenantRadarLoggingConfig = {
        "level": level,
    }

    # Support both REDIS__URL and REDIS_URL for compatibility
    redis_url = _parse_str("REDIS__URL", "")
    if not redis_url:
        redis_url = _parse_str("REDIS_URL", "redis://redis:6379/0")

    redis_cfg: CovenantRadarRedisConfig = {
        "enabled": _parse_bool("REDIS__ENABLED", True),
        "url": redis_url,
    }

    rq_cfg: CovenantRadarRQConfig = {
        "queue_name": _parse_str("RQ__QUEUE_NAME", "covenant"),
        "job_timeout_sec": _parse_int("RQ__JOB_TIMEOUT_SEC", 3600),
        "result_ttl_sec": _parse_int("RQ__RESULT_TTL_SEC", 86_400),
        "failure_ttl_sec": _parse_int("RQ__FAILURE_TTL_SEC", 7 * 86_400),
    }

    app_cfg: CovenantRadarAppConfig = {
        "data_root": _parse_str("APP__DATA_ROOT", "/data"),
        "models_root": _parse_str("APP__MODELS_ROOT", "/data/models"),
        "logs_root": _parse_str("APP__LOGS_ROOT", "/data/logs"),
        "active_model_path": _parse_str("APP__ACTIVE_MODEL_PATH", "/data/models/active.ubj"),
    }

    app_env_str = _parse_str("APP_ENV", "dev")
    app_env: Literal["dev", "prod"] = "prod" if app_env_str == "prod" else "dev"

    return {
        "app_env": app_env,
        "logging": logging_cfg,
        "redis": redis_cfg,
        "rq": rq_cfg,
        "app": app_cfg,
        "database_url": _parse_str("DATABASE_URL", ""),
    }


__all__ = [
    "CovenantRadarAppConfig",
    "CovenantRadarLoggingConfig",
    "CovenantRadarRQConfig",
    "CovenantRadarRedisConfig",
    "CovenantRadarSettings",
    "load_covenant_radar_settings",
]
