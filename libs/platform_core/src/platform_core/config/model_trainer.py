"""Model-Trainer's configuration.

The names here are unprefixed because the module already says whose they are.
They carried a ``ModelTrainer`` prefix until 2026-08-26, for one reason: the
package barrel re-exported every service's types side by side, where three
different ``RedisConfig`` would have collided. The prefix paid for that flat
namespace at every call site, and Model-Trainer paid it back by aliasing the
prefix away again in its own ``core/config/settings.py``.

The barrel no longer re-exports these, so import from this module by name:

    from platform_core.config.model_trainer import Settings, load_settings
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.logging import LogLevel

from ._utils import (
    _parse_bool,
    _parse_int,
    _parse_str,
)


class LoggingConfig(TypedDict, total=True):
    level: LogLevel


class RedisConfig(TypedDict, total=True):
    enabled: bool
    url: str


class RQConfig(TypedDict, total=True):
    queue_name: str
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int
    retry_max: int
    retry_intervals_sec: str


class CleanupConfig(TypedDict, total=True):
    enabled: bool
    verify_upload: bool
    grace_period_seconds: int
    dry_run: bool


class CorpusCacheCleanupConfig(TypedDict, total=True):
    enabled: bool
    max_bytes: int
    min_free_bytes: int
    eviction_policy: Literal["lru", "oldest"]


class TokenizerCleanupConfig(TypedDict, total=True):
    enabled: bool
    min_unused_days: int


class AppConfig(TypedDict, total=True):
    data_root: str
    artifacts_root: str
    runs_root: str
    logs_root: str
    threads: int
    tokenizer_sample_max_lines: int
    data_bank_api_url: str
    data_bank_api_key: str
    cleanup: CleanupConfig
    corpus_cache_cleanup: CorpusCacheCleanupConfig
    tokenizer_cleanup: TokenizerCleanupConfig


class SecurityConfig(TypedDict, total=True):
    api_key: str


class WandbConfig(TypedDict, total=True):
    enabled: bool
    project: str


class Settings(TypedDict, total=True):
    app_env: Literal["dev", "prod"]
    logging: LoggingConfig
    redis: RedisConfig
    rq: RQConfig
    app: AppConfig
    security: SecurityConfig
    wandb: WandbConfig


def load_settings() -> Settings:
    """Read Model-Trainer's settings from the process environment.

    Returns:
        The settings, with every unset variable at its documented default.
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

    logging_cfg: LoggingConfig = {
        "level": level,
    }

    redis_cfg: RedisConfig = {
        "enabled": _parse_bool("REDIS__ENABLED", True),
        "url": _parse_str("REDIS__URL", "redis://redis:6379/0"),
    }
    rq_cfg: RQConfig = {
        "queue_name": _parse_str("RQ__QUEUE_NAME", "trainer"),
        "job_timeout_sec": _parse_int("RQ__JOB_TIMEOUT_SEC", 86_400),
        "result_ttl_sec": _parse_int("RQ__RESULT_TTL_SEC", 86_400),
        "failure_ttl_sec": _parse_int("RQ__FAILURE_TTL_SEC", 7 * 86_400),
        "retry_max": _parse_int("RQ__RETRY_MAX", 1),
        "retry_intervals_sec": _parse_str("RQ__RETRY_INTERVALS_SEC", "300"),
    }

    cleanup_cfg: CleanupConfig = {
        "enabled": _parse_bool("APP__CLEANUP__ENABLED", True),
        "verify_upload": _parse_bool("APP__CLEANUP__VERIFY_UPLOAD", True),
        "grace_period_seconds": _parse_int("APP__CLEANUP__GRACE_PERIOD_SECONDS", 0),
        "dry_run": _parse_bool("APP__CLEANUP__DRY_RUN", False),
    }
    corpus_cache_cleanup_cfg: CorpusCacheCleanupConfig = {
        "enabled": _parse_bool("APP__CORPUS_CACHE_CLEANUP__ENABLED", False),
        "max_bytes": _parse_int("APP__CORPUS_CACHE_CLEANUP__MAX_BYTES", 10 * 1024 * 1024 * 1024),
        "min_free_bytes": _parse_int(
            "APP__CORPUS_CACHE_CLEANUP__MIN_FREE_BYTES", 2 * 1024 * 1024 * 1024
        ),
        "eviction_policy": (
            "oldest"
            if _parse_str("APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY", "lru") == "oldest"
            else "lru"
        ),
    }
    tokenizer_cleanup_cfg: TokenizerCleanupConfig = {
        "enabled": _parse_bool("APP__TOKENIZER_CLEANUP__ENABLED", False),
        "min_unused_days": _parse_int("APP__TOKENIZER_CLEANUP__MIN_UNUSED_DAYS", 30),
    }

    gateway_url = _parse_str("API_GATEWAY_URL", "")
    direct_url = _parse_str("APP__DATA_BANK_API_URL", "")
    data_bank_url = f"{gateway_url}/data-bank" if gateway_url else direct_url

    app_cfg: AppConfig = {
        "data_root": _parse_str("APP__DATA_ROOT", "/data"),
        "artifacts_root": _parse_str("APP__ARTIFACTS_ROOT", "/data/artifacts"),
        "runs_root": _parse_str("APP__RUNS_ROOT", "/data/runs"),
        "logs_root": _parse_str("APP__LOGS_ROOT", "/data/logs"),
        "threads": _parse_int("APP__THREADS", 0),
        "tokenizer_sample_max_lines": _parse_int("APP__TOKENIZER_SAMPLE_MAX_LINES", 10000),
        "data_bank_api_url": data_bank_url,
        "data_bank_api_key": _parse_str("APP__DATA_BANK_API_KEY", ""),
        "cleanup": cleanup_cfg,
        "corpus_cache_cleanup": corpus_cache_cleanup_cfg,
        "tokenizer_cleanup": tokenizer_cleanup_cfg,
    }

    security_cfg: SecurityConfig = {
        "api_key": _parse_str("SECURITY__API_KEY", ""),
    }

    wandb_cfg: WandbConfig = {
        "enabled": _parse_bool("WANDB__ENABLED", False),
        "project": _parse_str("WANDB__PROJECT", "model-trainer"),
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
        "wandb": wandb_cfg,
    }


__all__ = [
    "AppConfig",
    "CleanupConfig",
    "CorpusCacheCleanupConfig",
    "LoggingConfig",
    "RQConfig",
    "RedisConfig",
    "SecurityConfig",
    "Settings",
    "TokenizerCleanupConfig",
    "WandbConfig",
    "load_settings",
]
