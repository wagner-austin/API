from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.logging import LogLevel

from ._utils import _parse_bool, _parse_int, _parse_str, _require_env_str

# ML backend type - matches covenant_ml.types.BackendName
MLBackend = Literal["xgboost", "mlp", "lstm", "lightgbm"]


class LoggingConfig(TypedDict, total=True):
    """Logging configuration."""

    level: LogLevel


class RedisConfig(TypedDict, total=True):
    """Redis connection configuration."""

    enabled: bool
    url: str


class RQConfig(TypedDict, total=True):
    """RQ job queue configuration."""

    queue_name: str
    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int


class DatadogConfig(TypedDict, total=True):
    """Datadog APM and metrics configuration.

    Fields:
        enabled: Whether Datadog integration is enabled.
        service: Service name for traces and metrics.
        env: Environment name (dev, staging, production).
        version: Service version for trace filtering.
        agent_host: Datadog agent host for DogStatsD.
        dogstatsd_port: DogStatsD UDP port.
        trace_enabled: Whether APM tracing is enabled.
    """

    enabled: bool
    service: str
    env: Literal["dev", "staging", "production"]
    version: str
    agent_host: str
    dogstatsd_port: int
    trace_enabled: bool


class AppConfig(TypedDict, total=True):
    """Application configuration."""

    data_root: str
    models_root: str
    logs_root: str
    ml_backend: MLBackend
    active_model_path_xgb: str
    active_model_path_mlp: str
    data_bank_api_url: str
    data_bank_api_key: str
    data_bank_model_file_id: str


class Settings(TypedDict, total=True):
    """Configuration for covenant-radar-api service."""

    app_env: Literal["dev", "prod"]
    logging: LoggingConfig
    redis: RedisConfig
    rq: RQConfig
    app: AppConfig
    datadog: DatadogConfig
    database_url: str


def _parse_ml_backend(env_var: str, default: MLBackend) -> MLBackend:
    """Parse ML backend from environment variable."""
    value = _parse_str(env_var, default)
    if value == "xgboost":
        return "xgboost"
    if value == "mlp":
        return "mlp"
    if value == "lstm":
        return "lstm"
    if value == "lightgbm":
        return "lightgbm"
    raise ValueError(f"{env_var} must be 'xgboost', 'mlp', 'lstm', or 'lightgbm', got '{value}'")


DatadogEnv = Literal["dev", "staging", "production"]


def _parse_datadog_env(env_var: str, default: DatadogEnv) -> DatadogEnv:
    """Parse Datadog environment from environment variable."""
    value = _parse_str(env_var, default)
    if value == "dev":
        return "dev"
    if value == "staging":
        return "staging"
    if value == "production":
        return "production"
    raise ValueError(f"{env_var} must be 'dev', 'staging', or 'production', got '{value}'")


def load_settings() -> Settings:
    """Load covenant-radar settings from environment variables.

    Environment variables:
        APP_ENV: Application environment (dev/prod, default: dev)
        LOGGING__LEVEL: Log level (default: INFO)
        REDIS__ENABLED: Enable Redis (default: true)
        REDIS_URL: Redis connection URL (default: redis://redis:6379/0). The
            name this service's own compose file and .env set; there is no
            second spelling to fall back to.
        RQ__QUEUE_NAME: RQ queue name (default: covenant)
        RQ__JOB_TIMEOUT_SEC: Job timeout in seconds (default: 3600)
        RQ__RESULT_TTL_SEC: Result TTL in seconds (default: 86400)
        RQ__FAILURE_TTL_SEC: Failure TTL in seconds (default: 604800)
        APP__DATA_ROOT: Data root directory (default: /data)
        APP__MODELS_ROOT: Models directory (default: /data/models)
        APP__LOGS_ROOT: Logs directory (default: /data/logs)
        APP__ML_BACKEND: ML backend for inference (xgboost/mlp/lstm/lightgbm, default: xgboost)
        APP__ACTIVE_MODEL_PATH_XGB: Active XGBoost model path (default: /data/models/active_xgb.ubj)
        APP__ACTIVE_MODEL_PATH_MLP: Active MLP model path (default: /data/models/active_mlp.pt)
        DATA_BANK_API_URL: URL for data-bank-api (default: empty)
        DATA_BANK_API_KEY: API key for data-bank-api (default: empty)
        DATA_BANK_MODEL_FILE_ID: Model file_id to download from data-bank (default: empty)
        DATADOG__ENABLED: Enable Datadog integration (default: false)
        DATADOG__SERVICE: Service name for traces (default: covenant-radar-api)
        DATADOG__ENV: Environment name (dev/staging/production, default: dev)
        DATADOG__VERSION: Service version (default: 0.0.0)
        DATADOG__AGENT_HOST: Datadog agent host (default: localhost)
        DATADOG__DOGSTATSD_PORT: DogStatsD port (default: 8125)
        DATADOG__TRACE_ENABLED: Enable APM tracing (default: true)
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

    logging_cfg: LoggingConfig = {
        "level": level,
    }

    redis_cfg: RedisConfig = {
        "enabled": _parse_bool("REDIS__ENABLED", True),
        "url": _parse_str("REDIS_URL", "redis://redis:6379/0"),
    }

    rq_cfg: RQConfig = {
        "queue_name": _parse_str("RQ__QUEUE_NAME", "covenant"),
        "job_timeout_sec": _parse_int("RQ__JOB_TIMEOUT_SEC", 3600),
        "result_ttl_sec": _parse_int("RQ__RESULT_TTL_SEC", 86_400),
        "failure_ttl_sec": _parse_int("RQ__FAILURE_TTL_SEC", 7 * 86_400),
    }

    # Parse ML backend and backend-specific active model paths
    ml_backend = _parse_ml_backend("APP__ML_BACKEND", "xgboost")
    active_model_path_xgb = _parse_str("APP__ACTIVE_MODEL_PATH_XGB", "/data/models/active_xgb.ubj")
    active_model_path_mlp = _parse_str("APP__ACTIVE_MODEL_PATH_MLP", "/data/models/active_mlp.pt")

    app_cfg: AppConfig = {
        "data_root": _parse_str("APP__DATA_ROOT", "/data"),
        "models_root": _parse_str("APP__MODELS_ROOT", "/data/models"),
        "logs_root": _parse_str("APP__LOGS_ROOT", "/data/logs"),
        "ml_backend": ml_backend,
        "active_model_path_xgb": active_model_path_xgb,
        "active_model_path_mlp": active_model_path_mlp,
        "data_bank_api_url": _parse_str("DATA_BANK_API_URL", ""),
        "data_bank_api_key": _parse_str("DATA_BANK_API_KEY", ""),
        "data_bank_model_file_id": _parse_str("DATA_BANK_MODEL_FILE_ID", ""),
    }

    datadog_cfg: DatadogConfig = {
        "enabled": _parse_bool("DATADOG__ENABLED", False),
        "service": _parse_str("DATADOG__SERVICE", "covenant-radar-api"),
        "env": _parse_datadog_env("DATADOG__ENV", "dev"),
        "version": _parse_str("DATADOG__VERSION", "0.0.0"),
        "agent_host": _parse_str("DATADOG__AGENT_HOST", "localhost"),
        "dogstatsd_port": _parse_int("DATADOG__DOGSTATSD_PORT", 8125),
        "trace_enabled": _parse_bool("DATADOG__TRACE_ENABLED", True),
    }

    app_env_str = _parse_str("APP_ENV", "dev")
    app_env: Literal["dev", "prod"] = "prod" if app_env_str == "prod" else "dev"

    return {
        "app_env": app_env,
        "logging": logging_cfg,
        "redis": redis_cfg,
        "rq": rq_cfg,
        "app": app_cfg,
        "datadog": datadog_cfg,
        # Required, not defaulted: psycopg treats an empty DSN as "connect to
        # the local socket using PG* defaults", so a deployment that forgets
        # DATABASE_URL would silently attach to whatever Postgres is nearby
        # instead of failing. streaming_worker_entry and scripts/seed already
        # hard-fail on this variable; this makes the API agree with them.
        "database_url": _require_env_str("DATABASE_URL"),
    }


__all__ = [
    "AppConfig",
    "DatadogConfig",
    "LoggingConfig",
    "MLBackend",
    "RQConfig",
    "RedisConfig",
    "Settings",
    "load_settings",
]
