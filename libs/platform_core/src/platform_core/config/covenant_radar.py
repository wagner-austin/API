from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.logging import LogLevel

from ._utils import _parse_bool, _parse_int, _parse_str

# ML backend type - matches covenant_ml.types.BackendName
MLBackend = Literal["xgboost", "mlp"]


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
    ml_backend: MLBackend
    active_model_path_xgb: str
    active_model_path_mlp: str


class CovenantRadarSettings(TypedDict, total=True):
    """Configuration for covenant-radar-api service."""

    app_env: Literal["dev", "prod"]
    logging: CovenantRadarLoggingConfig
    redis: CovenantRadarRedisConfig
    rq: CovenantRadarRQConfig
    app: CovenantRadarAppConfig
    database_url: str


def _parse_ml_backend(env_var: str, default: MLBackend) -> MLBackend:
    """Parse ML backend from environment variable."""
    value = _parse_str(env_var, default)
    if value == "xgboost":
        return "xgboost"
    if value == "mlp":
        return "mlp"
    raise ValueError(f"{env_var} must be 'xgboost' or 'mlp', got '{value}'")


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
        APP__ML_BACKEND: ML backend for inference (xgboost/mlp, default: xgboost)
        APP__ACTIVE_MODEL_PATH_XGB: Active XGBoost model path (default: /data/models/active_xgb.ubj)
        APP__ACTIVE_MODEL_PATH_MLP: Active MLP model path (default: /data/models/active_mlp.pt)
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

    # Parse ML backend and backend-specific active model paths
    ml_backend = _parse_ml_backend("APP__ML_BACKEND", "xgboost")
    active_model_path_xgb = _parse_str("APP__ACTIVE_MODEL_PATH_XGB", "/data/models/active_xgb.ubj")
    active_model_path_mlp = _parse_str("APP__ACTIVE_MODEL_PATH_MLP", "/data/models/active_mlp.pt")

    # Resolve active_model_path based on backend for backward compatibility
    active_model_path = active_model_path_xgb if ml_backend == "xgboost" else active_model_path_mlp

    app_cfg: CovenantRadarAppConfig = {
        "data_root": _parse_str("APP__DATA_ROOT", "/data"),
        "models_root": _parse_str("APP__MODELS_ROOT", "/data/models"),
        "logs_root": _parse_str("APP__LOGS_ROOT", "/data/logs"),
        "active_model_path": active_model_path,
        "ml_backend": ml_backend,
        "active_model_path_xgb": active_model_path_xgb,
        "active_model_path_mlp": active_model_path_mlp,
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
    "MLBackend",
    "load_covenant_radar_settings",
]
