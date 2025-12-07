from __future__ import annotations

from platform_core.config import (
    ModelTrainerAppConfig,
    ModelTrainerCleanupConfig,
    ModelTrainerCorpusCacheCleanupConfig,
    ModelTrainerLoggingConfig,
    ModelTrainerRedisConfig,
    ModelTrainerRQConfig,
    ModelTrainerSecurityConfig,
    ModelTrainerSettings,
    ModelTrainerTokenizerCleanupConfig,
    ModelTrainerWandbConfig,
    load_model_trainer_settings,
)

# Re-export the shared TypedDicts so callers keep identical types.
LoggingConfig = ModelTrainerLoggingConfig
RedisConfig = ModelTrainerRedisConfig
RQConfig = ModelTrainerRQConfig
CleanupConfig = ModelTrainerCleanupConfig
CorpusCacheCleanupConfig = ModelTrainerCorpusCacheCleanupConfig
TokenizerCleanupConfig = ModelTrainerTokenizerCleanupConfig
AppConfig = ModelTrainerAppConfig
SecurityConfig = ModelTrainerSecurityConfig
WandbConfig = ModelTrainerWandbConfig
Settings = ModelTrainerSettings


def load_settings() -> Settings:
    """Load Model-Trainer settings from the centralized platform_core config."""
    return load_model_trainer_settings()


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
