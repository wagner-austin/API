"""Settings configuration for Art-Trainer.

Re-exports the shared TypedDicts from platform_core for service use.
"""

from __future__ import annotations

from platform_core.config import (
    ArtTrainerAppConfig,
    ArtTrainerLoggingConfig,
    ArtTrainerRedisConfig,
    ArtTrainerRQConfig,
    ArtTrainerSecurityConfig,
    ArtTrainerSettings,
    load_art_trainer_settings,
)

# Re-export the shared TypedDicts so callers keep identical types.
LoggingConfig = ArtTrainerLoggingConfig
RedisConfig = ArtTrainerRedisConfig
RQConfig = ArtTrainerRQConfig
AppConfig = ArtTrainerAppConfig
SecurityConfig = ArtTrainerSecurityConfig
Settings = ArtTrainerSettings


def load_settings() -> Settings:
    """Load Art-Trainer settings from the centralized platform_core config.

    Returns:
        Complete Settings TypedDict.
    """
    return load_art_trainer_settings()


__all__ = [
    "AppConfig",
    "LoggingConfig",
    "RQConfig",
    "RedisConfig",
    "SecurityConfig",
    "Settings",
    "load_settings",
]
