from __future__ import annotations

from platform_core.config import (
    DigitsConfig,
    DiscordbotSettings,
    DiscordConfig,
    GatewayConfig,
    HandwritingConfig,
    ModelTrainerConfig,
    QRConfig,
    RedisConfig,
    TranscriptConfig,
    require_discord_token,
)

from . import _test_hooks


def load_discordbot_settings() -> DiscordbotSettings:
    """Load DiscordBot settings via hook (allows test injection)."""
    return _test_hooks.load_settings()


__all__ = [
    "DigitsConfig",
    "DiscordConfig",
    "DiscordbotSettings",
    "GatewayConfig",
    "HandwritingConfig",
    "ModelTrainerConfig",
    "QRConfig",
    "RedisConfig",
    "TranscriptConfig",
    "load_discordbot_settings",
    "require_discord_token",
]
