"""DiscordBot's settings, re-exported from platform_core.

The types come from ``platform_core.config.discordbot`` by their own names --
the barrel no longer holds per-service types, because keeping every service's
side by side is what forced six of the seven to carry a service-name prefix.

``load_discordbot_settings`` is NOT a rename of the loader: it dispatches
through ``_test_hooks``, which is the seam a test replaces. That is a wrapper
doing the one job a wrapper is for.
"""

from __future__ import annotations

from platform_core.config import require_discord_token as require_discord_token
from platform_core.config.discordbot import DigitsConfig as DigitsConfig
from platform_core.config.discordbot import DiscordConfig as DiscordConfig
from platform_core.config.discordbot import GatewayConfig as GatewayConfig
from platform_core.config.discordbot import HandwritingConfig as HandwritingConfig
from platform_core.config.discordbot import ModelTrainerConfig as ModelTrainerConfig
from platform_core.config.discordbot import QRConfig as QRConfig
from platform_core.config.discordbot import RedisConfig as RedisConfig
from platform_core.config.discordbot import Settings as Settings
from platform_core.config.discordbot import TranscriptConfig as TranscriptConfig

from . import _test_hooks


def load_discordbot_settings() -> Settings:
    """Load DiscordBot settings via hook (allows test injection).

    Returns:
        The settings the installed hook produces.
    """
    return _test_hooks.load_settings()


__all__ = [
    "DigitsConfig",
    "DiscordConfig",
    "GatewayConfig",
    "HandwritingConfig",
    "ModelTrainerConfig",
    "QRConfig",
    "RedisConfig",
    "Settings",
    "TranscriptConfig",
    "load_discordbot_settings",
    "require_discord_token",
]
