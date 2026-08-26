"""Art-Trainer's settings, re-exported from platform_core.

Six aliases and a passthrough loader lived here, for the same reason
Model-Trainer's did: platform_core's barrel held every service's types in one
flat namespace, so each carried a service-name prefix, and each service
aliased the prefix off again at home. platform_core's modules now own
unprefixed names, so this is an explicit ``import X as X`` re-export -- one
name for one type.
"""

from __future__ import annotations

from platform_core.config.art_trainer import AppConfig as AppConfig
from platform_core.config.art_trainer import LoggingConfig as LoggingConfig
from platform_core.config.art_trainer import RedisConfig as RedisConfig
from platform_core.config.art_trainer import RQConfig as RQConfig
from platform_core.config.art_trainer import SecurityConfig as SecurityConfig
from platform_core.config.art_trainer import Settings as Settings
from platform_core.config.art_trainer import load_settings as load_settings

__all__ = [
    "AppConfig",
    "LoggingConfig",
    "RQConfig",
    "RedisConfig",
    "SecurityConfig",
    "Settings",
    "load_settings",
]
