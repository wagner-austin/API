"""Shared configuration helpers, and nothing service-specific.

This barrel used to re-export every service's config types side by side. That
flat namespace is the only reason those types carried their service name as a
prefix: three ``RedisConfig`` cannot live in one module, so six of the seven
services spelled theirs ``ArtTrainerRedisConfig``,
``CovenantRadarRedisConfig``, ``ModelTrainerRedisConfig``, and paid for it at
every call site. The seventh, ``discordbot``, arrived first and kept the plain
names -- which is the whole argument in one observation: the prefix encoded
import order, not meaning.

Model-Trainer and Art-Trainer then aliased the prefix back off again in their
own settings modules (``Settings = ModelTrainerSettings``), so the cost was
paid twice and the names were ambiguous in both directions.

So the per-service types are gone from here and each module owns unprefixed
names. Import from the module that says whose they are:

    from platform_core.config.model_trainer import Settings, load_settings
    from platform_core.config.art_trainer import Settings as ArtSettings

The module path carries the service, which is what a path is for. What stays
here is what is genuinely shared: env parsing helpers and the test hooks.
"""

from __future__ import annotations

from . import _test_hooks as config_test_hooks
from ._test_hooks import _default_get_env
from ._utils import (
    LogFormat,
    LogLevel,
    _decode_table,
    _decode_toml,
    _optional_env_str,
    _parse_bool,
    _parse_float,
    _parse_int,
    _parse_log_format,
    _parse_log_level,
    _parse_str,
    _require_env_csv,
    _require_env_str,
    _validate_log_format,
    _validate_log_level,
)
from .covenant_radar import MLBackend
from .discordbot import require_discord_token

__all__ = [
    "LogFormat",
    "LogLevel",
    "MLBackend",
    "_decode_table",
    "_decode_toml",
    "_default_get_env",
    "_optional_env_str",
    "_parse_bool",
    "_parse_float",
    "_parse_int",
    "_parse_log_format",
    "_parse_log_level",
    "_parse_str",
    "_require_env_csv",
    "_require_env_str",
    "_validate_log_format",
    "_validate_log_level",
    "config_test_hooks",
    "require_discord_token",
]
