"""Model-Trainer's settings, re-exported from platform_core.

This module used to bind ten aliases -- ``Settings = ModelTrainerSettings``
and nine more -- plus a ``load_settings`` that only called
``load_model_trainer_settings``. Those existed because platform_core's barrel
put every service's types in one flat namespace, so each type carried its
service name as a prefix, and this module aliased the prefix straight back off
again. Two names for one type, in both directions, and a reader grepping for
either found half the uses.

platform_core's config modules now own unprefixed names, so this is an
explicit re-export in the ``import X as X`` form: one name, one type, no
second spelling. The 551 call sites in this service did not change, because
the name they were already using is the real one now.
"""

from __future__ import annotations

from platform_core.config.model_trainer import AppConfig as AppConfig
from platform_core.config.model_trainer import CleanupConfig as CleanupConfig
from platform_core.config.model_trainer import CorpusCacheCleanupConfig as CorpusCacheCleanupConfig
from platform_core.config.model_trainer import LoggingConfig as LoggingConfig
from platform_core.config.model_trainer import RedisConfig as RedisConfig
from platform_core.config.model_trainer import RQConfig as RQConfig
from platform_core.config.model_trainer import SecurityConfig as SecurityConfig
from platform_core.config.model_trainer import Settings as Settings
from platform_core.config.model_trainer import TokenizerCleanupConfig as TokenizerCleanupConfig
from platform_core.config.model_trainer import WandbConfig as WandbConfig
from platform_core.config.model_trainer import load_settings as load_settings

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
