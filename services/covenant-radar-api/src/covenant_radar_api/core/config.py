"""Covenant-Radar's settings, re-exported from platform_core.

Was ``import CovenantRadarSettings as Settings`` -- the import form of a
renaming alias -- plus a ``settings_from_env`` whose body was
``return load_covenant_radar_settings()``. platform_core's config modules now
own unprefixed names, so both collapse into an explicit ``import X as X``
re-export.
"""

from __future__ import annotations

from platform_core.config.covenant_radar import Settings as Settings
from platform_core.config.covenant_radar import load_settings as load_settings

__all__ = ["Settings", "load_settings"]
