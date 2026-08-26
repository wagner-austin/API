"""Turkic-API's settings, re-exported from platform_core.

Was two defects in twelve lines: ``import TurkicApiSettings as Settings`` --
the import form of a renaming alias -- and a ``settings_from_env`` whose body
was ``return load_turkic_api_settings()``. platform_core's config modules now
own unprefixed names, so both collapse into an explicit ``import X as X``
re-export with no second spelling of anything.
"""

from __future__ import annotations

from platform_core.config.turkic_api import Settings as Settings
from platform_core.config.turkic_api import load_settings as load_settings

__all__ = ["Settings", "load_settings"]
