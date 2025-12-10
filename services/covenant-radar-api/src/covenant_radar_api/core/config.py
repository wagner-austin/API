"""Configuration loading for covenant-radar-api using platform_core TypedDict settings."""

from __future__ import annotations

from platform_core.config import CovenantRadarSettings as Settings
from platform_core.config import load_covenant_radar_settings


def settings_from_env() -> Settings:
    """Load covenant-radar settings from the shared platform_core config."""
    return load_covenant_radar_settings()


__all__ = ["Settings", "settings_from_env"]
