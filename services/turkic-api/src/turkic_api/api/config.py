from __future__ import annotations

from platform_core.config import TurkicApiSettings as Settings
from platform_core.config import load_turkic_api_settings


def settings_from_env() -> Settings:
    """Load settings via shared platform_core config helpers."""
    return load_turkic_api_settings()


__all__ = ["Settings", "settings_from_env"]
