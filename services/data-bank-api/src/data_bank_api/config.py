from __future__ import annotations

from platform_core.config import DataBankSettings as Settings
from platform_core.config import load_data_bank_settings


def settings_from_env() -> Settings:
    """Load data-bank settings from the shared platform_core config."""
    return load_data_bank_settings()


__all__ = ["Settings", "settings_from_env"]
