"""Data-Bank's settings, re-exported from platform_core.

Was ``import DataBankSettings as Settings`` -- the import form of a renaming
alias -- plus a ``settings_from_env`` whose body was
``return load_data_bank_settings()``. platform_core's config modules now own
unprefixed names, so both collapse into an explicit ``import X as X``
re-export.
"""

from __future__ import annotations

from platform_core.config.data_bank import Settings as Settings
from platform_core.config.data_bank import load_settings as load_settings

__all__ = ["Settings", "load_settings"]
