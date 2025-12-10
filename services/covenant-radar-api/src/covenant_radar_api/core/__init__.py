"""Core configuration and dependency injection."""

from __future__ import annotations

from . import _test_hooks
from .config import Settings, settings_from_env
from .container import ServiceContainer

__all__ = ["ServiceContainer", "Settings", "_test_hooks", "settings_from_env"]
