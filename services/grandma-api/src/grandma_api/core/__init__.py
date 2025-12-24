"""Core module for grandma-api.

Provides service container and foundational infrastructure.
Uses platform_core.errors for all error codes.
"""

from __future__ import annotations

from grandma_api.core.container import ServiceContainer

__all__ = [
    "ServiceContainer",
]
