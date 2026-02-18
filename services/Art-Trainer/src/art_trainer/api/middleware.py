"""API middleware for Art-Trainer.

This module provides middleware components for the API.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.errors import ErrorCode
from platform_core.security import create_api_key_dependency

from art_trainer.core.config.settings import Settings


def api_key_dependency(settings: Settings) -> Callable[[str | None], None]:
    """Create API key dependency for route protection.

    Args:
        settings: Application settings.

    Returns:
        FastAPI dependency function.
    """
    return create_api_key_dependency(
        required_key=settings["security"]["api_key"],
        error_code=ErrorCode.UNAUTHORIZED,
        http_status=401,
        header_name="X-API-Key",
        message="Unauthorized",
    )


__all__ = [
    "api_key_dependency",
]
