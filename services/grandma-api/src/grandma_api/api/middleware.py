"""Middleware and dependencies for grandma-api.

Provides API key authentication and request context middleware.
Uses platform_core.security for centralized security patterns.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.errors import ErrorCode
from platform_core.request_context import RequestIdMiddleware, request_id_var
from platform_core.security import create_api_key_dependency

from grandma_api.config import GrandmaApiSettings


def api_key_dependency(settings: GrandmaApiSettings) -> Callable[[str | None], None]:
    """Create FastAPI dependency for API key validation.

    Creates a dependency that validates the X-API-Key header against
    the configured API token. If API_TOKEN is empty, validation is skipped.

    Args:
        settings: Application settings containing the API token.

    Returns:
        A FastAPI dependency function for API key validation.
    """
    return create_api_key_dependency(
        required_key=settings["api_token"],
        error_code=ErrorCode.UNAUTHORIZED,
        http_status=401,
        header_name="X-API-Key",
        message="Unauthorized",
    )


__all__ = [
    "RequestIdMiddleware",
    "api_key_dependency",
    "request_id_var",
]
