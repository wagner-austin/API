from __future__ import annotations

from collections.abc import Callable

from platform_core.errors import ErrorCode
from platform_core.request_context import RequestIdMiddleware, request_id_var
from platform_core.security import create_api_key_dependency

from ..core.config.settings import Settings


def api_key_dependency(settings: Settings) -> Callable[[str | None], None]:
    return create_api_key_dependency(
        required_key=settings["security"]["api_key"],
        error_code=ErrorCode.UNAUTHORIZED,
        http_status=401,
        header_name="X-API-Key",
        message="Unauthorized",
    )


__all__ = ["RequestIdMiddleware", "api_key_dependency", "request_id_var"]
