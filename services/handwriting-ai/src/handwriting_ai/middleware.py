from __future__ import annotations

from platform_core.errors import AppError, ErrorCode
from platform_core.logging import get_logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class ExceptionNormalizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except BaseExceptionGroup:
            get_logger("handwriting_ai").error("exception_group_normalized", exc_info=True)
            # Raise AppError so the registered handler renders JSON without silent excepts
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                "Internal server error.",
            ) from None
