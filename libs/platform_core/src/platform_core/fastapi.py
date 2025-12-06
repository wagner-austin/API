from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from starlette.requests import Request
from starlette.responses import Response

from platform_core.errors import (
    ErrorCode,
    ErrorCodeBase,
    _ExceptionHandlerProto,
    _FastAPIAppProto,
    _JSONResponseProto,
    install_exception_handlers,
)
from platform_core.request_context import request_id_var as _global_request_id_var

# ---------------------------------------------------------------------------
# Protocols for our typed interface (what services use)
# ---------------------------------------------------------------------------


@runtime_checkable
class StarletteRequestProto(Protocol):
    """Protocol for Starlette Request - defines what our handlers need."""

    @property
    def url(self) -> _URLProto: ...

    @property
    def method(self) -> str: ...


class _URLProto(Protocol):
    @property
    def path(self) -> str: ...


@runtime_checkable
class StarletteResponseProto(Protocol):
    """Protocol for Starlette Response - defines what our handlers return."""

    @property
    def body(self) -> bytes: ...

    @property
    def status_code(self) -> int: ...


# ---------------------------------------------------------------------------
# Internal Protocol for FastAPI app's add_exception_handler signature
# ---------------------------------------------------------------------------


class _FastAPIAddExceptionHandler(Protocol):
    """Protocol matching FastAPI's add_exception_handler method signature."""

    def __call__(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: _StarletteExceptionHandler,
    ) -> None: ...


# Starlette's actual exception handler type (async callable)
class _StarletteExceptionHandler(Protocol):
    async def __call__(self, request: Request, exc: Exception) -> Response: ...


class _FastAPILike(Protocol):
    """Protocol for FastAPI-like apps with add_exception_handler."""

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: _StarletteExceptionHandler,
    ) -> None: ...


# ---------------------------------------------------------------------------
# FastAPI App Adapter - wraps FastAPI to satisfy _FastAPIAppProto
# ---------------------------------------------------------------------------


class FastAPIAppAdapter:
    """Adapter that wraps a FastAPI app to satisfy _FastAPIAppProto.

    This adapter converts between platform_core's Protocol-typed handlers
    and Starlette's concrete types.

    Usage:
        from fastapi import FastAPI
        from platform_core.fastapi import FastAPIAppAdapter
        from platform_core.errors import install_exception_handlers

        app = FastAPI()
        install_exception_handlers(FastAPIAppAdapter(app))
    """

    def __init__(self, app: _FastAPILike) -> None:
        self._app = app

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: _ExceptionHandlerProto,
    ) -> None:
        """Register exception handler, wrapping Protocol types to Starlette types."""

        async def _wrapped(request: Request, exc: Exception) -> Response:
            # Call our Protocol-typed handler
            proto_response: _JSONResponseProto = await handler(request, exc)
            # Convert to Starlette Response
            body_bytes = proto_response.body
            if isinstance(body_bytes, memoryview):
                body_bytes = bytes(body_bytes)
            return Response(
                content=body_bytes,
                status_code=proto_response.status_code,
                media_type="application/json",
            )

        self._app.add_exception_handler(exc_class_or_status_code, _wrapped)


# ---------------------------------------------------------------------------
# Convenience function for installing exception handlers on FastAPI
# ---------------------------------------------------------------------------


def install_exception_handlers_fastapi(
    app: _FastAPILike,
    *,
    request_id_var: ContextVar[str] | None = _global_request_id_var,
    logger_name: str = "app",
    log_user_errors: bool = True,
    internal_error_code: ErrorCodeBase = ErrorCode.INTERNAL_ERROR,
) -> None:
    """Install exception handlers on a FastAPI application.

    This is the recommended way to install exception handlers on FastAPI apps.
    It handles all the type conversion internally.

    Args:
        app: FastAPI application instance
        request_id_var: Optional ContextVar for request ID tracking
        logger_name: Logger name for error logging (default: "app")
        log_user_errors: Whether to log user errors at INFO level (default: True)
        internal_error_code: Error code to use for unhandled exceptions

    Example:
        from fastapi import FastAPI
        from platform_core.fastapi import install_exception_handlers_fastapi

        app = FastAPI()
        install_exception_handlers_fastapi(app, logger_name="my-api")
    """
    adapter: _FastAPIAppProto = FastAPIAppAdapter(app)
    install_exception_handlers(
        adapter,
        request_id_var=request_id_var,
        logger_name=logger_name,
        log_user_errors=log_user_errors,
        internal_error_code=internal_error_code,
    )


__all__ = [
    "FastAPIAppAdapter",
    "StarletteRequestProto",
    "StarletteResponseProto",
    "install_exception_handlers_fastapi",
]
