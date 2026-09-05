from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from fastapi import FastAPI as _FastAPI
from fastapi.responses import JSONResponse as _FastAPIJSONResponse
from starlette.requests import Request
from starlette.responses import Response

from platform_core._asgi_protocols import (
    _FastAPIAppProto,
    _JSONResponseProto,
    _RequestProto,
    _URLProto,
)
from platform_core.errors import (
    AppError,
    ErrorCode,
    ErrorCodeBase,
    _code_value,
)
from platform_core.json_utils import InvalidJsonError, JSONDecodeError
from platform_core.logging import get_logger
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
        handler: Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]],
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
# The handlers themselves
# ---------------------------------------------------------------------------


def install_exception_handlers(
    app: _FastAPIAppProto,
    *,
    request_id_var: ContextVar[str] | None = _global_request_id_var,
    logger_name: str = "app",
    log_user_errors: bool = True,
    internal_error_code: ErrorCodeBase = ErrorCode.INTERNAL_ERROR,
) -> None:
    """Install centralized exception handlers with platform_core logging integration.

    Lives here rather than beside the error codes because it is transport: it
    reaches for ``fastapi.responses`` and the ASGI protocols, and
    :mod:`platform_core.errors` is imported by hundreds of modules that never
    serve a request.

    Registers handlers for:
    - AppError: Structured application errors
    - Exception: All unhandled exceptions

    Logging behavior:
    - User errors (4xx): Logged at INFO level without traceback
    - System errors (5xx): Logged at ERROR level with full traceback (exc_info=True)
    - Unhandled exceptions: Logged at ERROR level with full traceback

    Args:
        app: An app satisfying the ASGI app protocol. For a real FastAPI
            instance use :func:`install_exception_handlers_fastapi`, which
            wraps it in :class:`FastAPIAppAdapter` first.
        request_id_var: Optional ContextVar for request ID tracking
        logger_name: Logger name for error logging (default: "app")
        log_user_errors: Whether to log user errors at INFO level (default: True)
        internal_error_code: Error code reported for unhandled exceptions
    """
    logger = get_logger(logger_name)

    def _json_response(*, content: dict[str, str], status_code: int) -> _JSONResponseProto:
        return _FastAPIJSONResponse(content=content, status_code=status_code)

    async def _app_error_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        """Handle AppError exceptions with structured logging and response."""
        if not isinstance(exc, AppError):
            # Should not happen, but handle gracefully by delegating
            return await _unhandled_handler(request, exc)

        # Extract request ID from context if available
        rid = request_id_var.get() if request_id_var is not None else ""

        # Determine if this is a user error (4xx) or system error (5xx)
        is_user_error = exc.http_status < 500

        code_value = _code_value(exc.code)
        if is_user_error and log_user_errors:
            # Log user errors at INFO level without traceback
            logger.info(
                "user_error",
                extra={
                    "error_code": code_value,
                    "request_id": rid,
                    "error_message": exc.message,
                    "path": request.url.path,
                    "method": request.method,
                },
            )
        elif not is_user_error:
            # Log system errors at ERROR level with full traceback
            logger.error(
                "system_error",
                extra={
                    "error_code": code_value,
                    "request_id": rid,
                    "error_message": exc.message,
                    "path": request.url.path,
                    "method": request.method,
                },
                exc_info=True,
            )

        # Return structured JSON error response
        response_body: dict[str, str] = {
            "code": code_value,
            "message": exc.message,
            "request_id": rid,
        }
        return _json_response(content=response_body, status_code=exc.http_status)

    async def _unhandled_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        """Handle all unhandled exceptions with full logging."""
        # Extract request ID from context if available
        rid = request_id_var.get() if request_id_var is not None else ""

        # Log all unhandled exceptions at ERROR level with full traceback
        logger.error(
            "unhandled_exception",
            extra={
                "error_type": type(exc).__name__,
                "request_id": rid,
                "path": request.url.path,
                "method": request.method,
                "error_message": str(exc),
            },
            exc_info=True,
        )

        # Return generic error response (don't expose internal details)
        code_value_internal = _code_value(internal_error_code)
        response_body: dict[str, str] = {
            "code": code_value_internal,
            "message": "Internal server error",
            "request_id": rid,
        }
        return _json_response(content=response_body, status_code=500)

    # Register handlers with FastAPI
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(Exception, _unhandled_handler)


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


def register_json_error_handler(
    app: _FastAPI, *, detail: str = "Invalid JSON body"
) -> Callable[[Request, Exception], Response | Awaitable[Response]]:
    """Translate a malformed request body into a 400 rather than a 500.

    Lived in :mod:`platform_core.json_utils` until 2026-08-28. It was the
    only thing in that module that needed a web framework, and because every
    JSON decode in this repository goes through that module, the import
    reached everything: the hpc3 CLI -- a command-line tool that talks to
    Slurm over SSH -- had FastAPI, Starlette and httpx installed because it
    validates JSON documents.

    Moving one function to the module named after the framework it needs
    leaves ``json_utils`` on the standard library alone, which is what lets
    the comparability stack it underpins be consumed by research code that
    has no business installing a web server.

    Args:
        app: The application to install the handlers on.
        detail: Message the 400 carries.

    Returns:
        The installed handler, so a caller can assert on identity rather
        than on the fact that registration was attempted.
    """
    handler_type = Callable[[Request, Exception], Response | Awaitable[Response]]

    def _handler(_: Request, exc: Exception) -> Response:
        if isinstance(exc, (InvalidJsonError, JSONDecodeError)):
            raise AppError(code=ErrorCode.INVALID_INPUT, message=detail, http_status=400) from exc
        raise exc

    handler: handler_type = _handler
    app.add_exception_handler(InvalidJsonError, handler)
    app.add_exception_handler(JSONDecodeError, handler)
    return handler


__all__ = [
    "FastAPIAppAdapter",
    "StarletteRequestProto",
    "StarletteResponseProto",
    "install_exception_handlers",
    "install_exception_handlers_fastapi",
    "register_json_error_handler",
]
