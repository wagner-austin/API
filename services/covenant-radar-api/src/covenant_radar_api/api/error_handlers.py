"""Maps decode failures onto HTTP status codes.

This handler lives at the transport boundary. It does not recover from
anything: it re-raises as an AppError carrying a specific code and the
originating message, so the failure still terminates the request. Core logic
raises and never catches.

Absent rows need no handler here — repositories raise
``AppError(ErrorCode.NOT_FOUND)`` directly, which platform_core already renders
as a 404. A bare ``KeyError`` is deliberately left unmapped: it signals a defect
rather than a missing row, and must keep surfacing as a 500.

Without this, platform_core's catch-all turns every malformed request body into
a generic 500, which is both wrong for the caller and undiagnosable.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONTypeError, register_json_error_handler

_HandlerType = Callable[[Request, Exception], Response | Awaitable[Response]]


def _json_type_error_handler(_: Request, exc: Exception) -> Response:
    """Translate a decode type error into a 400.

    Registered against JSONTypeError only, so the dispatcher guarantees the
    exception type; the handler needs nothing from `exc` but its message. No
    isinstance guard, because the alternative branch would be unreachable.

    Args:
        _: The active request; unused.
        exc: The JSONTypeError being handled.

    Returns:
        Never returns normally; the raised AppError is rendered by the
        AppError handler installed by platform_core.

    Raises:
        AppError: Always, with ErrorCode.INVALID_INPUT and http_status 400.
    """
    raise AppError(code=ErrorCode.INVALID_INPUT, message=str(exc), http_status=400) from exc


def install_covenant_error_handlers(app: FastAPI) -> None:
    """Register the service's decode-to-HTTP mappings on an application.

    Must be called on any app that serves these routers, including test
    harnesses, so tests observe the same error contract as production.

    Args:
        app: Application to register handlers on.
    """
    register_json_error_handler(app)
    json_type_handler: _HandlerType = _json_type_error_handler
    app.add_exception_handler(JSONTypeError, json_type_handler)


__all__ = ["install_covenant_error_handlers"]
