"""Tests for errors: _FakeURL."""

from __future__ import annotations

from contextvars import ContextVar

import pytest

from platform_core._asgi_protocols import _JSONResponseProto
from platform_core.errors import AppError, ErrorCode
from platform_core.fastapi import install_exception_handlers
from platform_core.logging import stdlib_logging
from tests._error_helpers import (
    FakeFastAPIApp,
    FakeRequest,
    parse_response_body,
)


def test_install_exception_handlers_registers_handlers() -> None:
    """Test install_exception_handlers registers AppError and Exception handlers."""
    app = FakeFastAPIApp()

    install_exception_handlers(app)

    assert AppError in app.handlers
    assert Exception in app.handlers


def test_install_exception_handlers_app_error_user_error_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers logs user errors at INFO level."""
    app = FakeFastAPIApp()
    request_id_var: ContextVar[str] = ContextVar("test_request_id", default="")
    request_id_var.set("req-123")

    caplog.set_level(stdlib_logging.INFO)

    install_exception_handlers(
        app,
        request_id_var=request_id_var,
        logger_name="test_errors",
        log_user_errors=True,
    )

    handler = app.handlers[AppError]
    assert callable(handler)

    request = FakeRequest(path="/api/test", method="GET")
    error = AppError(
        code=ErrorCode.NOT_FOUND,
        message="Resource not found",
    )

    # Manually invoke the handler (it's async)
    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 404
    content = parse_response_body(response)
    assert content["code"] == "NOT_FOUND"
    assert content["message"] == "Resource not found"
    assert content["request_id"] == "req-123"

    # Check logging
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "INFO"
    assert record.getMessage() == "user_error"


def test_install_exception_handlers_app_error_system_error_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers logs system errors at ERROR level with exc_info."""
    app = FakeFastAPIApp()
    request_id_var: ContextVar[str] = ContextVar("test_request_id", default="")
    request_id_var.set("req-456")

    caplog.set_level(stdlib_logging.ERROR)

    install_exception_handlers(
        app,
        request_id_var=request_id_var,
        logger_name="test_errors",
    )

    handler = app.handlers[AppError]
    assert callable(handler)

    request = FakeRequest(path="/api/system", method="POST")
    error = AppError(
        code=ErrorCode.INTERNAL_ERROR,
        message="System failure",
    )

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = parse_response_body(response)
    assert content["code"] == "INTERNAL_ERROR"
    assert content["message"] == "System failure"
    assert content["request_id"] == "req-456"

    # Check logging - verify ERROR level and message, and that exc_info was requested
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.getMessage() == "system_error"
    # Verify exc_info was requested (it's a tuple even if exception context is empty)
    assert record.exc_info is not None or record.exc_text is not None


def test_install_exception_handlers_app_error_no_request_id() -> None:
    """Test install_exception_handlers works without request_id_var."""
    app = FakeFastAPIApp()

    install_exception_handlers(app, request_id_var=None, logger_name="test_errors")

    handler = app.handlers[AppError]
    assert callable(handler)

    request = FakeRequest(path="/api/test", method="GET")
    error = AppError(code=ErrorCode.FORBIDDEN, message="Access denied")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 403
    content = parse_response_body(response)
    assert content["request_id"] == ""


def test_install_exception_handlers_app_error_no_user_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers skips user error logging when disabled."""
    app = FakeFastAPIApp()

    caplog.set_level(stdlib_logging.INFO)

    install_exception_handlers(
        app,
        request_id_var=None,
        logger_name="test_errors",
        log_user_errors=False,
    )

    handler = app.handlers[AppError]
    assert callable(handler)

    request = FakeRequest(path="/api/test", method="GET")
    error = AppError(code=ErrorCode.INVALID_INPUT, message="Bad input")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    asyncio.run(_run_handler())

    # Should not log user errors
    assert len(caplog.records) == 0


def test_install_exception_handlers_unhandled_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test install_exception_handlers logs unhandled exceptions at ERROR level."""
    app = FakeFastAPIApp()
    request_id_var: ContextVar[str] = ContextVar("test_request_id", default="")
    request_id_var.set("req-789")

    caplog.set_level(stdlib_logging.ERROR)

    install_exception_handlers(
        app,
        request_id_var=request_id_var,
        logger_name="test_errors",
    )

    handler = app.handlers[Exception]
    assert callable(handler)

    request = FakeRequest(path="/api/crash", method="DELETE")
    error = ValueError("Unexpected error")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = parse_response_body(response)
    assert content["code"] == "INTERNAL_ERROR"
    assert content["message"] == "Internal server error"
    assert content["request_id"] == "req-789"

    # Check logging - verify ERROR level and message, and that exc_info was requested
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.getMessage() == "unhandled_exception"
    # Verify exc_info was requested (it's a tuple even if exception context is empty)
    assert record.exc_info is not None or record.exc_text is not None


def test_install_exception_handlers_unhandled_exception_no_request_id() -> None:
    """Test install_exception_handlers handles unhandled exceptions without request_id_var."""
    app = FakeFastAPIApp()

    install_exception_handlers(app, request_id_var=None, logger_name="test_errors")

    handler = app.handlers[Exception]
    assert callable(handler)

    request = FakeRequest(path="/api/error", method="PUT")
    error = RuntimeError("Runtime error")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    assert response.status_code == 500
    content = parse_response_body(response)
    assert content["request_id"] == ""


def test_install_exception_handlers_app_error_handler_delegates_non_app_error() -> None:
    """Test AppError handler delegates non-AppError exceptions to unhandled handler."""
    app = FakeFastAPIApp()

    install_exception_handlers(app, request_id_var=None, logger_name="test_errors")

    app_error_handler = app.handlers[AppError]
    assert callable(app_error_handler)

    request = FakeRequest(path="/api/test", method="GET")
    # Pass a non-AppError exception to AppError handler
    error = ValueError("Not an AppError")

    import asyncio

    async def _run_handler() -> _JSONResponseProto:
        return await app_error_handler(request, error)

    response: _JSONResponseProto = asyncio.run(_run_handler())

    # Should delegate to unhandled handler
    assert response.status_code == 500
    content = parse_response_body(response)
    assert content["code"] == "INTERNAL_ERROR"
