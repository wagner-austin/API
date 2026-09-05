from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

from platform_core._asgi_protocols import _JSONResponseProto, _RequestProto
from platform_core.errors import AppError, ErrorCode
from platform_core.fastapi import (
    FastAPIAppAdapter,
    install_exception_handlers_fastapi,
)

# ---------------------------------------------------------------------------
# Fake classes for testing
# ---------------------------------------------------------------------------


class _FakeURL:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path


class _FakeRequest:
    """Fake request that satisfies _RequestProto."""

    def __init__(self, path: str, method: str) -> None:
        self._url = _FakeURL(path)
        self._method = method

    @property
    def url(self) -> _FakeURL:
        return self._url

    @property
    def method(self) -> str:
        return self._method


class _FakeJSONResponse:
    """Fake response that satisfies _JSONResponseProto."""

    def __init__(self, content: dict[str, str], status_code: int) -> None:
        from platform_core.json_utils import dump_json_str

        self._body: bytes = dump_json_str(content).encode("utf-8")
        self._status_code = status_code

    @property
    def body(self) -> bytes:
        return self._body

    @property
    def status_code(self) -> int:
        return self._status_code


# Handler type that matches what Starlette/FastAPI expects


class _FakeFastAPIApp:
    """Fake FastAPI app that records registered handlers."""

    def __init__(self) -> None:
        self.handlers: dict[
            int | type[Exception], Callable[[Request, Exception], Awaitable[Response]]
        ] = {}

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: Callable[[Request, Exception], Awaitable[Response]],
    ) -> None:
        self.handlers[exc_class_or_status_code] = handler


# ---------------------------------------------------------------------------
# Tests for FastAPIAppAdapter
# ---------------------------------------------------------------------------


def test_fastapi_app_adapter_registers_handler() -> None:
    """Test that FastAPIAppAdapter registers exception handlers correctly."""
    fake_app = _FakeFastAPIApp()
    adapter = FastAPIAppAdapter(fake_app)

    async def test_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        return _FakeJSONResponse({"code": "TEST", "message": str(exc)}, 500)

    adapter.add_exception_handler(ValueError, test_handler)

    assert ValueError in fake_app.handlers


def test_fastapi_app_adapter_handler_conversion() -> None:
    """Test that adapter properly converts Protocol response to Starlette Response."""
    fake_app = _FakeFastAPIApp()
    adapter = FastAPIAppAdapter(fake_app)

    async def test_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        return _FakeJSONResponse({"code": "TEST", "message": str(exc)}, 418)

    adapter.add_exception_handler(ValueError, test_handler)

    wrapped_handler = fake_app.handlers[ValueError]
    exc = ValueError("test error")

    async def run_handler() -> Response:
        # A real Starlette Request built from a minimal ASGI scope, so the
        # adapter is exercised against the type it actually receives.
        scope: dict[str, str | list[tuple[bytes, bytes]]] = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
        }
        request = Request(scope)
        return await wrapped_handler(request, exc)

    response = asyncio.run(run_handler())
    assert response.status_code == 418
    assert b"TEST" in response.body


def test_fastapi_app_adapter_memoryview_body() -> None:
    """Test that adapter handles memoryview body correctly."""
    fake_app = _FakeFastAPIApp()
    adapter = FastAPIAppAdapter(fake_app)

    class _MemoryViewResponse:
        def __init__(self) -> None:
            self._body = memoryview(b'{"code":"MV","message":"memoryview"}')
            self._status_code = 200

        @property
        def body(self) -> memoryview[int]:
            return self._body

        @property
        def status_code(self) -> int:
            return self._status_code

    async def memoryview_handler(request: _RequestProto, exc: Exception) -> _JSONResponseProto:
        return _MemoryViewResponse()

    adapter.add_exception_handler(RuntimeError, memoryview_handler)

    wrapped_handler = fake_app.handlers[RuntimeError]
    exc = RuntimeError("test")

    async def run_handler() -> Response:
        scope: dict[str, str | list[tuple[bytes, bytes]]] = {
            "type": "http",
            "method": "GET",
            "path": "/mv",
            "headers": [],
        }
        request = Request(scope)
        return await wrapped_handler(request, exc)

    response = asyncio.run(run_handler())
    assert response.status_code == 200
    assert b"MV" in response.body


# ---------------------------------------------------------------------------
# Tests for install_exception_handlers_fastapi
# ---------------------------------------------------------------------------


def test_install_exception_handlers_fastapi_registers_handlers() -> None:
    """Test that install_exception_handlers_fastapi registers both handlers."""
    fake_app = _FakeFastAPIApp()
    install_exception_handlers_fastapi(fake_app, logger_name="test-fastapi", request_id_var=None)

    assert AppError in fake_app.handlers
    assert Exception in fake_app.handlers


def test_install_exception_handlers_fastapi_app_error_handler() -> None:
    """Test that installed AppError handler returns correct response."""
    fake_app = _FakeFastAPIApp()
    install_exception_handlers_fastapi(fake_app, logger_name="test-fastapi", request_id_var=None)

    app_error_handler = fake_app.handlers[AppError]
    exc = AppError(ErrorCode.NOT_FOUND, "Resource not found")

    async def run_handler() -> Response:
        scope: dict[str, str | list[tuple[bytes, bytes]]] = {
            "type": "http",
            "method": "GET",
            "path": "/api/test",
            "headers": [],
        }
        request = Request(scope)
        return await app_error_handler(request, exc)

    response = asyncio.run(run_handler())
    assert response.status_code == 404
    assert b"NOT_FOUND" in response.body
    assert b"Resource not found" in response.body


def test_install_exception_handlers_fastapi_unhandled_exception_handler() -> None:
    """Test that installed Exception handler returns generic error response."""
    fake_app = _FakeFastAPIApp()
    install_exception_handlers_fastapi(fake_app, logger_name="test-fastapi", request_id_var=None)

    exception_handler = fake_app.handlers[Exception]
    exc = RuntimeError("unexpected error")

    async def run_handler() -> Response:
        scope: dict[str, str | list[tuple[bytes, bytes]]] = {
            "type": "http",
            "method": "POST",
            "path": "/api/crash",
            "headers": [],
        }
        request = Request(scope)
        return await exception_handler(request, exc)

    response = asyncio.run(run_handler())
    assert response.status_code == 500
    assert b"INTERNAL_ERROR" in response.body
    # Should not expose internal error message
    assert b"unexpected error" not in response.body


def test_install_exception_handlers_fastapi_custom_internal_error_code() -> None:
    """Test custom internal error code is used for unhandled exceptions."""
    from platform_core.errors import ErrorCodeBase

    class CustomErrorCode(ErrorCodeBase):
        CUSTOM_INTERNAL = "CUSTOM_INTERNAL"

    fake_app = _FakeFastAPIApp()
    install_exception_handlers_fastapi(
        fake_app,
        logger_name="test-fastapi",
        request_id_var=None,
        internal_error_code=CustomErrorCode.CUSTOM_INTERNAL,
    )

    exception_handler = fake_app.handlers[Exception]
    exc = ValueError("boom")

    async def run_handler() -> Response:
        scope: dict[str, str | list[tuple[bytes, bytes]]] = {
            "type": "http",
            "method": "GET",
            "path": "/api/crash",
            "headers": [],
        }
        request = Request(scope)
        return await exception_handler(request, exc)

    response = asyncio.run(run_handler())
    assert b"CUSTOM_INTERNAL" in response.body
