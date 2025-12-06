from __future__ import annotations

import uuid
from typing import Protocol, TypedDict, runtime_checkable

import pytest

import platform_core.request_context as request_context_module
from platform_core.request_context import (
    RequestIdMiddleware,
    _ASGIScope,
    install_request_id_middleware,
    request_id_var,
)


class Scope:
    def __init__(
        self,
        *,
        scope_type: str,
        method: str | None = None,
        path: str | None = None,
        headers: list[tuple[bytes, bytes]] | str | None = None,
    ) -> None:
        self.type = scope_type
        self.method = method if method is not None else ""
        self.path = path if path is not None else ""
        self.headers: list[tuple[bytes, bytes]] | str = headers if headers is not None else []

    def __getitem__(self, key: str) -> str | list[tuple[bytes, bytes]] | None:
        if key == "type":
            return self.type
        if key == "method":
            return self.method
        if key == "path":
            return self.path
        if key == "headers":
            return self.headers
        return None

    def get(
        self, key: str, default: str | list[tuple[bytes, bytes]] | None = None
    ) -> str | list[tuple[bytes, bytes]] | None:
        value = self.__getitem__(key)
        return value if value is not None else default


class ResponseMessage(TypedDict, total=False):
    type: str
    status: int
    headers: list[tuple[bytes, bytes]] | str
    body: bytes


def test_request_id_var_default() -> None:
    """Test request_id_var has empty string default."""
    # Create a new context to avoid contamination from other tests
    import contextvars

    ctx = contextvars.copy_context()

    def check() -> str:
        return request_id_var.get()

    result = ctx.run(check)
    assert result == ""


def test_request_id_var_set_get() -> None:
    """Test request_id_var set and get."""
    token = request_id_var.set("test-request-123")
    try:
        assert request_id_var.get() == "test-request-123"
    finally:
        request_id_var.reset(token)


def test_request_id_var_reset() -> None:
    """Test request_id_var reset restores previous value."""
    original = request_id_var.get()
    token = request_id_var.set("temporary-id")
    assert request_id_var.get() == "temporary-id"

    request_id_var.reset(token)
    assert request_id_var.get() == original


@runtime_checkable
class _ReceiveProto(Protocol):
    async def __call__(self) -> ResponseMessage: ...


@runtime_checkable
class _SendProto(Protocol):
    async def __call__(self, message: ResponseMessage) -> None: ...


@runtime_checkable
class _ASGIApp(Protocol):
    async def __call__(
        self,
        scope: _ASGIScope,
        receive: _ReceiveProto,
        send: _SendProto,
    ) -> None: ...


class MockASGIApp:
    def __init__(self) -> None:
        self.call_count = 0
        self.last_scope: _ASGIScope | None = None
        self.last_send: _SendProto | None = None

    async def __call__(
        self,
        scope: _ASGIScope,
        receive: _ReceiveProto,
        send: _SendProto,
    ) -> None:
        self.call_count += 1
        self.last_scope = scope
        self.last_send = send
        # Send a mock response only for HTTP requests so middleware can intercept it
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})


async def mock_receive() -> ResponseMessage:
    """Mock ASGI receive callable."""
    return {"type": "http.request", "body": b""}


class MockSend:
    def __init__(self) -> None:
        self.messages: list[ResponseMessage] = []

    async def __call__(self, message: ResponseMessage) -> None:
        self.messages.append(message)


def test_request_id_middleware_extracts_existing_header() -> None:
    """Test RequestIdMiddleware extracts X-Request-ID from headers."""
    app = MockASGIApp()
    middleware = RequestIdMiddleware(app)

    scope: _ASGIScope = Scope(
        scope_type="http",
        method="GET",
        path="/test",
        headers=[
            (b"content-type", b"application/json"),
            (b"x-request-id", b"existing-request-123"),
        ],
    )

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify middleware called the app
    assert app.call_count == 1

    # Verify response has X-Request-ID header
    response_start = next((m for m in send.messages if m["type"] == "http.response.start"), None)
    if response_start is None:
        pytest.fail("expected http.response.start message")
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    request_id_header = next(
        (v for k, v in headers if isinstance(k, bytes) and k == b"x-request-id"),
        None,
    )
    assert request_id_header == b"existing-request-123"


def test_request_id_middleware_generates_uuid() -> None:
    """Test RequestIdMiddleware generates UUID when no X-Request-ID header."""
    app = MockASGIApp()
    middleware = RequestIdMiddleware(app)

    scope: _ASGIScope = Scope(
        scope_type="http",
        method="POST",
        path="/api/test",
        headers=[(b"content-type", b"application/json")],
    )

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify middleware called the app
    assert app.call_count == 1

    # Verify response has X-Request-ID header with UUID
    response_start = next((m for m in send.messages if m["type"] == "http.response.start"), None)
    if response_start is None:
        pytest.fail("expected http.response.start message")
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    request_id_header = next(
        (v for k, v in headers if isinstance(k, bytes) and k == b"x-request-id"),
        None,
    )
    if not isinstance(request_id_header, bytes):
        pytest.fail("expected x-request-id header to be bytes")

    # Validate it's a valid UUID
    request_id_str = request_id_header.decode("utf-8")
    parsed_uuid = uuid.UUID(request_id_str)
    assert str(parsed_uuid) == request_id_str


def test_request_id_middleware_case_insensitive_header() -> None:
    """Test RequestIdMiddleware handles case-insensitive X-Request-ID header."""
    app = MockASGIApp()
    middleware = RequestIdMiddleware(app)

    scope: _ASGIScope = Scope(
        scope_type="http",
        method="GET",
        path="/test",
        headers=[(b"X-REQUEST-ID", b"case-test-123")],
    )

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify response has correct request ID
    response_start = next((m for m in send.messages if m["type"] == "http.response.start"), None)
    if response_start is None:
        pytest.fail("expected http.response.start message")
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    request_id_header = next(
        (v for k, v in headers if isinstance(k, bytes) and k == b"x-request-id"),
        None,
    )
    assert request_id_header == b"case-test-123"


def test_request_id_middleware_sets_context_var() -> None:
    """Test RequestIdMiddleware sets request_id_var context variable."""
    captured_request_id: str | None = None

    class CapturingApp:
        async def __call__(
            self,
            scope: _ASGIScope,
            receive: _ReceiveProto,
            send: _SendProto,
        ) -> None:
            nonlocal captured_request_id
            captured_request_id = request_id_var.get()

    middleware = RequestIdMiddleware(CapturingApp())

    scope: _ASGIScope = Scope(scope_type="http", headers=[(b"x-request-id", b"context-test-456")])

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify context variable was set during app execution
    assert captured_request_id == "context-test-456"


def test_request_id_middleware_cleans_up_context() -> None:
    """Test RequestIdMiddleware resets context variable after request."""
    # Set initial value
    token = request_id_var.set("initial-value")

    try:
        app = MockASGIApp()
        middleware = RequestIdMiddleware(app)

        scope: _ASGIScope = Scope(scope_type="http", headers=[(b"x-request-id", b"temporary-123")])

        send = MockSend()

        import asyncio

        asyncio.run(middleware(scope, mock_receive, send))

        # After middleware completes, context should be restored
        assert request_id_var.get() == "initial-value"
    finally:
        request_id_var.reset(token)


def test_request_id_middleware_non_http_passthrough() -> None:
    """Test RequestIdMiddleware passes through non-HTTP requests without modification."""
    app = MockASGIApp()
    middleware = RequestIdMiddleware(app)

    scope: _ASGIScope = Scope(scope_type="websocket", path="/ws")

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify middleware called the app
    assert app.call_count == 1

    # Verify no request ID header was added
    assert len(send.messages) == 0


def test_request_id_middleware_missing_headers_list() -> None:
    """Test RequestIdMiddleware handles missing headers list gracefully."""
    app = MockASGIApp()
    middleware = RequestIdMiddleware(app)

    scope: _ASGIScope = Scope(scope_type="http", method="GET", path="/test", headers=[])

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Should generate UUID
    assert app.call_count == 1

    response_start = next((m for m in send.messages if m["type"] == "http.response.start"), None)
    if response_start is None:
        pytest.fail("expected http.response.start message")
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    request_id_header = next(
        (v for k, v in headers if isinstance(k, bytes) and k == b"x-request-id"),
        None,
    )
    if not isinstance(request_id_header, bytes):
        pytest.fail("expected x-request-id header to be bytes")


def test_request_id_middleware_invalid_headers_type() -> None:
    """Test RequestIdMiddleware handles invalid headers type gracefully."""
    app = MockASGIApp()
    middleware = RequestIdMiddleware(app)

    scope: _ASGIScope = Scope(scope_type="http", method="GET", path="/test", headers="not a list")

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Should generate UUID
    assert app.call_count == 1


def test_request_id_middleware_adds_header_to_response_without_headers() -> None:
    """Test RequestIdMiddleware creates headers list if missing in response."""

    class AppWithoutHeaders:
        async def __call__(
            self,
            scope: _ASGIScope,
            receive: _ReceiveProto,
            send: _SendProto,
        ) -> None:
            assert callable(send)
            # Send response without headers key
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"OK"})

    middleware = RequestIdMiddleware(AppWithoutHeaders())

    scope: _ASGIScope = Scope(scope_type="http", headers=[(b"x-request-id", b"test-123")])

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify headers were created and request ID was added
    response_start = next((m for m in send.messages if m["type"] == "http.response.start"), None)
    if response_start is None:
        pytest.fail("expected http.response.start message")
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    assert len(headers) == 1
    assert headers[0] == (b"x-request-id", b"test-123")


def test_request_id_middleware_preserves_existing_response_headers() -> None:
    """Test RequestIdMiddleware preserves existing response headers."""

    class AppWithHeaders:
        async def __call__(
            self,
            scope: _ASGIScope,
            receive: _ReceiveProto,
            send: _SendProto,
        ) -> None:
            assert callable(send)
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

    middleware = RequestIdMiddleware(AppWithHeaders())

    scope: _ASGIScope = Scope(scope_type="http", headers=[(b"x-request-id", b"preserve-test-789")])

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify existing headers preserved and request ID added
    response_start = next((m for m in send.messages if m["type"] == "http.response.start"), None)
    if response_start is None:
        pytest.fail("expected http.response.start message")
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    assert len(headers) == 2
    assert (b"content-type", b"application/json") in headers
    assert (b"x-request-id", b"preserve-test-789") in headers


def test_request_id_middleware_only_adds_header_to_response_start() -> None:
    """Test RequestIdMiddleware only adds header to http.response.start messages."""

    class AppWithMultipleMessages:
        async def __call__(
            self,
            scope: _ASGIScope,
            receive: _ReceiveProto,
            send: _SendProto,
        ) -> None:
            assert callable(send)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"chunk1"})
            await send({"type": "http.response.body", "body": b"chunk2"})

    middleware = RequestIdMiddleware(AppWithMultipleMessages())

    scope: _ASGIScope = Scope(scope_type="http", headers=[(b"x-request-id", b"multi-msg-123")])

    send = MockSend()

    import asyncio

    asyncio.run(middleware(scope, mock_receive, send))

    # Verify only response.start has the header
    assert len(send.messages) == 3

    response_start = send.messages[0]
    assert response_start["type"] == "http.response.start"
    headers = response_start.get("headers")
    if not isinstance(headers, list):
        pytest.fail("expected headers to be a list")
    assert (b"x-request-id", b"multi-msg-123") in headers

    # Body messages should not have headers added
    assert "headers" not in send.messages[1]
    assert "headers" not in send.messages[2]


def test_install_request_id_middleware_decorator_sets_header_and_resets_context() -> None:
    """Decorator installer should wrap call_next and set/reset request_id_var."""

    class _RequestAdapter(request_context_module._RequestAdapter):
        def __init__(self, scope: _ASGIScope) -> None:
            self._scope = scope
            self._headers: dict[str, str] = {}

        @property
        def scope(self) -> _ASGIScope:
            return self._scope

        @property
        def headers(self) -> dict[str, str]:
            return self._headers

    class _ResponseAdapter(request_context_module._ResponseAdapter):
        def __init__(self) -> None:
            self._headers: dict[str, str] = {}

        @property
        def headers(self) -> dict[str, str]:
            return self._headers

    class _FakeApp(request_context_module._FastAPIAppProto):
        def __init__(self) -> None:
            self.handler: request_context_module._CallNextMiddleware | None = None

        def middleware(self, name: str) -> request_context_module._MiddlewareDecorator:
            assert name == "http"

            def _decorator(
                func: request_context_module._CallNextMiddleware,
            ) -> request_context_module._CallNextMiddleware:
                self.handler = func
                return func

            return _decorator

    app = _FakeApp()
    install_request_id_middleware(app)
    assert callable(app.handler)

    scope: _ASGIScope = Scope(scope_type="http", headers=[])

    async def _call_next(
        request: request_context_module._RequestAdapter,
    ) -> _ResponseAdapter:
        assert request.scope == scope
        return _ResponseAdapter()

    call_next: request_context_module._CallNext = _call_next

    import asyncio

    prior_request_id = request_id_var.get()
    response = asyncio.run(app.handler(_RequestAdapter(scope), call_next))
    headers = response.headers
    assert type(headers) is dict
    assert "x-request-id" in headers
    assert request_id_var.get() == prior_request_id
