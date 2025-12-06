from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Protocol, TypedDict, runtime_checkable

# Context variable for request ID tracking across async boundaries.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class _ResponseMessage(TypedDict, total=False):
    type: str
    status: int
    headers: list[tuple[bytes, bytes]] | str
    body: bytes


@runtime_checkable
class _ASGIScope(Protocol):
    """Minimal ASGI scope for HTTP requests."""

    def __getitem__(self, key: str) -> str | list[tuple[bytes, bytes]] | None: ...

    def get(
        self, key: str, default: str | list[tuple[bytes, bytes]] | None = None
    ) -> str | list[tuple[bytes, bytes]] | None: ...


@runtime_checkable
class _ReceiveProto(Protocol):
    """Protocol for ASGI receive callable."""

    async def __call__(self) -> _ResponseMessage: ...


@runtime_checkable
class _SendProto(Protocol):
    """Protocol for ASGI send callable."""

    async def __call__(self, message: _ResponseMessage) -> None: ...


@runtime_checkable
class _AppProto(Protocol):
    """Protocol for ASGI application."""

    async def __call__(
        self, scope: _ASGIScope, receive: _ReceiveProto, send: _SendProto
    ) -> None: ...


class _HeadersMutable(Protocol):
    """Minimal protocol for response headers mutation."""

    def __setitem__(self, key: str, value: str) -> None: ...


class _RequestAdapter(Protocol):
    """Protocol for FastAPI/Starlette Request in middleware decorator."""

    @property
    def scope(self) -> _ASGIScope: ...

    @property
    def headers(self) -> _HeadersMutable: ...


class _ResponseAdapter(Protocol):
    """Protocol for FastAPI/Starlette Response returned by call_next."""

    @property
    def headers(self) -> _HeadersMutable: ...


class _CallNext(Protocol):
    async def __call__(self, request: _RequestAdapter) -> _ResponseAdapter: ...


class _MiddlewareDecorator(Protocol):
    def __call__(self, func: _CallNextMiddleware) -> _CallNextMiddleware: ...


class _CallNextMiddleware(Protocol):
    async def __call__(
        self, request: _RequestAdapter, call_next: _CallNext
    ) -> _ResponseAdapter: ...


class _FastAPIAppProto(Protocol):
    """Minimal FastAPI app protocol for installing middleware."""

    def middleware(self, name: str) -> _MiddlewareDecorator: ...


class RequestIdMiddleware:
    """ASGI middleware for request ID tracking and injection."""

    def __init__(self, app: _AppProto) -> None:
        self._app = app

    async def __call__(self, scope: _ASGIScope, receive: _ReceiveProto, send: _SendProto) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        rid = _decode_request_id(scope)
        token = request_id_var.set(rid)

        try:

            async def send_with_request_id(message: _ResponseMessage) -> None:
                if message["type"] == "http.response.start":
                    headers = _attach_headers(message)
                    headers.append((b"x-request-id", rid.encode("utf-8")))
                await send(message)

            await self._app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)


def install_request_id_middleware(app: _FastAPIAppProto) -> None:
    """Install request ID middleware using FastAPI's decorator API."""

    decorator = app.middleware("http")

    @decorator
    async def _middleware(request: _RequestAdapter, call_next: _CallNext) -> _ResponseAdapter:
        rid = _decode_request_id(request.scope)
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            return response
        finally:
            request_id_var.reset(token)


def _decode_request_id(scope: _ASGIScope) -> str:
    """Extract request ID from ASGI scope headers or generate new UUID."""
    headers_raw = scope.get("headers")
    if not isinstance(headers_raw, list):
        return str(uuid.uuid4())

    for header_name_bytes, header_value_bytes in headers_raw:
        header_name = header_name_bytes.decode("latin1").lower()
        if header_name == "x-request-id":
            return header_value_bytes.decode("latin1")

    return str(uuid.uuid4())


def _attach_headers(message: _ResponseMessage) -> list[tuple[bytes, bytes]]:
    headers = message.get("headers")
    if not isinstance(headers, list):
        headers_list: list[tuple[bytes, bytes]] = []
        message["headers"] = headers_list
        return headers_list
    return headers


ASGIApp = _AppProto

__all__ = [
    "ASGIApp",
    "RequestIdMiddleware",
    "_ASGIScope",
    "_decode_request_id",
    "install_request_id_middleware",
    "request_id_var",
]
