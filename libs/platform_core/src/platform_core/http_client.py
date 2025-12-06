from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Protocol

from platform_core.json_utils import JSONValue

JsonObject = dict[str, JSONValue]


class HttpxResponse(Protocol):
    status_code: int
    text: str
    headers: Mapping[str, str]
    content: bytes | bytearray

    def json(self) -> JSONValue: ...


class Timeout(Protocol):
    def __repr__(self) -> str: ...


class _TimeoutCtor(Protocol):
    def __call__(self, timeout: float) -> Timeout: ...


class AsyncTransport(Protocol):
    async def aclose(self) -> None: ...


class SyncTransport(Protocol):
    def close(self) -> None: ...


class HttpxAsyncClient(Protocol):
    async def aclose(self) -> None: ...

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse: ...

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> HttpxResponse: ...


class _AsyncClientCtor(Protocol):
    def __call__(
        self,
        *,
        timeout: Timeout,
        transport: AsyncTransport | None = None,
    ) -> HttpxAsyncClient: ...


class HttpxClient(Protocol):
    def close(self) -> None: ...

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse: ...


class _ClientCtor(Protocol):
    def __call__(
        self,
        *,
        timeout: Timeout,
        transport: SyncTransport | None = None,
    ) -> HttpxClient: ...


def _load_httpx() -> tuple[_TimeoutCtor, _AsyncClientCtor, _ClientCtor]:
    mod: ModuleType = __import__("httpx")
    timeout_ctor: _TimeoutCtor = object.__getattribute__(mod, "Timeout")
    async_ctor: _AsyncClientCtor = object.__getattribute__(mod, "AsyncClient")
    client_ctor: _ClientCtor = object.__getattribute__(mod, "Client")
    return timeout_ctor, async_ctor, client_ctor


def build_async_client(
    timeout_seconds: float, transport: AsyncTransport | None = None
) -> HttpxAsyncClient:
    timeout_ctor, async_ctor, _ = _load_httpx()
    timeout_obj = timeout_ctor(float(timeout_seconds))
    if transport is None:
        return async_ctor(timeout=timeout_obj)
    return async_ctor(timeout=timeout_obj, transport=transport)


def build_client(timeout_seconds: float, transport: SyncTransport | None = None) -> HttpxClient:
    timeout_ctor, _, client_ctor = _load_httpx()
    timeout_obj = timeout_ctor(float(timeout_seconds))
    if transport is None:
        return client_ctor(timeout=timeout_obj)
    return client_ctor(timeout=timeout_obj, transport=transport)


__all__ = [
    "AsyncTransport",
    "HttpxAsyncClient",
    "HttpxClient",
    "HttpxResponse",
    "JsonObject",
    "SyncTransport",
    "Timeout",
    "build_async_client",
    "build_client",
]
