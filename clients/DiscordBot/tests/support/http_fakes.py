"""
Protocol-compliant fake implementations for HTTP client testing.

These fakes implement the protocols defined in platform_core.http_client
without importing httpx directly, avoiding Any types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn, Protocol

from platform_core.http_client import HttpxAsyncClient, HttpxClient, HttpxResponse, Timeout
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str


class FakeTimeout:
    """Protocol-compliant fake for httpx.Timeout."""

    __slots__ = ("_timeout",)

    def __init__(self, timeout: float) -> None:
        self._timeout = float(timeout)

    def __repr__(self) -> str:
        return f"Timeout({self._timeout})"


class FakeHttpxResponse:
    """Protocol-compliant fake for httpx.Response.

    Satisfies HttpxResponse Protocol from platform_core.http_client.
    Supports initialization from JSON body, raw bytes, or text.
    """

    __slots__ = ("_json", "content", "headers", "status_code", "text")

    def __init__(
        self,
        status: int,
        json_body: JSONValue | None = None,
        *,
        content: bytes | None = None,
        text: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code: int = int(status)
        self._json: JSONValue | None = json_body
        self.headers: Mapping[str, str] = dict(headers) if headers else {}

        # Determine content and text from inputs
        if content is not None:
            self.content: bytes | bytearray = content
            self.text: str = text if text is not None else content.decode("utf-8", errors="replace")
        elif json_body is not None:
            self.text = dump_json_str(json_body)
            self.content = self.text.encode("utf-8")
        elif text is not None:
            self.text = text
            self.content = text.encode("utf-8")
        else:
            self.text = ""
            self.content = b""

    def json(self) -> JSONValue:
        if self._json is not None:
            return self._json
        return load_json_str(self.text)


class FakeHttpxAsyncClient:
    """Protocol-compliant fake for httpx.AsyncClient.

    Satisfies HttpxAsyncClient Protocol from platform_core.http_client.
    Supports configurable responses and exception raising for testing.
    """

    __slots__ = (
        "_call_count",
        "_exception_count",
        "_exception_to_raise",
        "_response",
        "seen_headers",
        "seen_urls",
    )

    def __init__(
        self,
        response: HttpxResponse | None = None,
        *,
        exception_to_raise: Exception | None = None,
        exception_count: int = 0,
    ) -> None:
        self._response: HttpxResponse | None = response
        self._exception_to_raise: Exception | None = exception_to_raise
        self._exception_count: int = exception_count
        self._call_count: int = 0
        self.seen_headers: dict[str, str] = {}
        self.seen_urls: list[str] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self._call_count += 1
        self.seen_urls.append(url)
        self.seen_headers.update(headers)
        _ = (json, files)  # Unused but part of protocol

        # Raise exception if configured (for first N calls)
        should_raise = self._exception_to_raise is not None and (
            self._exception_count == 0 or self._call_count <= self._exception_count
        )
        if should_raise and self._exception_to_raise is not None:
            raise self._exception_to_raise

        if self._response is None:
            raise RuntimeError("No response configured for FakeHttpxAsyncClient")
        return self._response

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> HttpxResponse:
        return await self.post(url, headers=headers, json=None, files=None)


class FakeHttpxAsyncClientRaises:
    """Fake async client that always raises on post/get."""

    __slots__ = ("_exception",)

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> NoReturn:
        _ = (url, headers, json, files)
        raise self._exception

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> NoReturn:
        raise self._exception


class FakeHttpxClient:
    """Protocol-compliant fake for httpx.Client (sync).

    Satisfies HttpxClient Protocol from platform_core.http_client.
    """

    __slots__ = ("_response", "seen_headers", "seen_urls")

    def __init__(self, response: HttpxResponse | None = None) -> None:
        self._response: HttpxResponse | None = response
        self.seen_headers: dict[str, str] = {}
        self.seen_urls: list[str] = []

    def close(self) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self.seen_urls.append(url)
        self.seen_headers.update(headers)
        _ = (json, files)

        if self._response is None:
            raise RuntimeError("No response configured for FakeHttpxClient")
        return self._response


class FakeHttpxClientRaises:
    """Fake sync client that always raises on post."""

    __slots__ = ("_exception",)

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    def close(self) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> NoReturn:
        _ = (url, headers, json, files)
        raise self._exception


# Protocol for creating typed Timeout constructor
class TimeoutCtor(Protocol):
    """Protocol for Timeout constructor."""

    def __call__(self, timeout: float) -> Timeout: ...


# Protocol for creating typed async client constructor
class AsyncClientCtor(Protocol):
    """Protocol for AsyncClient constructor."""

    def __call__(self, *, timeout: Timeout) -> HttpxAsyncClient: ...


# Protocol for creating typed sync client constructor
class ClientCtor(Protocol):
    """Protocol for Client constructor."""

    def __call__(self, *, timeout: Timeout) -> HttpxClient: ...


def make_timeout_ctor() -> TimeoutCtor:
    """Create a typed Timeout constructor returning FakeTimeout."""

    def ctor(timeout: float) -> Timeout:
        return FakeTimeout(timeout)

    return ctor


def make_async_client_ctor(response: HttpxResponse) -> AsyncClientCtor:
    """Create a typed AsyncClient constructor returning FakeHttpxAsyncClient."""

    def ctor(*, timeout: Timeout) -> HttpxAsyncClient:
        _ = timeout
        return FakeHttpxAsyncClient(response)

    return ctor


def make_client_ctor(response: HttpxResponse) -> ClientCtor:
    """Create a typed Client constructor returning FakeHttpxClient."""

    def ctor(*, timeout: Timeout) -> HttpxClient:
        _ = timeout
        return FakeHttpxClient(response)

    return ctor


class FakeHttpxModule:
    """Fake httpx module for monkeypatching sys.modules.

    Use with: monkeypatch.setitem(sys.modules, "httpx", fake_module)
    """

    def __init__(
        self,
        response: HttpxResponse,
        *,
        async_client: bool = False,
    ) -> None:
        object.__setattr__(self, "Timeout", make_timeout_ctor())
        if async_client:
            object.__setattr__(self, "AsyncClient", make_async_client_ctor(response))
        else:
            object.__setattr__(self, "Client", make_client_ctor(response))


class FakeHttpxModuleSyncOnly:
    """Fake httpx module with only sync Client (for sync-only code)."""

    def __init__(self, response: HttpxResponse) -> None:
        object.__setattr__(self, "Timeout", make_timeout_ctor())
        object.__setattr__(self, "Client", make_client_ctor(response))


__all__ = [
    "AsyncClientCtor",
    "ClientCtor",
    "FakeHttpxAsyncClient",
    "FakeHttpxAsyncClientRaises",
    "FakeHttpxClient",
    "FakeHttpxClientRaises",
    "FakeHttpxModule",
    "FakeHttpxModuleSyncOnly",
    "FakeHttpxResponse",
    "FakeTimeout",
    "TimeoutCtor",
    "make_async_client_ctor",
    "make_client_ctor",
    "make_timeout_ctor",
]
