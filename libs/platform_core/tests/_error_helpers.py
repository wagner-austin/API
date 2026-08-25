"""Shared helpers for the platform_core error-handling tests.

``_parse_response_body`` decodes a handler's JSON response into a plain
str-to-str mapping. It lives here rather than in one test module so the
handler and service-code test files share a single definition.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from platform_core._asgi_protocols import _JSONResponseProto, _RequestProto
from platform_core.json_utils import load_json_bytes


def parse_response_body(response: _JSONResponseProto) -> dict[str, str]:
    """Parse a JSON response body into a mapping of strings.

    Args:
        response: Response whose body holds a JSON object of strings.

    Returns:
        The decoded object, with every key and value asserted to be a string.
    """
    body_bytes = response.body if isinstance(response.body, bytes) else bytes(response.body)
    content = load_json_bytes(body_bytes)
    assert isinstance(content, dict)
    result: dict[str, str] = {}
    for key, value in content.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        result[key] = value
    return result


@runtime_checkable
class _FakeURL(Protocol):
    @property
    def path(self) -> str: ...


@runtime_checkable
class _FakeRequest(Protocol):
    @property
    def url(self) -> _FakeURL: ...

    @property
    def method(self) -> str: ...


class FakeURL:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path


class FakeRequest:
    def __init__(self, path: str, method: str) -> None:
        self._url = FakeURL(path)
        self._method = method

    @property
    def url(self) -> FakeURL:
        return self._url

    @property
    def method(self) -> str:
        return self._method


class FakeJSONResponse:
    def __init__(self, content: dict[str, str], status_code: int) -> None:
        self.content = content
        self.status_code = status_code


@runtime_checkable
class _ExceptionHandlerProto(Protocol):
    """Protocol for exception handler callable."""

    async def __call__(self, request: _FakeRequest, exc: Exception) -> FakeJSONResponse: ...


class FakeFastAPIApp:
    def __init__(self) -> None:
        self.handlers: dict[
            type[Exception], Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]]
        ] = {}

    def add_exception_handler(
        self,
        exc_class_or_status_code: int | type[Exception],
        handler: Callable[[_RequestProto, Exception], Awaitable[_JSONResponseProto]],
    ) -> None:
        if not isinstance(exc_class_or_status_code, int):
            self.handlers[exc_class_or_status_code] = handler


__all__ = [
    "FakeFastAPIApp",
    "FakeJSONResponse",
    "FakeRequest",
    "FakeURL",
    "parse_response_body",
]
