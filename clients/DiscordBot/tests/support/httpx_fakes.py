from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str


class Request(Protocol):
    method: str
    url: str
    headers: Mapping[str, str]
    json: JSONValue | None


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        text: str,
        headers: Mapping[str, str],
        content: bytes,
        json_value: JSONValue,
    ) -> None:
        self.status_code = int(status_code)
        self.text = text if text else dump_json_str(json_value)
        self.headers: Mapping[str, str] = dict(headers)
        self.content: bytes | bytearray = content if content else self.text.encode("utf-8")
        self._json_value = json_value

    def json(self) -> JSONValue:
        return self._json_value


class RequestHandler(Protocol):
    def __call__(self, request: Request) -> HttpxResponse: ...


class FakeHttpxAsyncClient:
    def __init__(self, handler: RequestHandler) -> None:
        self._handler = handler

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
        _ = files  # unused in fake
        req: Request = _SimpleRequest(method="POST", url=url, headers=headers, json=json)
        return self._handler(req)

    async def get(self, url: str, *, headers: Mapping[str, str]) -> HttpxResponse:
        req: Request = _SimpleRequest(method="GET", url=url, headers=headers, json=None)
        return self._handler(req)


class _SimpleRequest:
    __slots__ = ("headers", "json", "method", "url")

    def __init__(
        self, *, method: str, url: str, headers: Mapping[str, str], json: JSONValue | None
    ) -> None:
        self.method = method
        self.url = url
        self.headers: Mapping[str, str] = dict(headers)
        self.json = json


__all__ = ["FakeHttpxAsyncClient", "FakeResponse", "Request", "RequestHandler"]
