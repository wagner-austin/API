from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Protocol

import pytest
from platform_core.http_client import HttpxClient, HttpxResponse, Timeout
from platform_core.json_utils import JSONValue

from clubbot.services.transcript.api_client import TranscriptApiClient, captions


class _FakeTimeout:
    """Fake timeout that satisfies Timeout Protocol."""

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"Timeout({self._timeout})"


class _FakeResp:
    """Fake response with 200 status and raw text."""

    def __init__(self, code: int, text: str) -> None:
        self.status_code = code
        self.text = text
        self.headers: Mapping[str, str] = {}
        self.content: bytes | bytearray = b""

    def json(self) -> JSONValue:
        raise ValueError("no json")


class _FakeClient:
    """Fake client that returns 200 response with non-dict text."""

    def __init__(self, *, text: str) -> None:
        self._resp = _FakeResp(200, text)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        _ = (url, headers, json, files)
        return self._resp

    def close(self) -> None:
        return None


class _TimeoutCtor(Protocol):
    def __call__(self, timeout: float) -> Timeout: ...


class _ClientCtor(Protocol):
    def __call__(self, *, timeout: Timeout) -> HttpxClient: ...


def _make_timeout_ctor() -> _TimeoutCtor:
    """Create a typed Timeout constructor."""

    def ctor(timeout: float) -> Timeout:
        return _FakeTimeout(timeout)

    return ctor


def _make_client_ctor(text: str) -> _ClientCtor:
    """Create a typed Client constructor."""

    def ctor(*, timeout: Timeout) -> HttpxClient:
        _ = timeout
        return _FakeClient(text=text)

    return ctor


class _FakeHttpxModule:
    """Fake httpx module with properly typed attributes."""

    def __init__(self, text: str) -> None:
        timeout_ctor = _make_timeout_ctor()
        client_ctor = _make_client_ctor(text)

        def _async_client(*, timeout: Timeout) -> HttpxClient:
            _ = timeout
            return client_ctor(timeout=timeout)

        object.__setattr__(self, "Timeout", timeout_ctor)
        object.__setattr__(self, "AsyncClient", _async_client)
        object.__setattr__(self, "Client", client_ctor)


def test_transcript_api_client_unexpected_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpxModule('"not a dict"'))

    client: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(client, url="https://x", preferred_langs=["en"])  # triggers unexpected format
