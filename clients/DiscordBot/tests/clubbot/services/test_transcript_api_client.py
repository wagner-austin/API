from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Protocol

import pytest
from platform_core.errors import AppError
from platform_core.http_client import HttpxClient, HttpxResponse, Timeout
from platform_core.json_utils import JSONValue, dump_json_str

from clubbot.services.transcript.api_client import TranscriptApiClient, captions, stt


class _FakeTimeout:
    """Fake timeout that satisfies Timeout Protocol."""

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"Timeout({self._timeout})"


class _FakeResp:
    """Fake response that satisfies HttpxResponse Protocol."""

    def __init__(self, status: int, payload: dict[str, JSONValue] | None) -> None:
        self.status_code = status
        self._payload: dict[str, JSONValue] | None = payload
        self.headers: Mapping[str, str] = {}
        self.content: bytes | bytearray = b""
        self.text = "{}" if payload is None else dump_json_str(payload)

    def json(self) -> JSONValue:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeClient:
    """Fake client that satisfies HttpxClient Protocol."""

    def __init__(self, *, status: int, payload: dict[str, JSONValue] | None) -> None:
        self._status = status
        self._payload = payload

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        _ = (url, headers, json, files)
        return _FakeResp(self._status, self._payload)

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


def _make_client_ctor(status: int, payload: dict[str, JSONValue] | None) -> _ClientCtor:
    """Create a typed Client constructor."""

    def ctor(*, timeout: Timeout) -> HttpxClient:
        _ = timeout
        return _FakeClient(status=status, payload=payload)

    return ctor


class _FakeHttpxModule:
    """Fake httpx module with properly typed attributes."""

    def __init__(self, status: int, payload: dict[str, JSONValue] | None) -> None:
        timeout_ctor = _make_timeout_ctor()
        client_ctor = _make_client_ctor(status, payload)

        def _async_client(*, timeout: Timeout) -> HttpxClient:
            _ = timeout
            return client_ctor(timeout=timeout)

        object.__setattr__(self, "Timeout", timeout_ctor)
        object.__setattr__(self, "AsyncClient", _async_client)
        object.__setattr__(self, "Client", client_ctor)


def _fake_httpx_module(status: int, payload: dict[str, JSONValue] | None) -> _FakeHttpxModule:
    return _FakeHttpxModule(status, payload)


def test_captions_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = _fake_httpx_module(200, {"url": "https://x", "video_id": "vid", "text": "ok"})
    monkeypatch.setitem(sys.modules, "httpx", fake_mod)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    out = captions(api, url="https://x", preferred_langs=["en"])
    assert out["video_id"] == "vid" and out["text"] == "ok"


def test_captions_400_maps_app_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = _fake_httpx_module(400, {"detail": "bad"})
    monkeypatch.setitem(sys.modules, "httpx", fake_mod)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(AppError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_captions_non_200_maps_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = _fake_httpx_module(500, {})
    monkeypatch.setitem(sys.modules, "httpx", fake_mod)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_stt_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = _fake_httpx_module(200, {"url": "https://x", "video_id": "vid", "text": "ok"})
    monkeypatch.setitem(sys.modules, "httpx", fake_mod)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    out = stt(api, url="https://x")
    assert out["video_id"] == "vid" and out["text"] == "ok"


def test_captions_400_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = _fake_httpx_module(400, None)
    monkeypatch.setitem(sys.modules, "httpx", fake_mod)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(AppError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_captions_200_non_dict_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test response that returns list instead of dict."""

    class _RespList(_FakeResp):
        def json(self) -> JSONValue:
            return [1, 2, 3]

    class _ClientList(_FakeClient):
        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            json: JSONValue | None = None,
            files: Mapping[str, tuple[str, bytes, str]] | None = None,
        ) -> HttpxResponse:
            _ = (url, headers, json, files)
            return _RespList(200, None)

    class _FakeHttpxModuleList:
        def __init__(self) -> None:
            timeout_ctor = _make_timeout_ctor()

            def _client_ctor(*, timeout: Timeout) -> HttpxClient:
                _ = timeout
                return _ClientList(status=200, payload=None)

            def _async_client(*, timeout: Timeout) -> HttpxClient:
                return _client_ctor(timeout=timeout)

            object.__setattr__(self, "Timeout", timeout_ctor)
            object.__setattr__(self, "AsyncClient", _async_client)
            object.__setattr__(self, "Client", _client_ctor)

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpxModuleList())

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_captions_200_invalid_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = _fake_httpx_module(200, {"url": 1, "video_id": 2, "text": 3})
    monkeypatch.setitem(sys.modules, "httpx", fake_mod)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(api, url="https://x", preferred_langs=None)
