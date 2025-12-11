from __future__ import annotations

from collections.abc import Mapping

import pytest
from platform_core.errors import AppError
from platform_core.http_client import HttpxClient, HttpxResponse
from platform_core.json_utils import JSONValue, dump_json_str

from clubbot import _test_hooks
from clubbot.services.transcript.api_client import TranscriptApiClient, captions, stt


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


def _make_fake_client_builder(
    status: int, payload: dict[str, JSONValue] | None
) -> _test_hooks.BuildClientProtocol:
    """Create a build_client hook that returns a fake client."""

    def builder(timeout: float) -> HttpxClient:
        _ = timeout
        return _FakeClient(status=status, payload=payload)

    return builder


def test_captions_success() -> None:
    _test_hooks.build_client = _make_fake_client_builder(
        200, {"url": "https://x", "video_id": "vid", "text": "ok"}
    )

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    out = captions(api, url="https://x", preferred_langs=["en"])
    assert out["video_id"] == "vid" and out["text"] == "ok"


def test_captions_400_maps_app_error() -> None:
    _test_hooks.build_client = _make_fake_client_builder(400, {"detail": "bad"})

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(AppError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_captions_non_200_maps_runtime() -> None:
    _test_hooks.build_client = _make_fake_client_builder(500, {})

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_stt_success() -> None:
    _test_hooks.build_client = _make_fake_client_builder(
        200, {"url": "https://x", "video_id": "vid", "text": "ok"}
    )

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    out = stt(api, url="https://x")
    assert out["video_id"] == "vid" and out["text"] == "ok"


def test_captions_400_invalid_json() -> None:
    _test_hooks.build_client = _make_fake_client_builder(400, None)

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(AppError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_captions_200_non_dict_payload() -> None:
    """Test response that returns list instead of dict."""

    class _RespList(_FakeResp):
        def json(self) -> JSONValue:
            return [1, 2, 3]

    class _ClientList:
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

        def close(self) -> None:
            return None

    def _builder(timeout: float) -> HttpxClient:
        _ = timeout
        return _ClientList()

    _test_hooks.build_client = _builder

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(api, url="https://x", preferred_langs=None)


def test_captions_200_invalid_fields() -> None:
    _test_hooks.build_client = _make_fake_client_builder(200, {"url": 1, "video_id": 2, "text": 3})

    api: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(RuntimeError):
        _ = captions(api, url="https://x", preferred_langs=None)
