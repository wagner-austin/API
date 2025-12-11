from __future__ import annotations

from collections.abc import Mapping

import pytest
from platform_core.errors import AppError
from platform_core.http_client import HttpxClient, HttpxResponse
from platform_core.json_utils import JSONValue

from clubbot import _test_hooks
from clubbot.services.transcript.api_client import TranscriptApiClient, captions


class _FakeResp:
    """Fake response with 400 status and raw text."""

    def __init__(self, code: int, text: str) -> None:
        self.status_code = code
        self.text = text
        self.headers: Mapping[str, str] = {}
        self.content: bytes | bytearray = b""

    def json(self) -> JSONValue:
        raise ValueError("no json")


class _FakeClient:
    """Fake client that returns 400 response."""

    def __init__(self, *, text: str) -> None:
        self._resp = _FakeResp(400, text)

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


def _make_fake_client_builder(text: str) -> _test_hooks.BuildClientProtocol:
    """Create a build_client hook that returns a fake client with given text."""

    def builder(timeout: float) -> HttpxClient:
        _ = timeout
        return _FakeClient(text=text)

    return builder


def test_transcript_api_client_400_generic_invalid() -> None:
    _test_hooks.build_client = _make_fake_client_builder("not-json")
    client: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(AppError):
        _ = captions(client, url="x", preferred_langs=None)


def test_transcript_api_client_400_json_no_detail() -> None:
    _test_hooks.build_client = _make_fake_client_builder("{}")
    client: TranscriptApiClient = {"base_url": "http://api", "timeout_seconds": 1.0}
    with pytest.raises(AppError):
        _ = captions(client, url="x", preferred_langs=None)
