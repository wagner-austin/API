from __future__ import annotations

from collections.abc import Mapping

import pytest
from platform_core.errors import AppError
from platform_core.http_client import HttpxResponse
from platform_core.json_utils import JSONValue, load_json_str
from tests.conftest import _build_settings

from clubbot.services.qr.client import QRHttpClient, QRService


class _FakeResp:
    """Fake response satisfying HttpxResponse Protocol."""

    def __init__(self, status: int, content: bytes, text: str | None = None) -> None:
        self.status_code: int = status
        self.content: bytes | bytearray = content
        self.text: str = text if text is not None else ""
        self.headers: Mapping[str, str] = {}

    def json(self) -> JSONValue:
        return load_json_str(self.text)


class _FakeClient:
    """Fake client satisfying HttpxClient Protocol."""

    def __init__(self, status: int, content: bytes, text: str | None = None) -> None:
        self._status = status
        self._content = content
        self._text = text

    def close(self) -> None:
        pass

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        return _FakeResp(self._status, self._content, self._text)


def _png() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"x" * 10


def test_qr_http_client_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QRHttpClient("http://api")
    client._client = _FakeClient(200, _png())
    out = client.png(
        payload={"url": "https://x"},
        request_id="req",
    )
    assert isinstance(out, (bytes | bytearray)) and out[:8] == b"\x89PNG\r\n\x1a\n"


def test_qr_http_client_non_200_raises() -> None:
    client = QRHttpClient("http://api")
    client._client = _FakeClient(500, b"")
    with pytest.raises(RuntimeError):
        client.png(
            payload={"url": "https://x"},
            request_id="req",
        )


def test_qr_http_client_invalid_png_raises() -> None:
    client = QRHttpClient("http://api")
    client._client = _FakeClient(200, b"notpng")
    with pytest.raises(RuntimeError):
        client.png(
            payload={"url": "https://x"},
            request_id="req",
        )


def test_qr_http_client_400_app_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QRHttpClient("http://api")
    err_json = '{"code":"INVALID_INPUT","message":"Invalid URL"}'
    client._client = _FakeClient(400, b"", err_json)
    with pytest.raises(AppError) as excinfo:
        client.png(payload={"url": "https://x"}, request_id="req")
    assert "Invalid URL" in str(excinfo.value)


def test_qr_service_missing_base_raises() -> None:
    cfg = _build_settings(qr_default_border=2, qr_api_url="")
    svc = QRService(cfg)
    with pytest.raises(RuntimeError):
        svc.generate_qr("https://x")
