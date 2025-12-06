from __future__ import annotations

import pytest

from clubbot.services.qr.client import QRHttpClient, QRRequestPayload, QRResult, QRService


def test_qrresult_getitem_key_error() -> None:
    res = QRResult(image_png=b"\x89PNG\r\n\x1a\n", url="u")
    with pytest.raises(KeyError):
        _ = res["nope"]


def test_decode_app_error_non_object_raises() -> None:
    client = QRHttpClient("http://api")
    with pytest.raises(RuntimeError):
        _ = client._decode_app_error("[1,2,3]", 400)


def test_decode_app_error_missing_fields_raises() -> None:
    client = QRHttpClient("http://api")
    with pytest.raises(RuntimeError):
        _ = client._decode_app_error("{}", 400)


def test_decode_app_error_unrecognized_code_raises() -> None:
    client = QRHttpClient("http://api")
    body = '{"code":"NOT_A_CODE","message":"m"}'
    with pytest.raises(RuntimeError):
        _ = client._decode_app_error(body, 400)


def test_qrservice_generate_uses_client() -> None:
    class _Client:
        def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
            _ = request_id
            return b"\x89PNG\r\n\x1a\nXXXX"

    from tests.support.settings import build_settings

    cfg = build_settings(qr_api_url="http://api", qr_public_responses=True)
    svc = QRService(cfg, client=_Client())
    out = svc.generate_qr_with_payload({"url": "https://x"})
    assert isinstance(out, QRResult) and out["url"] == "https://x"
