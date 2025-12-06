from __future__ import annotations

import pytest
from tests.support.settings import build_settings

from clubbot.services.qr.client import QRHttpClient, QRResult, QRService


def test_qrresult_getitem_image_png() -> None:
    res = QRResult(image_png=b"\x89PNG\r\n\x1a\nxxxx", url="u")
    # Access image_png branch
    head = res["image_png"][:8]
    assert isinstance(head, (bytes, bytearray)) and head == b"\x89PNG\r\n\x1a\n"


def test_qrhttpclient_close_no_error() -> None:
    client = QRHttpClient("http://api")
    client.close()


def test_qrhttpclient_png_missing_url_raises() -> None:
    client = QRHttpClient("http://api")
    with pytest.raises(ValueError):
        _ = client.png(payload={}, request_id="r")


def test_qrservice_generate_qr_with_payload_missing_url_raises() -> None:
    cfg = build_settings(qr_api_url="http://api")
    svc = QRService(cfg)
    with pytest.raises(ValueError):
        _ = svc.generate_qr_with_payload({})
