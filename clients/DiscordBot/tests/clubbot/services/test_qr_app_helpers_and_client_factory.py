from __future__ import annotations

from platform_core.http_client import JsonObject
from tests.support.settings import build_settings

from clubbot.services.qr.client import (
    QRHttpClient,
    QRRequestPayload,
    QRService,
    _QRHttpClientHelpers,
)


def test_qr_http_client_factory_and_helper_copies() -> None:
    # copy_str and copy_int cover positive branches
    payload: QRRequestPayload = {
        "url": "https://example.com",
        "ecc": "M",
        "box_size": 4,
        "border": 2,
        "fill_color": "black",
        "back_color": "white",
    }
    target: JsonObject = {}
    _QRHttpClientHelpers.copy_str(payload, target, "url")
    _QRHttpClientHelpers.copy_str(payload, target, "ecc")
    _QRHttpClientHelpers.copy_int(payload, target, "box_size")
    _QRHttpClientHelpers.copy_int(payload, target, "border")
    _QRHttpClientHelpers.copy_str(payload, target, "fill_color")
    _QRHttpClientHelpers.copy_str(payload, target, "back_color")
    assert target["url"] == "https://example.com" and target["box_size"] == 4

    # _get_client constructs QRHttpClient when not provided
    cfg = build_settings(qr_api_url="http://api")
    svc = QRService(cfg)
    client = svc._get_client()
    assert type(client) is QRHttpClient


def test_qr_http_client_helper_missing_keys_branches() -> None:
    empty: QRRequestPayload = {}
    target: JsonObject = {}
    # Keys not present; copy helpers should not mutate target
    _QRHttpClientHelpers.copy_str(empty, target, "url")
    _QRHttpClientHelpers.copy_int(empty, target, "box_size")
    assert target == {}
