from __future__ import annotations

from clubbot.services.qr.client import QRHttpClient, QRRequestPayload


def test_qr_http_build_body_ignores_wrong_types() -> None:
    # Use dynamic constructor to avoid typing conflicts while exercising wrong-type branches
    payload: QRRequestPayload = __import__("builtins").dict(
        url="https://example",
        ecc=123,
        box_size="7",
        border="2",
        fill_color=10,
        back_color=20,
    )
    body = QRHttpClient._build_body(payload)
    # Only 'url' should be included from the above
    assert body == {"url": "https://example"}
