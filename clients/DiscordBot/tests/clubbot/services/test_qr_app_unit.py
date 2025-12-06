from __future__ import annotations

from tests.conftest import _build_settings

from clubbot.services.qr.client import QRRequestPayload, QRService


def test_generate_qr_calls_build_and_returns_result() -> None:
    cfg = _build_settings(qr_default_border=2, qr_api_url="http://localhost:8080")

    # Track calls with proper types
    called_png = False
    called_payload: QRRequestPayload | None = None

    class _FakeClient:
        def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
            nonlocal called_png, called_payload
            called_png = True
            called_payload = payload
            return b"\x89PNG\r\n\x1a\n"

    svc = QRService(cfg, client=_FakeClient())

    out = svc.generate_qr("https://example.com")
    assert out.url == "https://example.com"
    assert called_png
    assert called_payload == {"url": "https://example.com"}
