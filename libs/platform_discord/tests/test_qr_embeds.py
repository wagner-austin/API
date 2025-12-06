from __future__ import annotations

from platform_discord.qr.embeds import build_qr_embed, build_qr_error_embed
from platform_discord.qr.types import QRRequest


def test_qr_embeds() -> None:
    _rq: QRRequest = {"url": "https://x"}
    e = build_qr_embed(url="https://x")
    d = e.to_dict()
    assert d.get("title") == "QR Code Generated"
    e2 = build_qr_error_embed(message="bad")
    d2 = e2.to_dict()
    assert d2.get("title") == "QR Generation Failed"
