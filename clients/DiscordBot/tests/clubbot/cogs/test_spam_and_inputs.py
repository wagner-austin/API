from __future__ import annotations

import asyncio
import logging
import time

import pytest
from platform_core.errors import AppError, ErrorCode
from tests.support.discord_fakes import FakeBot, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.qr import QRCog
from clubbot.config import DiscordbotSettings
from clubbot.services.qr.client import QRClient, QRRequestPayload, QRResult, QRService


def make_cfg(per: int = 1000, window: int = 1) -> DiscordbotSettings:
    return build_settings(
        qr_rate_limit=per,
        qr_rate_window_seconds=window,
        qr_default_border=2,
        qr_public_responses=True,
    )


class _QRResult(QRResult):
    def __init__(self, image_png: bytes, url: str) -> None:
        super().__init__(image_png=image_png, url=url)


class SlowQRService(QRService):
    def __init__(self, cfg: DiscordbotSettings, delay: float = 0.1) -> None:
        super().__init__(cfg)
        self.delay = delay

    def generate_qr(self, url: str) -> QRResult:
        time.sleep(self.delay)
        return _QRResult(image_png=b"\x89PNG\r\n\x1a\n", url=url)


class _RejectingClient(QRClient):
    def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
        _ = (payload, request_id)
        raise AppError(ErrorCode.INVALID_INPUT, "Invalid URL", http_status=400)


@pytest.mark.asyncio
async def test_qrcode_spam_concurrent_calls_complete_without_errors() -> None:
    bot = FakeBot()
    cfg = make_cfg(per=1000, window=1)
    svc = SlowQRService(cfg, delay=0.05)
    cog = QRCog(bot, cfg, svc)

    n = 10
    ctxs = [RecordingInteraction() for _ in range(n)]

    async def run_one(inter: RecordingInteraction) -> None:
        await cog._qrcode_impl(inter, "example.com")

    await asyncio.gather(*(run_one(c) for c in ctxs))

    assert all(c.sent for c in ctxs)
    assert all(c.sent[-1]["file"] is not None for c in ctxs)


@pytest.mark.asyncio
async def test_qrcode_handles_various_invalid_inputs_with_clear_messages() -> None:
    bot = FakeBot()
    cfg = make_cfg(per=1000, window=1)
    svc = QRService(cfg, client=_RejectingClient())
    cog = QRCog(bot, cfg, svc)

    bad_inputs = [
        "msn",
        "msn. c om",
        "msncom",
    ]

    for raw in bad_inputs:
        ctx = RecordingInteraction()
        await cog._qrcode_impl(ctx, raw)
        assert ctx.sent, f"Expected an error response for input: {raw!r}"
        msg = str(ctx.sent[-1]["content"] or "")
        assert (
            ("Please provide" in msg)
            or ("Invalid URL" in msg)
            or ("URL host is required" in msg)
            or ("Please check the URL and try again." in msg)
        )


logger = logging.getLogger(__name__)
