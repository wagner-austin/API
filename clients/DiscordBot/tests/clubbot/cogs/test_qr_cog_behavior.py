from __future__ import annotations

import logging

import pytest
from platform_core.errors import AppError, ErrorCode
from tests.support.discord_fakes import FakeBot, RecordedSend, RecordingInteraction
from tests.support.settings import SettingsFactory

from clubbot.cogs.qr import QRCog
from clubbot.services.qr.client import QRClient, QRRequestPayload, QRService


class _FakeClient(QRClient):
    """QR client that returns valid PNG bytes."""

    def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
        _ = (payload, request_id)
        return b"\x89PNG\r\n\x1a\n"


class _FailingClient(QRClient):
    """QR client that raises a RuntimeError."""

    def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
        _ = (payload, request_id)
        raise RuntimeError("boom")


class _InvalidUrlClient(QRClient):
    """QR client that raises AppError for invalid URLs."""

    def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
        _ = request_id
        url_val = payload.get("url")
        if not isinstance(url_val, str) or " " in url_val:
            raise AppError(ErrorCode.INVALID_INPUT, "Invalid URL", http_status=400)
        return b"\x89PNG\r\n\x1a\n"


def _last_send(inter: RecordingInteraction) -> RecordedSend:
    assert inter.sent, "Expected at least one send"
    return inter.sent[-1]


@pytest.mark.asyncio
async def test_qrcode_accepts_bare_hostname_and_replies_with_file(
    settings_factory: SettingsFactory,
) -> None:
    bot = FakeBot()
    cfg = settings_factory(qr_rate_limit=1_000_000, qr_rate_window_seconds=1)
    svc = QRService(cfg, client=_FakeClient())
    cog = QRCog(bot, cfg, svc)
    interaction = RecordingInteraction()

    await cog._qrcode_impl(interaction, "example.com")

    last = _last_send(interaction)
    if last["file"] is None:
        raise AssertionError("expected file")
    assert last["where"] == "followup"


@pytest.mark.asyncio
async def test_qrcode_invalid_url_returns_user_message(
    settings_factory: SettingsFactory,
) -> None:
    bot = FakeBot()
    cfg = settings_factory()
    svc = QRService(cfg, client=_InvalidUrlClient())
    cog = QRCog(bot, cfg, svc)
    interaction = RecordingInteraction()

    await cog._qrcode_impl(interaction, "not a url with spaces")

    last = _last_send(interaction)
    msg = str(last["content"]) if last["content"] is not None else ""
    has_error = (
        ("Invalid URL" in msg)
        or ("Please provide" in msg)
        or ("Please check the URL and try again later" in msg)
        or ("Please check the URL and try again." in msg)
    )
    assert has_error
    assert last["ephemeral"] is True


@pytest.mark.asyncio
async def test_qrcode_rate_limit_message_on_second_call(
    settings_factory: SettingsFactory,
) -> None:
    bot = FakeBot()
    cfg = settings_factory(qr_rate_limit=1, qr_rate_window_seconds=1, qr_public_responses=True)
    svc = QRService(cfg, client=_FakeClient())
    cog = QRCog(bot, cfg, svc)
    interaction = RecordingInteraction()

    await cog._qrcode_impl(interaction, "https://example.com")
    await cog._qrcode_impl(interaction, "https://example.com")

    last = _last_send(interaction)
    content = last["content"] or ""
    assert content.startswith("Please wait ")
    assert last["ephemeral"] == (not cfg["qr"]["public_responses"])


@pytest.mark.asyncio
async def test_qrcode_handles_internal_exception_with_generic_message(
    settings_factory: SettingsFactory,
) -> None:
    bot = FakeBot()
    cfg = settings_factory(qr_rate_limit=1_000_000, qr_rate_window_seconds=1)
    svc = QRService(cfg, client=_FailingClient())
    cog = QRCog(bot, cfg, svc)
    interaction = RecordingInteraction()

    await cog._qrcode_impl(interaction, "https://example.com")

    last = _last_send(interaction)
    content = last["content"] or ""
    assert content.startswith("An error occurred")
    assert last["ephemeral"] is True


logger = logging.getLogger(__name__)
