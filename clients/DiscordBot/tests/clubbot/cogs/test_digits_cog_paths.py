from __future__ import annotations

import logging

import pytest
from platform_core.errors import AppError
from platform_discord.protocols import InteractionProto
from tests.support.discord_fakes import (
    FakeAttachment,
    FakeBot,
    FakeDigitService,
    FakeInteraction,
    FakeUser,
)
from tests.support.settings import build_settings

from clubbot.cogs.base import _Logger
from clubbot.cogs.digits import DigitsCog
from clubbot.config import DiscordbotSettings
from clubbot.services.handai.client import HandwritingAPIError, PredictResult


class _RecordingDigitService(FakeDigitService):
    """DigitService fake that can raise configured errors."""

    def __init__(self) -> None:
        super().__init__(max_mb=1)
        self._err: Exception | None = None
        self._res: PredictResult = PredictResult(
            digit=7,
            confidence=0.9,
            probs=(0.1,) * 10,
            model_id="m",
            uncertain=False,
            latency_ms=10,
        )

    def set_error(self, err: Exception | None) -> None:
        self._err = err

    async def read_image(
        self, *, data: bytes, filename: str, content_type: str, request_id: str
    ) -> PredictResult:
        _ = (data, filename, content_type, request_id)
        if self._err is not None:
            raise self._err
        return self._res


def _cfg() -> DiscordbotSettings:
    return build_settings(
        handwriting_api_url="https://api",
        handwriting_api_timeout_seconds=1,
        handwriting_api_max_retries=0,
        digits_max_image_mb=1,
        digits_rate_limit=5,
    )


@pytest.mark.asyncio
async def test_read_early_ack_return() -> None:
    class _Cog(DigitsCog):
        async def _safe_defer(self, interaction: InteractionProto, *, ephemeral: bool) -> bool:
            _ = interaction
            _ = ephemeral
            return False

    inter = FakeInteraction()
    cfg = _cfg()
    service = _RecordingDigitService()
    cog = _Cog(bot=FakeBot(), config=cfg, service=service)
    attachment = FakeAttachment(filename="test.png", content_type="image/png", size=1024, data=b"x")
    await cog._read_impl(inter, inter.user, attachment)


@pytest.mark.asyncio
async def test_read_user_id_none_triggers_user_error() -> None:
    messages: list[str] = []

    class _Cog(DigitsCog):
        async def handle_user_error(
            self, interaction: InteractionProto, log: _Logger, message: str
        ) -> None:
            _ = (interaction, log)
            messages.append(message)

    inter = FakeInteraction()
    cfg = _cfg()
    service = _RecordingDigitService()
    cog = _Cog(bot=FakeBot(), config=cfg, service=service)
    attachment = FakeAttachment(filename="test.png", content_type="image/png", size=1024, data=b"x")
    await cog._read_impl(inter, None, attachment)
    assert messages


@pytest.mark.asyncio
async def test_read_handles_api_and_generic_errors() -> None:
    messages: list[str] = []

    class _Cog(DigitsCog):
        async def handle_user_error(
            self, interaction: InteractionProto, log: _Logger, message: str
        ) -> None:
            _ = (interaction, log)
            messages.append(message)

        async def handle_exception(
            self, interaction: InteractionProto, log: _Logger, exc: Exception
        ) -> None:
            _ = (interaction, log, exc)
            messages.append("exc")

    cfg = _cfg()
    svc = _RecordingDigitService()
    inter = FakeInteraction(user=FakeUser(user_id=1))
    cog = _Cog(bot=FakeBot(), config=cfg, service=svc)

    svc.set_error(HandwritingAPIError(400, "bad", code="invalid_image"))
    attachment1 = FakeAttachment(filename="x", content_type="image/png", size=1024, data=b"1")
    await cog._read_impl(inter, inter.user, attachment1)

    svc.set_error(RuntimeError("boom"))
    attachment2 = FakeAttachment(filename="x", content_type="image/png", size=1024, data=b"2")
    await cog._read_impl(inter, inter.user, attachment2)

    svc.set_error(HandwritingAPIError(500, "oops"))
    attachment3 = FakeAttachment(filename="x", content_type="image/png", size=1024, data=b"3")
    await cog._read_impl(inter, inter.user, attachment3)

    assert messages and any(msg for msg in messages if msg == "exc")


def test_extract_int_attr_and_validate_attachment_size() -> None:
    cfg = _cfg()
    service = _RecordingDigitService()
    cog = DigitsCog(bot=FakeBot(), config=cfg, service=service)
    assert DigitsCog.decode_int_attr(None, "id") is None

    att = FakeAttachment(
        filename="large.png",
        content_type="image/png",
        size=(cfg["digits"]["max_image_mb"] * 1024 * 1024) + 1,
        data=b"x",
    )
    with pytest.raises(AppError):
        cog._validate_attachment(att)


logger = logging.getLogger(__name__)
