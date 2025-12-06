from __future__ import annotations

import logging

import pytest
from platform_discord.protocols import InteractionProto
from tests.support.discord_fakes import (
    FakeAttachment,
    FakeBot,
    FakeDigitService,
    FakeUser,
    RecordingInteraction,
)
from tests.support.settings import build_settings

from clubbot.cogs.base import _Logger
from clubbot.cogs.digits import DigitsCog
from clubbot.config import DiscordbotSettings
from clubbot.services.handai.client import HandwritingAPIError, PredictResult


class _ConfigurableDigitService(FakeDigitService):
    """DigitService fake that can return custom results or raise errors."""

    def __init__(
        self,
        *,
        result: PredictResult | None = None,
        err: Exception | None = None,
        max_mb: int = 2,
    ) -> None:
        super().__init__(max_mb=max_mb)
        self._result = result
        self._err = err

    def set_error(self, err: Exception | None) -> None:
        self._err = err

    async def read_image(
        self, *, data: bytes, filename: str, content_type: str, request_id: str
    ) -> PredictResult:
        _ = (data, filename, content_type, request_id)
        if self._err is not None:
            raise self._err
        if self._result is not None:
            return self._result
        return await super().read_image(
            data=data, filename=filename, content_type=content_type, request_id=request_id
        )


def _make_cfg(public: bool = True, limit: int = 5, window: int = 60) -> DiscordbotSettings:
    return build_settings(
        qr_default_border=2,
        qr_public_responses=True,
        digits_public_responses=public,
        digits_rate_limit=limit,
        digits_rate_window_seconds=window,
        digits_max_image_mb=2,
        handwriting_api_url="http://localhost:7000",
    )


@pytest.mark.asyncio
async def test_read_happy_path_formats_reply() -> None:
    cfg = _make_cfg(public=True)
    service = _ConfigurableDigitService()
    cog = DigitsCog(FakeBot(), cfg, service)
    inter = RecordingInteraction(user=FakeUser(user_id=123))
    att = FakeAttachment(filename="d.png", content_type="image/png", size=10, data=b"image")
    await cog._read_impl(inter, inter.user, att)
    assert inter.sent and "Digit:" in str(inter.sent[-1]["content"])


@pytest.mark.asyncio
async def test_read_rejects_unsupported_type() -> None:
    cfg = _make_cfg()
    service = _ConfigurableDigitService()
    cog = DigitsCog(FakeBot(), cfg, service)
    inter = RecordingInteraction(user=FakeUser(user_id=123))
    att = FakeAttachment(filename="x.txt", content_type="text/plain", size=10, data=b"x")
    await cog._read_impl(inter, inter.user, att)
    last = inter.sent[-1]
    assert "Unsupported file type" in str(last["content"]) and last["ephemeral"] is True


@pytest.mark.asyncio
async def test_read_rate_limit_message_on_second_call() -> None:
    cfg = _make_cfg(limit=1, window=1)
    service = _ConfigurableDigitService()
    cog = DigitsCog(FakeBot(), cfg, service)
    inter = RecordingInteraction(user=FakeUser(user_id=99))
    att = FakeAttachment(filename="a.png", content_type="image/png", size=10, data=b"a")
    await cog._read_impl(inter, inter.user, att)
    await cog._read_impl(inter, inter.user, att)
    msg = str(inter.sent[-1]["content"])
    assert msg.startswith("Please wait")


@pytest.mark.asyncio
async def test_read_handles_5xx_surfaces_api_error() -> None:
    class _Cog(DigitsCog):
        def __init__(self, service: _ConfigurableDigitService, cfg: DiscordbotSettings) -> None:
            super().__init__(FakeBot(), cfg, service, autostart_subscriber=False)
            self.seen: list[str] = []
            self.msgs: list[str] = []

        async def handle_user_error(
            self, interaction: InteractionProto, log: _Logger, message: str
        ) -> None:
            _ = (interaction, log)
            self.seen.append("user")
            self.msgs.append(message)

    cfg = _make_cfg(public=True)
    service = _ConfigurableDigitService(err=HandwritingAPIError(500, "boom"))
    cog = _Cog(service, cfg)
    inter = RecordingInteraction(user=FakeUser(user_id=77))
    att = FakeAttachment(filename="d.png", content_type="image/png", size=10, data=b"d")
    await cog._read_impl(inter, inter.user, att)
    assert "user" in cog.seen
    assert any("boom" in m or "HTTP" in m or "internal_error" in m for m in cog.msgs)


logger = logging.getLogger(__name__)
