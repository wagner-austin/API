from __future__ import annotations

import logging

import pytest
from platform_core.errors import AppError
from platform_discord.protocols import InteractionProto
from tests.support.discord_fakes import (
    FakeAttachment,
    FakeBot,
    FakeDigitService,
    FakeFollowup,
    FakeResponse,
    FakeUser,
)
from tests.support.settings import build_settings

import clubbot.cogs.digits as digits_mod
from clubbot.cogs.base import _Logger
from clubbot.cogs.digits import DigitsCog, _format_result, _top_k_indices
from clubbot.config import DiscordbotSettings
from clubbot.services.handai.client import HandwritingAPIError, PredictResult


class _FakeHTTPError(Exception):
    """Fake HTTP error for testing."""

    def __init__(self, code: int) -> None:
        super().__init__(f"http {code}")
        self.code = code


class _FakeNotFoundError(Exception):
    """Fake NotFound error for testing."""

    pass


class _RespRaises(FakeResponse):
    """Response that raises a configured exception when deferred."""

    def __init__(self, exc: Exception | None = None) -> None:
        super().__init__(done=False)
        self._exc = exc

    async def defer(self, *, ephemeral: bool = False) -> None:
        if self._exc is not None:
            raise self._exc
        await super().defer(ephemeral=ephemeral)


class _InteractionWithResp:
    """Interaction composed of provided response/followup fakes."""

    def __init__(self, resp: FakeResponse | None = None, *, user: FakeUser | None = None) -> None:
        self._response = resp if resp is not None else FakeResponse()
        self._followup = FakeFollowup()
        self._user = user if user is not None else FakeUser()

    @property
    def response(self) -> FakeResponse:
        return self._response

    @property
    def followup(self) -> FakeFollowup:
        return self._followup

    @property
    def user(self) -> FakeUser:
        return self._user


def _make_cfg(public: bool = False) -> DiscordbotSettings:
    return build_settings(
        qr_public_responses=True,
        digits_public_responses=public,
        digits_max_image_mb=1,
        handwriting_api_url="http://localhost:7000",
    )


@pytest.mark.asyncio
async def test_safe_defer_handles_not_found() -> None:
    cfg = _make_cfg()
    cog = DigitsCog(FakeBot(), cfg, FakeDigitService(), autostart_subscriber=False)
    inter: InteractionProto = _InteractionWithResp(_RespRaises(_FakeNotFoundError()))

    ok = await cog._safe_defer(inter, ephemeral=True)
    assert ok is False


@pytest.mark.asyncio
async def test_safe_defer_http_exception_already_ack() -> None:
    cfg = _make_cfg()
    cog = DigitsCog(FakeBot(), cfg, FakeDigitService(), autostart_subscriber=False)
    inter: InteractionProto = _InteractionWithResp(_RespRaises(_FakeHTTPError(40060)))

    ok = await cog._safe_defer(inter, ephemeral=True)
    assert ok is True


@pytest.mark.asyncio
async def test_safe_defer_http_exception_other() -> None:
    cfg = _make_cfg()
    cog = DigitsCog(FakeBot(), cfg, FakeDigitService(), autostart_subscriber=False)
    inter: InteractionProto = _InteractionWithResp(_RespRaises(_FakeHTTPError(1)))

    ok = await cog._safe_defer(inter, ephemeral=True)
    assert ok is False


@pytest.mark.asyncio
async def test_user_error_mappings_and_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    class _Cog(DigitsCog):
        async def handle_user_error(
            self, interaction: InteractionProto, log: _Logger, message: str
        ) -> None:
            messages.append(message)

        async def handle_exception(
            self, interaction: InteractionProto, log: _Logger, exc: Exception
        ) -> None:
            messages.append("EXC")

    cfg = _make_cfg()
    cog = _Cog(FakeBot(), cfg, FakeDigitService(), autostart_subscriber=False)

    att = FakeAttachment(
        filename="a.png",
        content_type="image/png",
        size=10 * 1024 * 1024,
        data=b"x",
    )
    log = logging.LoggerAdapter(logging.getLogger(__name__), {})

    with pytest.raises(AppError) as exc_info:
        cog._validate_attachment(att)
    inter: InteractionProto = _InteractionWithResp()
    await cog.handle_user_error(inter, log, str(exc_info.value))

    msgs: list[str] = [
        digits_mod._user_message_from_api_error(HandwritingAPIError(401, "", code="unauthorized")),
        digits_mod._user_message_from_api_error(HandwritingAPIError(413, "", code="too_large")),
        digits_mod._user_message_from_api_error(
            HandwritingAPIError(415, "", code="unsupported_media_type")
        ),
        digits_mod._user_message_from_api_error(HandwritingAPIError(400, "", code="invalid_image")),
        digits_mod._user_message_from_api_error(HandwritingAPIError(504, "", code="timeout")),
        digits_mod._user_message_from_api_error(HandwritingAPIError(500, "", code=None)),
    ]
    assert "authorized" in msgs[0].lower()
    assert "large" in msgs[1].lower()
    assert "unsupported" in msgs[2].lower()
    assert "process" in msgs[3].lower()
    assert "timed out" in msgs[4].lower()
    assert "internal_error" in msgs[5].lower() or "http" in msgs[5].lower()
    assert messages


def test_format_helpers() -> None:
    result = _format_result(
        PredictResult(
            digit=2,
            confidence=0.6,
            probs=(0.1, 0.2, 0.6, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
            model_id="m",
            uncertain=True,
            latency_ms=1,
        )
    )
    assert "Digit: 2" in result and "Low confidence" in result
    assert _top_k_indices((0.1, 0.2, 0.6, 0.05, 0.05), 3) == [2, 1, 0]


logger = logging.getLogger(__name__)
