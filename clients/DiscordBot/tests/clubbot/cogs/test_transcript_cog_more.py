from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_discord.protocols import InteractionProto
from tests.support.discord_fakes import (
    FakeBot,
    FakeUser,
    NoIdUser,
    RecordedSend,
    RecordingInteraction,
)
from tests.support.settings import build_settings

from clubbot.cogs.base import _Logger
from clubbot.cogs.transcript import TranscriptCog
from clubbot.config import DiscordbotSettings
from clubbot.services.transcript.client import (
    TranscriptResult,
    TranscriptService,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _cfg(*, provider: str = "api", attach_mb: int | None = None) -> DiscordbotSettings:
    return build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        transcript_public_responses=False,
        transcript_provider=provider,
        redis_url="redis://fake",
        transcript_max_file_mb=1,
        transcript_enable_chunking=False,
        transcript_max_video_seconds=60,
        transcript_max_attachment_mb=(attach_mb if attach_mb is not None else 25),
    )


class _FakeTranscriptService(TranscriptService):
    def __init__(self, cfg: DiscordbotSettings) -> None:
        super().__init__(cfg)

    def fetch_cleaned(self, url: str) -> TranscriptResult:
        return TranscriptResult(url=url, video_id="vid", text="ok")


def _make_interaction() -> RecordingInteraction:
    return RecordingInteraction()


def _run_in_thread_sync(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return func(*args, **kwargs)


async def _run_in_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return func(*args, **kwargs)


def _validate_url(u: str) -> str:
    return u


def _last_send(inter: RecordingInteraction) -> RecordedSend:
    assert inter.sent, "Expected at least one send"
    return inter.sent[-1]


def test_limits_and_decode() -> None:
    cfg = _cfg(attach_mb=2)
    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = TranscriptCog(bot=bot, config=cfg, transcript_service=svc)
    user_with_id = FakeUser(user_id=5)
    assert cog.decode_int_attr(user_with_id, "id") == 5
    assert cog.decode_int_attr(NoIdUser(), "id") is None
    assert cog._get_attachment_limit_mb() == 2
    assert cog._is_attachment_too_large(b"x" * (3 * 1024 * 1024)) is True
    assert cog._is_attachment_too_large(b"x") is False


class _TestingCog(TranscriptCog):
    """Cog subclass that captures errors for testing."""

    def __init__(
        self,
        errors: list[str],
        exceptions: list[Exception],
        bot: FakeBot,
        config: DiscordbotSettings,
        svc: TranscriptService,
    ) -> None:
        super().__init__(bot=bot, config=config, transcript_service=svc)
        self._errors = errors
        self._exceptions = exceptions

    async def handle_user_error(
        self, interaction: InteractionProto, log: _Logger, message: str
    ) -> None:
        _ = (interaction, log)
        self._errors.append(message)

    async def handle_exception(
        self, interaction: InteractionProto, log: _Logger, e: Exception
    ) -> None:
        _ = (interaction, log)
        self._exceptions.append(e)


@pytest.mark.asyncio
async def test_transcript_attachment_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(provider="api", attach_mb=1)
    messages: list[str] = []

    def fake_fetch_cleaned(_svc: TranscriptService, _url: str) -> TranscriptResult:
        text = "x" * (2 * 1024 * 1024)
        return TranscriptResult(url="https://v", video_id="vid1", text=text)

    def validate_url(u: str) -> str:
        return u

    monkeypatch.setattr("clubbot.cogs.transcript.validate_youtube_url", validate_url, raising=True)
    monkeypatch.setattr("clubbot.cogs.transcript.asyncio.to_thread", _run_in_thread, raising=True)

    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = _TestingCog(messages, [], bot, cfg, svc)

    def _fetch_cleaned_bound(url: str) -> TranscriptResult:
        return fake_fetch_cleaned(svc, url)

    monkeypatch.setattr(svc, "fetch_cleaned", _fetch_cleaned_bound, raising=True)
    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    assert messages and "too large" in messages[-1]


@pytest.mark.asyncio
async def test_transcript_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()

    def allow_rate(_user_id: int, _action: str) -> tuple[bool, float]:
        return (False, 3.0)

    monkeypatch.setattr("clubbot.cogs.transcript.asyncio.to_thread", _run_in_thread, raising=True)
    monkeypatch.setattr("clubbot.cogs.transcript.validate_youtube_url", _validate_url, raising=True)

    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = _TestingCog([], [], bot, cfg, svc)
    monkeypatch.setattr(cog.rate_limiter, "allow", allow_rate, raising=True)
    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    last = _last_send(inter)
    content = str(last["content"] or "")
    assert "Please wait" in content


@pytest.mark.asyncio
async def test_transcript_user_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    errors: list[str] = []

    def bad_fetch(_svc: TranscriptService, _url: str) -> TranscriptResult:
        raise AppError(ErrorCode.INVALID_INPUT, "bad", http_status=400)

    def allow_all(_user_id: int, _action: str) -> tuple[bool, float]:
        return (True, 0.0)

    monkeypatch.setattr("clubbot.cogs.transcript.asyncio.to_thread", _run_in_thread, raising=True)
    monkeypatch.setattr("clubbot.cogs.transcript.validate_youtube_url", _validate_url, raising=True)

    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = _TestingCog(errors, [], bot, cfg, svc)

    def _bad_fetch_bound(url: str) -> TranscriptResult:
        return bad_fetch(svc, url)

    monkeypatch.setattr(svc, "fetch_cleaned", _bad_fetch_bound, raising=True)
    monkeypatch.setattr(cog.rate_limiter, "allow", allow_all, raising=True)
    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    assert errors and "bad" in errors[-1]


@pytest.mark.asyncio
async def test_transcript_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    exceptions: list[Exception] = []

    def boom_fetch(_svc: TranscriptService, _url: str) -> TranscriptResult:
        raise RuntimeError("boom")

    def allow_all(_user_id: int, _action: str) -> tuple[bool, float]:
        return (True, 0.0)

    monkeypatch.setattr("clubbot.cogs.transcript.asyncio.to_thread", _run_in_thread, raising=True)
    monkeypatch.setattr("clubbot.cogs.transcript.validate_youtube_url", _validate_url, raising=True)

    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = _TestingCog([], exceptions, bot, cfg, svc)

    def _boom_fetch_bound(url: str) -> TranscriptResult:
        return boom_fetch(svc, url)

    monkeypatch.setattr(svc, "fetch_cleaned", _boom_fetch_bound, raising=True)
    monkeypatch.setattr(cog.rate_limiter, "allow", allow_all, raising=True)
    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    assert exceptions and isinstance(exceptions[-1], RuntimeError)


@pytest.mark.asyncio
async def test_transcript_success(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()

    def ok_fetch(_svc: TranscriptService, _url: str) -> TranscriptResult:
        return TranscriptResult(url="https://v", video_id="vid1", text="hi")

    def allow_all(_user_id: int, _action: str) -> tuple[bool, float]:
        return (True, 0.0)

    monkeypatch.setattr("clubbot.cogs.transcript.asyncio.to_thread", _run_in_thread, raising=True)
    monkeypatch.setattr("clubbot.cogs.transcript.validate_youtube_url", _validate_url, raising=True)

    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = _TestingCog([], [], bot, cfg, svc)

    def _ok_fetch_bound(url: str) -> TranscriptResult:
        return ok_fetch(svc, url)

    monkeypatch.setattr(svc, "fetch_cleaned", _ok_fetch_bound, raising=True)
    monkeypatch.setattr(cog.rate_limiter, "allow", allow_all, raising=True)
    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    last = _last_send(inter)
    if last["file"] is None:
        raise AssertionError("expected file")
