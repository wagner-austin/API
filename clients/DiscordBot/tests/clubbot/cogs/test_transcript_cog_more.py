from __future__ import annotations

import logging
from typing import ParamSpec, TypeVar

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_discord.protocols import InteractionProto
from platform_discord.rate_limiter import RateLimiter
from tests.support.discord_fakes import (
    FakeBot,
    FakeUser,
    NoIdUser,
    RecordedSend,
    RecordingInteraction,
)
from tests.support.settings import build_settings

from clubbot import _test_hooks
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
    """Configurable fake transcript service for testing."""

    def __init__(
        self,
        cfg: DiscordbotSettings,
        *,
        result: TranscriptResult | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        super().__init__(cfg)
        self._result = result
        self._raise_error = raise_error

    def fetch_cleaned(self, url: str) -> TranscriptResult:
        if self._raise_error is not None:
            raise self._raise_error
        if self._result is not None:
            return self._result
        return TranscriptResult(url=url, video_id="vid", text="ok")


def _make_interaction() -> RecordingInteraction:
    return RecordingInteraction()


async def _run_sync(func: _test_hooks._SyncCallable, url: str) -> _test_hooks.TranscriptResultLike:
    """Fake asyncio.to_thread that runs synchronously."""
    result: _test_hooks.TranscriptResultLike = func(url)
    return result


def _passthrough_url(url: str) -> str:
    """Skip URL validation for testing."""
    return url


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


def _make_exhausted_rate_limiter() -> RateLimiter:
    """Create a rate limiter that's already exhausted."""
    rl = RateLimiter(per_window=1, window_seconds=60)
    # Pre-exhaust for common test user IDs
    rl.allow(67890, "transcript")
    return rl


@pytest.mark.asyncio
async def test_transcript_attachment_too_large() -> None:
    cfg = _cfg(provider="api", attach_mb=1)
    messages: list[str] = []

    # Configure hooks
    _test_hooks.validate_youtube_url = _passthrough_url
    _test_hooks.asyncio_to_thread = _run_sync

    # Large text result
    large_text = "x" * (2 * 1024 * 1024)
    result = TranscriptResult(url="https://v", video_id="vid1", text=large_text)
    svc = _FakeTranscriptService(cfg, result=result)

    bot = FakeBot()
    cog = _TestingCog(messages, [], bot, cfg, svc)

    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    assert messages and "too large" in messages[-1]


@pytest.mark.asyncio
async def test_transcript_rate_limit() -> None:
    cfg = _cfg()

    # Configure hooks
    _test_hooks.validate_youtube_url = _passthrough_url
    _test_hooks.asyncio_to_thread = _run_sync

    bot = FakeBot()
    svc = _FakeTranscriptService(cfg)
    cog = _TestingCog([], [], bot, cfg, svc)
    # Replace rate limiter with exhausted one
    cog.rate_limiter = _make_exhausted_rate_limiter()

    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    last = _last_send(inter)
    content = str(last["content"] or "")
    assert "Please wait" in content


@pytest.mark.asyncio
async def test_transcript_user_error() -> None:
    cfg = _cfg()
    errors: list[str] = []

    # Configure hooks
    _test_hooks.validate_youtube_url = _passthrough_url
    _test_hooks.asyncio_to_thread = _run_sync

    # Service that raises user error
    svc = _FakeTranscriptService(
        cfg, raise_error=AppError(ErrorCode.INVALID_INPUT, "bad", http_status=400)
    )

    bot = FakeBot()
    cog = _TestingCog(errors, [], bot, cfg, svc)

    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    assert errors and "bad" in errors[-1]


@pytest.mark.asyncio
async def test_transcript_runtime_error() -> None:
    cfg = _cfg()
    exceptions: list[Exception] = []

    # Configure hooks
    _test_hooks.validate_youtube_url = _passthrough_url
    _test_hooks.asyncio_to_thread = _run_sync

    # Service that raises runtime error
    svc = _FakeTranscriptService(cfg, raise_error=RuntimeError("boom"))

    bot = FakeBot()
    cog = _TestingCog([], exceptions, bot, cfg, svc)

    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    assert exceptions and isinstance(exceptions[-1], RuntimeError)


@pytest.mark.asyncio
async def test_transcript_success() -> None:
    cfg = _cfg()

    # Configure hooks
    _test_hooks.validate_youtube_url = _passthrough_url
    _test_hooks.asyncio_to_thread = _run_sync

    result = TranscriptResult(url="https://v", video_id="vid1", text="hi")
    svc = _FakeTranscriptService(cfg, result=result)

    bot = FakeBot()
    cog = _TestingCog([], [], bot, cfg, svc)

    inter = _make_interaction()
    await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
    last = _last_send(inter)
    if last["file"] is None:
        raise AssertionError("expected file")


class _FakeTranscriptResultLike:
    """A class that matches TranscriptResultLike but is NOT a TranscriptResult instance.

    This is used to trigger the isinstance check failure in transcript.py line 124.
    """

    url: str = "https://v"
    video_id: str = "vid"
    text: str = "hello"


@pytest.mark.asyncio
async def test_transcript_unexpected_result_type_raises() -> None:
    """Test that unexpected result type from asyncio_to_thread raises RuntimeError.

    This covers transcript.py line 124:
        raise RuntimeError("Unexpected result type from task")
    """
    cfg = _cfg()

    # Configure hooks
    _test_hooks.validate_youtube_url = _passthrough_url

    # Make asyncio_to_thread return something that's NOT a TranscriptResult
    async def _return_wrong_type(
        func: _test_hooks._SyncCallable, url: str
    ) -> _test_hooks.TranscriptResultLike:
        _ = (func, url)
        # Return a class instance that matches the protocol but is not TranscriptResult
        # The isinstance check will fail
        return _FakeTranscriptResultLike()

    _test_hooks.asyncio_to_thread = _return_wrong_type

    # Service won't actually be called since we're overriding asyncio_to_thread
    svc = _FakeTranscriptService(cfg)

    bot = FakeBot()
    cog = TranscriptCog(bot=bot, config=cfg, transcript_service=svc)

    inter = _make_interaction()
    with pytest.raises(RuntimeError, match="Unexpected result type from task"):
        await cog._transcript_impl(inter, inter.user, None, "https://example.com/watch?v=1")
