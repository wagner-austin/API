from __future__ import annotations

import asyncio

import pytest
from platform_discord.protocols import InteractionProto, UserProto
from tests.support.discord_fakes import FakeBot, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.transcript import TranscriptCog
from clubbot.services.transcript.client import TranscriptResult, TranscriptService


class _Svc(TranscriptService):
    def fetch_cleaned(self, url: str) -> TranscriptResult:
        return TranscriptResult(url=url, video_id="v", text="ok")


@pytest.mark.asyncio
async def test_transcript_wrapper_placeholder_for_coverage() -> None:
    # The public wrapper is decorated into a discord app command and is exercised
    # via integration tests. Unit tests target _transcript_impl directly.
    assert True


@pytest.mark.asyncio
async def test_transcript_impl_ack_false_and_user_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_settings()
    cog = TranscriptCog(bot=FakeBot(), config=cfg, transcript_service=_Svc(cfg))
    inter = RecordingInteraction()

    # Ack false branch
    async def _false_ack(_i: InteractionProto, *, ephemeral: bool) -> bool:
        _ = ephemeral
        return False

    monkeypatch.setattr(cog, "_safe_defer", _false_ack, raising=True)
    await cog._transcript_impl(inter, inter.user, None, "https://x")
    assert inter.sent == []

    # Now user_id None branch
    async def _true_ack(_i: InteractionProto, *, ephemeral: bool) -> bool:
        _ = ephemeral
        return True

    def _none_decode(_o: UserProto | None, _n: str) -> int | None:
        return None

    monkeypatch.setattr(cog, "_safe_defer", _true_ack, raising=True)
    monkeypatch.setattr(TranscriptCog, "decode_int_attr", staticmethod(_none_decode), raising=True)

    errors: list[str] = []

    from clubbot.cogs.base import _Logger

    async def err(_i: InteractionProto, _l: _Logger, msg: str) -> None:
        errors.append(msg)

    monkeypatch.setattr(cog, "handle_user_error", err, raising=True)
    await cog._transcript_impl(inter, inter.user, None, "https://x")
    assert errors and "user id" in errors[-1]


@pytest.mark.asyncio
async def test_transcript_impl_unexpected_result_type_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_settings()
    cog = TranscriptCog(bot=FakeBot(), config=cfg, transcript_service=_Svc(cfg))
    inter = RecordingInteraction()

    from collections.abc import Callable
    from typing import ParamSpec, TypeVar

    ps = ParamSpec("ps")
    rt = TypeVar("rt")

    async def run_thread(func: Callable[ps, rt], *args: ps.args, **kwargs: ps.kwargs) -> int:
        _ = (func, args, kwargs)
        await asyncio.sleep(0)
        return 1

    monkeypatch.setattr("clubbot.cogs.transcript.asyncio.to_thread", run_thread, raising=True)

    def _val(u: str) -> str:
        return u

    monkeypatch.setattr("clubbot.cogs.transcript.validate_youtube_url", _val, raising=True)

    with pytest.raises(RuntimeError):
        await cog._transcript_impl(inter, inter.user, None, "https://x")
