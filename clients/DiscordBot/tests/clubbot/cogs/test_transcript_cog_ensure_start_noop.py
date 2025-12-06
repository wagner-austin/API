from __future__ import annotations

import logging

import pytest
from tests.support.discord_fakes import FakeBot
from tests.support.settings import build_settings

from clubbot.cogs.transcript import TranscriptCog
from clubbot.config import DiscordbotSettings
from clubbot.services.transcript.client import TranscriptResult, TranscriptService


class _FakeTranscriptService(TranscriptService):
    def __init__(self, cfg: DiscordbotSettings) -> None:
        super().__init__(cfg)

    def fetch_cleaned(self, url: str) -> TranscriptResult:
        return TranscriptResult(url=url, video_id="vid", text="ok")


@pytest.mark.asyncio
async def test_transcript_cog_ensure_subscriber_started_noop() -> None:
    cfg = build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        transcript_public_responses=False,
        transcript_provider="api",
    )

    cog = TranscriptCog(bot=FakeBot(), config=cfg, transcript_service=_FakeTranscriptService(cfg))
    cog.ensure_subscriber_started()
    await cog.cog_unload()


logger = logging.getLogger(__name__)
