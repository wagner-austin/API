from __future__ import annotations

import logging

from tests.support.discord_fakes import FakeBot
from tests.support.settings import build_settings

from clubbot.cogs.transcript import TranscriptCog
from clubbot.config import DiscordbotSettings
from clubbot.services.transcript.client import TranscriptResult, TranscriptService

logger = logging.getLogger(__name__)


class _FakeTranscriptService(TranscriptService):
    def __init__(self, cfg: DiscordbotSettings) -> None:
        super().__init__(cfg)

    def fetch_cleaned(self, url: str) -> TranscriptResult:
        return TranscriptResult(url=url, video_id="vid", text="ok")


def _cfg() -> DiscordbotSettings:
    return build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        transcript_public_responses=False,
        transcript_provider="api",
    )


def test_transcript_helper_methods_executed() -> None:
    cfg = _cfg()
    cog = TranscriptCog(bot=FakeBot(), config=cfg, transcript_service=_FakeTranscriptService(cfg))
    # Exercise helper return paths to ensure statement coverage
    assert type(cog._get_attachment_limit_mb()) is int
    assert cog._is_attachment_too_large(b"") is False
