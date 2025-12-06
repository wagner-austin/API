from __future__ import annotations

import logging

import pytest
from tests.support.discord_fakes import FakeBot, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.transcript import TranscriptCog
from clubbot.config import DiscordbotSettings
from clubbot.services.transcript.client import TranscriptResult, TranscriptService


class FakeTranscriptService(TranscriptService):
    def __init__(
        self,
        cfg: DiscordbotSettings,
        *,
        text: str = "hello world",
        vid: str = "dQw4w9WgXcQ",
    ) -> None:
        super().__init__(cfg)
        self._text = text
        self._vid = vid

    def fetch_cleaned(self, url: str) -> TranscriptResult:
        return TranscriptResult(
            url=f"https://www.youtube.com/watch?v={self._vid}",
            video_id=self._vid,
            text=self._text,
        )


def make_cfg() -> DiscordbotSettings:
    return build_settings(
        qr_default_border=2,
        qr_public_responses=True,
        transcript_public_responses=True,
    )


@pytest.mark.asyncio
async def test_transcript_command_always_responds_with_file() -> None:
    bot = FakeBot()
    cfg = make_cfg()
    svc = FakeTranscriptService(cfg, text="short text")
    cog = TranscriptCog(bot, cfg, svc)
    interaction = RecordingInteraction()
    guild_obj: None = None

    await cog._transcript_impl(
        interaction,
        interaction.user,
        guild_obj,
        "https://youtu.be/dQw4w9WgXcQ",
    )

    assert interaction.sent, "Expected a response"
    last = interaction.sent[-1]
    if last["file"] is None:
        raise AssertionError("Transcript should always be sent as a file")


logger = logging.getLogger(__name__)
