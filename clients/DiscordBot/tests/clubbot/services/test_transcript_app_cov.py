from __future__ import annotations

import pytest
from tests.support.settings import build_settings

from clubbot.services.transcript.client import TranscriptService


def test_transcript_service_requires_api_provider() -> None:
    cfg = build_settings(transcript_provider="stt")
    with pytest.raises(RuntimeError):
        _ = TranscriptService(cfg)


def test_transcript_service_requires_base_url() -> None:
    cfg = build_settings(transcript_provider="api", transcript_api_url="")
    with pytest.raises(RuntimeError):
        _ = TranscriptService(cfg)
