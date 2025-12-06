from __future__ import annotations

import pytest
from tests.support.settings import build_settings

import clubbot.services.transcript.client as app_mod
from clubbot.services.transcript.api_client import TranscriptApiClient
from clubbot.services.transcript.client import TranscriptService


def _pass_through_url(url: str) -> str:
    return url


def _extract_vid(url: str) -> str:
    return "vid"


def test_transcript_service_api_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_client: TranscriptApiClient | None = None
    captured_langs: list[str] | None = None

    def fake_captions(
        client: TranscriptApiClient, *, url: str, preferred_langs: list[str]
    ) -> dict[str, str]:
        nonlocal captured_client, captured_langs
        captured_client = client
        captured_langs = list(preferred_langs)
        return {"url": url, "video_id": "vid", "text": "OK"}

    monkeypatch.setattr(app_mod, "validate_youtube_url", _pass_through_url, raising=True)
    monkeypatch.setattr(app_mod, "extract_video_id", _extract_vid, raising=True)
    monkeypatch.setattr(app_mod, "captions", fake_captions, raising=True)
    cfg = build_settings(
        transcript_api_url="http://api",
        transcript_provider="api",
    )

    svc = TranscriptService(cfg)
    out = svc.fetch_cleaned("https://youtu.be/dQw4w9WgXcQ")
    assert out.video_id == "vid" and out.text == "OK"
    assert type(captured_client) is dict
    assert captured_langs == ["en", "en-US"]


def test_transcript_service_respects_preferred_langs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_langs: list[str] | None = None

    def fake_captions(
        client: TranscriptApiClient, *, url: str, preferred_langs: list[str]
    ) -> dict[str, str]:
        nonlocal captured_langs
        captured_langs = preferred_langs
        return {"url": url, "video_id": "vid", "text": "OK"}

    monkeypatch.setattr(app_mod, "validate_youtube_url", _pass_through_url, raising=True)
    monkeypatch.setattr(app_mod, "extract_video_id", _extract_vid, raising=True)
    monkeypatch.setattr(app_mod, "captions", fake_captions, raising=True)
    cfg = build_settings(
        transcript_api_url="http://api",
        transcript_provider="api",
        transcript_preferred_langs="en,fr",
    )
    svc = TranscriptService(cfg)
    _ = svc.fetch_cleaned("https://youtu.be/dQw4w9WgXcQ")
    assert captured_langs == ["en", "fr"]


def test_transcript_service_api_provider_missing_url() -> None:
    # Empty string means explicitly set api_url to None
    cfg = build_settings(transcript_api_url="")
    with pytest.raises(RuntimeError):
        _ = TranscriptService(cfg)


def test_transcript_service_provider_mismatch_raises() -> None:
    cfg = build_settings(transcript_provider="youtube")
    with pytest.raises(RuntimeError):
        _ = TranscriptService(cfg)
