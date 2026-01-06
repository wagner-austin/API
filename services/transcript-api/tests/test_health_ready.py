"""Tests for transcript-api health endpoints."""

from __future__ import annotations

from typing import BinaryIO

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from transcript_api.api.main import AppDeps, create_app
from transcript_api.provider import TranscriptListing, TranscriptResource
from transcript_api.service import Clients, Config
from transcript_api.types import RawTranscriptItem, SubtitleResultTD, VerboseResponseTD, YtInfoTD


class _StubResource:
    def fetch(self) -> list[RawTranscriptItem]:
        return []


class _StubListing:
    def find_transcript(self, languages: list[str]) -> TranscriptResource | None:
        return _StubResource()

    def translate(self, language: str) -> TranscriptResource:
        return _StubResource()


class _StubYTClient:
    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        return []

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        return _StubListing()


class _StubSTTClient:
    def transcribe_verbose(self, *, file: BinaryIO, timeout: float | None) -> VerboseResponseTD:
        return {"text": "", "segments": []}


class _StubProbeClient:
    def probe(self, url: str) -> YtInfoTD:
        return {"duration": 0, "formats": []}

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        return ""

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        return None


def _make_test_deps() -> AppDeps:
    """Create test dependencies for health endpoint tests.

    Returns:
        AppDeps with stub clients.
    """
    cfg: Config = {
        "TRANSCRIPT_MAX_VIDEO_SECONDS": 0,
        "TRANSCRIPT_MAX_FILE_MB": 0,
        "TRANSCRIPT_ENABLE_CHUNKING": False,
        "TRANSCRIPT_CHUNK_THRESHOLD_MB": 0.0,
        "TRANSCRIPT_TARGET_CHUNK_MB": 0.0,
        "TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS": 0.0,
        "TRANSCRIPT_MAX_CONCURRENT_CHUNKS": 0,
        "TRANSCRIPT_SILENCE_THRESHOLD_DB": -40.0,
        "TRANSCRIPT_SILENCE_DURATION_SECONDS": 0.5,
        "TRANSCRIPT_STT_RTF": 0.0,
        "TRANSCRIPT_DL_MIB_PER_SEC": 0.0,
        "TRANSCRIPT_PREFERRED_LANGS": None,
    }
    cls: Clients = {
        "youtube": _StubYTClient(),
        "stt": _StubSTTClient(),
        "probe": _StubProbeClient(),
    }
    return {"config": cfg, "clients": cls}


def test_healthz_returns_ok() -> None:
    """Test /healthz returns status ok."""
    app = create_app(_make_test_deps())
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = narrow_json_to_dict(load_json_str(response.text))
    assert body.get("status") == "ok"
