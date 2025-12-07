from __future__ import annotations

from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis, FakeRedisNoPong
from pytest import MonkeyPatch

from transcript_api.app import AppDeps, create_app
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


def _deps() -> AppDeps:
    # Minimal deps for app creation; endpoints under test don't use service
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


def _client(
    monkeypatch: MonkeyPatch, *, workers: int, pong: bool = True
) -> tuple[TestClient, FakeRedis]:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    from transcript_api import app as app_mod

    fake_redis: FakeRedis = FakeRedisNoPong() if not pong else FakeRedis()
    if pong:
        for i in range(workers):
            fake_redis.sadd("rq:workers", f"worker-{i}")

    def _fake(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(app_mod, "redis_for_kv", _fake)
    return TestClient(create_app(_deps())), fake_redis


def test_healthz_ok(monkeypatch: MonkeyPatch) -> None:
    client, fake_redis = _client(monkeypatch, workers=1)
    r = client.get("/healthz")
    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "ok"
    fake_redis.assert_only_called({"sadd"})


def test_readyz_degraded_without_worker(monkeypatch: MonkeyPatch) -> None:
    client, fake_redis = _client(monkeypatch, workers=0)
    r = client.get("/readyz")
    assert r.status_code == 503
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "degraded"
    assert body.get("reason") == "no-worker"
    fake_redis.assert_only_called({"ping", "scard", "close"})


def test_readyz_ready_with_worker(monkeypatch: MonkeyPatch) -> None:
    client, fake_redis = _client(monkeypatch, workers=1)
    r = client.get("/readyz")
    assert r.status_code == 200
    body_raw = load_json_str(r.text)
    if type(body_raw) is not dict:
        pytest.fail("expected dict response body")
    body: dict[str, JSONValue] = body_raw
    assert body.get("status") == "ready"
    fake_redis.assert_only_called({"sadd", "ping", "scard", "close"})
