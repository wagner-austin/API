from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import BinaryIO

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import stdlib_logging
from platform_workers.testing import FakeRedis

from transcript_api.jobs import (
    STTConfig,
    STTJobParams,
    STTJobResult,
    process_stt_impl,
)
from transcript_api.types import SubtitleResultTD, VerboseResponseTD, YtInfoTD


class _StubSTTClient:
    def transcribe_verbose(self, *, file: BinaryIO, timeout: float | None) -> VerboseResponseTD:
        return {"text": "hello world", "segments": []}


class _StubProbeClient:
    def probe(self, url: str) -> YtInfoTD:
        return {"id": "vid", "duration": 1.0, "formats": [], "requested_downloads": []}

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        return "/tmp/audio"

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        return None


def test_process_stt_impl_roundtrip_publishes_events_and_saves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubProvider:
        def __init__(
            self,
            *,
            stt_client: _StubSTTClient,
            probe_client: _StubProbeClient,
            max_video_seconds: int,
            max_file_mb: int,
            enable_chunking: bool,
            chunk_threshold_mb: float,
            target_chunk_mb: float,
            max_chunk_duration: float,
            max_concurrent_chunks: int,
            silence_threshold_db: float,
            silence_duration: float,
            stt_rtf: float,
            dl_mib_per_sec: float,
            cookies_text: str | None,
        ) -> None:
            self._stt_client = stt_client
            self._probe_client = probe_client
            self._created_at = datetime.utcnow()

        def fetch(
            self, video_id: str, opts: Mapping[str, JSONValue]
        ) -> list[Mapping[str, JSONValue]]:
            return [{"text": "hello", "start": 0.0, "duration": 1.0}]

    redis = FakeRedis()
    params: STTJobParams = {"url": "https://youtu.be/dQw4w9WgXcQ", "user_id": 5}
    config: STTConfig = {
        "max_video_seconds": 1,
        "max_file_mb": 1,
        "enable_chunking": False,
        "chunk_threshold_mb": 10.0,
        "target_chunk_mb": 5.0,
        "max_chunk_duration_seconds": 100.0,
        "max_concurrent_chunks": 1,
        "silence_threshold_db": -20.0,
        "silence_duration_seconds": 0.5,
        "stt_rtf": 0.5,
        "dl_mib_per_sec": 1.0,
    }
    stt_client = _StubSTTClient()
    probe_client = _StubProbeClient()
    logger = stdlib_logging.getLogger("test-logger")

    import transcript_api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "STTTranscriptProvider", _StubProvider)
    result: STTJobResult = process_stt_impl(
        "job-xyz",
        params,
        redis=redis,
        stt_client=stt_client,
        probe_client=probe_client,
        config=config,
        logger=logger,
    )

    assert result["job_id"] == "job-xyz"
    assert result["status"] == "completed"
    assert redis.published  # events were emitted
    # Ensure status persisted with expected fields
    key = next(iter(redis._hashes))
    saved = redis._hashes[key]
    assert saved["status"] == "completed"
    assert saved["progress"] == "100"
    assert saved["video_id"] != ""
    redis.assert_only_called({"publish", "hset", "expire"})


def test_process_stt_impl_requires_url_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubProvider:
        def __init__(
            self,
            *,
            stt_client: _StubSTTClient,
            probe_client: _StubProbeClient,
            max_video_seconds: int,
            max_file_mb: int,
            enable_chunking: bool,
            chunk_threshold_mb: float,
            target_chunk_mb: float,
            max_chunk_duration: float,
            max_concurrent_chunks: int,
            silence_threshold_db: float,
            silence_duration: float,
            stt_rtf: float,
            dl_mib_per_sec: float,
            cookies_text: str | None,
        ) -> None:
            self._stt_client = stt_client
            self._probe_client = probe_client
            self._created_at = datetime.utcnow()

        def fetch(
            self, video_id: str, opts: Mapping[str, JSONValue]
        ) -> list[Mapping[str, JSONValue]]:
            return [{"text": "hi", "start": 0.0, "duration": 1.0}]

    redis = FakeRedis()
    stt_client = _StubSTTClient()
    probe_client = _StubProbeClient()
    config: STTConfig = {
        "max_video_seconds": 1,
        "max_file_mb": 1,
        "enable_chunking": False,
        "chunk_threshold_mb": 1.0,
        "target_chunk_mb": 1.0,
        "max_chunk_duration_seconds": 1.0,
        "max_concurrent_chunks": 1,
        "silence_threshold_db": -1.0,
        "silence_duration_seconds": 0.1,
        "stt_rtf": 0.1,
        "dl_mib_per_sec": 1.0,
    }
    logger = stdlib_logging.getLogger("test-logger-bad")
    import transcript_api.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "STTTranscriptProvider", _StubProvider)
    with pytest.raises(AppError):
        process_stt_impl(
            "job-bad",
            {"url": "", "user_id": 1},
            redis=redis,
            stt_client=stt_client,
            probe_client=probe_client,
            config=config,
            logger=logger,
        )
    redis.assert_only_called({"publish"})


def test_load_stt_config_returns_typed_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _load_stt_config loads values from environment."""
    import transcript_api.jobs as jobs_mod

    monkeypatch.setenv("TRANSCRIPT_MAX_VIDEO_SECONDS", "300")
    monkeypatch.setenv("TRANSCRIPT_MAX_FILE_MB", "50")
    monkeypatch.setenv("TRANSCRIPT_ENABLE_CHUNKING", "true")
    monkeypatch.setenv("TRANSCRIPT_CHUNK_THRESHOLD_MB", "30.0")
    monkeypatch.setenv("TRANSCRIPT_TARGET_CHUNK_MB", "25.0")
    monkeypatch.setenv("TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS", "500.0")
    monkeypatch.setenv("TRANSCRIPT_MAX_CONCURRENT_CHUNKS", "8")
    monkeypatch.setenv("TRANSCRIPT_SILENCE_THRESHOLD_DB", "-30.0")
    monkeypatch.setenv("TRANSCRIPT_SILENCE_DURATION_SECONDS", "0.3")
    monkeypatch.setenv("TRANSCRIPT_STT_RTF", "0.4")
    monkeypatch.setenv("TRANSCRIPT_DL_MIB_PER_SEC", "10.0")

    config = jobs_mod._load_stt_config()

    assert config["max_video_seconds"] == 300
    assert config["max_file_mb"] == 50
    assert config["enable_chunking"] is True
    assert config["chunk_threshold_mb"] == 30.0
    assert config["target_chunk_mb"] == 25.0
    assert config["max_chunk_duration_seconds"] == 500.0
    assert config["max_concurrent_chunks"] == 8
    assert config["silence_threshold_db"] == -30.0
    assert config["silence_duration_seconds"] == 0.3
    assert config["stt_rtf"] == 0.4
    assert config["dl_mib_per_sec"] == 10.0


def test_decode_stt_params_success() -> None:
    """Test _decode_stt_params with valid input."""
    import transcript_api.jobs as jobs_mod

    raw: dict[str, JSONValue] = {"url": "https://youtu.be/abc", "user_id": 42}
    result = jobs_mod._decode_stt_params(raw)
    assert result["url"] == "https://youtu.be/abc"
    assert result["user_id"] == 42


def test_decode_stt_params_strips_whitespace() -> None:
    """Test _decode_stt_params strips whitespace from url."""
    import transcript_api.jobs as jobs_mod

    raw: dict[str, JSONValue] = {"url": "  https://youtu.be/abc  ", "user_id": 1}
    result = jobs_mod._decode_stt_params(raw)
    assert result["url"] == "https://youtu.be/abc"


def test_decode_stt_params_raises_on_missing_url() -> None:
    """Test _decode_stt_params raises on missing url."""
    import transcript_api.jobs as jobs_mod

    with pytest.raises(JSONTypeError, match="url must be a non-empty string"):
        jobs_mod._decode_stt_params({"user_id": 1})


def test_decode_stt_params_raises_on_empty_url() -> None:
    """Test _decode_stt_params raises on empty url."""
    import transcript_api.jobs as jobs_mod

    with pytest.raises(JSONTypeError, match="url must be a non-empty string"):
        jobs_mod._decode_stt_params({"url": "   ", "user_id": 1})


def test_decode_stt_params_raises_on_non_string_url() -> None:
    """Test _decode_stt_params raises on non-string url."""
    import transcript_api.jobs as jobs_mod

    with pytest.raises(JSONTypeError, match="url must be a non-empty string"):
        jobs_mod._decode_stt_params({"url": 123, "user_id": 1})


def test_decode_stt_params_raises_on_missing_user_id() -> None:
    """Test _decode_stt_params raises on missing user_id."""
    import transcript_api.jobs as jobs_mod

    with pytest.raises(JSONTypeError, match="user_id must be an integer"):
        jobs_mod._decode_stt_params({"url": "https://youtu.be/abc"})


def test_decode_stt_params_raises_on_non_int_user_id() -> None:
    """Test _decode_stt_params raises on non-integer user_id."""
    import transcript_api.jobs as jobs_mod

    with pytest.raises(JSONTypeError, match="user_id must be an integer"):
        jobs_mod._decode_stt_params({"url": "https://youtu.be/abc", "user_id": "42"})


_redis_url_captured: list[str] = []
_redis_instance_captured: list[FakeRedis] = []


def _capturing_redis_factory(url: str) -> FakeRedis:
    """Factory that captures URL and returns FakeRedis."""
    _redis_url_captured.append(url)
    redis = FakeRedis()
    _redis_instance_captured.append(redis)
    return redis


def test_get_redis_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_redis_client returns a redis client."""
    import transcript_api.jobs as jobs_mod

    _redis_url_captured.clear()
    _redis_instance_captured.clear()
    monkeypatch.setattr(jobs_mod, "redis_for_kv", _capturing_redis_factory)
    jobs_mod._get_redis_client("redis://localhost:6379")
    assert _redis_url_captured == ["redis://localhost:6379"]
    _redis_instance_captured[0].assert_only_called(set())


_stt_api_key_captured: list[str] = []


class _TestSTTClient:
    """Mock STT client that captures API key."""

    def __init__(self, *, api_key: str) -> None:
        _stt_api_key_captured.append(api_key)


def test_build_stt_client_with_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _build_stt_client with OPENAI_API_KEY env var."""
    import transcript_api.jobs as jobs_mod

    _stt_api_key_captured.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setattr("transcript_api.adapters.openai_client.OpenAISttClient", _TestSTTClient)
    jobs_mod._build_stt_client()
    assert _stt_api_key_captured == ["sk-test-key"]


def test_build_stt_client_with_alt_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _build_stt_client with OPEN_AI_API_KEY env var."""
    import transcript_api.jobs as jobs_mod

    _stt_api_key_captured.clear()
    # Unset OPENAI_API_KEY to force fallback to OPEN_AI_API_KEY
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_AI_API_KEY", "sk-alt-key")
    monkeypatch.setattr("transcript_api.adapters.openai_client.OpenAISttClient", _TestSTTClient)
    jobs_mod._build_stt_client()
    assert _stt_api_key_captured == ["sk-alt-key"]


def test_build_stt_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _build_stt_client raises when no API key is set."""
    import transcript_api.jobs as jobs_mod

    # Unset both API key env vars
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        jobs_mod._build_stt_client()


def test_build_probe_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _build_probe_client returns a probe client."""
    import transcript_api.jobs as jobs_mod

    class _MockAdapter:
        pass

    monkeypatch.setattr("transcript_api.adapters.yt_dlp_client.YtDlpAdapter", _MockAdapter)
    client = jobs_mod._build_probe_client()
    assert type(client).__name__ == "_MockAdapter"


_decode_redis_captured: list[FakeRedis] = []


class _DecodeTestSTTClient:
    """Mock STT client for decode test."""

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key


class _DecodeTestProbeClient:
    """Mock probe client for decode test."""


class _DecodeTestProvider:
    """Mock provider for decode test."""

    def __init__(self, **kwargs: str | int | float | bool | None) -> None:
        pass

    def fetch(self, video_id: str, opts: Mapping[str, JSONValue]) -> list[Mapping[str, JSONValue]]:
        return [{"text": "hello", "start": 0.0, "duration": 1.0}]


def _make_decode_redis(url: str) -> FakeRedis:
    redis = FakeRedis()
    _decode_redis_captured.append(redis)
    return redis


def test_decode_process_stt_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _decode_process_stt loads deps and processes job."""
    import transcript_api.jobs as jobs_mod

    _decode_redis_captured.clear()

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    monkeypatch.setattr(jobs_mod, "redis_for_kv", _make_decode_redis)
    monkeypatch.setattr(
        "transcript_api.adapters.openai_client.OpenAISttClient", _DecodeTestSTTClient
    )
    monkeypatch.setattr(
        "transcript_api.adapters.yt_dlp_client.YtDlpAdapter", _DecodeTestProbeClient
    )
    monkeypatch.setattr(jobs_mod, "STTTranscriptProvider", _DecodeTestProvider)

    result = jobs_mod._decode_process_stt(
        "job-123", {"url": "https://youtu.be/dQw4w9WgXcQ", "user_id": 5}
    )

    assert result["job_id"] == "job-123"
    assert result["status"] == "completed"
    assert _decode_redis_captured[0].closed
    _decode_redis_captured[0].assert_only_called({"publish", "hset", "expire", "close"})


def test_process_stt_delegates_to_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test process_stt delegates to _decode_process_stt."""
    import transcript_api.jobs as jobs_mod

    call_args: dict[str, JSONValue] = {}

    def mock_decode(job_id: str, params: dict[str, JSONValue]) -> jobs_mod.STTJobResult:
        call_args["job_id"] = job_id
        call_args["params"] = params
        return {"job_id": job_id, "status": "completed", "video_id": "vid", "text": "hi"}

    monkeypatch.setattr(jobs_mod, "_decode_process_stt", mock_decode)

    result = jobs_mod.process_stt("job-456", {"url": "https://youtu.be/abc", "user_id": 10})

    assert result["job_id"] == "job-456"
    assert call_args["job_id"] == "job-456"
    assert call_args["params"] == {"url": "https://youtu.be/abc", "user_id": 10}
