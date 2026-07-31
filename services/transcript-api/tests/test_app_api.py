from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from platform_core.errors import AppError, TranscriptErrorCode
from platform_core.json_utils import load_json_str

from transcript_api.api.main import (
    AppDeps,
    CaptionsPayload,
    Clients,
    Config,
    STTPayload,
    build_captions_handler,
    build_stt_handler,
    create_app,
)
from transcript_api.provider import (
    TranscriptListing,
    TranscriptResource,
    YouTubeTranscriptClient,
)
from transcript_api.service import TranscriptService
from transcript_api.stt_provider import ProbeDownloadClient, STTClient
from transcript_api.types import (
    RawTranscriptItem,
    SubtitleResultTD,
    VerboseResponseTD,
    VerboseSegmentTD,
    YtInfoTD,
)


class _FakeResource:
    def __init__(self, data: list[RawTranscriptItem]) -> None:
        self._data = data

    def fetch(self) -> list[RawTranscriptItem]:
        return self._data


class _FakeListing:
    def __init__(self, res: TranscriptResource) -> None:
        self._res = res

    def find_transcript(self, languages: list[str]) -> TranscriptResource:
        assert languages
        return self._res

    def translate(self, language: str) -> TranscriptResource:
        return self._res


class _FakeYTClient(YouTubeTranscriptClient):
    def __init__(self, data: list[RawTranscriptItem]) -> None:
        self._data = data

    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]:
        assert len(video_id) == 11
        assert languages
        return self._data

    def list_transcripts(self, video_id: str) -> TranscriptListing:
        return _FakeListing(_FakeResource(self._data))


class _FakeSTTClient(STTClient):
    def __init__(self, segments: list[VerboseSegmentTD]) -> None:
        self._segments = segments

    def transcribe_verbose(
        self,
        *,
        file: BinaryIO,
        timeout: float | None,
    ) -> VerboseResponseTD:
        # Return structure compatible with convert_verbose_to_segments
        return {"text": "", "segments": self._segments}


class _FakeProbeDownload(ProbeDownloadClient):
    def __init__(
        self,
        info: YtInfoTD,
        path: str,
        subtitle_path: str | None = None,
    ) -> None:
        self._info = info
        self._path = path
        self._subtitle_path = subtitle_path

    def probe(self, url: str) -> YtInfoTD:
        return self._info

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        return self._path

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        if self._subtitle_path is None:
            return None
        return {"path": self._subtitle_path, "lang": "en", "is_auto": False}


def _mk_service(tmp_path: Path) -> tuple[FastAPI, TranscriptService]:
    # Prepare fake backends
    yt = _FakeYTClient(
        [
            {"text": "Hello", "start": 0.0, "duration": 0.5},
            {"text": "world", "start": 0.5, "duration": 0.5},
        ]
    )
    stt = _FakeSTTClient(
        [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]
    )
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")

    # Create VTT subtitle file for captions
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:00.500
Hello

00:00:00.500 --> 00:00:01.000
world
"""
    vtt_path = tmp_path / "subs.vtt"
    vtt_path.write_text(vtt_content, encoding="utf-8")

    probe = _FakeProbeDownload({"duration": 5}, str(audio), str(vtt_path))

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    app = create_app(deps)
    # Build a service matching create_app wiring
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    return app, service


def test_captions_success(tmp_path: Path) -> None:
    _app, service = _mk_service(tmp_path)
    handler = build_captions_handler(service)
    payload: CaptionsPayload = {
        "url": "https://youtu.be/dQw4w9WgXcQ",
        "preferred_langs": ["en"],
    }
    out = handler(payload)
    assert out["video_id"] == "dQw4w9WgXcQ"
    assert out["text"] == "Hello world"


def test_captions_invalid_url(tmp_path: Path) -> None:
    _, service = _mk_service(tmp_path)
    handler = build_captions_handler(service)
    payload: CaptionsPayload = {
        "url": "https://example.com/x",
        "preferred_langs": None,
    }
    with pytest.raises(Exception) as excinfo:
        _ = handler(payload)
    assert "YouTube, Vimeo, or a direct video file" in str(excinfo.value)


def test_stt_success(tmp_path: Path) -> None:
    _, service = _mk_service(tmp_path)
    handler = build_stt_handler(service)
    payload: STTPayload = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    }
    out = handler(payload)
    assert out["video_id"] == "dQw4w9WgXcQ"
    assert out["text"] == "Hello world"


def test_stt_user_error_invalid_url(tmp_path: Path) -> None:
    _, service = _mk_service(tmp_path)
    handler = build_stt_handler(service)
    bad: STTPayload = {"url": "https://example.com/x"}
    with pytest.raises(Exception) as excinfo:
        _ = handler(bad)
    assert "YouTube, Vimeo, or a direct video file" in str(excinfo.value)


def test_healthz_endpoint(tmp_path: Path) -> None:
    from platform_core.health import HealthResponse

    from transcript_api.health import healthz_endpoint

    out: HealthResponse = healthz_endpoint()
    assert out == {"status": "ok"}


def test_stt_job_routes_are_mounted_on_the_app(tmp_path: Path) -> None:
    """Test the async STT job endpoints are reachable through create_app.

    The router was fully implemented and unit-tested for a long time while
    create_app never included it, so every one of these URLs 404'd in the
    running service. This exercises them through the production app factory.
    """
    from platform_workers.testing import FakeLogger, FakeQueue, FakeRedis

    from transcript_api.dependencies import provider_context

    app, _ = _mk_service(tmp_path)
    redis = FakeRedis()
    queue = FakeQueue()
    provider_context.redis_provider = lambda: redis
    provider_context.queue_provider = lambda: queue
    provider_context.logger_provider = lambda: FakeLogger()
    try:
        client = TestClient(app)

        submit_payload: dict[str, str | int] = {
            "url": "https://youtu.be/dQw4w9WgXcQ",
            "user_id": 42,
        }
        created = client.post("/v1/stt/jobs", json=submit_payload)
        assert created.status_code == 202
        body: dict[str, str | int] = created.json()
        job_id = body["job_id"]
        assert body["status"] == "queued"
        assert queue.jobs[0].func == "transcript_api.jobs.process_stt"

        # The status route resolves the id the create route just handed back.
        status = client.get(f"/v1/stt/jobs/{job_id}")
        assert status.status_code == 200
        status_body: dict[str, str | int | None] = status.json()
        assert status_body["job_id"] == job_id
        assert status_body["status"] == "queued"
        assert status_body["user_id"] == 42

        missing = client.get("/v1/stt/jobs/no-such-job")
        assert missing.status_code == 404

        # Submit writes the record (hset + expire); both reads are a hgetall.
        redis.assert_only_called({"hset", "expire", "hgetall"})
    finally:
        provider_context.redis_provider = None
        provider_context.queue_provider = None
        provider_context.logger_provider = None


def test_app_error_handled_by_adapter(tmp_path: Path) -> None:
    deps_app, _ = _mk_service(tmp_path)

    def _boom() -> None:
        raise AppError(TranscriptErrorCode.TRANSCRIPT_UNAVAILABLE, "boom", 400)

    deps_app.add_api_route("/boom", _boom, methods=["GET"])

    with TestClient(deps_app) as client:
        resp = client.get("/boom")
    assert resp.status_code == 400
    raw = load_json_str(resp.text)
    if type(raw) is not dict:
        pytest.fail("expected dict response")
    assert raw["code"] == "TRANSCRIPT_UNAVAILABLE"
    assert raw["message"] == "boom"


class _UrlCapturingProbe(ProbeDownloadClient):
    """Fake probe client that captures the URL passed to it for verification."""

    def __init__(
        self,
        info: YtInfoTD,
        path: str,
        subtitle_path: str | None = None,
    ) -> None:
        self._info = info
        self._path = path
        self._subtitle_path = subtitle_path
        self.captured_probe_url: str | None = None
        self.captured_download_url: str | None = None
        self.captured_subtitle_url: str | None = None

    def probe(self, url: str) -> YtInfoTD:
        self.captured_probe_url = url
        return self._info

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        self.captured_download_url = url
        return self._path

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        self.captured_subtitle_url = url
        if self._subtitle_path is None:
            return None
        return {"path": self._subtitle_path, "lang": "en", "is_auto": False}


def test_captions_direct_url_passes_canonical_url(tmp_path: Path) -> None:
    """Verify that direct/download URLs pass the canonical URL to probe client."""
    yt = _FakeYTClient(
        [
            {"text": "Hello", "start": 0.0, "duration": 0.5},
            {"text": "world", "start": 0.5, "duration": 0.5},
        ]
    )
    stt = _FakeSTTClient(
        [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]
    )
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:00.500
Hello

00:00:00.500 --> 00:00:01.000
world
"""
    vtt_path = tmp_path / "subs.vtt"
    vtt_path.write_text(vtt_content, encoding="utf-8")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), str(vtt_path))

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_captions_handler(service)

    # Canvas-style download URL
    canvas_url = "https://canvas.edu/files/123/download?verifier=abc123"
    payload: CaptionsPayload = {"url": canvas_url, "preferred_langs": ["en"]}
    out = handler(payload)

    # Verify the canonical URL was passed through, not an MD5 hash
    assert probe.captured_subtitle_url == canvas_url
    assert out["url"] == canvas_url
    assert len(out["video_id"]) == 32  # MD5 hash
    assert out["text"] == "Hello world"


class _FakeSubprocessResult:
    """Fake subprocess result for ffprobe mock."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout: str | bytes | None = stdout
        self.stderr: str | bytes | None = stderr


def test_stt_direct_url_passes_canonical_url(tmp_path: Path) -> None:
    """Verify that direct/download URLs pass the canonical URL to STT provider."""
    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient(
        [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]
    )
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), None)

    # Mock subprocess_run to return fake ffprobe duration for direct URLs
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        # Return fake ffprobe duration output
        if args and args[0] == "ffprobe":
            return _FakeSubprocessResult(0, "5.0\n")
        return _FakeSubprocessResult(1, "")

    _test_hooks.subprocess_run = fake_subprocess_run

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    # Canvas-style download URL
    canvas_url = "https://canvas.edu/files/456/download?download_frd=1&verifier=xyz"
    payload: STTPayload = {"url": canvas_url}
    out = handler(payload)

    # For direct URLs, probe() is not called - only download_audio()
    # Duration is obtained from ffprobe on the downloaded file
    assert probe.captured_probe_url is None  # probe() not called for direct URLs
    assert probe.captured_download_url == canvas_url
    assert out["url"] == canvas_url
    assert len(out["video_id"]) == 32  # MD5 hash
    assert out["text"] == "Hello world"


def test_captions_direct_mp4_url(tmp_path: Path) -> None:
    """Verify that direct .mp4 URLs work for captions."""
    yt = _FakeYTClient([])
    stt = _FakeSTTClient([])
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:00.500
Direct

00:00:00.500 --> 00:00:01.000
video
"""
    vtt_path = tmp_path / "subs.vtt"
    vtt_path.write_text(vtt_content, encoding="utf-8")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), str(vtt_path))

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_captions_handler(service)

    direct_url = "https://cdn.example.com/video.mp4"
    payload: CaptionsPayload = {"url": direct_url, "preferred_langs": ["en"]}
    out = handler(payload)

    assert probe.captured_subtitle_url == direct_url
    assert out["url"] == direct_url
    assert out["text"] == "Direct video"


def test_stt_direct_mp4_url(tmp_path: Path) -> None:
    """Verify that direct .mp4 URLs work for STT."""
    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient(
        [
            {"text": "Transcribed", "start": 0.0, "end": 0.5},
            {"text": "audio", "start": 0.5, "end": 1.0},
        ]
    )
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), None)

    # Mock subprocess_run to return fake ffprobe duration for direct URLs
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        # Return fake ffprobe duration output
        if args and args[0] == "ffprobe":
            return _FakeSubprocessResult(0, "5.0\n")
        return _FakeSubprocessResult(1, "")

    _test_hooks.subprocess_run = fake_subprocess_run

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    direct_url = "https://cdn.example.com/lecture.mp4"
    payload: STTPayload = {"url": direct_url}
    out = handler(payload)

    # For direct URLs, probe() is not called - only download_audio()
    # Duration is obtained from ffprobe on the downloaded file
    assert probe.captured_probe_url is None  # probe() not called for direct URLs
    assert probe.captured_download_url == direct_url
    assert out["url"] == direct_url
    assert out["text"] == "Transcribed audio"


def test_stt_direct_url_ffprobe_fails(tmp_path: Path) -> None:
    """Test that ffprobe failure for direct URLs raises duration unknown error."""
    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient([])
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), None)

    # Mock subprocess_run to fail (non-zero return code)
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        # Return failed ffprobe
        return _FakeSubprocessResult(1, "")

    _test_hooks.subprocess_run = fake_subprocess_run

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    direct_url = "https://cdn.example.com/video.mp4"
    payload: STTPayload = {"url": direct_url}

    with pytest.raises(AppError) as excinfo:
        handler(payload)
    err: AppError[TranscriptErrorCode] = excinfo.value
    assert err.code == TranscriptErrorCode.STT_DURATION_UNKNOWN


def test_stt_direct_url_ffprobe_exception(tmp_path: Path) -> None:
    """Test that ffprobe OSError for direct URLs is handled."""

    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient([])
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), None)

    # Mock subprocess_run to raise OSError
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        raise OSError("ffprobe not found")

    _test_hooks.subprocess_run = fake_subprocess_run

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    direct_url = "https://cdn.example.com/video.mp4"
    payload: STTPayload = {"url": direct_url}

    with pytest.raises(AppError) as excinfo:
        handler(payload)
    err: AppError[TranscriptErrorCode] = excinfo.value
    assert err.code == TranscriptErrorCode.STT_DURATION_UNKNOWN


def test_stt_direct_url_duration_too_long(tmp_path: Path) -> None:
    """Test that direct URL with duration exceeding max raises error."""
    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient([])
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"fake")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), None)

    # Mock subprocess_run to return duration longer than max
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        # Return duration of 120 seconds (exceeds max of 60)
        return _FakeSubprocessResult(0, "120.0\n")

    _test_hooks.subprocess_run = fake_subprocess_run

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,  # Max 60 seconds
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    direct_url = "https://cdn.example.com/video.mp4"
    payload: STTPayload = {"url": direct_url}

    with pytest.raises(AppError) as excinfo:
        handler(payload)
    err: AppError[TranscriptErrorCode] = excinfo.value
    assert err.code == TranscriptErrorCode.STT_TOO_LONG


class _LargeFileProbe(ProbeDownloadClient):
    """Probe client that returns a large file size for testing over-limit behavior."""

    def __init__(self, path: str, file_size_bytes: int) -> None:
        self._path = path
        self._file_size_bytes = file_size_bytes
        self.captured_download_url: str | None = None

    def probe(self, url: str) -> YtInfoTD:
        return {"duration": 5}

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        self.captured_download_url = url
        return self._path

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        return None


def test_stt_direct_url_over_file_limit(tmp_path: Path) -> None:
    """Test that direct URL with file size over limit triggers chunking path.

    When chunking is enabled and file exceeds threshold, _handle_over_limit is called.
    """
    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient(
        [
            {"text": "Chunked", "start": 0.0, "end": 0.5},
            {"text": "result", "start": 0.5, "end": 1.0},
        ]
    )
    audio = tmp_path / "large.m4a"
    # Create a file that will appear large via os_stat mock
    audio.write_bytes(b"fake")

    # Mock file size to be over the limit (25MB config, so use 30MB)
    large_file_size = 30 * 1024 * 1024  # 30 MB

    def fake_os_stat(path: str) -> os.stat_result:
        # os.stat_result tuple: (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
        return os.stat_result((0, 0, 0, 0, 0, 0, large_file_size, 0, 0, 0))

    def fake_os_path_getsize(path: str) -> int:
        return large_file_size

    # Mock subprocess_run for ffprobe duration
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        if args and args[0] == "ffprobe":
            return _FakeSubprocessResult(0, "5.0\n")
        return _FakeSubprocessResult(1, "")

    _test_hooks.subprocess_run = fake_subprocess_run
    _test_hooks.os_stat = fake_os_stat
    _test_hooks.os_path_getsize = fake_os_path_getsize

    probe = _LargeFileProbe(str(audio), large_file_size)

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,  # 25 MB limit, file is 30 MB
            TRANSCRIPT_ENABLE_CHUNKING=False,  # Disabled, so should raise error
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    direct_url = "https://cdn.example.com/large_video.mp4"
    payload: STTPayload = {"url": direct_url}

    # With chunking disabled, over-limit file should raise STT_CHUNKING_DISABLED
    with pytest.raises(AppError) as excinfo:
        handler(payload)
    err: AppError[TranscriptErrorCode] = excinfo.value
    assert err.code == TranscriptErrorCode.STT_CHUNKING_DISABLED


def test_stt_direct_url_cleanup_oserror(tmp_path: Path) -> None:
    """Test that OSError during cleanup is properly raised for direct URLs."""
    from transcript_api import _test_hooks
    from transcript_api._test_hooks import SubprocessRunResult

    yt = _FakeYTClient([])
    stt = _FakeSTTClient(
        [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]
    )
    # Create audio in system temp dir so cleanup is triggered
    import tempfile

    tmp_dir = tempfile.gettempdir()
    audio = Path(tmp_dir) / "test_cleanup_audio.m4a"
    audio.write_bytes(b"fake")

    probe = _UrlCapturingProbe({"duration": 5}, str(audio), None)

    # Mock subprocess_run for ffprobe duration
    def fake_subprocess_run(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult:
        if args and args[0] == "ffprobe":
            return _FakeSubprocessResult(0, "5.0\n")
        return _FakeSubprocessResult(1, "")

    # Mock os_remove to raise OSError
    def fake_os_remove(path: str) -> None:
        raise OSError("Permission denied")

    _test_hooks.subprocess_run = fake_subprocess_run
    _test_hooks.os_remove = fake_os_remove

    deps = AppDeps(
        config=Config(
            TRANSCRIPT_MAX_VIDEO_SECONDS=60,
            TRANSCRIPT_MAX_FILE_MB=25,
            TRANSCRIPT_ENABLE_CHUNKING=False,
            TRANSCRIPT_CHUNK_THRESHOLD_MB=20.0,
            TRANSCRIPT_TARGET_CHUNK_MB=20.0,
            TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600.0,
            TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3,
            TRANSCRIPT_SILENCE_THRESHOLD_DB=-40.0,
            TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5,
            TRANSCRIPT_STT_RTF=0.5,
            TRANSCRIPT_DL_MIB_PER_SEC=4.0,
            TRANSCRIPT_PREFERRED_LANGS=None,
        ),
        clients=Clients(youtube=yt, stt=stt, probe=probe),
    )
    from transcript_api.service import TranscriptService as _Svc

    service = _Svc(deps["config"], deps["clients"])
    handler = build_stt_handler(service)

    direct_url = "https://cdn.example.com/video.mp4"
    payload: STTPayload = {"url": direct_url}

    # Cleanup OSError should be raised
    with pytest.raises(OSError):
        handler(payload)
