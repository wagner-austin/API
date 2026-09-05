"""App API direct-URL handling: canonical URLs and direct MP4s."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from fastapi import FastAPI
from platform_stt import VerboseResponse, VerboseSegment

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
    def __init__(self, segments: list[VerboseSegment]) -> None:
        self._segments = segments

    def transcribe_verbose(
        self,
        *,
        file: BinaryIO,
        timeout: float | None,
    ) -> VerboseResponse:
        # Return structure compatible with convert_verbose_to_segments
        return {"text": "", "language": None, "segments": self._segments}


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
