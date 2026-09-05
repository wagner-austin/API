"""STTTranscriptProvider: fetch lifecycle, estimates, cookies."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol

import pytest
from platform_stt import VerboseResponse, VerboseSegment

from transcript_api import _test_hooks
from transcript_api.stt_provider import (
    STTTranscriptProvider,
)
from transcript_api.types import (
    AudioChunk,
    SubtitleResultTD,
    TranscriptOptions,
    YtInfoTD,
)


class _TranscribeLike(Protocol):
    pass


class _StubSTTClient:
    def __init__(self, responses: list[dict[str, float | int | str]]) -> None:
        self._responses = responses
        self.calls = 0

    def transcribe_verbose(self, *, file: BinaryIO, timeout: float | None) -> VerboseResponse:
        data = file.read()
        self.calls += 1
        size = len(data)
        segments: list[VerboseSegment] = []
        for idx, base in enumerate(self._responses):
            text = f"{base.get('text', '')} {size}" if idx == 0 else str(base.get("text", ""))
            segments.append(
                {
                    "text": text.strip(),
                    "start": float(base.get("start", 0.0)),
                    "end": float(base.get("end", 1.0)),
                }
            )
        return {"text": "", "language": None, "segments": segments}


class _StubProbeDownloadClient:
    def __init__(self, info: YtInfoTD, download_path: str) -> None:
        self._info = info
        self._path = download_path
        self.probe_calls = 0
        self.download_calls = 0
        self.cookies_used: list[str | None] = []

    def probe(self, url: str) -> YtInfoTD:
        self.probe_calls += 1
        return self._info

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        self.download_calls += 1
        self.cookies_used.append(cookies_path)
        return self._path

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        return None


def _make_provider(tmp_file_size: int = 0) -> tuple[STTTranscriptProvider, str]:
    fd, path = tempfile.mkstemp(prefix="stt_unit_", suffix=".bin")
    os.close(fd)
    if tmp_file_size > 0:
        with open(path, "wb") as f:
            f.write(b"x" * tmp_file_size)
    stt = _StubSTTClient(
        [
            {"text": "seg", "start": 0, "end": 1},
        ]
    )
    info0: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info0, path)
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    return prov, path


def test_transcribe_uses_stt_client(tmp_path: os.PathLike[str]) -> None:
    fd, path = tempfile.mkstemp(prefix="stt_unit_trans_", suffix=".bin")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"abc")
    stt = _StubSTTClient(
        [
            {"text": "hello", "start": 0, "end": 1},
        ]
    )
    probe = _StubProbeDownloadClient({"duration": 10}, path)
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    segs = prov._transcribe(path)
    assert segs and segs[0]["text"].endswith("3")
    assert stt.calls == 1


def test_transcribe_with_strategy_passthrough() -> None:
    """Test that _transcribe_with_strategy passes through to normal transcribe when not chunking."""
    prov, path = _make_provider(tmp_file_size=16)
    prov.enable_chunking = True
    # Set chunk threshold high so file doesn't trigger chunking
    prov.chunk_threshold_mb = 100.0

    # Make os_path_getsize return small size to avoid chunking
    _test_hooks.os_path_getsize = lambda p: 16

    out = prov._transcribe_with_strategy(path)
    # The stub STT client returns a segment with "seg" text
    assert len(out) == 1 and "seg" in out[0]["text"]


def test_estimate_and_eta_minutes_branching() -> None:
    """Test estimate and ETA calculation branches."""
    stt = _StubSTTClient(
        [
            {"text": "x", "start": 0, "end": 1},
        ]
    )
    info: YtInfoTD = {
        "duration": 120,
        "formats": [
            {"vcodec": "none", "acodec": "aac", "abr": 64.0, "filesize": 1024 * 1024 * 5},
        ],
    }
    probe = _StubProbeDownloadClient(info, "unused")
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=600,
        max_file_mb=50,
        enable_chunking=False,
    )
    dur, approx = prov.estimate("https://x")
    assert dur == 120 and approx > 0.0
    eta_no_chunk = prov.estimate_eta_minutes(dur, approx)
    assert eta_no_chunk >= 1

    # Use hook to set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True
    prov.enable_chunking = True
    eta_with_chunk = prov.estimate_eta_minutes(dur, approx)
    assert eta_with_chunk <= eta_no_chunk

    info2: YtInfoTD = {"duration": 60}  # formats ignored when not a list
    probe2 = _StubProbeDownloadClient(info2, "unused")
    prov2 = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe2,
        max_video_seconds=600,
        max_file_mb=50,
    )
    dur2, approx2 = prov2.estimate("https://x")
    assert dur2 == 60 and approx2 == 0.0


def test_estimate_formats_edge_cases() -> None:
    stt = _StubSTTClient(
        [
            {"text": "x", "start": 0, "end": 1},
        ]
    )
    info: YtInfoTD = {
        "duration": 100,
        "formats": [
            {"vcodec": "h264", "acodec": "aac", "abr": 64.0, "filesize": 1024 * 1024},
            {"vcodec": "none", "acodec": "none", "abr": 64.0, "filesize": 1024 * 1024},
            {"vcodec": "none", "acodec": "aac", "abr": 32.0, "filesize": 2 * 1024 * 1024},
            {"vcodec": "none", "acodec": "aac", "abr": 16.0, "filesize": 512 * 1024},
        ],
    }
    probe = _StubProbeDownloadClient(info, "unused")
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=600,
        max_file_mb=50,
    )
    dur, approx = prov.estimate("https://x")
    assert dur == 100 and approx > 0.0

    info2: YtInfoTD = {
        "duration": 50,
        "formats": [
            {"vcodec": "none", "acodec": "aac", "abr": 64.0},
        ],
    }
    probe2 = _StubProbeDownloadClient(info2, "unused")
    prov2 = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe2,
        max_video_seconds=600,
        max_file_mb=50,
    )
    dur2, approx2 = prov2.estimate("https://x")
    assert dur2 == 50 and approx2 > 0.0


# Note: non-dict format branch in estimate removed.


def test_estimate_eta_minutes_chunk_branch() -> None:
    """Test ETA calculation with chunking enabled."""
    stt = _StubSTTClient(
        [
            {"text": "x", "start": 0, "end": 1},
        ]
    )
    info_eta: YtInfoTD = {"duration": 120}
    probe = _StubProbeDownloadClient(info_eta, "unused")
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=600,
        max_file_mb=5,
        enable_chunking=True,
    )
    # Use hook to set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True
    eta = prov.estimate_eta_minutes(120, 50.0)
    assert eta >= 1


def test_post_init_cookies_text_success_and_cleanup(tmp_path: Path) -> None:
    import base64

    encoded = base64.b64encode(b"cookie-data").decode("ascii")
    stt = _StubSTTClient([{"text": "x", "start": 0, "end": 1}])
    info_cookies: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info_cookies, str(tmp_path / "a.m4a"))
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
        cookies_text=encoded,
        cookies_path=None,
    )
    path = prov._temp_cookies_file
    assert type(path) is str
    cookie_path = Path(path)
    assert cookie_path.exists()
    body = cookie_path.read_text(encoding="utf-8")
    assert "cookie-data" in body
    prov.__del__()
    assert not cookie_path.exists()


def test_post_init_cookies_text_invalid() -> None:
    stt = _StubSTTClient([{"text": "x", "start": 0, "end": 1}])
    info_invalid: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info_invalid, "unused")
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
        cookies_text="!!not-base64!!",
        cookies_path=None,
    )
    assert prov._temp_cookies_file is None


def test_is_over_limit_branches() -> None:
    prov, _ = _make_provider()
    assert prov._is_over_limit(0) is False
    big_bytes = 25 * 1024 * 1024
    assert prov._is_over_limit(big_bytes) is True


def test_fetch_success_and_cleanup() -> None:
    """Test that fetch successfully transcribes and cleans up temp files."""
    # Create a provider with a temp file that will be cleaned up
    fd, path = tempfile.mkstemp(prefix="stt_cleanup_test_", suffix=".bin")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"x" * 16)

    stt = _StubSTTClient([{"text": "ok", "start": 0, "end": 1}])
    info: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info, path)

    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    prov.enable_chunking = False

    # Track removal calls
    removed: list[str] = []

    def _remove(p: str) -> None:
        removed.append(os.path.abspath(p))
        # Actually remove the file
        os.remove(p)

    _test_hooks.os_remove = _remove

    out = prov.fetch("vid", TranscriptOptions(preferred_langs=["en"]))
    assert len(out) == 1
    # Check cleanup happened
    assert os.path.abspath(path) in removed
    assert not os.path.exists(path)


def test_fetch_cleanup_raises() -> None:
    """Test that OSError during cleanup is propagated."""
    # Create a provider with a temp file
    fd, path = tempfile.mkstemp(prefix="stt_cleanup_fail_", suffix=".bin")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"x" * 8)

    stt = _StubSTTClient([{"text": "ok", "start": 0, "end": 1}])
    info: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info, path)

    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    prov.enable_chunking = False

    def _remove_raise(p: str) -> None:
        raise OSError(f"rm {p}")

    _test_hooks.os_remove = _remove_raise

    with pytest.raises(OSError):
        _ = prov.fetch("vid", TranscriptOptions(preferred_langs=["en"]))

    # Clean up the temp file that didn't get removed due to error
    if os.path.exists(path):
        os.remove(path)


def test_fetch_over_limit_uses_handle(tmp_path: Path) -> None:
    """Test that fetch uses chunking when file is over limit."""
    from transcript_api._test_hooks import AudioChunkerProto

    # Create a file that appears to be over the limit
    audio = tmp_path / "big.m4a"
    audio.write_bytes(b"x" * 100)

    # Create provider with low max_file_mb so file appears over limit
    stt = _StubSTTClient([{"text": "big", "start": 0, "end": 1}])
    info: YtInfoTD = {"duration": 20}
    probe = _StubProbeDownloadClient(info, str(audio))
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=600,
        max_file_mb=0,  # Very small limit so file is "over"
        enable_chunking=True,
    )

    # Set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True

    # Set os_stat to return a large size to trigger over-limit
    def _stat_large(pth: str) -> os.stat_result:
        return os.stat_result((0, 0, 0, 0, 0, 0, 10 * 1024 * 1024, 0, 0, 0))

    _test_hooks.os_stat = _stat_large

    # Create chunker that returns the original file (single chunk)
    class _SingleChunker:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            return [
                AudioChunk(path=path, start_seconds=0.0, duration_seconds=duration, size_bytes=100)
            ]

    def _single_chunker_factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto:
        return _SingleChunker()

    _test_hooks.audio_chunker_factory = _single_chunker_factory

    # Hook subprocess for ffprobe
    def _fake_subprocess(
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _test_hooks.SubprocessRunResult:
        class _Proc:
            returncode = 0
            stdout: bytes | str | None = '{"format": {"duration": "20.0"}}'
            stderr: bytes | str | None = None

        return _Proc()

    _test_hooks.subprocess_run = _fake_subprocess

    out = prov.fetch("vid", TranscriptOptions(preferred_langs=["en"]))
    # Should have a result from the chunked transcription
    assert len(out) == 1


def test_ffmpeg_available() -> None:
    prov, _ = _make_provider()
    _ = prov._ffmpeg_available()


logger = logging.getLogger(__name__)
