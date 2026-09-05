"""STTTranscriptProvider: chunked transcription paths."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol

import pytest
from platform_core.errors import AppError
from platform_stt import VerboseResponse, VerboseSegment

from transcript_api import _test_hooks
from transcript_api.stt_provider import (
    STTTranscriptProvider,
)
from transcript_api.types import (
    AudioChunk,
    SubtitleResultTD,
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


def test_transcribe_chunked_ffmpeg_unavailable(tmp_path: Path) -> None:
    """Test that _transcribe_chunked raises when ffmpeg is unavailable."""
    prov, _ = _make_provider()
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    prov.enable_chunking = True

    # Set ffmpeg as unavailable via hook
    _test_hooks.ffmpeg_available = lambda: False

    with pytest.raises(AppError):
        _ = prov._transcribe_chunked(str(audio))


def test_transcribe_chunked_single_chunk_passthrough(tmp_path: Path) -> None:
    """Test that single-chunk case passes through to normal transcribe."""
    from transcript_api._test_hooks import AudioChunkerProto

    # Create a real audio file
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x" * 100)

    # Create provider with test data
    stt = _StubSTTClient([{"text": "single", "start": 0, "end": 10}])
    info: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info, str(audio))
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )

    # Set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True

    # Create a fake chunker that returns a single chunk pointing to the original file
    class _OneChunker:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            return [
                AudioChunk(
                    path=path,
                    start_seconds=0.0,
                    duration_seconds=duration,
                    size_bytes=0,
                )
            ]

    def _one_chunker_factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto:
        return _OneChunker()

    _test_hooks.audio_chunker_factory = _one_chunker_factory

    # Hook subprocess to simulate ffprobe returning duration
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
            stdout: bytes | str | None = '{"format": {"duration": "10.0"}}'
            stderr: bytes | str | None = None

        return _Proc()

    _test_hooks.subprocess_run = _fake_subprocess

    out = prov._transcribe_chunked(str(audio))
    assert len(out) == 1


def test_transcribe_chunked_multi_chunk_merges_and_cleans(tmp_path: Path) -> None:
    """Test that multi-chunk transcription merges results and cleans up chunk files."""
    from transcript_api._test_hooks import AudioChunkerProto

    audio = tmp_path / "src.m4a"
    audio.write_bytes(b"x" * 100)

    # Create provider with stub clients
    stt = _StubSTTClient([{"text": "chunk", "start": 0, "end": 10}])
    info: YtInfoTD = {"duration": 20}
    probe = _StubProbeDownloadClient(info, str(audio))
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    prov.enable_chunking = True

    # Set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True

    # Create stub chunker that creates 2 chunk files
    class _StubChunker:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            p1 = tmp_path / "c1.m4a"
            p2 = tmp_path / "c2.m4a"
            p1.write_bytes(b"a")
            p2.write_bytes(b"b")
            return [
                AudioChunk(path=str(p1), start_seconds=0.0, duration_seconds=10.0, size_bytes=1),
                AudioChunk(path=str(p2), start_seconds=10.0, duration_seconds=10.0, size_bytes=1),
            ]

    def _stub_chunker_factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto:
        return _StubChunker()

    _test_hooks.audio_chunker_factory = _stub_chunker_factory

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

    out = prov._transcribe_chunked(str(audio))
    starts = [s["start"] for s in out]
    assert starts == [0.0, 10.0]
    # Verify chunk files were cleaned up
    assert not (tmp_path / "c1.m4a").exists()
    assert not (tmp_path / "c2.m4a").exists()


def test_transcribe_chunked_missing_chunk_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that missing chunk files during cleanup are logged (not crashed)."""
    from transcript_api._test_hooks import AudioChunkerProto

    audio = tmp_path / "src2.m4a"
    audio.write_bytes(b"x" * 100)

    # Create two chunk files - one will be deleted when the other is cleaned up
    chunk1 = tmp_path / "chunk1.m4a"
    chunk2 = tmp_path / "chunk2.m4a"
    chunk1.write_bytes(b"test1")
    chunk2.write_bytes(b"test2")

    # Create provider with stub clients that return segments for both chunks
    stt = _StubSTTClient(
        [
            {"text": "x", "start": 0, "end": 5},
            {"text": "y", "start": 5, "end": 10},
        ]
    )
    info: YtInfoTD = {"duration": 10}
    probe = _StubProbeDownloadClient(info, str(audio))
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    prov.enable_chunking = True

    # Set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True

    # Create chunker that returns two chunk paths
    class _TwoChunker:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            return [
                AudioChunk(
                    path=str(chunk1),
                    start_seconds=0.0,
                    duration_seconds=5.0,
                    size_bytes=100,
                ),
                AudioChunk(
                    path=str(chunk2),
                    start_seconds=5.0,
                    duration_seconds=5.0,
                    size_bytes=100,
                ),
            ]

    def _two_chunker_factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto:
        return _TwoChunker()

    _test_hooks.audio_chunker_factory = _two_chunker_factory

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
            stdout: bytes | str | None = '{"format": {"duration": "10.0"}}'
            stderr: bytes | str | None = None

        return _Proc()

    _test_hooks.subprocess_run = _fake_subprocess

    # Hook os_remove to delete chunk1, but also delete chunk2 (simulating race)
    def _removing_os_remove(path: str) -> None:
        import os

        os.remove(path)
        # When cleaning up chunk1, also delete chunk2 to simulate race condition
        if path == str(chunk1) and chunk2.exists():
            os.remove(str(chunk2))

    _test_hooks.os_remove = _removing_os_remove

    # Run transcription - cleanup should hit missing file path for chunk2
    with caplog.at_level(logging.WARNING):
        prov._transcribe_chunked(str(audio))

    # Check that warning was logged about missing chunk file
    assert "Chunk file missing during cleanup" in caplog.text


def test_transcribe_chunked_skips_original_path_cleanup(tmp_path: Path) -> None:
    """Test that original audio file is not removed during cleanup, only chunk files."""
    from transcript_api._test_hooks import AudioChunkerProto

    audio = tmp_path / "src3.m4a"
    audio.write_bytes(b"x" * 100)

    # Create provider with stub clients that return 2 results
    stt = _StubSTTClient(
        [
            {"text": "p", "start": 0, "end": 10},
        ]
    )
    info: YtInfoTD = {"duration": 20}
    probe = _StubProbeDownloadClient(info, str(audio))
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=60,
        max_file_mb=10,
    )
    prov.enable_chunking = True

    # Set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True

    # Create chunker that returns original path as one chunk + another file
    class _ChunkerWithOriginal:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            other = tmp_path / "other.m4a"
            other.write_bytes(b"y" * 100)
            return [
                AudioChunk(
                    path=str(path),  # Original file - should NOT be deleted
                    start_seconds=0.0,
                    duration_seconds=10.0,
                    size_bytes=1,
                ),
                AudioChunk(
                    path=str(other),  # Other file - SHOULD be deleted
                    start_seconds=10.0,
                    duration_seconds=10.0,
                    size_bytes=1,
                ),
            ]

    def _chunker_factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto:
        return _ChunkerWithOriginal()

    _test_hooks.audio_chunker_factory = _chunker_factory

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

    out = prov._transcribe_chunked(str(audio))
    # Should have results from both chunks
    assert len(out) >= 2
    # Original audio file should still exist
    assert audio.exists()
    # Other chunk file should be cleaned up
    assert not (tmp_path / "other.m4a").exists()
