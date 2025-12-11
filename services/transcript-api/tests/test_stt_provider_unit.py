from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from transcript_api import _test_hooks
from transcript_api.stt_provider import (
    STTTranscriptProvider,
    _as_float,
    _is_numeric_str,
)
from transcript_api.types import (
    AudioChunk,
    SubtitleResultTD,
    TranscriptOptions,
    VerboseResponseTD,
    VerboseSegmentTD,
    YtInfoTD,
)


class _TranscribeLike(Protocol):
    pass


class _StubSTTClient:
    def __init__(self, responses: list[dict[str, float | int | str]]) -> None:
        self._responses = responses
        self.calls = 0

    def transcribe_verbose(self, *, file: BinaryIO, timeout: float | None) -> VerboseResponseTD:
        data = file.read()
        self.calls += 1
        size = len(data)
        segments: list[VerboseSegmentTD] = []
        for idx, base in enumerate(self._responses):
            text = f"{base.get('text', '')} {size}" if idx == 0 else str(base.get("text", ""))
            segments.append(
                {
                    "text": text.strip(),
                    "start": float(base.get("start", 0.0)),
                    "end": float(base.get("end", 1.0)),
                }
            )
        return {"text": "", "segments": segments}


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


def test_numeric_helpers_edges() -> None:
    assert _is_numeric_str("10")
    assert _is_numeric_str("+3.5")
    assert not _is_numeric_str("")
    assert not _is_numeric_str("+")
    assert not _is_numeric_str("1.2.3")
    assert _as_float("5.0") == 5.0
    assert _as_float("bad") == 0.0
    assert _as_float("") == 0.0
    assert _as_float(None) == 0.0
    assert _as_float(2) == 2.0
    # object() is not in the accepted types (int | float | str)


def test_should_cleanup_variants(tmp_path: Path) -> None:
    prov, _ = _make_provider()

    assert prov._should_cleanup("") is False

    owned_dir = tmp_path / "ytstt_owned"
    owned_dir.mkdir()
    owned_file = owned_dir / "a.m4a"
    owned_file.write_bytes(b"x")
    prov._owned_tmp_dirs.add(os.path.abspath(str(owned_dir)))
    assert prov._should_cleanup(str(owned_file)) is True

    other_dir = tmp_path / "ytstt_other"
    other_dir.mkdir()
    other_file = other_dir / "b.m4a"
    other_file.write_bytes(b"y")
    assert prov._should_cleanup(str(other_file)) is True


def test_probe_or_error_rejects_invalid_and_too_long() -> None:
    stt = _StubSTTClient([{"text": "x", "start": 0, "end": 1}])
    info_short: YtInfoTD = {"duration": 0}
    probe_short = _StubProbeDownloadClient(info_short, "unused")
    prov_short = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe_short,
        max_video_seconds=60,
        max_file_mb=10,
    )
    with pytest.raises(AppError):
        _ = prov_short._probe_or_error("vid", "https://youtu.be/dQw4w9WgXcQ")

    info_long: YtInfoTD = {"duration": 120}
    probe_long = _StubProbeDownloadClient(info_long, "unused")
    prov_long = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe_long,
        max_video_seconds=10,
        max_file_mb=10,
    )
    with pytest.raises(AppError):
        _ = prov_long._probe_or_error("vid", "https://youtu.be/dQw4w9WgXcQ")


def test_probe_or_error_success() -> None:
    stt = _StubSTTClient([{"text": "x", "start": 0, "end": 1}])
    info_probe: YtInfoTD = {"duration": 42}
    probe = _StubProbeDownloadClient(info_probe, "unused")
    prov = STTTranscriptProvider(
        stt_client=stt,
        probe_client=probe,
        max_video_seconds=100,
        max_file_mb=10,
    )
    dur = prov._probe_or_error("vid", "https://youtu.be/dQw4w9WgXcQ")
    assert dur == 42


def test_download_or_error_stat_happy_and_retry() -> None:
    prov, path = _make_provider(tmp_file_size=8)

    calls = {"n": 0}

    def _stat_retry(pth: str) -> os.stat_result:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("first fail")
        # Create a minimal stat result with st_size=123
        return os.stat_result((0, 0, 0, 0, 0, 0, 123, 0, 0, 0))

    _test_hooks.os_stat = _stat_retry
    out_path, size = prov._download_or_error("https://x")
    assert out_path == path and size == 123 and calls["n"] == 2


def test_download_or_error_stat_failure() -> None:
    prov, _ = _make_provider(tmp_file_size=4)

    def _stat_fail(pth: str) -> os.stat_result:
        raise OSError("fail")

    _test_hooks.os_stat = _stat_fail
    with pytest.raises(AppError):
        _ = prov._download_or_error("https://x")


def test_transcribe_with_strategy_chunk_error() -> None:
    """Test that chunking error raises AppError."""
    from transcript_api._test_hooks import AudioChunkerProto

    prov, path = _make_provider(tmp_file_size=16)
    prov.enable_chunking = True
    # Set chunk threshold low so file triggers chunking
    prov.chunk_threshold_mb = 0.00001

    # Set ffmpeg as available so chunking path is taken
    _test_hooks.ffmpeg_available = lambda: True

    # Create chunker that raises an error
    class _ErrorChunker:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            raise RuntimeError("boom")

    def _error_chunker_factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto:
        return _ErrorChunker()

    _test_hooks.audio_chunker_factory = _error_chunker_factory

    # Hook subprocess for ffprobe (needed to get duration)
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

    # Set os_path_getsize to return small file to trigger chunking
    _test_hooks.os_path_getsize = lambda p: 16

    with pytest.raises(AppError):
        _ = prov._transcribe_with_strategy(path)


def test_handle_over_limit_branches(tmp_path: Path) -> None:
    """Test _handle_over_limit behavior with chunking enabled/disabled."""
    from transcript_api._test_hooks import AudioChunkerProto

    prov, _ = _make_provider()
    prov.enable_chunking = False
    with pytest.raises(AppError):
        _ = prov._handle_over_limit("a.m4a", 1024)

    prov.enable_chunking = True

    # Create audio file for chunking
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x" * 100)

    # Set ffmpeg as available
    _test_hooks.ffmpeg_available = lambda: True

    # Create chunker that returns a single chunk
    class _SingleChunker:
        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            return [AudioChunk(path=path, start_seconds=0.0, duration_seconds=1.0, size_bytes=100)]

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
            stdout: bytes | str | None = '{"format": {"duration": "1.0"}}'
            stderr: bytes | str | None = None

        return _Proc()

    _test_hooks.subprocess_run = _fake_subprocess

    out = prov._handle_over_limit(str(audio), 1024)
    assert len(out) == 1


def test_should_chunk_branches(tmp_path: Path) -> None:
    prov, _ = _make_provider()
    prov.enable_chunking = False
    assert prov._should_chunk(str(tmp_path / "x.m4a")) is False

    prov.enable_chunking = True

    def _size_fail(_: str) -> int:
        raise OSError("fail")

    _test_hooks.os_path_getsize = _size_fail
    assert prov._should_chunk(str(tmp_path / "x.m4a")) is False

    def _size_ok(_: str) -> int:
        return 2 * 1024 * 1024

    _test_hooks.os_path_getsize = _size_ok
    prov.chunk_threshold_mb = 1.0
    assert prov._should_chunk(str(tmp_path / "x.m4a")) is True


def test_get_audio_duration_success_and_error(tmp_path: Path) -> None:
    import subprocess

    from transcript_api._test_hooks import SubprocessRunResult

    prov, _ = _make_provider()
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")

    class _Proc:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout: bytes | str | None = stdout
            self.stderr: bytes | str | None = None

    def _run_ok(
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
        data = {"format": {"duration": "2.5"}}
        return _Proc(stdout=dump_json_str(data))

    _test_hooks.subprocess_run = _run_ok
    dur = prov._get_audio_duration(str(audio))
    assert dur == 2.5

    def _run_fail(
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
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    _test_hooks.subprocess_run = _run_fail
    dur2 = prov._get_audio_duration(str(audio))
    assert dur2 == 0.0

    def _run_list(
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
        items: list[str] = ["not-a-dict"]
        body = dump_json_str(items)
        return _Proc(stdout=body)

    _test_hooks.subprocess_run = _run_list
    dur3 = prov._get_audio_duration(str(audio))
    assert dur3 == 0.0

    def _run_format_not_dict(
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
        return _Proc(stdout='{"format": "bad"}')

    _test_hooks.subprocess_run = _run_format_not_dict
    dur4 = prov._get_audio_duration(str(audio))
    assert dur4 == 0.0

    def _run_duration_not_str(
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
        return _Proc(stdout='{"format": {"duration": 5}}')

    _test_hooks.subprocess_run = _run_duration_not_str
    dur5 = prov._get_audio_duration(str(audio))
    assert dur5 == 0.0


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
