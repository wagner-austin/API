"""STTTranscriptProvider: helpers, probe/download, limits, duration."""

from __future__ import annotations

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
