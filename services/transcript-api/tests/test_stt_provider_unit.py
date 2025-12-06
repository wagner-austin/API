from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from transcript_api.stt_provider import (
    STTTranscriptProvider,
    _as_float,
    _is_numeric_str,
)
from transcript_api.types import (
    AudioChunk,
    SubtitleResultTD,
    TranscriptOptions,
    TranscriptSegment,
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


def test_download_or_error_stat_happy_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, path = _make_provider(tmp_file_size=8)

    class _Stat:
        def __init__(self, size: int) -> None:
            self.st_size = size

    calls = {"n": 0}

    def _stat_retry(pth: str) -> _Stat:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("first fail")
        return _Stat(123)

    monkeypatch.setattr(os, "stat", _stat_retry, raising=True)
    out_path, size = prov._download_or_error("https://x")
    assert out_path == path and size == 123 and calls["n"] == 2


def test_download_or_error_stat_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, _ = _make_provider(tmp_file_size=4)

    def _stat_fail(pth: str) -> None:
        raise OSError("fail")

    monkeypatch.setattr(os, "stat", _stat_fail, raising=True)
    with pytest.raises(AppError):
        _ = prov._download_or_error("https://x")


def test_transcribe_with_strategy_chunk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, path = _make_provider(tmp_file_size=16)
    prov.enable_chunking = True

    def _always_chunk(_: str) -> bool:
        return True

    monkeypatch.setattr(prov, "_should_chunk", _always_chunk, raising=True)

    def _raise_chunk(p: str) -> list[TranscriptSegment]:
        raise RuntimeError("boom")

    monkeypatch.setattr(prov, "_transcribe_chunked", _raise_chunk, raising=True)
    with pytest.raises(AppError):
        _ = prov._transcribe_with_strategy(path)


def test_handle_over_limit_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, _ = _make_provider()
    prov.enable_chunking = False
    with pytest.raises(AppError):
        _ = prov._handle_over_limit("a.m4a", 1024)

    prov.enable_chunking = True

    def _fake_chunk(path: str) -> list[TranscriptSegment]:
        return [TranscriptSegment(text="ok", start=0.0, duration=1.0)]

    monkeypatch.setattr(prov, "_transcribe_chunked", _fake_chunk, raising=True)
    out = prov._handle_over_limit("a.m4a", 1024)
    assert len(out) == 1 and out[0]["text"] == "ok"


def test_should_chunk_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prov, _ = _make_provider()
    prov.enable_chunking = False
    assert prov._should_chunk(str(tmp_path / "x.m4a")) is False

    prov.enable_chunking = True

    def _size_fail(_: str) -> int:
        raise OSError("fail")

    monkeypatch.setattr(os.path, "getsize", _size_fail, raising=True)
    assert prov._should_chunk(str(tmp_path / "x.m4a")) is False

    def _size_ok(_: str) -> int:
        return 2 * 1024 * 1024

    monkeypatch.setattr(os.path, "getsize", _size_ok, raising=True)
    prov.chunk_threshold_mb = 1.0
    assert prov._should_chunk(str(tmp_path / "x.m4a")) is True


def test_get_audio_duration_success_and_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prov, _ = _make_provider()
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")

    class _Proc:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _run_ok(cmd: list[str], capture_output: bool, text: bool, timeout: int) -> _Proc:
        data = {"format": {"duration": "2.5"}}
        return _Proc(stdout=dump_json_str(data))

    monkeypatch.setattr(subprocess, "run", _run_ok, raising=True)
    dur = prov._get_audio_duration(str(audio))
    assert dur == 2.5

    def _run_fail(cmd: list[str], capture_output: bool, text: bool, timeout: int) -> _Proc:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(subprocess, "run", _run_fail, raising=True)
    dur2 = prov._get_audio_duration(str(audio))
    assert dur2 == 0.0

    class _ProcList:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _run_list(cmd: list[str], capture_output: bool, text: bool, timeout: int) -> _ProcList:
        items: list[str] = ["not-a-dict"]
        body = dump_json_str(items)
        return _ProcList(stdout=body)

    monkeypatch.setattr(subprocess, "run", _run_list, raising=True)
    dur3 = prov._get_audio_duration(str(audio))
    assert dur3 == 0.0

    def _run_format_not_dict(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> _Proc:
        return _Proc(stdout='{"format": "bad"}')

    monkeypatch.setattr(subprocess, "run", _run_format_not_dict, raising=True)
    dur4 = prov._get_audio_duration(str(audio))
    assert dur4 == 0.0

    def _run_duration_not_str(
        cmd: list[str], capture_output: bool, text: bool, timeout: int
    ) -> _Proc:
        return _Proc(stdout='{"format": {"duration": 5}}')

    monkeypatch.setattr(subprocess, "run", _run_duration_not_str, raising=True)
    dur5 = prov._get_audio_duration(str(audio))
    assert dur5 == 0.0


def test_transcribe_chunked_ffmpeg_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prov, _ = _make_provider()
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    prov.enable_chunking = True
    monkeypatch.setattr(prov, "_ffmpeg_available", lambda: False, raising=True)
    with pytest.raises(AppError):
        _ = prov._transcribe_chunked(str(audio))


def test_transcribe_chunked_single_chunk_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prov, _ = _make_provider()
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")

    def _always_ffmpeg() -> bool:
        return True

    monkeypatch.setattr(prov, "_ffmpeg_available", _always_ffmpeg, raising=True)

    def _duration_10(_: str) -> float:
        return 10.0

    monkeypatch.setattr(prov, "_get_audio_duration", _duration_10, raising=True)

    class _OneChunker:
        def __init__(
            self,
            *,
            target_chunk_mb: float,
            max_chunk_duration_seconds: float,
            silence_threshold_db: float,
            silence_duration_seconds: float,
        ) -> None:
            pass

        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            return [
                AudioChunk(
                    path=path,
                    start_seconds=0.0,
                    duration_seconds=duration,
                    size_bytes=0,
                )
            ]

    from transcript_api import stt_provider as sp

    monkeypatch.setattr(sp, "AudioChunker", _OneChunker, raising=True)

    def _stub_transcribe(p: str) -> list[TranscriptSegment]:
        assert p == str(audio)
        return [TranscriptSegment(text="single", start=0.0, duration=10.0)]

    monkeypatch.setattr(prov, "_transcribe", _stub_transcribe, raising=True)
    out = prov._transcribe_chunked(str(audio))
    assert len(out) == 1 and out[0]["text"] == "single"


def test_transcribe_chunked_multi_chunk_merges_and_cleans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prov, _ = _make_provider()
    audio = tmp_path / "src.m4a"
    audio.write_bytes(b"x")
    prov.enable_chunking = True

    def _always_ffmpeg() -> bool:
        return True

    monkeypatch.setattr(prov, "_ffmpeg_available", _always_ffmpeg, raising=True)

    def _duration_20(_: str) -> float:
        return 20.0

    monkeypatch.setattr(prov, "_get_audio_duration", _duration_20, raising=True)

    class _StubChunker:
        def __init__(
            self,
            *,
            target_chunk_mb: float,
            max_chunk_duration_seconds: float,
            silence_threshold_db: float,
            silence_duration_seconds: float,
        ) -> None:
            pass

        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            p1 = tmp_path / "c1.m4a"
            p2 = tmp_path / "c2.m4a"
            p1.write_bytes(b"a")
            p2.write_bytes(b"b")
            return [
                AudioChunk(path=str(p1), start_seconds=0.0, duration_seconds=10.0, size_bytes=1),
                AudioChunk(path=str(p2), start_seconds=10.0, duration_seconds=10.0, size_bytes=1),
            ]

    from transcript_api import stt_provider as sp

    monkeypatch.setattr(sp, "AudioChunker", _StubChunker, raising=True)
    out = prov._transcribe_chunked(str(audio))
    starts = [s["start"] for s in out]
    assert starts == [0.0, 10.0]
    assert not (tmp_path / "c1.m4a").exists()
    assert not (tmp_path / "c2.m4a").exists()


def test_transcribe_chunked_missing_chunk_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prov, _ = _make_provider()
    audio = tmp_path / "src2.m4a"
    audio.write_bytes(b"x")
    prov.enable_chunking = True

    def _always_ffmpeg() -> bool:
        return True

    monkeypatch.setattr(prov, "_ffmpeg_available", _always_ffmpeg, raising=True)

    def _duration(_: str) -> float:
        return 10.0

    monkeypatch.setattr(prov, "_get_audio_duration", _duration, raising=True)

    class _MissingChunker:
        def __init__(
            self,
            *,
            target_chunk_mb: float,
            max_chunk_duration_seconds: float,
            silence_threshold_db: float,
            silence_duration_seconds: float,
        ) -> None:
            pass

        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            missing = tmp_path / "missing.m4a"
            return [
                AudioChunk(
                    path=str(missing),
                    start_seconds=0.0,
                    duration_seconds=duration,
                    size_bytes=0,
                )
            ]

    from transcript_api import stt_provider as sp

    class _StubTranscriber:
        def __init__(
            self,
            *,
            transcribe: _TranscribeLike,
            max_concurrent: int,
            max_retries: int,
            timeout_seconds: float,
        ) -> None:
            pass

        def transcribe_chunks(self, chunks: list[AudioChunk]) -> list[list[TranscriptSegment]]:
            return [[TranscriptSegment(text="x", start=0.0, duration=1.0)] for _ in chunks]

    monkeypatch.setattr(sp, "AudioChunker", _MissingChunker, raising=True)
    monkeypatch.setattr(sp, "ParallelTranscriber", _StubTranscriber, raising=True)
    prov._transcribe_chunked(str(audio))


def test_transcribe_chunked_skips_original_path_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prov, _ = _make_provider()
    audio = tmp_path / "src3.m4a"
    audio.write_bytes(b"x")
    prov.enable_chunking = True

    def _always_ffmpeg() -> bool:
        return True

    monkeypatch.setattr(prov, "_ffmpeg_available", _always_ffmpeg, raising=True)

    def _duration(_: str) -> float:
        return 20.0

    monkeypatch.setattr(prov, "_get_audio_duration", _duration, raising=True)

    class _ChunkerWithOriginal:
        def __init__(
            self,
            *,
            target_chunk_mb: float,
            max_chunk_duration_seconds: float,
            silence_threshold_db: float,
            silence_duration_seconds: float,
        ) -> None:
            pass

        def chunk_audio(self, path: str, duration: float, size_mb: float) -> list[AudioChunk]:
            other = tmp_path / "other.m4a"
            other.write_bytes(b"y")
            return [
                AudioChunk(
                    path=str(path),
                    start_seconds=0.0,
                    duration_seconds=10.0,
                    size_bytes=1,
                ),
                AudioChunk(
                    path=str(other),
                    start_seconds=10.0,
                    duration_seconds=10.0,
                    size_bytes=1,
                ),
            ]

    class _StubTranscriber2:
        def __init__(
            self,
            *,
            transcribe: _TranscribeLike,
            max_concurrent: int,
            max_retries: int,
            timeout_seconds: float,
        ) -> None:
            pass

        def transcribe_chunks(self, chunks: list[AudioChunk]) -> list[list[TranscriptSegment]]:
            return [
                [TranscriptSegment(text="p", start=0.0, duration=1.0)],
                [TranscriptSegment(text="q", start=0.0, duration=1.0)],
            ]

    from transcript_api import stt_provider as sp

    monkeypatch.setattr(sp, "AudioChunker", _ChunkerWithOriginal, raising=True)
    monkeypatch.setattr(sp, "ParallelTranscriber", _StubTranscriber2, raising=True)
    out = prov._transcribe_chunked(str(audio))
    assert [s["text"] for s in out] == ["p", "q"]
    assert audio.exists()
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


def test_transcribe_with_strategy_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, path = _make_provider(tmp_file_size=16)
    prov.enable_chunking = True

    def _never_chunk(_: str) -> bool:
        return False

    monkeypatch.setattr(prov, "_should_chunk", _never_chunk, raising=True)

    def _stub_transcribe(p: str) -> list[TranscriptSegment]:
        return [TranscriptSegment(text="plain", start=0.0, duration=1.0)]

    monkeypatch.setattr(prov, "_transcribe", _stub_transcribe, raising=True)
    out = prov._transcribe_with_strategy(path)
    assert len(out) == 1 and out[0]["text"] == "plain"


def test_estimate_and_eta_minutes_branching(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _ffmpeg_true() -> bool:
        return True

    monkeypatch.setattr(prov, "_ffmpeg_available", _ffmpeg_true, raising=True)
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


def test_estimate_eta_minutes_chunk_branch(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(prov, "_ffmpeg_available", lambda: True, raising=True)
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


def test_fetch_success_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, path = _make_provider(tmp_file_size=16)
    prov.enable_chunking = False

    def _probe_ok(video_id: str, url: str) -> int:
        assert video_id == "vid"
        return 10

    monkeypatch.setattr(prov, "_probe_or_error", _probe_ok, raising=True)

    def _download_ok(_: str) -> tuple[str, int]:
        return path, 1024

    def _not_over(_: int) -> bool:
        return False

    monkeypatch.setattr(prov, "_download_or_error", _download_ok, raising=True)
    monkeypatch.setattr(prov, "_is_over_limit", _not_over, raising=True)

    def _strategy(p: str) -> list[TranscriptSegment]:
        return [TranscriptSegment(text="ok", start=0.0, duration=1.0)]

    monkeypatch.setattr(prov, "_transcribe_with_strategy", _strategy, raising=True)

    removed: list[str] = []

    def _remove(p: str) -> None:
        removed.append(os.path.abspath(p))

    prov._owned_tmp_dirs.add(os.path.dirname(os.path.abspath(path)))
    monkeypatch.setattr(os, "remove", _remove, raising=True)

    out = prov.fetch("vid", TranscriptOptions(preferred_langs=["en"]))
    assert len(out) == 1 and out[0]["text"] == "ok"
    assert os.path.abspath(path) in removed


def test_fetch_cleanup_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, path = _make_provider(tmp_file_size=8)
    prov.enable_chunking = False

    def _probe_ok2(_: str, __: str) -> int:
        return 5

    def _download_ok2(_: str) -> tuple[str, int]:
        return path, 1024

    def _not_over2(_: int) -> bool:
        return False

    monkeypatch.setattr(prov, "_probe_or_error", _probe_ok2, raising=True)
    monkeypatch.setattr(prov, "_download_or_error", _download_ok2, raising=True)
    monkeypatch.setattr(prov, "_is_over_limit", _not_over2, raising=True)

    def _raise_strategy(_: str) -> list[TranscriptSegment]:
        raise RuntimeError("x")

    monkeypatch.setattr(prov, "_transcribe_with_strategy", _raise_strategy, raising=True)

    def _remove_raise(p: str) -> None:
        raise OSError(f"rm {p}")

    prov._owned_tmp_dirs.add(os.path.dirname(os.path.abspath(path)))
    monkeypatch.setattr(os, "remove", _remove_raise, raising=True)

    with pytest.raises(OSError):
        _ = prov.fetch("vid", TranscriptOptions(preferred_langs=["en"]))


def test_fetch_over_limit_uses_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    prov, path = _make_provider(tmp_file_size=32)
    prov.enable_chunking = True

    def _probe_ok3(video_id: str, url: str) -> int:
        assert video_id == "vid"
        return 20

    def _download_ok3(_: str) -> tuple[str, int]:
        return path, 10 * 1024 * 1024

    def _always_over(size_bytes: int) -> bool:
        assert size_bytes == 10 * 1024 * 1024
        return True

    called: list[str] = []

    def _handle(path_arg: str, size_arg: int) -> list[TranscriptSegment]:
        called.append(path_arg)
        return [TranscriptSegment(text="big", start=0.0, duration=1.0)]

    monkeypatch.setattr(prov, "_probe_or_error", _probe_ok3, raising=True)
    monkeypatch.setattr(prov, "_download_or_error", _download_ok3, raising=True)
    monkeypatch.setattr(prov, "_is_over_limit", _always_over, raising=True)
    monkeypatch.setattr(prov, "_handle_over_limit", _handle, raising=True)

    out = prov.fetch("vid", TranscriptOptions(preferred_langs=["en"]))
    assert called and path in called
    assert len(out) == 1 and out[0]["text"] == "big"


def test_ffmpeg_available() -> None:
    prov, _ = _make_provider()
    _ = prov._ffmpeg_available()


logger = logging.getLogger(__name__)
