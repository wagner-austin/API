from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Sequence

import pytest
from platform_core.json_utils import dump_json_str

from transcript_api.chunker import AudioChunker, _FfprobeOutputDict


def _touch(path: str, size: int = 0) -> None:
    with open(path, "wb") as f:
        f.write(b"x" * size)


def test_chunker_passthrough_when_below_threshold() -> None:
    fd, path = tempfile.mkstemp(prefix="aud_", suffix=".m4a")
    os.close(fd)
    _touch(path, size=1024)
    ch = AudioChunker(target_chunk_mb=10.0, max_chunk_duration_seconds=600.0)
    chunks = ch.chunk_audio(path, total_duration=30.0, estimated_mb=0.5)
    assert len(chunks) == 1 and os.path.abspath(chunks[0]["path"]) == os.path.abspath(path)


def test_detect_silence_parses_ffmpeg_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ch = AudioChunker()
    sample = "\n".join(
        [
            "silence_start: 1.0",
            "silence_end: 2.50 | silence_duration: 1.50",
            "silence_end: 4.00 | silence_duration: 0.50",
        ]
    )
    import subprocess

    def _run_silence(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> _ProcRes:
        return _ProcRes(stdout=sample, stderr="")

    monkeypatch.setattr(subprocess, "run", _run_silence, raising=True)
    points = ch._detect_silence("/tmp/a.m4a", 10.0)
    assert points == [2.5, 4.0]


def test_split_audio_copy_then_reencode(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()

    def _probe(_p: str) -> tuple[str, str]:
        return "webm", "opus"

    monkeypatch.setattr(ch, "_probe_stream_info", _probe)

    fd, in_path = tempfile.mkstemp(prefix="aud_", suffix=".webm")
    os.close(fd)
    _touch(in_path, size=2048)

    calls: list[str] = []

    def fake_run(
        cmd: Sequence[str] | str,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
    ) -> _ProcRes:
        nonlocal calls
        if "-c:a" in cmd:
            calls.append("reencode")
        else:
            calls.append("copy")
            import subprocess

            raise subprocess.CalledProcessError(1, cmd)
        out_path = cmd[-1]
        _touch(out_path, size=512)
        return _ProcRes(stdout="", stderr="")

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run, raising=True)

    created = ch._split_audio(in_path, [1.0], total_duration=2.0)
    assert len(created) == 2
    assert calls and calls[0] == "copy"
    assert "reencode" in calls
    try:
        os.remove(in_path)
    except OSError:
        logging.getLogger(__name__).exception("cleanup failed for %s", in_path)
        raise


def test_probe_stream_info_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()
    payload = {
        "format": {"format_name": "m4a", "format_long_name": "MPEG-4 AAC"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    import subprocess

    def _run_probe(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> _ProcRes:
        return _ProcRes(stdout=dump_json_str(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", _run_probe, raising=True)
    container, codec = ch._probe_stream_info("/tmp/a.m4a")
    assert container == "m4a" and codec == "aac"


def test_safe_size_mb_missing_file() -> None:
    ch = AudioChunker()
    assert ch._safe_size_mb("/path/does/not/exist.xyz") == 0.0


def test_detect_silence_skips_bad_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()
    sample = "\n".join(
        [
            "silence_end: notanumber | silence_duration: 1.0",
            "silence_end: 2.00 | silence_duration: 0.5",
        ]
    )
    import subprocess

    def _run_bad(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> _ProcRes:
        return _ProcRes(stdout=sample, stderr="")

    monkeypatch.setattr(subprocess, "run", _run_bad, raising=True)
    points = ch._detect_silence("/tmp/a.m4a", 10.0)
    assert points == [2.0]


def test_detect_silence_handles_unexpected_token(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()
    sample = "silence_end: bad_token | silence_duration: 1.0"

    import subprocess

    import transcript_api.chunker as ch_mod

    def _run_bad(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> _ProcRes:
        return _ProcRes(stdout=sample, stderr="")

    monkeypatch.setattr(subprocess, "run", _run_bad, raising=True)
    monkeypatch.setattr(
        ch_mod, "_SILENCE_END_RE", re.compile(r"silence_end:\s*(?P<ts>bad_token)"), raising=True
    )

    points = ch._detect_silence("/tmp/a.m4a", 10.0)
    assert points == []


def test_calculate_split_points_variants(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    ch = AudioChunker(target_chunk_mb=20.0, max_chunk_duration_seconds=10.0)
    empty = ch._calculate_split_points([], total_duration=5.0, estimated_mb=1.0)
    assert empty == []
    pts = ch._calculate_split_points([], total_duration=25.0, estimated_mb=25.0)
    assert pts and all(0 < p < 25.0 for p in pts)
    pts2 = ch._calculate_split_points([8.0, 16.6], total_duration=25.0, estimated_mb=25.0)
    assert any(abs(p - 16.6) < 1e-6 for p in pts2)


def test_cleanup_dir_invalid_path_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    ch = AudioChunker()
    ch._cleanup_dir("")
    messages = [record.getMessage() for record in caplog.records]
    assert any("Invalid directory for cleanup" in msg for msg in messages)


def test_ffprobe_extract_helpers_cover_empty_and_missing() -> None:
    raw_empty: _FfprobeOutputDict = {"format": {"format_name": "webm"}, "streams": []}
    assert AudioChunker._extract_container_format(raw_empty) == "webm"
    assert AudioChunker._extract_audio_codec(raw_empty) == ""

    raw_missing_format: _FfprobeOutputDict = {"streams": []}
    assert AudioChunker._extract_container_format(raw_missing_format) == ""

    raw_format_int: _FfprobeOutputDict = {"format": {"format_name": 123}, "streams": []}
    assert AudioChunker._extract_container_format(raw_format_int) == ""

    raw_format_str: _FfprobeOutputDict = {"format": "bad", "streams": []}
    assert AudioChunker._extract_container_format(raw_format_str) == ""

    raw_video_only: _FfprobeOutputDict = {
        "format": {"format_name": ""},
        "streams": [{"codec_type": "video", "codec_name": "h264"}],
    }
    assert AudioChunker._extract_audio_codec(raw_video_only) == ""

    raw_streams_as_str: _FfprobeOutputDict = {"format": {"format_name": "mp4"}, "streams": "bad"}
    assert AudioChunker._extract_audio_codec(raw_streams_as_str) == ""

    raw_streams: _FfprobeOutputDict = {
        "format": {"format_name": ""},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }
    assert AudioChunker._extract_audio_codec(raw_streams) == "aac"

    raw_audio_non_str: _FfprobeOutputDict = {
        "format": {"format_name": "mp4"},
        "streams": [{"codec_type": "audio", "codec_name": 5}],
    }
    assert AudioChunker._extract_audio_codec(raw_audio_non_str) == ""


def test_load_ffprobe_json_validation_paths() -> None:
    assert AudioChunker._load_ffprobe_json("[]") is None

    out_default = AudioChunker._load_ffprobe_json('{"format": "bad", "streams": "bad"}')
    if out_default is None:
        pytest.fail("expected ffprobe output")
    fmt_default = out_default.get("format")
    if type(fmt_default) is dict:
        assert fmt_default.get("format_name") == ""
    streams_default = out_default.get("streams")
    assert streams_default == [] or streams_default == "bad"

    payload: _FfprobeOutputDict = {
        "format": {"format_name": "mp4"},
        "streams": [
            {"codec_type": "audio", "codec_name": "aac"},
            {"codec_type": "video", "codec_name": "h264"},
        ],
    }
    out_valid = AudioChunker._load_ffprobe_json(dump_json_str(payload))
    if out_valid is None:
        pytest.fail("expected ffprobe output")
    fmt_valid = out_valid.get("format")
    assert type(fmt_valid) is dict and fmt_valid.get("format_name") == "mp4"
    assert out_valid["streams"] == [
        {"codec_type": "audio", "codec_name": "aac"},
        {"codec_type": "video", "codec_name": "h264"},
    ]

    mixed_streams = {"format": {"format_name": "mp4"}, "streams": ["nope"]}
    out_mixed = AudioChunker._load_ffprobe_json(dump_json_str(mixed_streams))
    if out_mixed is None:
        pytest.fail("expected ffprobe output")
    assert out_mixed["streams"] == []


logger = logging.getLogger(__name__)


class _ProcRes:
    def __init__(self, stdout: str, stderr: str) -> None:
        self.stdout = stdout
        self.stderr = stderr
