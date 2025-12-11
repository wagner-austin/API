from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import pytest
from platform_core.json_utils import dump_json_str

from transcript_api import _test_hooks
from transcript_api.chunker import AudioChunker, _FfprobeOutputDict


def _touch(path: str, size: int = 0) -> None:
    with open(path, "wb") as f:
        f.write(b"x" * size)


class _ProcRes:
    """Fake subprocess result."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ) -> None:
        self.stdout: bytes | str | None = stdout
        self.stderr: bytes | str | None = stderr
        self.returncode = returncode


def test_chunker_passthrough_when_below_threshold() -> None:
    fd, path = tempfile.mkstemp(prefix="aud_", suffix=".m4a")
    os.close(fd)
    _touch(path, size=1024)
    ch = AudioChunker(target_chunk_mb=10.0, max_chunk_duration_seconds=600.0)
    chunks = ch.chunk_audio(path, total_duration=30.0, estimated_mb=0.5)
    assert len(chunks) == 1 and os.path.abspath(chunks[0]["path"]) == os.path.abspath(path)


def test_detect_silence_parses_ffmpeg_output() -> None:
    """Test parsing of ffmpeg silencedetect output."""
    ch = AudioChunker()
    sample = "\n".join(
        [
            "silence_start: 1.0",
            "silence_end: 2.50 | silence_duration: 1.50",
            "silence_end: 4.00 | silence_duration: 0.50",
        ]
    )

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
        return _ProcRes(stdout=sample, stderr="")

    _test_hooks.subprocess_run = _fake_subprocess
    points = ch._detect_silence("/tmp/a.m4a", 10.0)
    assert points == [2.5, 4.0]


def test_split_audio_copy_then_reencode() -> None:
    """Test that split falls back to reencode when copy fails."""
    ch = AudioChunker()

    fd, in_path = tempfile.mkstemp(prefix="aud_", suffix=".webm")
    os.close(fd)
    _touch(in_path, size=2048)

    calls: list[str] = []

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
        args_str = " ".join(args)
        # Handle ffprobe for codec info
        if "ffprobe" in args_str:
            opus_info = (
                '{"format": {"format_name": "webm"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "opus"}]}'
            )
            return _ProcRes(stdout=opus_info)
        # Handle ffmpeg split operations
        if "-c:a" in args:
            calls.append("reencode")
            out_path = args[-1]
            _touch(out_path, size=512)
            return _ProcRes()
        calls.append("copy")
        if check:
            raise subprocess.CalledProcessError(1, args)
        return _ProcRes(returncode=1)

    _test_hooks.subprocess_run = _fake_subprocess

    created = ch._split_audio(in_path, [1.0], total_duration=2.0)
    assert len(created) == 2
    assert calls and calls[0] == "copy"
    assert calls[1] == "reencode"
    try:
        os.remove(in_path)
    except OSError:
        logging.getLogger(__name__).exception("cleanup failed for %s", in_path)
        raise


def test_probe_stream_info_parses_json() -> None:
    """Test that ffprobe JSON output is parsed correctly."""
    ch = AudioChunker()
    payload = {
        "format": {"format_name": "m4a", "format_long_name": "MPEG-4 AAC"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

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
        return _ProcRes(stdout=dump_json_str(payload), stderr="")

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = ch._probe_stream_info("/tmp/a.m4a")
    assert container == "m4a" and codec == "aac"


def test_safe_size_mb_missing_file() -> None:
    ch = AudioChunker()
    assert ch._safe_size_mb("/path/does/not/exist.xyz") == 0.0


def test_detect_silence_skips_bad_numbers() -> None:
    """Test that bad numbers in silence detection are skipped."""
    ch = AudioChunker()
    sample = "\n".join(
        [
            "silence_end: notanumber | silence_duration: 1.0",
            "silence_end: 2.00 | silence_duration: 0.5",
        ]
    )

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
        return _ProcRes(stdout=sample, stderr="")

    _test_hooks.subprocess_run = _fake_subprocess
    points = ch._detect_silence("/tmp/a.m4a", 10.0)
    assert points == [2.0]


def test_detect_silence_handles_unexpected_token() -> None:
    """Test that unexpected tokens (non-numeric) result in empty points."""
    # This test verifies the ValueError handling path - when a matched group
    # can't be converted to float, the point is skipped
    ch = AudioChunker()
    # The regex will match but the value can't be converted to float
    sample = "silence_end: not_a_number | silence_duration: 1.0"

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
        return _ProcRes(stdout=sample, stderr="")

    _test_hooks.subprocess_run = _fake_subprocess
    points = ch._detect_silence("/tmp/a.m4a", 10.0)
    # "not_a_number" won't match the regex pattern for silence_end timestamps
    # so points should be empty
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
