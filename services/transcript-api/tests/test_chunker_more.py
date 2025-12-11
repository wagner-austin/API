from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from platform_core.json_utils import dump_json_str

from transcript_api import _test_hooks
from transcript_api.chunker import AudioChunker, _FfprobeOutputDict


class _FakeProc:
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


def _make_tmp_audio(tmp_path: Path) -> str:
    p = tmp_path / "audio.m4a"
    p.write_bytes(b"data")
    return str(p)


def test_calculate_split_points_ideal_empty_returns_empty() -> None:
    c = AudioChunker(target_chunk_mb=20.0)
    out = c._calculate_split_points([], total_duration=30.0, estimated_mb=5.0)
    assert out == []


def test_split_audio_clamps_points_and_copy_path(tmp_path: Path) -> None:
    """Test that split audio clamps points and creates files."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

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
        if "ffprobe" in args_str:
            aac_info = (
                '{"format": {"format_name": "mp4"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "aac"}]}'
            )
            return _FakeProc(stdout=aac_info)
        # ffmpeg split - create output file
        out_path = args[-1]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return _FakeProc()

    _test_hooks.subprocess_run = _fake_subprocess
    created = c._split_audio(audio, split_points=[-5.0, 9999.0], total_duration=3.0)
    assert created and all(0.0 <= seg["start_seconds"] <= 3.0 for seg in created)


def test_cleanup_dir_nonexistent_no_raise(tmp_path: Path) -> None:
    c = AudioChunker()
    missing = str(tmp_path / "does-not-exist")
    c._cleanup_dir(missing)


def test_probe_stream_info_timeout_returns_empty_more(tmp_path: Path) -> None:
    """Test that probe timeout returns empty strings."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    def _raise_timeout(
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
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    _test_hooks.subprocess_run = _raise_timeout
    container, codec = c._probe_stream_info(audio)
    assert container == "" and codec == ""


def test_detect_silence_ignores_bad_timestamp() -> None:
    """Test that bad timestamps in silence detection are skipped."""
    c = AudioChunker()

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
        return _FakeProc(
            stdout="",
            stderr="silence_end: not_a_number\nmore\nsilence_end: 1.0\n",
        )

    _test_hooks.subprocess_run = _fake_subprocess
    out = c._detect_silence("/tmp/a.m4a", duration=10.0)
    assert out == [1.0]


def test_split_audio_prefers_webm_for_opus_and_handles_no_splits(tmp_path: Path) -> None:
    """Test that opus uses webm and empty splits returns original."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

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
        if "ffprobe" in args_str:
            opus_info = (
                '{"format": {"format_name": "matroska,webm"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "opus"}]}'
            )
            return _FakeProc(stdout=opus_info)
        out_path = args[-1]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return _FakeProc()

    _test_hooks.subprocess_run = _fake_subprocess
    res = c._split_audio(audio, split_points=[], total_duration=5.0)
    assert len(res) == 1 and res[0]["path"] == audio


def test_calculate_split_points_logs_no_nearby_silence() -> None:
    c = AudioChunker(target_chunk_mb=1.0)
    out = c._calculate_split_points(
        silence_points=[1000.0], total_duration=100.0, estimated_mb=50.0
    )
    assert out


def test_probe_stream_info_non_dict_streams_skip(tmp_path: Path) -> None:
    """Test probing with valid streams."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    payload: _FfprobeOutputDict = {
        "format": {"format_name": "mp4"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    body = dump_json_str(payload)

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
        return _FakeProc(stdout=body)

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == "aac"


def test_probe_stream_info_streams_not_list(tmp_path: Path) -> None:
    """Test probing when streams is not a list."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    payload: dict[str, dict[str, str]] = {"format": {"format_name": "mp4"}, "streams": {"k": "v"}}
    body = dump_json_str(payload)

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
        return _FakeProc(stdout=body)

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == ""


def test_probe_stream_info_streams_empty(tmp_path: Path) -> None:
    """Test probing with empty streams list."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    payload: _FfprobeOutputDict = {"format": {"format_name": "mp4"}, "streams": []}
    body = dump_json_str(payload)

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
        return _FakeProc(stdout=body)

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == ""


def test_probe_stream_info_root_not_dict(tmp_path: Path) -> None:
    """Test probing when root is not a dict."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    items: list[str] = ["not-a-dict"]
    body = dump_json_str(items)

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
        return _FakeProc(stdout=body)

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = c._probe_stream_info(audio)
    assert container == "" and codec == ""


def test_probe_stream_info_format_name_not_str(tmp_path: Path) -> None:
    """Test probing when format_name is not a string."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    payload: dict[str, dict[str, int] | list[dict[str, str]]] = {
        "format": {"format_name": 123},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }
    body = dump_json_str(payload)

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
        return _FakeProc(stdout=body)

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = c._probe_stream_info(audio)
    assert container == "" and codec == "aac"


def test_probe_stream_info_codec_name_not_str(tmp_path: Path) -> None:
    """Test probing when codec_name is not a string."""
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    payload: dict[str, dict[str, str] | list[dict[str, str | int]]] = {
        "format": {"format_name": "mp4"},
        "streams": [{"codec_type": "audio", "codec_name": 123}],
    }
    body = dump_json_str(payload)

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
        return _FakeProc(stdout=body)

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == ""


logger = logging.getLogger(__name__)
