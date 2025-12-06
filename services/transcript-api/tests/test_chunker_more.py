from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str

from transcript_api.chunker import AudioChunker, _FfprobeOutputDict


def _make_tmp_audio(tmp_path: Path) -> str:
    p = tmp_path / "audio.m4a"
    p.write_bytes(b"data")
    return str(p)


def test_calculate_split_points_ideal_empty_returns_empty() -> None:
    c = AudioChunker(target_chunk_mb=20.0)
    out = c._calculate_split_points([], total_duration=30.0, estimated_mb=5.0)
    assert out == []


def test_split_audio_clamps_points_and_copy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    def _probe(_p: str) -> tuple[str, str]:
        return "mp4", "aac"

    monkeypatch.setattr(c, "_probe_stream_info", _probe)

    def _fake_run(
        cmd: list[str],
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        out_path = cmd[-1]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    created = c._split_audio(audio, split_points=[-5.0, 9999.0], total_duration=3.0)
    assert created and all(0.0 <= seg["start_seconds"] <= 3.0 for seg in created)


def test_cleanup_dir_nonexistent_no_raise(tmp_path: Path) -> None:
    c = AudioChunker()
    missing = str(tmp_path / "does-not-exist")
    c._cleanup_dir(missing)


def test_probe_stream_info_timeout_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    def _boom(
        cmd: Sequence[str], capture_output: bool, text: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(subprocess, "run", _boom)
    container, codec = c._probe_stream_info(audio)
    assert container == "" and codec == ""


def test_detect_silence_ignores_bad_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    c = AudioChunker()

    class _R:
        def __init__(self) -> None:
            self.stdout = ""
            self.stderr = "silence_end: not_a_number\nmore\nsilence_end: 1.0\n"
            self.returncode = 0

    def _fake_run(cmd: list[str], capture_output: bool, text: bool, timeout: int) -> _R:
        return _R()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    out = c._detect_silence("/tmp/a.m4a", duration=10.0)
    assert out == [1.0]


def test_split_audio_prefers_webm_for_opus_and_handles_no_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    def _probe(_p: str) -> tuple[str, str]:
        return "matroska,webm", "opus"

    monkeypatch.setattr(c, "_probe_stream_info", _probe)

    def _fake_run_copy(
        cmd: list[str], **kwargs: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        out_path = cmd[-1]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run_copy)
    res = c._split_audio(audio, split_points=[], total_duration=5.0)
    assert len(res) == 1 and res[0]["path"] == audio


def test_calculate_split_points_logs_no_nearby_silence() -> None:
    c = AudioChunker(target_chunk_mb=1.0)
    out = c._calculate_split_points(
        silence_points=[1000.0], total_duration=100.0, estimated_mb=50.0
    )
    assert out


def test_probe_stream_info_non_dict_streams_skip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    class _PR:
        def __init__(self, s: str) -> None:
            self.stdout = s
            self.stderr = ""

    payload: _FfprobeOutputDict = {
        "format": {"format_name": "mp4"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    body = dump_json_str(payload)

    def _run_probe(cmd: Sequence[str], capture_output: bool, text: bool, timeout: float) -> _PR:
        return _PR(body)

    monkeypatch.setattr(subprocess, "run", _run_probe)
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == "aac"


def test_probe_stream_info_streams_not_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    class _PR:
        def __init__(self, s: str) -> None:
            self.stdout = s
            self.stderr = ""

    payload: dict[str, dict[str, str]] = {"format": {"format_name": "mp4"}, "streams": {"k": "v"}}
    body = dump_json_str(payload)

    def _run_probe(cmd: Sequence[str], capture_output: bool, text: bool, timeout: float) -> _PR:
        return _PR(body)

    monkeypatch.setattr(subprocess, "run", _run_probe)
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == ""


def test_probe_stream_info_streams_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    class _PR:
        def __init__(self, s: str) -> None:
            self.stdout = s
            self.stderr = ""

    payload: _FfprobeOutputDict = {"format": {"format_name": "mp4"}, "streams": []}
    body = dump_json_str(payload)

    def _run_probe(cmd: Sequence[str], capture_output: bool, text: bool, timeout: float) -> _PR:
        return _PR(body)

    monkeypatch.setattr(subprocess, "run", _run_probe)
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == ""


def test_probe_stream_info_root_not_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    class _PR:
        def __init__(self, s: str) -> None:
            self.stdout = s
            self.stderr = ""

    items: list[str] = ["not-a-dict"]
    body = dump_json_str(items)

    def _run_probe(cmd: Sequence[str], capture_output: bool, text: bool, timeout: float) -> _PR:
        return _PR(body)

    monkeypatch.setattr(subprocess, "run", _run_probe)
    container, codec = c._probe_stream_info(audio)
    assert container == "" and codec == ""


def test_probe_stream_info_format_name_not_str(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    class _PR:
        def __init__(self, s: str) -> None:
            self.stdout = s
            self.stderr = ""

    payload: dict[str, dict[str, int] | list[dict[str, str]]] = {
        "format": {"format_name": 123},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }
    body = dump_json_str(payload)

    def _run_probe(cmd: Sequence[str], capture_output: bool, text: bool, timeout: float) -> _PR:
        return _PR(body)

    monkeypatch.setattr(subprocess, "run", _run_probe)
    container, codec = c._probe_stream_info(audio)
    assert container == "" and codec == "aac"


def test_probe_stream_info_codec_name_not_str(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c = AudioChunker()
    audio = _make_tmp_audio(tmp_path)

    class _PR:
        def __init__(self, s: str) -> None:
            self.stdout = s
            self.stderr = ""

    payload: dict[str, dict[str, str] | list[dict[str, str | int]]] = {
        "format": {"format_name": "mp4"},
        "streams": [{"codec_type": "audio", "codec_name": 123}],
    }
    body = dump_json_str(payload)

    def _run_probe(cmd: Sequence[str], capture_output: bool, text: bool, timeout: float) -> _PR:
        return _PR(body)

    monkeypatch.setattr(subprocess, "run", _run_probe)
    container, codec = c._probe_stream_info(audio)
    assert container == "mp4" and codec == ""


logger = logging.getLogger(__name__)
