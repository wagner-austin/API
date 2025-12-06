from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence

import pytest

from transcript_api.chunker import AudioChunker


def test_detect_silence_timeout_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()
    import subprocess

    def raise_timeout(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> None:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=90)

    monkeypatch.setattr(subprocess, "run", raise_timeout, raising=True)
    assert ch._detect_silence("/tmp/a.m4a", 10.0) == []


def test_split_audio_copy_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()

    def _probe(_p: str) -> tuple[str, str]:
        return "m4a", "aac"

    monkeypatch.setattr(ch, "_probe_stream_info", _probe, raising=True)

    fd, in_path = tempfile.mkstemp(prefix="aud_", suffix=".m4a")
    os.close(fd)

    import subprocess

    def raise_timeout(
        cmd: Sequence[str] | str,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
    ) -> None:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=90)

    monkeypatch.setattr(subprocess, "run", raise_timeout, raising=True)

    with pytest.raises(subprocess.TimeoutExpired):
        _ = ch._split_audio(in_path, [1.0], total_duration=2.0)


def test_split_audio_reencode_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()

    def _probe(_p: str) -> tuple[str, str]:
        return "m4a", "aac"

    monkeypatch.setattr(ch, "_probe_stream_info", _probe, raising=True)

    fd, in_path = tempfile.mkstemp(prefix="aud_", suffix=".m4a")
    os.close(fd)

    calls = {"n": 0}
    import subprocess

    def run_with_fallback(
        cmd: Sequence[str] | str,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
    ) -> None:
        if "-c:a" in cmd:
            calls["n"] += 1
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)
        calls["n"] += 1
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", run_with_fallback, raising=True)

    with pytest.raises(subprocess.TimeoutExpired):
        _ = ch._split_audio(in_path, [1.0], total_duration=2.0)
    assert calls["n"] >= 2


def test_probe_stream_info_timeout_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()
    import subprocess

    def raise_timeout(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> None:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    monkeypatch.setattr(subprocess, "run", raise_timeout, raising=True)
    container, codec = ch._probe_stream_info("/tmp/a.m4a")
    assert container == "" and codec == ""


def test_probe_stream_info_bad_json_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()
    import subprocess

    class _Res:
        def __init__(self) -> None:
            self.stdout = "not json"
            self.stderr = ""

    def bad_json(
        cmd: Sequence[str],
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> _Res:
        return _Res()

    monkeypatch.setattr(subprocess, "run", bad_json, raising=True)
    container, codec = ch._probe_stream_info("/tmp/a.m4a")
    assert container == "" and codec == ""


logger = logging.getLogger(__name__)
