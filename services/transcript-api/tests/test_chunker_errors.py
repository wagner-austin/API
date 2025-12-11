from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import pytest

from transcript_api import _test_hooks
from transcript_api.chunker import AudioChunker


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


def test_detect_silence_timeout_returns_empty() -> None:
    """Test that silence detection timeout returns empty list."""
    ch = AudioChunker()

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
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=90)

    _test_hooks.subprocess_run = _raise_timeout
    assert ch._detect_silence("/tmp/a.m4a", 10.0) == []


def test_split_audio_copy_timeout_raises() -> None:
    """Test that split audio copy timeout raises."""
    ch = AudioChunker()

    fd, in_path = tempfile.mkstemp(prefix="aud_", suffix=".m4a")
    os.close(fd)

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
                '{"format": {"format_name": "m4a"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "aac"}]}'
            )
            return _FakeProc(stdout=aac_info)
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=90)

    _test_hooks.subprocess_run = _fake_subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        _ = ch._split_audio(in_path, [1.0], total_duration=2.0)


def test_split_audio_reencode_timeout_raises() -> None:
    """Test that split audio reencode timeout raises."""
    ch = AudioChunker()

    fd, in_path = tempfile.mkstemp(prefix="aud_", suffix=".m4a")
    os.close(fd)

    calls = {"n": 0}

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
                '{"format": {"format_name": "m4a"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "aac"}]}'
            )
            return _FakeProc(stdout=aac_info)
        if "-c:a" in args:
            calls["n"] += 1
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)
        calls["n"] += 1
        if check:
            raise subprocess.CalledProcessError(1, args)
        return _FakeProc(returncode=1)

    _test_hooks.subprocess_run = _fake_subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        _ = ch._split_audio(in_path, [1.0], total_duration=2.0)
    assert calls["n"] >= 2


def test_probe_stream_info_timeout_returns_empty() -> None:
    """Test that probe timeout returns empty strings."""
    ch = AudioChunker()

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
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    _test_hooks.subprocess_run = _raise_timeout
    container, codec = ch._probe_stream_info("/tmp/a.m4a")
    assert container == "" and codec == ""


def test_probe_stream_info_bad_json_returns_empty() -> None:
    """Test that bad JSON from probe returns empty strings."""
    ch = AudioChunker()

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
        return _FakeProc(stdout="not json", stderr="")

    _test_hooks.subprocess_run = _fake_subprocess
    container, codec = ch._probe_stream_info("/tmp/a.m4a")
    assert container == "" and codec == ""


logger = logging.getLogger(__name__)
