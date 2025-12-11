from __future__ import annotations

import logging
import os
import tempfile

from transcript_api import _test_hooks
from transcript_api.chunker import AudioChunker


class _FakeProc:
    """Fake subprocess result for hook-based testing."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout: bytes | str | None = stdout
        self.stderr: bytes | str | None = stderr


def test_chunker_selects_webm_for_opus() -> None:
    """Test that opus audio uses webm container for chunks."""
    ch = AudioChunker()

    # Create fake subprocess that returns opus codec info for ffprobe
    # and succeeds for ffmpeg split operations
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
            # Return opus codec info
            opus_info = (
                '{"format": {"format_name": "matroska,webm"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "opus"}]}'
            )
            return _FakeProc(stdout=opus_info)
        if "silencedetect" in args_str:
            # No silence points
            return _FakeProc(stdout="", stderr="")
        # ffmpeg split command - just succeed
        return _FakeProc()

    _test_hooks.subprocess_run = _fake_subprocess

    fd, audio_path = tempfile.mkstemp(prefix="chunker_in_", suffix=".webm")
    os.close(fd)
    try:
        chunks = ch.chunk_audio(audio_path, total_duration=60.0, estimated_mb=100.0)
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            logging.getLogger(__name__).exception("cleanup failed for %s", audio_path)
            raise
    assert len(chunks) == 5
    assert all(c["path"].endswith(".webm") for c in chunks)


def test_chunker_selects_m4a_for_aac() -> None:
    """Test that aac audio uses m4a container for chunks."""
    ch = AudioChunker()

    # Create fake subprocess that returns aac codec info for ffprobe
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
            # Return aac codec info
            aac_info = (
                '{"format": {"format_name": "mp4"}, '
                '"streams": [{"codec_type": "audio", "codec_name": "aac"}]}'
            )
            return _FakeProc(stdout=aac_info)
        if "silencedetect" in args_str:
            # No silence points
            return _FakeProc(stdout="", stderr="")
        # ffmpeg split command - just succeed
        return _FakeProc()

    _test_hooks.subprocess_run = _fake_subprocess

    fd, audio_path = tempfile.mkstemp(prefix="chunker_in_", suffix=".m4a")
    os.close(fd)
    try:
        chunks = ch.chunk_audio(audio_path, total_duration=60.0, estimated_mb=60.0)
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            logging.getLogger(__name__).exception("cleanup failed for %s", audio_path)
            raise
    assert len(chunks) == 3
    assert all(c["path"].endswith(".m4a") for c in chunks)
