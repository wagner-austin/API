from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import pytest

from transcript_api.chunker import AudioChunker


def _fake_run_ok(*args: list[str], **kwargs: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")


def test_chunker_selects_webm_for_opus(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()

    def _probe(_p: str) -> tuple[str, str]:
        return "matroska,webm", "opus"

    def _silence(_p: str, _d: float) -> list[float]:
        return []

    monkeypatch.setattr(ch, "_probe_stream_info", _probe)
    monkeypatch.setattr(ch, "_detect_silence", _silence)
    monkeypatch.setattr("subprocess.run", _fake_run_ok)
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


def test_chunker_selects_m4a_for_aac(monkeypatch: pytest.MonkeyPatch) -> None:
    ch = AudioChunker()

    def _probe(_p: str) -> tuple[str, str]:
        return "mp4", "aac"

    def _silence(_p: str, _d: float) -> list[float]:
        return []

    monkeypatch.setattr(ch, "_probe_stream_info", _probe)
    monkeypatch.setattr(ch, "_detect_silence", _silence)
    monkeypatch.setattr("subprocess.run", _fake_run_ok)
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
