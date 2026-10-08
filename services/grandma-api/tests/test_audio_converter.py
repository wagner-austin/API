"""Tests for grandma_api.core.audio module."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from grandma_api.core.audio import _test_hooks as audio_hooks
from grandma_api.core.audio.converter import _default_convert_to_wav, conversion_prefix

from .conftest import generate_test_wav


def test_convert_to_wav_with_real_wav() -> None:
    """Test converting a real WAV file with ffmpeg."""
    wav_bytes = generate_test_wav()
    result = _default_convert_to_wav(wav_bytes, "test.wav")

    # Result should be valid WAV
    assert result[:4] == b"RIFF"
    assert result[8:12] == b"WAVE"
    assert _conversion_dirs() == frozenset()


def test_convert_to_wav_raises_when_ffmpeg_refuses_and_leaves_nothing() -> None:
    """Bytes that are not audio are ffmpeg's nonzero exit, raised, and the
    conversion's working files are gone afterwards, as after a success."""
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _default_convert_to_wav(b"this is not audio", "upload.webm")
    assert caught.value.returncode != 0
    assert _conversion_dirs() == frozenset()


def _conversion_dirs() -> frozenset[str]:
    """Name this process's conversion directories left in the temp directory.

    Returns:
        The names that start with this process's conversion prefix.
    """
    prefix = conversion_prefix()
    return frozenset(
        entry.name
        for entry in Path(tempfile.gettempdir()).iterdir()
        if entry.name.startswith(prefix)
    )


def test_audio_hooks_reset() -> None:
    """Test that audio hooks can be reset to defaults."""
    original = audio_hooks.convert_to_wav

    # Replace with a different function
    def fake_converter(audio_bytes: bytes, source_filename: str) -> bytes:
        return b"fake"

    audio_hooks.convert_to_wav = fake_converter
    assert audio_hooks.convert_to_wav is fake_converter

    # Reset should restore original
    audio_hooks.reset_hooks()
    assert audio_hooks.convert_to_wav is _default_convert_to_wav

    # Restore for other tests
    audio_hooks.convert_to_wav = original
