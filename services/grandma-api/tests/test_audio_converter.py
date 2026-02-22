"""Tests for grandma_api.core.audio module."""

from __future__ import annotations

from grandma_api.core.audio import _test_hooks as audio_hooks
from grandma_api.core.audio.converter import _default_convert_to_wav

from .conftest import generate_test_wav


def test_convert_to_wav_with_real_wav() -> None:
    """Test converting a real WAV file with ffmpeg."""
    wav_bytes = generate_test_wav()
    result = _default_convert_to_wav(wav_bytes, "test.wav")

    # Result should be valid WAV
    assert result[:4] == b"RIFF"
    assert result[8:12] == b"WAVE"


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
