"""Audio conversion utilities for grandma-api.

Provides audio format conversion using ffmpeg with dependency injection via _test_hooks.

Usage in production:
    from grandma_api.core.audio import convert_to_wav
    wav_bytes = convert_to_wav(webm_bytes, "audio.webm")

Usage in tests:
    from grandma_api.core.audio import _test_hooks
    _test_hooks.convert_to_wav = fake_converter
"""

from __future__ import annotations

from grandma_api.core.audio.converter import (
    DEFAULT_SAMPLE_RATE,
    AudioConverterProtocol,
    _default_convert_to_wav,
)

# Module-level converter - tests replace via _test_hooks
convert_to_wav: AudioConverterProtocol = _default_convert_to_wav


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "AudioConverterProtocol",
    "convert_to_wav",
]
