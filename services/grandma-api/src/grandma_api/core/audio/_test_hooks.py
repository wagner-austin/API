"""Internal test hooks for audio conversion - allows injecting test dependencies.

This module provides dependency injection for audio conversion.
Production code uses the default ffmpeg-based converter;
tests can replace with fakes.

Usage in tests:
    from grandma_api.core.audio import _test_hooks
    _test_hooks.convert_to_wav = lambda audio, filename: b"RIFF..."
"""

from __future__ import annotations

from grandma_api.core.audio.converter import (
    AudioConverterProtocol,
    _default_convert_to_wav,
)

# Hook for audio converter. Tests can replace to skip ffmpeg.
convert_to_wav: AudioConverterProtocol = _default_convert_to_wav


def reset_hooks() -> None:
    """Reset all hooks to their production defaults.

    Call this in test teardown to ensure clean state.
    """
    global convert_to_wav
    convert_to_wav = _default_convert_to_wav


__all__ = [
    "convert_to_wav",
    "reset_hooks",
]
