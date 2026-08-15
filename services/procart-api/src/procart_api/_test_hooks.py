"""Test hooks for procart-api.

Expose injectable module-level hooks that production sets at startup and tests override.
"""

from __future__ import annotations

from typing import Protocol

from procart.ffmpeg_runner import RealFfmpegRunner


class _FfmpegRunnerProto(Protocol):
    def encode_frames_to_video(self, frames_dir: str, fps: int, output_path: str) -> None: ...


# Bound to the real runner at import time so callers invoke it directly.
# Tests rebind it to a fake and restore it afterwards.
FFMPEG_RUNNER: _FfmpegRunnerProto = RealFfmpegRunner()

__all__ = ["FFMPEG_RUNNER"]
