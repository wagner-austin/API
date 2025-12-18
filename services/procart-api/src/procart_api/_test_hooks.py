"""Test hooks for procart-api.

Expose injectable module-level hooks that production sets at startup and tests override.
"""

from __future__ import annotations

from typing import Protocol


class _FfmpegRunnerProto(Protocol):
    def encode_frames_to_video(self, frames_dir: str, fps: int, output_path: str) -> None: ...


# Default to None; main.py sets real instance. Tests inject fakes.
FFMPEG_RUNNER: _FfmpegRunnerProto | None = None

__all__ = ["FFMPEG_RUNNER"]
