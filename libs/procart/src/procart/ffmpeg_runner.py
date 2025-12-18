from __future__ import annotations

import logging
import subprocess
from typing import Protocol

from .types import Fps

_logger = logging.getLogger(__name__)


class FfmpegRunner(Protocol):
    """Protocol for encoding a sequence of frames into a video file.

    Args:
        frames_dir: Directory containing PNG frames named as frame_%06d.png.
        fps: Frames per second for the output video.
        output_path: Target video path (e.g., an .mp4 file).

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If ffmpeg invocation fails.
    """

    def encode_frames_to_video(self, frames_dir: str, fps: Fps, output_path: str) -> None: ...


def build_ffmpeg_args(frames_dir: str, fps: Fps, output_path: str) -> list[str]:
    """Build ffmpeg command-line arguments for encoding frames to MP4.

    Args:
        frames_dir: Directory containing frame_%06d.png files.
        fps: Output frames per second.
        output_path: Output video file path.

    Returns:
        list[str]: The ffmpeg CLI argument vector.

    Raises:
        ValueError: If fps is not positive or paths are empty.
    """
    if int(fps) <= 0:
        raise ValueError("fps must be positive")
    if not frames_dir:
        raise ValueError("frames_dir must be non-empty")
    if not output_path:
        raise ValueError("output_path must be non-empty")
    pattern = f"{frames_dir}\\frame_%06d.png"
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]


class RealFfmpegRunner:
    """Real ffmpeg runner that shells out to the ffmpeg binary.

    Uses subprocess.run with check=True. Errors propagate.
    """

    def encode_frames_to_video(self, frames_dir: str, fps: Fps, output_path: str) -> None:
        _logger.info("Encoding video: %s -> %s @ %d fps", frames_dir, output_path, fps)
        args = build_ffmpeg_args(frames_dir, fps, output_path)
        self._run(args)
        _logger.info("Video encoding complete: %s", output_path)

    def _run(self, args: list[str]) -> None:
        subprocess.run(args, check=True)


__all__ = ["FfmpegRunner", "RealFfmpegRunner", "build_ffmpeg_args"]
