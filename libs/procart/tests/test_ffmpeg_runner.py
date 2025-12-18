from __future__ import annotations

import os

import pytest

from procart.ffmpeg_runner import RealFfmpegRunner, build_ffmpeg_args


def test_build_ffmpeg_args_cmd_vector_and_runner_invocation() -> None:
    args = build_ffmpeg_args("C:/frames", 24, "C:/out.mp4")
    assert args[0] == "ffmpeg" and "-framerate" in args and "-i" in args

    # Exercise RealFfmpegRunner._run with harmless command (echo), without try/except
    r = RealFfmpegRunner()
    # Replace the command with a safe built-in to avoid external dependency
    echo_cmd = ["cmd", "/c", "echo", "hello"] if os.name == "nt" else ["echo", "hello"]
    r._run(echo_cmd)


class _SpyRunner(RealFfmpegRunner):
    def __init__(self) -> None:
        self.args: list[str] | None = None

    def _run(self, args: list[str]) -> None:
        self.args = args


def test_encode_frames_to_video_builds_args_no_exec() -> None:
    spy = _SpyRunner()
    spy.encode_frames_to_video("C:/frames", 24, "C:/out.mp4")
    args = spy.args or []
    assert args and args[0] == "ffmpeg"


@pytest.mark.parametrize(
    "bad_fps, frames_dir, output",
    [
        (0, "C:/frames", "C:/out.mp4"),
        (24, "", "C:/out.mp4"),
        (24, "C:/frames", ""),
    ],
)
def test_build_ffmpeg_args_invalid_inputs_raise(bad_fps: int, frames_dir: str, output: str) -> None:
    with pytest.raises(ValueError):
        build_ffmpeg_args(frames_dir, bad_fps, output)
