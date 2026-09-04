"""Measure a live-view MJPEG stream and say what its numbers support.

Exists because "the video looks laggy" was answered four times by
argument in one session and three of those answers were wrong. Rate
alone cannot tell a slow stream from a stuttering one, and the average
hides the difference: 30 frames in a tenth of a second followed by a
second of nothing reports the same "3 fps" as a steady three.

Point it at a stream and it reports rate, duplicate share, the gap
distribution, stalls, and how many gaps sit at the sampler's own
interval -- the last being the one that catches "we are measuring our
own capture rate, not the source", which is what 12 Hz sampling was
doing to a game that animates faster.

Usage::

    tankpit-stream-probe                                  # the demo slot, 30 s
    tankpit-stream-probe --url http://127.0.0.1:27100/video
    tankpit-stream-probe --seconds 60 --fps 30

``--fps`` is the sender's configured capture rate
(``TANKPIT_BOT_VIDEO_FPS``); it only feeds the sampling-floor count and
may be omitted, which reports zero rather than guessing.

Exit code is 0 whatever the measurement says. This is a report, never a
gate: a stream is allowed to be idle because the game is idle, and a
tool that failed on that would be asserting something it cannot know.
"""

from __future__ import annotations

import sys

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.diagnostics.mjpeg_reader import (
    boundary_from_content_type,
    frames_from_buffer,
)
from tankpit_bot.diagnostics.stream_quality import render_report, summarize_stream
from tankpit_bot.service.fleet_config import FLEET_PORT_DEFAULT

DEFAULT_URL = f"http://127.0.0.1:{FLEET_PORT_DEFAULT}/demo/video/demo-1"
DEFAULT_SECONDS = 30.0

_USAGE = (
    "usage: tankpit-stream-probe [--url URL] [--seconds N] [--fps N]\n"
    "\n"
    f"  --url      stream to measure (default: {DEFAULT_URL})\n"
    f"  --seconds  observation window (default: {DEFAULT_SECONDS:.0f})\n"
    "  --fps      the sender's configured capture rate, for the\n"
    "             sampling-floor count; omitted means do not guess\n"
)


def parse_args(argv: list[str]) -> tuple[str, float, float]:
    """Parse the command line.

    Args:
        argv: Arguments after the program name.

    Returns:
        The URL, the window in seconds, and the declared sender fps
        (``0.0`` when not given).

    Raises:
        ValueError: On an unknown flag, a flag missing its value, or a
            non-numeric or non-positive number. Refused rather than
            defaulted, because a probe that silently measured a
            different window than asked for would produce numbers
            nobody could reproduce.
    """
    url = DEFAULT_URL
    seconds = DEFAULT_SECONDS
    fps = 0.0
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag not in ("--url", "--seconds", "--fps"):
            raise ValueError(f"unknown argument {flag!r}\n\n{_USAGE}")
        if index + 1 >= len(argv):
            raise ValueError(f"{flag} needs a value\n\n{_USAGE}")
        value = argv[index + 1]
        if flag == "--url":
            url = value
        else:
            number = float(value)
            if number <= 0:
                raise ValueError(f"{flag} must be positive, got {value}")
            if flag == "--seconds":
                seconds = number
            else:
                fps = number
        index += 2
    return url, seconds, fps


def collect(url: str, seconds: float) -> tuple[list[bytes], list[float], float]:
    """Read a stream for ``seconds`` and timestamp every frame.

    Args:
        url: Stream URL.
        seconds: How long to read.

    Returns:
        The frames, their arrival times in seconds from the first read,
        and the actual elapsed window (which may exceed ``seconds`` by
        one read).

    Raises:
        ValueError: If the endpoint refused, or the response carries no
            multipart boundary. The status is checked FIRST, because a
            refusal is a valid text/plain response and blaming the
            boundary for it points at the stream when the target was
            simply not there.
    """
    frames: list[bytes] = []
    arrivals: list[float] = []
    buffer = b""
    stream = core_hooks.open_http_stream(url)
    try:
        if stream.status != 200:
            raise ValueError(f"{url} answered {stream.status}, not a stream")
        boundary = boundary_from_content_type(stream.content_type)
        start = core_hooks.get_current_time_ms()
        while True:
            elapsed = (core_hooks.get_current_time_ms() - start) / 1000.0
            if elapsed >= seconds:
                return frames, arrivals, elapsed
            chunk = stream.read(4096)
            if not chunk:
                return frames, arrivals, elapsed
            buffer += chunk
            found, buffer = frames_from_buffer(buffer, boundary)
            now = (core_hooks.get_current_time_ms() - start) / 1000.0
            for frame in found:
                frames.append(frame)
                arrivals.append(now)
    finally:
        stream.close()


def main() -> None:
    """Entry point for ``tankpit-stream-probe``.

    Returns:
        None. Always exits 0 on a completed measurement; exits 2 on a
        bad command line.
    """
    argv = core_hooks.get_argv()[1:]
    try:
        url, seconds, fps = parse_args(argv)
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(2) from error
    frames, arrivals, elapsed = collect(url, seconds)
    if not frames:
        sys.stdout.write(f"no frames from {url} in {elapsed:.1f}s\n")
        return
    report = summarize_stream(frames, arrivals, elapsed, 1.0 / fps if fps > 0 else 0.0)
    sys.stdout.write(f"{url}\n{render_report(report)}\n")


__all__ = [
    "DEFAULT_SECONDS",
    "DEFAULT_URL",
    "collect",
    "main",
    "parse_args",
]
