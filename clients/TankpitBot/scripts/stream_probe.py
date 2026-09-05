"""Measure a live HLS stream and say what its numbers support.

Exists because "the video looks laggy" was answered four times by
argument in one session and three of those answers were wrong. For the
HLS pipeline the observable that separates encoder faults from viewer
faults is the playlist: a healthy live stream advances its media
sequence once per segment length, so polling the playlist measures the
encoder directly, upstream of any player, buffer, or network.

Usage::

    tankpit-stream-probe                          # the demo slot, 30 s
    tankpit-stream-probe --url http://127.0.0.1:27300/demo/video/demo-1/index.m3u8
    tankpit-stream-probe --seconds 60 --interval 0.25

Exit code is 0 whatever the measurement says. This is a report, never a
gate: a playlist is allowed to answer "warming" because the bot is
still logging in, and a tool that failed on that would be asserting
something it cannot know.
"""

from __future__ import annotations

import sys

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.diagnostics.hls_quality import (
    PlaylistSampleDict,
    parse_playlist,
    render_report,
    summarize_samples,
)
from tankpit_bot.service.fleet_config import FLEET_PORT_DEFAULT

DEFAULT_URL = f"http://127.0.0.1:{FLEET_PORT_DEFAULT}/demo/video/demo-1/index.m3u8"
DEFAULT_SECONDS = 30.0
DEFAULT_INTERVAL = 0.5

_USAGE = (
    "usage: tankpit-stream-probe [--url URL] [--seconds N] [--interval N]\n"
    "\n"
    f"  --url       playlist to measure (default: {DEFAULT_URL})\n"
    f"  --seconds   observation window (default: {DEFAULT_SECONDS:.0f})\n"
    f"  --interval  seconds between playlist polls (default: {DEFAULT_INTERVAL})\n"
)


def parse_args(argv: list[str]) -> tuple[str, float, float]:
    """Parse the command line.

    Args:
        argv: Arguments after the program name.

    Returns:
        The playlist URL, the window in seconds, and the poll interval
        in seconds.

    Raises:
        ValueError: On an unknown flag, a flag missing its value, or a
            non-numeric or non-positive number. Refused rather than
            defaulted, because a probe that silently measured a
            different window than asked for would produce numbers
            nobody could reproduce.
    """
    url = DEFAULT_URL
    seconds = DEFAULT_SECONDS
    interval = DEFAULT_INTERVAL
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag not in ("--url", "--seconds", "--interval"):
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
                interval = number
        index += 2
    return url, seconds, interval


def collect(url: str, seconds: float, interval: float) -> list[PlaylistSampleDict]:
    """Poll the playlist for ``seconds`` and record every observation.

    A 503 (warming) or 404 (nothing streaming) is a RECORDED sample,
    not a failure — those answers are part of what the report exists
    to count. A connection refusal propagates: the probe's target not
    existing at all is a fault of the probe run, and dressing it as a
    measurement would produce a report about nothing.

    Args:
        url: Playlist URL.
        seconds: How long to observe.
        interval: Seconds between polls.

    Returns:
        The samples, in time order.

    Raises:
        ValueError: A 200 answer that is not an m3u8 playlist — the
            URL points at something, but not at a stream.
    """
    samples: list[PlaylistSampleDict] = []
    start_ms = core_hooks.get_current_time_ms()
    while True:
        now_ms = core_hooks.get_current_time_ms()
        if (now_ms - start_ms) / 1000.0 >= seconds:
            return samples
        response = script_hooks.http_get(url)
        playlist = (
            parse_playlist(response.content.decode("utf-8"))
            if response.status_code == 200
            else None
        )
        samples.append(
            PlaylistSampleDict(
                at_ms=core_hooks.get_current_time_ms(),
                status=response.status_code,
                playlist=playlist,
            )
        )
        script_hooks.sleep_seconds(interval)


def main() -> None:
    """Entry point for ``tankpit-stream-probe``.

    Returns:
        None. Always exits 0 on a completed measurement; exits 2 on a
        bad command line.
    """
    argv = core_hooks.get_argv()[1:]
    try:
        url, seconds, interval = parse_args(argv)
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(2) from error
    samples = collect(url, seconds, interval)
    sys.stdout.write(f"{url}\n{render_report(summarize_samples(samples))}\n")


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_SECONDS",
    "DEFAULT_URL",
    "collect",
    "main",
    "parse_args",
]
