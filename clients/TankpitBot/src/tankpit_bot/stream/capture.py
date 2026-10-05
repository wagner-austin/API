"""Xvfb and ffmpeg lifecycle for one bot's display capture.

The bot owns two helper processes here while streaming: an Xvfb server
that IS the display Chromium renders onto, and an ffmpeg that records
that display, and the game sound the bot's
:class:`~tankpit_bot.stream.audio.AudioSink` carries, into HLS
segments. This module builds their command lines from one
:class:`~tankpit_bot.stream.types.StreamConfigDict` and walks both
through start and stop, in the only order that works — display up
before the browser launches, encoder up once there is something to
record, encoder down before the display it records.

The capture is a REAL screen recording of the game rendering itself at
its own cadence. Nothing here touches the page, the tick loop, or the
CDP connection; the slideshow of 2026-09-04 came from doing capture
work on those, and the whole point of this design is that it cannot.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot.stream import _test_hooks
from tankpit_bot.stream.audio import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SINK_NAME,
    pulse_server_address,
)
from tankpit_bot.stream.helper_process import CaptureError, await_socket, end_process
from tankpit_bot.stream.types import StreamConfigDict

log = get_logger(__name__)

XVFB_PROGRAM = "Xvfb"
"""The X virtual framebuffer server, expected on PATH in the image."""

FFMPEG_PROGRAM = "ffmpeg"
"""The encoder, expected on PATH in the image."""

AUDIO_BITRATE_KBPS = 128
"""AAC bitrate of the game sound. The effects are short and sharp;
128k stereo carries them without the smearing lower rates add, and is
a small fraction of the video's bitrate."""

INPUT_QUEUE_PACKETS = 1024
"""ffmpeg's per-input packet queue (``-thread_queue_size``).

Two live inputs are read on their own threads; the default queue of 8
fills while the encoder is busy on a keyframe, and a full queue drops
packets from a live source, which is heard as clicks and seen as
stutter."""

HLS_PLAYLIST_FILENAME = "index.m3u8"
"""The playlist ffmpeg maintains, and the file viewers ask for first."""

HLS_SEGMENT_TEMPLATE = "seg%05d.ts"
"""ffmpeg's segment naming template. The zero-padded counter is what
:data:`tankpit_bot.stream.hls.SEGMENT_NAME_PATTERN` validates against."""

HLS_LIST_SEGMENTS = 6
"""Live-window length in segments. Six two-second segments is twelve
seconds of joinable history — enough for a player to buffer, small
enough that the directory never accumulates a session's worth of video."""


def x11_socket_path(display: int) -> Path:
    """Return the socket path an X server for ``display`` binds.

    Args:
        display: X display number.

    Returns:
        The abstract-namespace-free Unix socket path X clients dial.
    """
    return Path(f"/tmp/.X11-unix/X{display}")


def xvfb_command(config: StreamConfigDict) -> list[str]:
    """Build the Xvfb argv for one capture session.

    ``-nolisten tcp`` because the display's only clients are this
    process's own children; a TCP listener would be a port nobody
    asked for on a container that publishes exactly one.

    Args:
        config: The capture session's parameters.

    Returns:
        Full argv, program first.
    """
    return [
        XVFB_PROGRAM,
        f":{config['display']}",
        "-screen",
        "0",
        f"{config['width']}x{config['height']}x24",
        "-nolisten",
        "tcp",
    ]


def ffmpeg_command(config: StreamConfigDict) -> list[str]:
    """Build the ffmpeg argv for one capture session.

    The choices that matter:

    * ``x11grab`` at the configured rate records the display itself —
      capture rides the compositor, not the page's main thread.
    * The second input is the monitor of the bot's own sound sink
      (:mod:`tankpit_bot.stream.audio`). Both live inputs stamp their
      packets with the wall clock as they are read, so the sound lands
      in the same segment as the frame it belongs to with no offset to
      tune.
    * ``libx264 veryfast`` with ``yuv420p`` and AAC: software encode is
      cheap at this resolution and every browser decodes both.
    * The keyframe interval equals one segment exactly
      (``fps * segment_seconds``, scene-cut detection off), so every
      segment opens decodable and a viewer can join at any boundary.
    * ``delete_segments`` keeps the directory at the live window;
      ``temp_file`` makes each segment appear atomically, so the HTTP
      surface can never serve a half-written file.
    * ``program_date_time`` writes each segment's wall-clock start into
      the playlist, which is how the demo page lines the bot's caption
      up with the frame on screen rather than with the live edge.

    Args:
        config: The capture session's parameters.

    Returns:
        Full argv, program first.
    """
    hls_dir = Path(config["hls_dir"])
    keyframe_interval = config["fps"] * config["segment_seconds"]
    return [
        FFMPEG_PROGRAM,
        "-loglevel",
        "error",
        "-nostdin",
        "-thread_queue_size",
        str(INPUT_QUEUE_PACKETS),
        "-f",
        "x11grab",
        # The X cursor parks wherever it last was — the bot plays over
        # the wire and never moves it — and drawing it into the public
        # stream reads as a ghost hand on the game (operator report,
        # 2026-09-05, first live viewing).
        "-draw_mouse",
        "0",
        "-framerate",
        str(config["fps"]),
        "-video_size",
        f"{config['width']}x{config['height']}",
        "-i",
        f":{config['display']}",
        "-thread_queue_size",
        str(INPUT_QUEUE_PACKETS),
        "-f",
        "pulse",
        "-server",
        pulse_server_address(config["display"]),
        "-sample_rate",
        str(AUDIO_SAMPLE_RATE),
        "-channels",
        str(AUDIO_CHANNELS),
        "-i",
        f"{AUDIO_SINK_NAME}.monitor",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(keyframe_interval),
        "-keyint_min",
        str(keyframe_interval),
        "-sc_threshold",
        "0",
        "-b:v",
        f"{config['bitrate_kbps']}k",
        "-maxrate",
        f"{config['bitrate_kbps'] * 3 // 2}k",
        "-bufsize",
        f"{config['bitrate_kbps'] * 3}k",
        "-c:a",
        "aac",
        "-b:a",
        f"{AUDIO_BITRATE_KBPS}k",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        str(AUDIO_CHANNELS),
        "-f",
        "hls",
        "-hls_time",
        str(config["segment_seconds"]),
        "-hls_list_size",
        str(HLS_LIST_SEGMENTS),
        "-hls_flags",
        "delete_segments+independent_segments+temp_file+program_date_time",
        "-hls_segment_filename",
        str(hls_dir / HLS_SEGMENT_TEMPLATE),
        str(hls_dir / HLS_PLAYLIST_FILENAME),
    ]


class DisplayCapture:
    """One bot's Xvfb + ffmpeg pair, started apart and stopped together.

    Started apart because the two have different prerequisites: the
    display must exist before Chromium launches, while the encoder
    only has something to record once the game is on screen. Stopped
    together, encoder first, because an encoder whose display vanishes
    dies mid-segment instead of finalising it.
    """

    def __init__(self, config: StreamConfigDict) -> None:
        """Hold the configuration; start nothing yet.

        Args:
            config: The capture session's parameters.
        """
        self._config = config
        self._xvfb: _test_hooks.CaptureProcessProtocol | None = None
        self._ffmpeg: _test_hooks.CaptureProcessProtocol | None = None

    @property
    def display_env(self) -> str:
        """The ``DISPLAY`` value Chromium must launch under."""
        return f":{self._config['display']}"

    def start_display(self) -> None:
        """Start Xvfb and block until its display accepts clients.

        Raises:
            CaptureError: Already started, the server died on launch,
                or it never came ready.
            OSError: Xvfb is not installed.
        """
        if self._xvfb is not None:
            raise CaptureError("display already started")
        log_path = Path(self._config["hls_dir"]).parent / "xvfb.log"
        process = _test_hooks.spawn_capture_process(xvfb_command(self._config), log_path)
        self._xvfb = process
        log.info("Capture: Xvfb pid %d serving display %s", process.pid, self.display_env)
        await_socket(process, XVFB_PROGRAM, x11_socket_path(self._config["display"]), log_path)

    def start_encoder(self) -> None:
        """Start ffmpeg recording the display into a fresh HLS dir.

        Fresh means fresh: segments and playlist from an earlier run
        of this instance are removed first, because a playlist that
        interleaves two sessions' segments decodes as neither.

        Raises:
            CaptureError: The display was never started, the encoder
                already was, or the display server has died.
            OSError: ffmpeg is not installed.
        """
        if self._xvfb is None:
            raise CaptureError("start_display must run before start_encoder")
        if self._ffmpeg is not None:
            raise CaptureError("encoder already started")
        code = self._xvfb.poll()
        if code is not None:
            raise CaptureError(f"Xvfb exited {code}; there is no display to record")
        hls_dir = Path(self._config["hls_dir"])
        hls_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(hls_dir.glob("*.ts")) + sorted(hls_dir.glob("*.m3u8")):
            stale.unlink()
        process = _test_hooks.spawn_capture_process(
            ffmpeg_command(self._config), hls_dir.parent / "ffmpeg.log"
        )
        self._ffmpeg = process
        log.info(
            "Capture: ffmpeg pid %d recording %s into %s", process.pid, self.display_env, hls_dir
        )

    def stop(self) -> None:
        """End whatever is running, encoder first. Safe to call twice."""
        if self._ffmpeg is not None:
            end_process(self._ffmpeg, FFMPEG_PROGRAM)
            self._ffmpeg = None
        if self._xvfb is not None:
            end_process(self._xvfb, XVFB_PROGRAM)
            self._xvfb = None


__all__ = [
    "AUDIO_BITRATE_KBPS",
    "FFMPEG_PROGRAM",
    "HLS_LIST_SEGMENTS",
    "HLS_PLAYLIST_FILENAME",
    "HLS_SEGMENT_TEMPLATE",
    "INPUT_QUEUE_PACKETS",
    "XVFB_PROGRAM",
    "DisplayCapture",
    "ffmpeg_command",
    "x11_socket_path",
    "xvfb_command",
]
