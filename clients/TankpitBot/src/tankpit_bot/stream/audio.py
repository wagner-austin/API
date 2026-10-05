"""The PulseAudio server one streamed bot's game sound plays into.

TankPit plays its effects (shots, explosions, radar, pickups, the
engine) through WebAudio, and Chromium on Linux hands WebAudio to
PulseAudio. A container has no sound server at all, so before this
module a streamed bot's audio went nowhere and the public stream was
silent (operator, 2026-10-05: "there's no sound?").

Each streamed bot therefore gets its own PulseAudio server with exactly
one output, a null sink named :data:`AUDIO_SINK_NAME`. Chromium plays
into it because its ``PULSE_SERVER`` names this server's socket, and
the encoder records the sink's monitor source
(:func:`tankpit_bot.stream.capture.ffmpeg_command`) beside the display,
so sound and picture are cut into the same HLS segments on one clock.
One server per bot, never one shared: a shared sink would mix every
bot's game into every stream.

The socket lives under ``/tmp`` (:data:`~tankpit_bot.stream._test_hooks.socket_root`)
keyed by the bot's X display number, which the fleet already allocates
uniquely per live bot. Not beside the HLS directory: ``runs/`` is a
bind mount from the Windows host on the fleet, and a Unix socket cannot
be created on that filesystem.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot.stream import _test_hooks
from tankpit_bot.stream.helper_process import CaptureError, await_socket, end_process
from tankpit_bot.stream.types import StreamConfigDict

log = get_logger(__name__)

PULSEAUDIO_PROGRAM = "pulseaudio"
"""The sound server, expected on PATH in the image."""

AUDIO_SINK_NAME = "game"
"""The one sink the server creates. Chromium plays into it as the
default output; the encoder records ``<name>.monitor``."""

AUDIO_SAMPLE_RATE = 48000
"""Sample rate of the sink, and of the encoded stream, in hertz.

One rate end to end, so nothing between the game and the viewer
resamples."""

AUDIO_CHANNELS = 2
"""Stereo: the sink, the recording and the encoded stream agree."""


def pulse_socket_path(display: int) -> Path:
    """Return the socket path a bot's sound server binds.

    Args:
        display: The bot's X display number, unique per live bot.

    Returns:
        The Unix socket path Chromium and ffmpeg dial.
    """
    return _test_hooks.socket_root() / f"tankpit-pulse-{display}" / "native"


def pulse_server_address(display: int) -> str:
    """Return the server string libpulse clients take.

    The same string is ``PULSE_SERVER`` in Chromium's environment and
    ffmpeg's ``-server`` input option.

    Args:
        display: The bot's X display number.

    Returns:
        ``unix:<socket path>``.
    """
    return f"unix:{pulse_socket_path(display).as_posix()}"


def pulseaudio_command(config: StreamConfigDict) -> list[str]:
    """Build the PulseAudio argv for one bot.

    The choices that matter:

    * ``-n`` loads no default script, so the server has exactly the two
      modules named here and never probes for hardware the container
      does not have.
    * ``--daemonize=no`` keeps it the child the capture lifecycle owns,
      and ``--use-pid-file=no`` lets several servers run under the one
      container user, one per bot.
    * ``--exit-idle-time=-1`` keeps it up while the game is quiet; an
      idle exit would end the encoder's audio input mid-session.
    * ``auth-anonymous=1`` because the socket's only clients are this
      bot's own Chromium and ffmpeg, and a cookie would be one more
      file both have to find.
    * ``--disable-shm=yes`` sends audio over the socket rather than
      through ``/dev/shm``, which Chromium already uses heavily and a
      container sizes small.

    Args:
        config: The capture session's parameters.

    Returns:
        Full argv, program first.
    """
    socket = pulse_socket_path(config["display"]).as_posix()
    return [
        PULSEAUDIO_PROGRAM,
        "-n",
        "--daemonize=no",
        "--use-pid-file=no",
        "--exit-idle-time=-1",
        "--disallow-exit",
        "--disable-shm=yes",
        "--log-target=stderr",
        f"--load=module-native-protocol-unix socket={socket} auth-anonymous=1",
        f"--load=module-null-sink sink_name={AUDIO_SINK_NAME}"
        f" rate={AUDIO_SAMPLE_RATE} channels={AUDIO_CHANNELS}",
    ]


class AudioSink:
    """One bot's PulseAudio server, started before Chromium launches.

    Started before the browser so the browser finds a server on its
    first sound, and stopped after the encoder, whose audio input is
    this server's monitor: an encoder whose input vanishes dies
    mid-segment instead of finalising it.
    """

    def __init__(self, config: StreamConfigDict) -> None:
        """Hold the configuration; start nothing yet.

        Args:
            config: The capture session's parameters.
        """
        self._config = config
        self._process: _test_hooks.CaptureProcessProtocol | None = None

    @property
    def server_address(self) -> str:
        """The ``PULSE_SERVER`` value Chromium must launch under."""
        return pulse_server_address(self._config["display"])

    def start(self) -> None:
        """Start the server and block until its socket accepts clients.

        A socket left by an earlier server on the same display is
        removed first: it would satisfy the readiness wait before the
        new server had bound anything.

        Raises:
            CaptureError: Already started, the server died on launch,
                or it never came ready.
            OSError: PulseAudio is not installed.
        """
        if self._process is not None:
            raise CaptureError("audio sink already started")
        socket = pulse_socket_path(self._config["display"])
        socket.parent.mkdir(parents=True, exist_ok=True)
        socket.unlink(missing_ok=True)
        log_path = Path(self._config["hls_dir"]).parent / "pulseaudio.log"
        process = _test_hooks.spawn_capture_process(pulseaudio_command(self._config), log_path)
        self._process = process
        log.info("Capture: PulseAudio pid %d serving %s", process.pid, self.server_address)
        await_socket(process, PULSEAUDIO_PROGRAM, socket, log_path)

    def stop(self) -> None:
        """End the server if it runs. Safe to call twice."""
        if self._process is not None:
            end_process(self._process, PULSEAUDIO_PROGRAM)
            self._process = None


__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_SAMPLE_RATE",
    "AUDIO_SINK_NAME",
    "PULSEAUDIO_PROGRAM",
    "AudioSink",
    "pulse_server_address",
    "pulse_socket_path",
    "pulseaudio_command",
]
