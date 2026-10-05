"""The capture helpers' command lines, whole, and their agreements.

Split from ``test_capture.py`` at the 600-line ceiling, by role: that
module drives the lifecycles with real children, this one pins what
each helper is asked to run. Every argv is asserted whole, because a
flag dropped from the encoder is a stream that silently changes for
every viewer.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.audio import (
    pulse_server_address,
    pulse_socket_path,
    pulseaudio_command,
)
from tankpit_bot.stream.capture import (
    HLS_LIST_SEGMENTS,
    HLS_PLAYLIST_FILENAME,
    HLS_SEGMENT_TEMPLATE,
    ffmpeg_command,
    x11_socket_path,
    xvfb_command,
)
from tankpit_bot.stream.hls import SEGMENT_NAME_PATTERN
from tests.stream._capture_fixtures import stream_config


def test_xvfb_command_is_exactly_the_documented_argv(tmp_path: Path) -> None:
    """The server argv, whole: display, screen geometry, no TCP."""
    assert xvfb_command(stream_config(tmp_path / "hls")) == [
        "Xvfb",
        ":91",
        "-screen",
        "0",
        "704x544x24",
        "-nolisten",
        "tcp",
    ]


def test_pulseaudio_command_is_exactly_the_documented_argv(
    tmp_path: Path, _socket_root: Path
) -> None:
    """The sound server argv, whole: no default script, one socket, one sink."""
    socket = (_socket_root / "tankpit-pulse-91" / "native").as_posix()
    assert pulseaudio_command(stream_config(tmp_path / "hls")) == [
        "pulseaudio",
        "-n",
        "--daemonize=no",
        "--use-pid-file=no",
        "--exit-idle-time=-1",
        "--disallow-exit",
        "--disable-shm=yes",
        "--log-target=stderr",
        f"--load=module-native-protocol-unix socket={socket} auth-anonymous=1",
        "--load=module-null-sink sink_name=game rate=48000 channels=2",
    ]


def test_ffmpeg_command_is_exactly_the_documented_argv(tmp_path: Path, _socket_root: Path) -> None:
    """The encoder argv, whole — display and sound in, keyframes aligned
    to segments, atomic segment writes, wall-clock stamps, and the
    rolling live window."""
    hls_dir = tmp_path / "hls"
    socket = (_socket_root / "tankpit-pulse-91" / "native").as_posix()
    assert ffmpeg_command(stream_config(hls_dir)) == [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-thread_queue_size",
        "1024",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-framerate",
        "30",
        "-video_size",
        "704x544",
        "-i",
        ":91",
        "-thread_queue_size",
        "1024",
        "-f",
        "pulse",
        "-server",
        f"unix:{socket}",
        "-sample_rate",
        "48000",
        "-channels",
        "2",
        "-i",
        "game.monitor",
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
        "60",
        "-keyint_min",
        "60",
        "-sc_threshold",
        "0",
        "-b:v",
        "1000k",
        "-maxrate",
        "1500k",
        "-bufsize",
        "3000k",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-f",
        "hls",
        "-hls_time",
        "2",
        "-hls_list_size",
        str(HLS_LIST_SEGMENTS),
        "-hls_flags",
        "delete_segments+independent_segments+temp_file+program_date_time",
        "-hls_segment_filename",
        str(hls_dir / HLS_SEGMENT_TEMPLATE),
        str(hls_dir / HLS_PLAYLIST_FILENAME),
    ]


def test_chromium_and_the_encoder_dial_the_same_sound_server(_socket_root: Path) -> None:
    """``PULSE_SERVER`` and ffmpeg's ``-server`` name the socket the server binds."""
    socket = pulse_socket_path(91)
    assert socket == _socket_root / "tankpit-pulse-91" / "native"
    assert pulse_server_address(91) == f"unix:{socket.as_posix()}"


def test_the_production_socket_root_is_the_containers_tmp() -> None:
    """Not the runs bind mount, which cannot hold a Unix socket."""
    assert stream_hooks._real_socket_root() == Path("/tmp")


def test_the_segment_template_matches_the_serving_grammar() -> None:
    """What the encoder names, the HTTP filename gate admits."""
    example = HLS_SEGMENT_TEMPLATE % 7
    assert example == "seg00007.ts"
    if SEGMENT_NAME_PATTERN.fullmatch(example) is None:
        raise AssertionError(f"{example!r} does not match the serving grammar")


def test_x11_socket_path_is_the_display_socket() -> None:
    """The readiness poll watches the socket X clients dial."""
    assert x11_socket_path(91) == Path("/tmp/.X11-unix/X91")
