"""The per-bot PulseAudio server's lifecycle, driven with real children.

The same seam discipline as the display tests: the spawn hook runs a
real ``sys.executable`` stand-in through the production spawner in
place of ``pulseaudio``, and the readiness poll reads the filesystem
seam. The socket root is the test's own directory (``conftest.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as root_hooks
from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.audio import AudioSink, pulse_socket_path, pulseaudio_command
from tankpit_bot.stream.helper_process import PROCESS_END_TIMEOUT_SECONDS, CaptureError
from tests.stream._capture_fixtures import SubstitutingSpawner, sleeper_argv, stream_config


def test_start_clears_a_stale_socket_spawns_the_server_and_waits_for_it(
    tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
) -> None:
    """A dead server's socket cannot pass for the new one's readiness."""
    socket = pulse_socket_path(91)
    socket.parent.mkdir(parents=True)
    socket.write_bytes(b"left by an earlier server")
    seen_at_spawn: list[bool] = []
    spawner = SubstitutingSpawner([sleeper_argv()])

    def spawn_after_clearing(
        command: list[str], log_path: Path
    ) -> stream_hooks.CaptureProcessProtocol:
        seen_at_spawn.append(socket.exists())
        return spawner(command, log_path)

    def socket_only(path: Path) -> bool:
        return path == socket

    stream_hooks.spawn_capture_process = spawn_after_clearing
    root_hooks.path_exists = socket_only
    config = stream_config(tmp_path / "hls")
    sink = AudioSink(config)

    sink.start()
    _reap.extend(spawner.processes)

    assert seen_at_spawn == [False]
    assert spawner.commands == [pulseaudio_command(config)]
    assert spawner.log_paths == [tmp_path / "pulseaudio.log"]
    assert sink.server_address == f"unix:{socket.as_posix()}"


def test_a_second_start_is_refused(
    tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
) -> None:
    """One sink owns one server."""
    spawner = SubstitutingSpawner([sleeper_argv()])
    stream_hooks.spawn_capture_process = spawner
    root_hooks.path_exists = lambda path: True
    sink = AudioSink(stream_config(tmp_path / "hls"))
    sink.start()
    _reap.extend(spawner.processes)

    with pytest.raises(CaptureError, match="audio sink already started"):
        sink.start()


def test_a_server_that_dies_is_reported_with_its_exit_code(tmp_path: Path) -> None:
    """A PulseAudio that exits reads as what it is, naming its log."""
    spawner = SubstitutingSpawner([[sys.executable, "-c", "raise SystemExit(2)"]])

    def spawn_and_wait(command: list[str], log_path: Path) -> stream_hooks.CaptureProcessProtocol:
        process = spawner(command, log_path)
        process.wait(30.0)
        return process

    stream_hooks.spawn_capture_process = spawn_and_wait
    root_hooks.path_exists = lambda path: False
    sink = AudioSink(stream_config(tmp_path / "hls"))

    with pytest.raises(CaptureError, match=r"pulseaudio exited 2 .*pulseaudio\.log"):
        sink.start()


def test_stop_ends_the_server_once_and_is_idempotent(
    tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
) -> None:
    """The server ends with the production bound; a second stop does nothing."""
    spawner = SubstitutingSpawner([sleeper_argv()])
    stream_hooks.spawn_capture_process = spawner
    root_hooks.path_exists = lambda path: True
    sink = AudioSink(stream_config(tmp_path / "hls"))
    sink.start()
    _reap.extend(spawner.processes)

    sink.stop()
    sink.stop()

    if spawner.processes[0].poll() is None:
        raise AssertionError("the pulseaudio stand-in still runs after stop()")
    assert spawner.processes[0].wait_timeouts == [PROCESS_END_TIMEOUT_SECONDS]


def test_stop_with_nothing_started_is_a_noop(tmp_path: Path) -> None:
    """A sink that never started stops cleanly."""
    AudioSink(stream_config(tmp_path / "hls")).stop()
