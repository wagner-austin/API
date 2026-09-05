"""Xvfb + ffmpeg lifecycle, driven with real child processes.

The DI seam chooses WHICH process runs, never whether one does: every
handle the tests hand the capture code wraps a real ``sys.executable``
child, so terminate/kill/wait/poll semantics are the operating
system's own. Only the clocks are injected, because a test that spent
the real ten-second readiness deadline would cost ten seconds to
prove one branch.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as root_hooks
from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.capture import (
    DISPLAY_READY_TIMEOUT_SECONDS,
    HLS_LIST_SEGMENTS,
    HLS_PLAYLIST_FILENAME,
    HLS_SEGMENT_TEMPLATE,
    CaptureError,
    DisplayCapture,
    ffmpeg_command,
    x11_socket_path,
    xvfb_command,
)
from tankpit_bot.stream.hls import SEGMENT_NAME_PATTERN
from tankpit_bot.stream.types import StreamConfigDict


def _config(hls_dir: Path) -> StreamConfigDict:
    """Build one capture configuration rooted in a test directory.

    Args:
        hls_dir: Where the encoder would write.

    Returns:
        The configuration.
    """
    return StreamConfigDict(
        display=91,
        width=704,
        height=544,
        fps=30,
        bitrate_kbps=1000,
        segment_seconds=2,
        hls_dir=str(hls_dir),
    )


def _sleeper_argv() -> list[str]:
    """A child that runs until terminated.

    Returns:
        Argv for a 60-second sleeper.
    """
    return [sys.executable, "-c", "import time; time.sleep(60)"]


class _SubstitutingSpawner:
    """Spawn REAL children while recording the commands asked for.

    The capture code asks for ``Xvfb``/``ffmpeg``, which do not exist
    on the test host; this seam records that request and runs a
    ``sys.executable`` stand-in through the REAL spawner, so log-file
    plumbing and process semantics stay production code.
    """

    def __init__(self, argv_per_call: list[list[str]]) -> None:
        """Bind the substitute argv for each successive call.

        Args:
            argv_per_call: What to actually run, call by call.
        """
        self._argv_per_call = argv_per_call
        self.commands: list[list[str]] = []
        self.log_paths: list[Path] = []
        self.processes: list[stream_hooks.CaptureProcessProtocol] = []

    def __call__(self, command: list[str], log_path: Path) -> stream_hooks.CaptureProcessProtocol:
        """Record the request and spawn the substitute.

        Args:
            command: What the capture code wanted to run.
            log_path: Where it wanted the console.

        Returns:
            The substitute process handle.
        """
        self.commands.append(command)
        self.log_paths.append(log_path)
        process = stream_hooks._real_spawn_capture_process(
            self._argv_per_call[len(self.processes)], log_path
        )
        self.processes.append(process)
        return process


class _SteppingClock:
    """Monotonic clock that advances a fixed step per read."""

    def __init__(self, step: float) -> None:
        """Start at zero, advancing ``step`` per read.

        Args:
            step: Seconds each read advances.
        """
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        """Read and advance the clock.

        Returns:
            The pre-advance reading.
        """
        now = self._now
        self._now += self._step
        return now


def _noop_sleep(seconds: float) -> None:
    """Sleep hook that spends no wall clock.

    Args:
        seconds: Ignored.
    """
    del seconds


@pytest.fixture(autouse=True)
def _reap() -> Generator[list[stream_hooks.CaptureProcessProtocol], None, None]:
    """Kill every child a test's spawner left running.

    Yields:
        The list the test's spawner should append processes to.
    """
    spawned: list[stream_hooks.CaptureProcessProtocol] = []
    yield spawned
    for process in spawned:
        if process.poll() is None:
            process.kill()
            process.wait(10.0)


class TestCommandLines:
    """The argv builders and their agreement with the serving layer."""

    def test_xvfb_command_is_exactly_the_documented_argv(self, tmp_path: Path) -> None:
        """The server argv, whole: display, screen geometry, no TCP."""
        assert xvfb_command(_config(tmp_path / "hls")) == [
            "Xvfb",
            ":91",
            "-screen",
            "0",
            "704x544x24",
            "-nolisten",
            "tcp",
        ]

    def test_ffmpeg_command_is_exactly_the_documented_argv(self, tmp_path: Path) -> None:
        """The encoder argv, whole — keyframes aligned to segments,
        atomic segment writes, and the rolling live window."""
        hls_dir = tmp_path / "hls"
        assert ffmpeg_command(_config(hls_dir)) == [
            "ffmpeg",
            "-loglevel",
            "error",
            "-nostdin",
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
            "-f",
            "hls",
            "-hls_time",
            "2",
            "-hls_list_size",
            str(HLS_LIST_SEGMENTS),
            "-hls_flags",
            "delete_segments+independent_segments+temp_file",
            "-hls_segment_filename",
            str(hls_dir / HLS_SEGMENT_TEMPLATE),
            str(hls_dir / HLS_PLAYLIST_FILENAME),
        ]

    def test_the_segment_template_matches_the_serving_grammar(self) -> None:
        """What the encoder names, the HTTP filename gate admits."""
        example = HLS_SEGMENT_TEMPLATE % 7
        assert example == "seg00007.ts"
        if SEGMENT_NAME_PATTERN.fullmatch(example) is None:
            raise AssertionError(f"{example!r} does not match the serving grammar")

    def test_x11_socket_path_is_the_display_socket(self) -> None:
        """The readiness poll watches the socket X clients dial."""
        assert x11_socket_path(91) == Path("/tmp/.X11-unix/X91")


class TestRealClockHooks:
    """The production clock seams are the stdlib's, exercised once."""

    def test_the_real_sleep_advances_the_real_clock(self) -> None:
        """``_real_sleep_seconds`` blocks; ``_real_monotonic_seconds`` sees it.

        The sleep is 50 ms, not 1: under a loaded xdist run this box's
        clock read identical values across a 1 ms sleep (measured
        2026-09-05, two workers at once), and a duration safely past
        the ~15.6 ms Windows scheduler tick is what makes the strict
        inequality a fact rather than a race.
        """
        before = stream_hooks._real_monotonic_seconds()
        stream_hooks._real_sleep_seconds(0.05)
        after = stream_hooks._real_monotonic_seconds()
        assert after > before


class TestRealSpawner:
    """The production spawner against a real child."""

    def test_spawn_captures_the_console_to_the_log_file(self, tmp_path: Path) -> None:
        """stdout and stderr land in the named file, dir created."""
        log_path = tmp_path / "deep" / "capture.log"
        process = stream_hooks._real_spawn_capture_process(
            [
                sys.executable,
                "-c",
                "import sys; print('out-line'); print('err-line', file=sys.stderr)",
            ],
            log_path,
        )
        assert process.wait(30.0) == 0
        text = log_path.read_text()
        assert "out-line" in text
        assert "err-line" in text


class TestStartDisplay:
    """Xvfb bring-up and the readiness wait."""

    def test_ready_display_returns_and_records_the_command(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """A socket that exists ends the wait; the Xvfb argv was asked for."""
        spawner = _SubstitutingSpawner([_sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner

        def socket_only(path: Path) -> bool:
            return path == x11_socket_path(91)

        root_hooks.path_exists = socket_only
        capture = DisplayCapture(_config(tmp_path / "hls"))

        capture.start_display()
        _reap.extend(spawner.processes)

        assert capture.display_env == ":91"
        assert spawner.commands == [xvfb_command(_config(tmp_path / "hls"))]
        assert spawner.log_paths == [tmp_path / "xvfb.log"]

    def test_second_start_is_refused(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """One capture owns one display; a second start is a defect."""
        spawner = _SubstitutingSpawner([_sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(tmp_path / "hls"))
        capture.start_display()
        _reap.extend(spawner.processes)

        with pytest.raises(CaptureError, match="already started"):
            capture.start_display()

    def test_a_server_that_dies_is_reported_with_its_exit_code(self, tmp_path: Path) -> None:
        """An Xvfb that exits reads as what it is, not as a timeout."""
        spawner = _SubstitutingSpawner([[sys.executable, "-c", "raise SystemExit(3)"]])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: False
        capture = DisplayCapture(_config(tmp_path / "hls"))
        # The stand-in must be DEAD before the poll reads it, or the
        # test races its own child.
        original_spawn = spawner.__call__

        def spawn_and_wait(
            command: list[str], log_path: Path
        ) -> stream_hooks.CaptureProcessProtocol:
            process = original_spawn(command, log_path)
            process.wait(30.0)
            return process

        stream_hooks.spawn_capture_process = spawn_and_wait

        with pytest.raises(CaptureError, match="Xvfb exited 3"):
            capture.start_display()

    def test_a_server_that_never_binds_times_out(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """Past the deadline with a live server, the wait gives up loudly."""
        spawner = _SubstitutingSpawner([_sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: False
        stream_hooks.monotonic_seconds = _SteppingClock(DISPLAY_READY_TIMEOUT_SECONDS)
        stream_hooks.sleep_seconds = _noop_sleep
        capture = DisplayCapture(_config(tmp_path / "hls"))

        with pytest.raises(CaptureError, match="not ready after"):
            capture.start_display()
        _reap.extend(spawner.processes)

    def test_a_slow_socket_is_polled_into_readiness(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """A socket appearing on the second look ends the wait normally."""
        spawner = _SubstitutingSpawner([_sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        answers = [False, True]
        root_hooks.path_exists = lambda path: answers.pop(0)
        stream_hooks.monotonic_seconds = _SteppingClock(0.01)
        slept: list[float] = []

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        stream_hooks.sleep_seconds = record_sleep
        capture = DisplayCapture(_config(tmp_path / "hls"))

        capture.start_display()
        _reap.extend(spawner.processes)

        assert len(slept) == 1


class TestStartEncoder:
    """ffmpeg bring-up over a fresh directory."""

    def test_encoder_before_display_is_refused(self, tmp_path: Path) -> None:
        """There is nothing to record without a display."""
        capture = DisplayCapture(_config(tmp_path / "hls"))
        with pytest.raises(CaptureError, match="start_display must run"):
            capture.start_encoder()

    def test_encoder_with_a_dead_display_is_refused(self, tmp_path: Path) -> None:
        """A display that died between the two starts is reported."""
        spawner = _SubstitutingSpawner([[sys.executable, "-c", "raise SystemExit(0)"]])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(tmp_path / "hls"))
        capture.start_display()
        spawner.processes[0].wait(30.0)

        with pytest.raises(CaptureError, match="no display to record"):
            capture.start_encoder()

    def test_encoder_clears_stale_files_and_records_the_command(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """An earlier session's playlist and segments never interleave."""
        hls_dir = tmp_path / "hls"
        hls_dir.mkdir(parents=True)
        (hls_dir / "seg00007.ts").write_bytes(b"stale")
        (hls_dir / HLS_PLAYLIST_FILENAME).write_bytes(b"stale")
        (hls_dir / "unrelated.txt").write_bytes(b"kept")
        spawner = _SubstitutingSpawner([_sleeper_argv(), _sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(hls_dir))
        capture.start_display()

        capture.start_encoder()
        _reap.extend(spawner.processes)

        assert not (hls_dir / "seg00007.ts").exists()
        assert not (hls_dir / HLS_PLAYLIST_FILENAME).exists()
        assert (hls_dir / "unrelated.txt").read_bytes() == b"kept"
        assert spawner.commands[1] == ffmpeg_command(_config(hls_dir))
        assert spawner.log_paths[1] == tmp_path / "ffmpeg.log"

    def test_second_encoder_is_refused(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """One capture owns one encoder."""
        spawner = _SubstitutingSpawner([_sleeper_argv(), _sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(tmp_path / "hls"))
        capture.start_display()
        capture.start_encoder()
        _reap.extend(spawner.processes)

        with pytest.raises(CaptureError, match="encoder already started"):
            capture.start_encoder()


class _StuckProcess:
    """A protocol-complete handle over a real child that ignores terminate.

    Models a helper stuck past SIGTERM: ``terminate`` does nothing and
    the first ``wait`` reports the timeout the real call would spend
    five seconds discovering. ``kill`` and the second ``wait`` are the
    real operations against the real child, so the escalation being
    tested actually ends a process.
    """

    def __init__(self, process: stream_hooks.CaptureProcessProtocol) -> None:
        """Wrap the real child.

        Args:
            process: The real handle.
        """
        self._process = process
        self.terminates = 0
        self.kills = 0
        self._waits = 0

    @property
    def pid(self) -> int:
        """The real child's pid."""
        return self._process.pid

    def poll(self) -> int | None:
        """Delegate to the real child."""
        return self._process.poll()

    def terminate(self) -> None:
        """Ignore the polite request, as a stuck process does."""
        self.terminates += 1

    def kill(self) -> None:
        """Really end the child."""
        self.kills += 1
        self._process.kill()

    def wait(self, timeout: float) -> int:
        """Time out once, then delegate.

        Args:
            timeout: Forwarded to the real wait on the second call.

        Returns:
            The real exit code, on the second call.

        Raises:
            subprocess.TimeoutExpired: On the first call.
        """
        self._waits += 1
        if self._waits == 1:
            raise subprocess.TimeoutExpired(cmd="stuck", timeout=timeout)
        return self._process.wait(timeout)


class TestStop:
    """Teardown ordering, idempotence, and the kill escalation."""

    def test_stop_ends_encoder_then_display_and_is_idempotent(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """Both children end; a second stop has nothing left to do."""
        spawner = _SubstitutingSpawner([_sleeper_argv(), _sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(tmp_path / "hls"))
        capture.start_display()
        capture.start_encoder()
        _reap.extend(spawner.processes)

        capture.stop()

        for name, process in zip(("Xvfb", "ffmpeg"), spawner.processes, strict=True):
            if process.poll() is None:
                raise AssertionError(f"{name} stand-in still running after stop()")
        capture.stop()  # nothing to do, nothing to raise

    def test_stop_with_nothing_started_is_a_noop(self, tmp_path: Path) -> None:
        """A capture that never started stops cleanly."""
        DisplayCapture(_config(tmp_path / "hls")).stop()

    def test_an_already_exited_helper_is_not_terminated_again(self, tmp_path: Path) -> None:
        """A child that ended on its own is logged, not signalled."""
        spawner = _SubstitutingSpawner([[sys.executable, "-c", "raise SystemExit(0)"]])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(tmp_path / "hls"))
        capture.start_display()
        spawner.processes[0].wait(30.0)

        capture.stop()

        assert spawner.processes[0].poll() == 0

    def test_a_helper_that_ignores_terminate_is_killed(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """SIGTERM refusal escalates to SIGKILL and still ends the child."""
        real_holder: list[_StuckProcess] = []

        def stuck_spawner(
            command: list[str], log_path: Path
        ) -> stream_hooks.CaptureProcessProtocol:
            del command
            stuck = _StuckProcess(
                stream_hooks._real_spawn_capture_process(_sleeper_argv(), log_path)
            )
            real_holder.append(stuck)
            return stuck

        stream_hooks.spawn_capture_process = stuck_spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(_config(tmp_path / "hls"))
        capture.start_display()
        _reap.extend(real_holder)

        capture.stop()

        assert real_holder[0].terminates == 1
        assert real_holder[0].kills == 1
        if real_holder[0].poll() is None:
            raise AssertionError("the stuck helper survived the kill escalation")
