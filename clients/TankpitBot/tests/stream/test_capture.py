"""Xvfb + ffmpeg lifecycle, driven with real child processes.

The DI seam chooses WHICH process runs, never whether one does: every
handle the tests hand the capture code wraps a real ``sys.executable``
child, so terminate/kill/poll semantics are the operating system's own.
Only the clocks are injected, because a test that spent the real
ten-second readiness deadline would cost ten seconds to prove one
branch. Waits are the one exception: a bounded wait is recorded, and
asserted, instead of spent against the host's load
(``tests/_capture_process.py`` says why). The argv builders are pinned
in ``test_capture_commands.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks as root_hooks
from tankpit_bot.stream import _test_hooks as stream_hooks
from tankpit_bot.stream.capture import (
    HLS_PLAYLIST_FILENAME,
    DisplayCapture,
    ffmpeg_command,
    x11_socket_path,
    xvfb_command,
)
from tankpit_bot.stream.helper_process import (
    PROCESS_END_TIMEOUT_SECONDS,
    SOCKET_READY_TIMEOUT_SECONDS,
    CaptureError,
)
from tests._capture_process import PatientCaptureProcess
from tests.stream._capture_fixtures import (
    SteppingClock,
    SubstitutingSpawner,
    noop_sleep,
    sleeper_argv,
    stream_config,
)


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
        spawner = SubstitutingSpawner([sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner

        def socket_only(path: Path) -> bool:
            return path == x11_socket_path(91)

        root_hooks.path_exists = socket_only
        capture = DisplayCapture(stream_config(tmp_path / "hls"))

        capture.start_display()
        _reap.extend(spawner.processes)

        assert capture.display_env == ":91"
        assert spawner.commands == [xvfb_command(stream_config(tmp_path / "hls"))]
        assert spawner.log_paths == [tmp_path / "xvfb.log"]

    def test_second_start_is_refused(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """One capture owns one display; a second start is a defect."""
        spawner = SubstitutingSpawner([sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        capture.start_display()
        _reap.extend(spawner.processes)

        with pytest.raises(CaptureError, match="already started"):
            capture.start_display()

    def test_a_server_that_dies_is_reported_with_its_exit_code(self, tmp_path: Path) -> None:
        """An Xvfb that exits reads as what it is, not as a timeout."""
        spawner = SubstitutingSpawner([[sys.executable, "-c", "raise SystemExit(3)"]])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: False
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
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
        spawner = SubstitutingSpawner([sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: False
        stream_hooks.monotonic_seconds = SteppingClock(SOCKET_READY_TIMEOUT_SECONDS)
        stream_hooks.sleep_seconds = noop_sleep
        capture = DisplayCapture(stream_config(tmp_path / "hls"))

        with pytest.raises(CaptureError, match="not ready after"):
            capture.start_display()
        _reap.extend(spawner.processes)

    def test_a_slow_socket_is_polled_into_readiness(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """A socket appearing on the second look ends the wait normally."""
        spawner = SubstitutingSpawner([sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        answers = [False, True]
        root_hooks.path_exists = lambda path: answers.pop(0)
        stream_hooks.monotonic_seconds = SteppingClock(0.01)
        slept: list[float] = []

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        stream_hooks.sleep_seconds = record_sleep
        capture = DisplayCapture(stream_config(tmp_path / "hls"))

        capture.start_display()
        _reap.extend(spawner.processes)

        assert len(slept) == 1


class TestStartEncoder:
    """ffmpeg bring-up over a fresh directory."""

    def test_encoder_before_display_is_refused(self, tmp_path: Path) -> None:
        """There is nothing to record without a display."""
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        with pytest.raises(CaptureError, match="start_display must run"):
            capture.start_encoder()

    def test_encoder_with_a_dead_display_is_refused(self, tmp_path: Path) -> None:
        """A display that died between the two starts is reported."""
        # The child lives until the test releases it, AFTER start_display
        # returned: a child that exited on its own could be seen dead by
        # start_display's first poll on a loaded machine, which failed
        # this test there with "Xvfb exited 0 before display :91 came up"
        # (MCPs board task 077204e8).
        release = tmp_path / "release"
        wait_for_release = (
            f"import os, time\nwhile not os.path.exists({str(release)!r}): time.sleep(0.01)"
        )
        spawner = SubstitutingSpawner([[sys.executable, "-c", wait_for_release]])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        capture.start_display()
        release.touch()
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
        spawner = SubstitutingSpawner([sleeper_argv(), sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(hls_dir))
        capture.start_display()

        capture.start_encoder()
        _reap.extend(spawner.processes)

        assert not (hls_dir / "seg00007.ts").exists()
        assert not (hls_dir / HLS_PLAYLIST_FILENAME).exists()
        assert (hls_dir / "unrelated.txt").read_bytes() == b"kept"
        assert spawner.commands[1] == ffmpeg_command(stream_config(hls_dir))
        assert spawner.log_paths[1] == tmp_path / "ffmpeg.log"

    def test_second_encoder_is_refused(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """One capture owns one encoder."""
        spawner = SubstitutingSpawner([sleeper_argv(), sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        capture.start_display()
        capture.start_encoder()
        _reap.extend(spawner.processes)

        with pytest.raises(CaptureError, match="encoder already started"):
            capture.start_encoder()


class _StuckProcess:
    """A protocol-complete handle over a real child that ignores terminate.

    Models a helper stuck past SIGTERM: ``terminate`` does nothing and
    the first ``wait`` reports the timeout the real call would spend
    five seconds discovering. ``kill`` and the second ``wait`` reach the
    real child through a :class:`PatientCaptureProcess`, so the
    escalation being tested actually ends a process, and the bound the
    second wait carries is recorded for the test to assert.
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

    def wait(self, timeout: float | None = None) -> int:
        """Time out once, then delegate.

        Args:
            timeout: Forwarded to the real wait on the second call; the
                reap fixture's unbounded wait forwards None.

        Returns:
            The real exit code, on the second call.

        Raises:
            subprocess.TimeoutExpired: On the first call.
        """
        self._waits += 1
        if self._waits == 1:
            raise subprocess.TimeoutExpired(cmd="stuck", timeout=PROCESS_END_TIMEOUT_SECONDS)
        return self._process.wait(timeout)


class TestStop:
    """Teardown ordering, idempotence, and the kill escalation."""

    def test_stop_ends_encoder_then_display_and_is_idempotent(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """Both children end; a second stop has nothing left to do."""
        spawner = SubstitutingSpawner([sleeper_argv(), sleeper_argv()])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        capture.start_display()
        capture.start_encoder()
        _reap.extend(spawner.processes)

        capture.stop()

        for name, process in zip(("Xvfb", "ffmpeg"), spawner.processes, strict=True):
            if process.poll() is None:
                raise AssertionError(f"{name} stand-in still running after stop()")
        # Each helper was waited on once, with the production bound: the
        # polite end, never the escalation.
        assert [process.wait_timeouts for process in spawner.processes] == [
            [PROCESS_END_TIMEOUT_SECONDS],
            [PROCESS_END_TIMEOUT_SECONDS],
        ]
        capture.stop()  # nothing to do, nothing to raise
        assert [process.wait_timeouts for process in spawner.processes] == [
            [PROCESS_END_TIMEOUT_SECONDS],
            [PROCESS_END_TIMEOUT_SECONDS],
        ]

    def test_stop_with_nothing_started_is_a_noop(self, tmp_path: Path) -> None:
        """A capture that never started stops cleanly."""
        DisplayCapture(stream_config(tmp_path / "hls")).stop()

    def test_an_already_exited_helper_is_not_terminated_again(self, tmp_path: Path) -> None:
        """A child that ended on its own is logged, not signalled."""
        spawner = SubstitutingSpawner([[sys.executable, "-c", "raise SystemExit(0)"]])
        stream_hooks.spawn_capture_process = spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        capture.start_display()
        spawner.processes[0].wait()

        capture.stop()

        assert spawner.processes[0].poll() == 0
        # The test's own wait is the only one: stop neither signalled
        # nor waited on a helper that had already ended.
        assert spawner.processes[0].wait_timeouts == [None]

    def test_a_helper_that_ignores_terminate_is_killed(
        self, tmp_path: Path, _reap: list[stream_hooks.CaptureProcessProtocol]
    ) -> None:
        """SIGTERM refusal escalates to SIGKILL and still ends the child."""
        real_holder: list[_StuckProcess] = []
        inner_holder: list[PatientCaptureProcess] = []

        def stuck_spawner(
            command: list[str], log_path: Path
        ) -> stream_hooks.CaptureProcessProtocol:
            del command
            inner = PatientCaptureProcess(
                stream_hooks._real_spawn_capture_process(sleeper_argv(), log_path)
            )
            stuck = _StuckProcess(inner)
            inner_holder.append(inner)
            real_holder.append(stuck)
            return stuck

        stream_hooks.spawn_capture_process = stuck_spawner
        root_hooks.path_exists = lambda path: True
        capture = DisplayCapture(stream_config(tmp_path / "hls"))
        capture.start_display()
        _reap.extend(real_holder)

        capture.stop()

        assert real_holder[0].terminates == 1
        assert real_holder[0].kills == 1
        if real_holder[0].poll() is None:
            raise AssertionError("the stuck helper survived the kill escalation")
        # The wait after the kill reached the real child with the
        # production bound, recorded rather than spent.
        assert inner_holder[0].wait_timeouts == [PROCESS_END_TIMEOUT_SECONDS]
