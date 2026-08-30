"""The hook implementations, against the real operating system.

These are the only tests in this package that do not use a fake, and that is
the point: everywhere else a fake stands in for these functions, so if they
themselves were tested against a fake nothing would ever exercise the real
socket, the real process or the real filesystem, and the whole suite would be
green against an operating system that does not exist.

The pure decisions each of these APPLIES -- which command fells a tree, which
flags isolate a spawn -- are checked exhaustively and cross-platform in
``test_harness_process_tree.py``. What is checked here is that the application
works: that a real child really is felled, that a real socket really is seen.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rw_bot.harness._hook_defaults import (
    COPY_EXCLUDES,
    _copy_entry_impl,
    _kill_tree_impl,
    _new_stamp_impl,
    _read_environment_impl,
    _read_executable_impl,
    _read_platform_impl,
    _remove_path_impl,
    _resolve_root_impl,
    _run_inherited_impl,
    _sleep_impl,
    _spawn_game_impl,
    _wait_for_port_impl,
)
from rw_bot.harness._hook_protocols import SpawnedMatchProto
from rw_bot.harness.launch import PORT_POLL_SECONDS
from rw_bot.harness.process_tree import spawn_isolation
from rw_bot.platform_id import WINDOWS, is_windows

#: Long enough that a slow machine still connects, short enough that the
#: negative case does not dominate the suite.
_WAIT_SECONDS = 5.0

#: Ports nothing in this repository leases. The launcher's own band starts at
#: 27510 and the recipe's random draw lives at 27600-27999, so these sit well
#: clear of both.
#:
#: Stated rather than read back from a port-0 bind, because
#: ``socket.getsockname()`` is typed ``Any`` and this package forbids an ``Any``
#: expression outright. A bind that fails here fails the test loudly, which is
#: the right outcome -- a port that is unexpectedly occupied is a fact worth
#: reporting, not one to route around.
_LISTEN_PORT = 28998
_UNUSED_PORT = 28999

#: How long the sleep test asks for, and the floor it holds the result to.
#: Windows' timer granularity returns a few milliseconds early, so the floor
#: sits below the request -- far enough above zero that a no-op still fails.
_SLEEP_SECONDS = 0.05
_SLEEP_FLOOR_SECONDS = 0.03


class TestReadingTheMachine:
    def test_the_platform_is_the_interpreters_own(self) -> None:
        assert _read_platform_impl() == sys.platform

    def test_the_executable_is_the_running_interpreter(self) -> None:
        """The planner runs in the environment the harness runs in, which is
        what removes ``poetry`` from a path that must work inside an image."""
        assert _read_executable_impl() == sys.executable

    def test_the_root_is_absolute_so_composed_paths_survive_a_cwd_change(self) -> None:
        """The engine runs with the GAME directory as its working directory,
        so every path handed to it is made absolute against this first."""
        root = _resolve_root_impl()
        assert Path(str(root)).is_absolute()

    def test_the_environment_carries_what_a_child_needs(self) -> None:
        environment = _read_environment_impl()
        key = "Path" if is_windows(sys.platform) else "PATH"
        assert key in {name.title() if is_windows(sys.platform) else name for name in environment}

    def test_the_environment_is_a_copy_a_caller_may_overlay(self) -> None:
        """A caller adds PYTHONPATH to it; reaching back into this process's
        own environment would change the launcher rather than the child."""
        first = dict(_read_environment_impl())
        first["RW_BOT_TEST_ONLY"] = "1"
        assert "RW_BOT_TEST_ONLY" not in _read_environment_impl()


class TestStamps:
    def test_a_stamp_is_eight_hex_characters(self) -> None:
        stamp = _new_stamp_impl()
        assert len(stamp) == 8
        assert set(stamp) <= set("0123456789abcdef")

    def test_two_stamps_differ(self) -> None:
        """Concurrent matches name their build directories by it, so two
        launches in one instant must not collide."""
        assert _new_stamp_impl() != _new_stamp_impl()


class TestSleeping:
    def test_it_actually_waits(self) -> None:
        """The tolerance is not slack, it is the platform: Windows' timer
        granularity lets ``time.sleep`` return a few milliseconds early, and
        this asserted the full interval once and failed at 0.047s against
        0.05. The floor still separates a real wait from a no-op by an order
        of magnitude, which is the property that matters -- the teardown pause
        exists so a file handle can be released before the delete.
        """
        started = time.monotonic()
        _sleep_impl(_SLEEP_SECONDS)
        elapsed = time.monotonic() - started
        assert elapsed >= _SLEEP_FLOOR_SECONDS


class TestRemovingPaths:
    def test_a_file_is_removed(self, tmp_path: Path) -> None:
        target = tmp_path / "rw-agent-play-x.jar"
        target.write_bytes(b"jar")
        _remove_path_impl(target)
        assert not target.exists()

    def test_a_whole_tree_is_removed(self, tmp_path: Path) -> None:
        """A per-match compile leaves a directory of class files."""
        classes = tmp_path / "play-x" / "rwbot" / "agent"
        classes.mkdir(parents=True)
        (classes / "Agent.class").write_bytes(b"cafebabe")
        _remove_path_impl(tmp_path / "play-x")
        assert not (tmp_path / "play-x").exists()

    def test_a_path_that_is_already_gone_is_not_an_error(self, tmp_path: Path) -> None:
        """This runs in teardown, where the interesting failure is the
        match's, not the cleanup's."""
        absent = tmp_path / "never-existed"
        _remove_path_impl(absent)
        assert not absent.exists()


class TestWaitingForTheChannel:
    def test_a_listening_socket_is_seen(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", _LISTEN_PORT))
        listener.listen(1)
        try:
            assert _wait_for_port_impl(_LISTEN_PORT, _WAIT_SECONDS, PORT_POLL_SECONDS) is None
        finally:
            listener.close()

    def test_a_closed_port_reports_the_reason_rather_than_a_bare_failure(self) -> None:
        """A refused connection means the engine is up and the agent never
        bound; a timeout with no route means the engine died during boot.
        Ninety seconds of silence used to report neither."""
        failure = _wait_for_port_impl(_UNUSED_PORT, 0.3, 0.05)
        if failure is None:
            raise AssertionError(f"port {_UNUSED_PORT} unexpectedly had a listener")
        assert "Error" in failure


#: A child that exits 0 only when the marker reached it, and 9 otherwise, so
#: the assertion is on the CHILD's view of its environment rather than on the
#: dictionary this process handed over.
_REPORT_MARKER = "import os, sys; sys.exit(0 if os.environ.get('RW_MARK') == 'y' else 9)"


class TestRunningAChild:
    def test_a_child_sees_the_environment_it_was_given(self) -> None:
        """This is how the frozen tree reaches the planner: the launcher adds
        one variable to the environment it forwards, and the child is the only
        thing that can confirm it arrived."""
        status = _run_inherited_impl(
            [sys.executable, "-c", _REPORT_MARKER],
            {**_read_environment_impl(), "RW_MARK": "y"},
        )
        assert status == 0

    def test_a_child_without_the_marker_reports_its_absence(self) -> None:
        """The negative half: without it the same child exits 9, so the test
        above is measuring the environment rather than always passing."""
        status = _run_inherited_impl(
            [sys.executable, "-c", _REPORT_MARKER], dict(_read_environment_impl())
        )
        assert status == 9

    def test_a_childs_status_is_returned_unchanged(self) -> None:
        """The planner's exit status IS the match's status."""
        status = _run_inherited_impl(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            dict(_read_environment_impl()),
        )
        assert status == 7


class TestSpawningTheEngine:
    def test_the_two_streams_land_in_their_own_files(self, tmp_path: Path) -> None:
        """A crash interleaved into a merged stream is a crash nobody found."""
        out = tmp_path / "deep" / "engine.log.agent"
        err = tmp_path / "deep" / "engine.log.err"
        program = "import sys; print('to-stdout'); print('to-stderr', file=sys.stderr)"
        child = _spawn_game_impl(
            [sys.executable, "-c", program], tmp_path, out, err, _read_environment_impl()
        )
        _await_exit(child)
        assert out.read_text(encoding="utf-8").strip() == "to-stdout"
        assert err.read_text(encoding="utf-8").strip() == "to-stderr"

    def test_the_engine_runs_in_the_directory_it_was_given(self, tmp_path: Path) -> None:
        """The engine writes three fixed-name paths inside its own directory,
        so the working directory IS the isolation between concurrent matches."""
        workdir = tmp_path / "clone"
        workdir.mkdir()
        out = tmp_path / "cwd.out"
        err = tmp_path / "cwd.err"
        program = "import os; open('here.txt','w').write('x')"
        child = _spawn_game_impl(
            [sys.executable, "-c", program], workdir, out, err, _read_environment_impl()
        )
        _await_exit(child)
        assert (workdir / "here.txt").exists()

    def test_a_spawned_child_reports_a_pid_and_is_running(self, tmp_path: Path) -> None:
        out = tmp_path / "live.out"
        err = tmp_path / "live.err"
        child = _spawn_game_impl(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            out,
            err,
            _read_environment_impl(),
        )
        try:
            assert child.pid > 0
            assert child.poll() is None
        finally:
            _kill_tree_impl(child.pid)


def _await_exit(child: SpawnedMatchProto) -> None:
    """Wait for a spawned child to finish.

    Polls the handle's own declared surface rather than reaching for
    ``Popen.wait``: the hook's contract is
    :class:`~rw_bot.harness._hook_protocols.SpawnedMatchProto`, and a test that
    used a method outside it would pass while the contract was unsatisfiable.

    Args:
        child: The handle :func:`_spawn_game_impl` returned.

    Raises:
        AssertionError: When it does not finish in time, which would leave the
            stream assertions reading a half-written file.
    """
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return
        time.sleep(0.02)
    raise AssertionError("the spawned child did not exit within 30s")


@pytest.mark.skipif(sys.platform != WINDOWS, reason="the priority class exists on Windows only")
def test_the_spawn_flags_are_accepted_by_the_standard_library() -> None:
    """The pair this package passes -- a priority class and a session flag --
    is legal on both platforms, which is what lets one call site serve both.
    Windows ignores the session; POSIX rejects any non-zero creation flag.
    """
    isolation = spawn_isolation(sys.platform)
    assert isolation["creationflags"] == subprocess.ABOVE_NORMAL_PRIORITY_CLASS
    assert isolation["start_new_session"] is False


class TestCopyingAnEntry:
    def test_a_file_lands_under_its_own_name(self, tmp_path: Path) -> None:
        source = tmp_path / "rw-agent.jar"
        source.write_bytes(b"cafebabe")
        (tmp_path / "tree").mkdir()
        _copy_entry_impl(source, tmp_path / "tree")
        assert (tmp_path / "tree" / "rw-agent.jar").read_bytes() == b"cafebabe"

    def test_a_tree_is_copied_whole(self, tmp_path: Path) -> None:
        (tmp_path / "doctrines" / "nested").mkdir(parents=True)
        (tmp_path / "doctrines" / "a.doctrine").write_bytes(b"attack")
        (tmp_path / "doctrines" / "nested" / "b.doctrine").write_bytes(b"defend")
        (tmp_path / "tree").mkdir()
        _copy_entry_impl(tmp_path / "doctrines", tmp_path / "tree")
        assert (tmp_path / "tree" / "doctrines" / "a.doctrine").read_bytes() == b"attack"
        assert (tmp_path / "tree" / "doctrines" / "nested" / "b.doctrine").exists()

    def test_bytecode_caches_are_not_copied(self, tmp_path: Path) -> None:
        """A ``.pyc`` is not source, and copying it makes a frozen tree's
        identity depend on whether anything imported a module before the
        freeze ran. 140 of one payload's 408 files were bytecode -- a third of
        it -- and a ``.pyc`` embeds the source's timestamp, so two freezes of
        identical source digested differently.
        """
        package = tmp_path / "rw_bot"
        (package / "__pycache__").mkdir(parents=True)
        (package / "planner.py").write_bytes(b"code")
        (package / "__pycache__" / "planner.cpython-311.pyc").write_bytes(b"bytecode")
        (tmp_path / "tree").mkdir()
        _copy_entry_impl(package, tmp_path / "tree")
        assert (tmp_path / "tree" / "rw_bot" / "planner.py").exists()
        assert not (tmp_path / "tree" / "rw_bot" / "__pycache__").exists()

    def test_a_nested_cache_is_excluded_too(self, tmp_path: Path) -> None:
        """The package this freezes has eight of them, at every level."""
        deep = tmp_path / "rw_bot" / "policy" / "__pycache__"
        deep.mkdir(parents=True)
        (deep / "doom.cpython-311.pyc").write_bytes(b"bytecode")
        (tmp_path / "rw_bot" / "policy" / "doom.py").write_bytes(b"code")
        (tmp_path / "tree").mkdir()
        _copy_entry_impl(tmp_path / "rw_bot", tmp_path / "tree")
        assert list((tmp_path / "tree").rglob("__pycache__")) == []
        assert (tmp_path / "tree" / "rw_bot" / "policy" / "doom.py").exists()

    def test_the_exclusion_names_what_it_drops(self) -> None:
        assert COPY_EXCLUDES == ("__pycache__",)
