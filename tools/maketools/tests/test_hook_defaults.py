"""The real hook bindings, exercised against real children, files and git.

Nothing here is faked: the point is that the production bindings do what
the fakes claim they do.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from maketools import _test_hooks
from maketools.cli import COMMANDS, repository_root

LAUNCHER = repository_root() / "tools" / "maketools" / "scripts" / "run.py"

#: Wall for the children these cases start, every one of which exits at
#: once. Generous so a loaded box never fails a case that is not about
#: timing; the case that IS about timing passes its own tiny wall.
PROBE_WALL_SECONDS = 120


def test_run_inheriting_returns_the_childs_status(tmp_path: Path) -> None:
    code = _test_hooks.run_inheriting(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        cwd=tmp_path,
        env=_test_hooks.environ(),
        new_session=sys.platform != "win32",
        timeout_seconds=PROBE_WALL_SECONDS,
    )
    assert code == 7


def test_run_inheriting_fells_a_child_that_outlives_its_wall(tmp_path: Path) -> None:
    """WATCH THE BOUND BITE, on a real child that really sleeps.

    Every other case here runs a child that exits immediately, so all of them
    would pass with no deadline at all. This one is the only evidence that
    the wall is wired to the call rather than merely present in the
    signature.

    It RAISES rather than returning a status: an uncaptured child has no
    output to hand back alongside a number, and a caller reading one would
    not know the run had been cut short.
    """
    with pytest.raises(subprocess.TimeoutExpired):
        _test_hooks.run_inheriting(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=_test_hooks.environ(),
            new_session=False,
            timeout_seconds=1,
        )


def test_run_inheriting_passes_the_environment_and_cwd(tmp_path: Path) -> None:
    marker = tmp_path / "seen.txt"
    environment = _test_hooks.environ()
    environment["MAKETOOLS_PROBE"] = "yes"
    code = _test_hooks.run_inheriting(
        [
            sys.executable,
            "-c",
            "import os, pathlib; "
            "pathlib.Path('seen.txt').write_text(os.environ['MAKETOOLS_PROBE'])",
        ],
        cwd=tmp_path,
        env=environment,
        new_session=False,
        timeout_seconds=PROBE_WALL_SECONDS,
    )
    assert code == 0
    assert marker.read_text() == "yes"


def test_run_capturing_collects_both_streams_and_the_status(tmp_path: Path) -> None:
    result = _test_hooks.run_capturing(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('out'); sys.stderr.write('err'); sys.exit(3)",
        ],
        cwd=tmp_path,
    )
    assert result == {"returncode": 3, "stdout": "out", "stderr": "err"}


def test_the_clock_sleep_pid_and_token_answer() -> None:
    before = _test_hooks.now()
    _test_hooks.sleep(0.01)
    assert _test_hooks.now() >= before
    assert _test_hooks.process_id() > 0
    token = _test_hooks.token()
    assert len(token) == 8
    assert token != _test_hooks.token()


def test_draw_stays_inside_the_inclusive_range_and_covers_it() -> None:
    assert _test_hooks.draw(7, 7) == 7
    seen = {_test_hooks.draw(1, 3) for _ in range(200)}
    assert seen == {1, 2, 3}


def test_platform_is_the_interpreters() -> None:
    assert _test_hooks.platform() == sys.platform


def test_the_writers_reach_the_streams(capsys: pytest.CaptureFixture[str]) -> None:
    _test_hooks.write_line("to stdout")
    _test_hooks.write_error("to stderr")
    captured = capsys.readouterr()
    assert captured.out == "to stdout\n"
    assert captured.err == "to stderr\n"


def test_environ_is_a_copy() -> None:
    first = _test_hooks.environ()
    first["MAKETOOLS_MUTATED"] = "1"
    assert "MAKETOOLS_MUTATED" not in _test_hooks.environ()


def test_the_real_process_table_lists_this_process() -> None:
    rows = _test_hooks.process_table()
    me = [r for r in rows if r["pid"] == _test_hooks.process_id()]
    assert len(me) == 1
    assert me[0]["name"] != ""
    assert me[0]["parent_pid"] > 0
    assert me[0]["created_unix"] > 0
    assert _test_hooks.process_alive(_test_hooks.process_id())
    assert not _test_hooks.process_alive(max(r["pid"] for r in rows) + 1_000_000)


def test_kill_ends_a_real_child_and_refuses_a_gone_one() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _test_hooks.kill(child.pid)
    assert child.wait(timeout=10) != 0
    with pytest.raises(OSError):
        _test_hooks.kill(child.pid)


def test_remove_tree_and_remove_file_delete(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "inner").mkdir(parents=True)
    (tree / "inner" / "f").write_bytes(b"")
    _test_hooks.remove_tree(tree)
    assert not tree.exists()
    lone = tmp_path / "lone"
    lone.write_bytes(b"")
    _test_hooks.remove_file(lone)
    assert not lone.exists()


def test_tracked_files_asks_git_and_refuses_outside_a_repository(tmp_path: Path) -> None:
    root = repository_root()
    found = _test_hooks.tracked_files(root, "*Makefile")
    assert root / "Makefile" in found
    assert root / "libs" / "platform_core" / "Makefile" in found
    assert all(path.is_absolute() for path in found)
    with pytest.raises(RuntimeError, match=r"git ls-files failed in"):
        _test_hooks.tracked_files(tmp_path, "*Makefile")


@pytest.mark.skipif(sys.platform == "win32", reason="/proc exists off Windows only")
def test_the_linux_table_reader_reads_the_real_proc_root() -> None:
    assert any(r["pid"] == _test_hooks.process_id() for r in _test_hooks._linux_table())


@pytest.mark.skipif(sys.platform != "win32", reason="there is no /proc on Windows")
def test_the_linux_table_reader_has_no_proc_root_on_windows() -> None:
    with pytest.raises(FileNotFoundError):
        _test_hooks._linux_table()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell exists on Windows only")
def test_the_windows_table_reader_runs_powershell() -> None:
    assert any(r["pid"] == _test_hooks.process_id() for r in _test_hooks._windows_table())


@pytest.mark.skipif(sys.platform == "win32", reason="the refusal is the off-Windows behaviour")
def test_the_windows_table_reader_has_no_powershell_off_windows() -> None:
    with pytest.raises(FileNotFoundError):
        _test_hooks._windows_table()


def test_the_launcher_runs_as_a_child_and_forwards_the_exit_code(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "no-such-command"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr == (
        f"MAKETOOLS_USAGE: unknown command 'no-such-command'; one of {sorted(COMMANDS)}\n"
    )
    ok = subprocess.run(
        [sys.executable, str(LAUNCHER), "guard"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0
    assert ok.stdout == (
        f"guard: not applicable: {tmp_path.name} has no scripts/guard.py or "
        "scripts/guard/__main__.py, 0 rules run\n"
    )
