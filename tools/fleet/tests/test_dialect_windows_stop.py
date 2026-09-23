"""How a Windows node stops a dispatch: the build's whole tree, by verified id.

MEASURED BEFORE IT WAS WRITTEN (MCPs board task fd5cabfa). On sedona,
2026-09-23, a probe task whose ``build.ps1`` ran a native child read parent
alive=False, child alive=True after ``Stop-ScheduledTask``: stopping a task
ends the process it started and nothing below it, which is how a cancelled
slime run's vitest held sedona for four hours. The same probe with the build
recording ``$PID`` and the stop running ``taskkill /PID <pid> /T /F`` ended the
parent, its child and its grandchild.

The text tests pin the order and the guard; the class below RUNS the stop
script on this hub, which is a Windows machine, against a real process tree
started the way a node starts one, so the claim that the tree dies is
measured here on every run rather than remembered from sedona.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

import pytest

from fleet.core import names
from fleet.core.dialect_windows import POWERSHELL_INVOCATION, WindowsDialect
from tests.conftest import DEMO_RUN_ID

DIALECT = WindowsDialect()

#: How long a test waits for a process it started to write its id.
STARTUP_SECONDS = 60


def _stop(target: str) -> str:
    """Render the stop script for the fixture's run under ``target``.

    Args:
        target: The dispatch's directory.

    Returns:
        The script's text.
    """
    return DIALECT.stop_script(target=target, run_id=DEMO_RUN_ID)


class TestTheBuildRecordsItself:
    def test_the_build_writes_its_process_id_before_anything_else(self) -> None:
        """First, so a stop that arrives during a slow install still finds
        the id; a build that recorded it later would be unstoppable for as
        long as ``npm ci`` ran."""
        body = DIALECT.build_script(
            target="C:/s/run-1", path="", workers=2, install=(("npm", "ci"),), cache_root="C:/c"
        )

        assert body.splitlines()[0] == (
            f"$PID | Set-Content -LiteralPath 'C:/s/run-1/{names.PID_NAME}'"
        )


class TestTheStopScriptText:
    def test_the_tree_is_ended_before_the_task_is_stopped(self) -> None:
        """Stopping the task first would end the build process and orphan
        its children, leaving nothing whose tree ``taskkill /T`` could walk."""
        body = _stop("C:/s/run-1")

        assert body.index("taskkill.exe /PID $buildPid /T /F") < body.index("Stop-ScheduledTask")
        assert body.index("Stop-ScheduledTask") < body.index("Unregister-ScheduledTask")

    def test_only_a_process_running_this_dispatchs_build_is_killed(self) -> None:
        body = _stop("C:/s/run-1")

        assert f"Test-Path -LiteralPath 'C:/s/run-1/{names.PID_NAME}'" in body
        assert "-like '*C:/s/run-1/build.ps1*'" in body

    def test_a_kill_that_fails_fails_the_script(self) -> None:
        assert 'throw "taskkill of $buildPid exited $LASTEXITCODE"' in _stop("C:/s/run-1")

    def test_the_task_name_is_the_one_the_launch_registers(self) -> None:
        """Both come from names.task_name, so a rename cannot make a stop
        report success having stopped nothing."""
        launched = DIALECT.launch_script(target="C:/s/run-1", run_id=DEMO_RUN_ID)

        assert names.task_name(DEMO_RUN_ID) in launched
        assert names.task_name(DEMO_RUN_ID) in _stop("C:/s/run-1")

    def test_the_stop_script_never_prompts(self) -> None:
        """There is nobody at the node to answer, and a prompt would hang."""
        assert "-Confirm:$false" in _stop("C:/s/run-1")


def _alive(pid: int) -> bool:
    """Whether a process with this id exists, asked of PowerShell.

    Args:
        pid: The process id.

    Returns:
        True while it runs.
    """
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[bool](Get-Process -Id {pid} -ErrorAction SilentlyContinue)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip() == "True"


def _await_id(path: pathlib.Path) -> int:
    """Wait for a started process to write its id, and read it.

    Args:
        path: The file it writes.

    Returns:
        The id.

    Raises:
        AssertionError: When nothing is written within
            :data:`STARTUP_SECONDS`.
    """
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8-sig").strip().isdigit():
            return int(path.read_text(encoding="utf-8-sig").strip())
        time.sleep(0.2)
    raise AssertionError(f"{path} was not written within {STARTUP_SECONDS}s")


def _run_stop(tmp_path: pathlib.Path, target: str) -> subprocess.CompletedProcess[str]:
    """Write the stop script beside the run and execute it the way ssh does.

    Args:
        tmp_path: Where the script is written.
        target: The dispatch's directory.

    Returns:
        The finished PowerShell process.
    """
    script = tmp_path / "stop.ps1"
    script.write_text(_stop(target), encoding="utf-8")
    return subprocess.run(
        [*POWERSHELL_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="the stop is PowerShell; run it here")
class TestTheStopForReal:
    def test_it_ends_the_build_and_the_child_it_is_waiting_on(self, tmp_path: pathlib.Path) -> None:
        """The case sedona measured: a build blocked on a native child. The
        build's first line is the real build script's own, so the id is
        written where the stop reads it."""
        target = tmp_path.as_posix()
        child_id = tmp_path / "child.pid"
        first_line = DIALECT.build_script(
            target=target, path="", workers=1, install=(), cache_root=target
        ).splitlines()[0]
        (tmp_path / "build.ps1").write_text(
            f"{first_line}\n"
            f"& powershell -NoProfile -Command "
            f"\"`$PID | Set-Content -LiteralPath '{child_id.as_posix()}'; Start-Sleep 300\"\n",
            encoding="utf-8",
        )
        build = subprocess.Popen([*POWERSHELL_INVOCATION, f"{target}/build.ps1"])
        child = 0
        try:
            recorded = _await_id(tmp_path / names.PID_NAME)
            child = _await_id(child_id)
            assert recorded == build.pid
            assert _alive(child)

            stopped = _run_stop(tmp_path, target)

            assert stopped.returncode == 0, stopped.stderr
            assert build.wait(timeout=60) != 0
            assert not _alive(child)
        finally:
            for pid in (build.pid, child):
                if pid != 0 and _alive(pid):
                    subprocess.run(
                        ["taskkill.exe", "/PID", str(pid), "/T", "/F"], check=False, timeout=60
                    )

    def test_a_recorded_id_that_names_another_process_kills_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Ids are reused, so an id outliving its build may name a stranger;
        here it names this test's own interpreter, which must survive."""
        (tmp_path / names.PID_NAME).write_text(f"{os.getpid()}\n", encoding="utf-8")

        stopped = _run_stop(tmp_path, tmp_path.as_posix())

        assert stopped.returncode == 0, stopped.stderr
        assert stopped.stdout.strip() == f"stopped {names.task_name(DEMO_RUN_ID)}"
        assert _alive(os.getpid())

    def test_a_build_that_never_recorded_an_id_is_stopped_by_task_alone(
        self, tmp_path: pathlib.Path
    ) -> None:
        stopped = _run_stop(tmp_path, tmp_path.as_posix())

        assert stopped.returncode == 0, stopped.stderr
        assert stopped.stdout.strip() == f"stopped {names.task_name(DEMO_RUN_ID)}"
