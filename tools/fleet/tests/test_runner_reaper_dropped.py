"""The reaper's ``--dropped`` mode: one Worker GitHub dropped, ended by pid.

MCPs board task 715b9ec5, A2. Two halves. The first runs the rendered script
over the laid-out table of :mod:`tests.test_runner_reaper`, whose ``kill``
stand-in records what it
would kill, so every refusal and every path through the mode is seen without
a signal sent. The second, bound to Linux (:mod:`tests._host`), runs it
against a REAL process tree: a listener started as ``.../bin/Runner.Listener``,
its Worker as ``.../bin/Runner.Worker`` and the Worker's two children, real
pids read through the real ``/proc``, with only the unit's cgroup listing laid
out and ``systemctl`` answered, so the kill the hub's reap sends is the one
that runs.
"""

from __future__ import annotations

import pathlib
import subprocess
import time

import pytest

from fleet.core import runner_reaper_render
from tests._host_bash import host_bash
from tests.test_runner_reaper import (
    OWN,
    SCOPE,
    UNIT,
    WORKER,
    Process,
    _killed,
    _lay_out,
    _run,
    _scope,
)

#: The Worker's step, and the step's own child.
STEP = Process(pid=14, ppid=13, cmdline="python -m pytest", start_ticks=91000)
GRANDCHILD = Process(pid=15, ppid=14, cmdline="node vitest", start_ticks=92000)

#: What the job daemonized, reparented to init: no descendant of the Worker,
#: which the timer's pass reaps once no Worker is alive.
DAEMON = Process(pid=22, ppid=1, cmdline="python forkserver", start_ticks=95000)

#: The Worker's start, as the hub read it.
STARTED = str(WORKER["start_ticks"])


def _ending(pid: int, cmdline: str) -> str:
    """The line the mode writes for a pid before any signal.

    Args:
        pid: The pid.
        cmdline: Its command line.

    Returns:
        The line.
    """
    return f"fleet-runner-reaper: {UNIT}: ending pid {pid}: {cmdline}\n"


class TestOverTheTable:
    def test_the_worker_and_its_descendants_are_ended_and_the_listener_is_not(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER, STEP, GRANDCHILD, DAEMON))
        ran = _run(tmp_path, "--dropped", UNIT, "13", STARTED)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            _ending(13, WORKER["cmdline"])
            + _ending(14, STEP["cmdline"])
            + _ending(15, GRANDCHILD["cmdline"])
            + f"fleet-runner-reaper: {UNIT}: ended Runner.Worker 13 and its tree, 3 process(es), "
            "whose job GitHub dropped; its listener runs on\n"
        )
        assert _killed(tmp_path) == ["13", "14", "15"]
        assert not (tmp_path / "slept").exists()

    def test_a_worker_in_its_job_scope_is_ended_there(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN)
        _scope(tmp_path, SCOPE, (WORKER, STEP))
        ran = _run(tmp_path, "--dropped", UNIT, "13", STARTED)
        assert ran.returncode == 0, ran.stderr
        assert _killed(tmp_path) == ["13", "14"]

    @pytest.mark.parametrize(
        ("pid", "started"), [("13", "90001"), ("99", STARTED)], ids=["pid-reused", "gone"]
    )
    def test_a_worker_no_longer_there_is_said_and_nothing_is_killed(
        self, tmp_path: pathlib.Path, pid: str, started: str
    ) -> None:
        """The pid names another process now, or none: the job ended between
        the hub's reading and this."""
        _lay_out(tmp_path, (*OWN, WORKER, STEP))
        ran = _run(tmp_path, "--dropped", UNIT, pid, started)
        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (
            0,
            f"fleet-runner-reaper: {UNIT}: Runner.Worker {pid}, started at {started}, has ended\n",
            [],
        )

    def test_a_stopped_unit_has_nothing_to_end(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, (*OWN, WORKER))
        (tmp_path / "show" / f"{UNIT}.ControlGroup").write_bytes(b"\n")
        ran = _run(tmp_path, "--dropped", UNIT, "13", STARTED)
        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (
            0,
            f"fleet-runner-reaper: {UNIT} runs in no cgroup, so Runner.Worker 13 ended with it\n",
            [],
        )

    def test_an_unreadable_unit_cgroup_is_refused(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, (*OWN, WORKER))
        (tmp_path / "cgroup" / "runners.slice" / UNIT / "cgroup.procs").unlink()
        ran = _run(tmp_path, "--dropped", UNIT, "13", STARTED)
        assert (ran.returncode, ran.stderr, _killed(tmp_path)) == (
            1,
            f"FLEET_RUNNER_REAPER_REFUSED: {UNIT}'s cgroup /runners.slice/{UNIT} cannot be read\n",
            [],
        )

    def test_a_pid_that_is_no_worker_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Named by the hub as a Worker, the process at that pid and start is
        the listener: the hub's reading is wrong, and nothing is killed."""
        _lay_out(tmp_path, (*OWN, WORKER))
        ran = _run(tmp_path, "--dropped", UNIT, "12", "1000")
        assert (ran.returncode, ran.stderr, _killed(tmp_path)) == (
            1,
            f"FLEET_RUNNER_REAPER_REFUSED: {UNIT}'s pid 12, started at 1000, is no Runner.Worker\n",
            [],
        )

    def test_a_process_that_outlives_sigkill_is_refused_after_the_wait(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER, STEP))
        (tmp_path / "unkillable" / "14").write_bytes(b"")
        ran = _run(tmp_path, "--dropped", UNIT, "13", STARTED)
        assert (ran.returncode, ran.stderr) == (
            1,
            f"FLEET_RUNNER_REAPER_REFUSED: {UNIT}: pid(s) 14 of Runner.Worker 13's tree outlived "
            "SIGKILL\n",
        )
        wait = runner_reaper_render.DROPPED_KILL_WAIT_SECONDS
        assert (tmp_path / "slept").read_text(encoding="utf-8").split() == ["1"] * wait

    def test_a_zombie_awaiting_its_parent_is_ended(self, tmp_path: pathlib.Path) -> None:
        """A killed Worker stays in /proc as a zombie until its listener
        reaps it, and is not a survivor."""
        _lay_out(tmp_path, (*OWN, WORKER))
        (tmp_path / "unkillable" / "13").write_bytes(b"")
        stat = tmp_path / "proc" / "13" / "stat"
        stat.write_bytes(stat.read_bytes().replace(b") S ", b") Z "))
        ran = _run(tmp_path, "--dropped", UNIT, "13", STARTED)
        assert ran.returncode == 0, ran.stderr
        assert not (tmp_path / "slept").exists()

    @pytest.mark.parametrize(
        "arguments",
        [(), (UNIT, "13"), (UNIT, "13", STARTED, "x"), (UNIT, "1a", STARTED), (UNIT, "13", "")],
    )
    def test_a_malformed_call_is_refused_with_its_usage(
        self, tmp_path: pathlib.Path, arguments: tuple[str, ...]
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER))
        ran = _run(tmp_path, "--dropped", *arguments)
        assert (ran.returncode, ran.stderr, _killed(tmp_path)) == (
            2,
            "fleet-runner-reaper: usage: fleet-runner-reaper --dropped UNIT PID START\n",
            [],
        )


#: How long the real tree may take to start, and to be seen gone.
TREE_DEADLINE_SECONDS = 10.0


def _children(pid: int) -> list[int]:
    """A real process's children, as the kernel lists them.

    Args:
        pid: The process.

    Returns:
        Its children's pids.
    """
    text = pathlib.Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    return [int(word) for word in text.split()]


def _alive(pid: int) -> bool:
    """Whether a real process runs, a zombie counting as ended.

    Args:
        pid: The process.

    Returns:
        True while it exists in a state other than Z.
    """
    stat = pathlib.Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    return stat.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()[0] != "Z"


@pytest.mark.host_linux
class TestAgainstARealTree:
    def test_the_worker_s_real_tree_dies_and_its_listener_runs_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (tmp_path / "worker.sh").write_text("sleep 300 & sleep 300 & wait\n", encoding="utf-8")
        # The listener outlives its Worker as a real one does: once the Worker
        # is reaped, it becomes a sleep under the same pid.
        (tmp_path / "listener.sh").write_text(
            f'(exec -a "{bin_dir}/Runner.Worker" bash "{tmp_path}/worker.sh") &\n'
            "wait\nexec sleep 300\n",
            encoding="utf-8",
        )
        listener = subprocess.Popen(
            [
                host_bash(),
                "-c",
                f'exec -a "{bin_dir}/Runner.Listener" bash "{tmp_path}/listener.sh"',
            ],
            stdin=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + TREE_DEADLINE_SECONDS
            tree: list[int] = []
            while time.monotonic() < deadline:
                workers = _children(listener.pid)
                tree = [*workers, *(child for worker in workers for child in _children(worker))]
                if len(tree) == 3:
                    break
                time.sleep(0.05)
            assert len(tree) == 3, f"the tree did not start: {tree}"
            worker = tree[0]
            started = (
                pathlib.Path(f"/proc/{worker}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1]
            ).split()[19]
            cgroup = tmp_path / "cgroup" / "listeners.slice" / UNIT
            cgroup.mkdir(parents=True)
            (cgroup / "cgroup.procs").write_text(
                "".join(f"{pid}\n" for pid in (listener.pid, *tree)), encoding="utf-8"
            )
            script = runner_reaper_render.REAPER_SCRIPT.replace(
                "\ncgroup_root=/sys/fs/cgroup\n", f'\ncgroup_root="{tmp_path}/cgroup"\n'
            )
            (tmp_path / "reaper.sh").write_text(script, encoding="utf-8")
            (tmp_path / "stand-ins.sh").write_text(
                f'systemctl() {{\n    echo "/listeners.slice/{UNIT}"\n}}\n', encoding="utf-8"
            )
            ran = subprocess.run(
                [
                    host_bash(),
                    "-c",
                    'source stand-ins.sh; source reaper.sh "$@"',
                    "x",
                    "--dropped",
                    UNIT,
                    str(worker),
                    started,
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=60,
            )
            assert ran.returncode == 0, ran.stderr
            lines = ran.stdout.splitlines()
            # Every pid's argv is written before the kill, the Worker's first.
            assert lines[0] == (
                f"fleet-runner-reaper: {UNIT}: ending pid {worker}: "
                f"{bin_dir}/Runner.Worker {tmp_path}/worker.sh "
            )
            assert [line.split(": ")[2] for line in lines[:3]] == [
                f"ending pid {pid}" for pid in tree
            ]
            assert lines[3] == (
                f"fleet-runner-reaper: {UNIT}: ended Runner.Worker {worker} and its tree, "
                "3 process(es), whose job GitHub dropped; its listener runs on"
            )
            deadline = time.monotonic() + TREE_DEADLINE_SECONDS
            while time.monotonic() < deadline and any(_alive(pid) for pid in tree):
                time.sleep(0.05)
            assert [pid for pid in tree if _alive(pid)] == []
            assert listener.poll() is None
        finally:
            listener.kill()
            listener.wait(timeout=TREE_DEADLINE_SECONDS)
