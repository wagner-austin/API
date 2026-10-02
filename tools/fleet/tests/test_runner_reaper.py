"""The reaper that kills what a finished job left in its runner unit (MCPs board task 53528106).

These execute the rendered script itself under bash, over a process table
laid out as files: ``/proc/<pid>/stat`` and ``cmdline``, ``/proc/uptime``
and the unit's ``cgroup.procs``. Only its two root paths are pointed at that
table; ``systemctl``, ``kill`` and ``getconf`` are shell functions defined
before the script runs, which answer from the table and record what was
killed. The table mirrors lavender on 2026-10-02: a runner unit whose main
process is ``runsvc.sh``, under it the node service and ``Runner.Listener``,
and beside them processes a cancelled job reparented to init, one started
before the next job's Worker and one after it.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from typing_extensions import TypedDict

from fleet.cli.runners import load_runner_spec
from fleet.contracts.runners import RunnerInstall
from fleet.core import runner_reaper_render, runner_render
from tests._host_bash import host_bash

#: The unit every case runs.
UNIT = "actions.runner.wagner-austin-API.lavender-wsl.service"

#: The stand-ins, defined before the script runs. A command substitution
#: inherits functions, so ``$(getconf CLK_TCK)`` reaches the stand-in too.
STAND_INS = """
systemctl() {
    case "$1" in
        list-units) cat "$TABLE/units" ;;
        show) cat "$TABLE/show/$5.$3" ;;
    esac
}
kill() {
    if [ -e "$TABLE/unkillable/$2" ] || [ ! -e "$TABLE/proc/$2" ]; then
        return 1
    fi
    echo "$2" >> "$TABLE/killed"
    rm -rf "$TABLE/proc/$2"
}
getconf() {
    echo 100
}
"""


class Process(TypedDict):
    """One row of the laid-out process table.

    Attributes:
        pid: Its id.
        ppid: Its parent's.
        cmdline: Its command line.
        start_ticks: When it started, in clock ticks since boot; the table's
            uptime is 1000 s at 100 ticks a second.
    """

    pid: int
    ppid: int
    cmdline: str
    start_ticks: int


#: The unit's own processes: main, the node service, the Listener.
OWN: tuple[Process, ...] = (
    Process(pid=10, ppid=1, cmdline="/bin/bash ./runsvc.sh", start_ticks=1000),
    Process(pid=11, ppid=10, cmdline="node ./bin/RunnerService.js", start_ticks=1000),
    Process(pid=12, ppid=11, cmdline="./bin/Runner.Listener run", start_ticks=1000),
)

#: What a cancelled job left: a forkserver reparented to init, 900 s old,
#: and its worker, 50 s old.
LEFT: tuple[Process, ...] = (
    Process(pid=20, ppid=1, cmdline="python -m forkserver", start_ticks=10000),
    Process(pid=21, ppid=20, cmdline="python pt_data_worker", start_ticks=95000),
)

#: A job's worker under the Listener.
WORKER = Process(pid=13, ppid=12, cmdline="./bin/Runner.Worker spawnclient", start_ticks=90000)


def _lay_out(table: pathlib.Path, processes: tuple[Process, ...], *, main: int = 10) -> None:
    """Write the process table and the unit's systemd answers.

    Args:
        table: The directory to lay it out in.
        processes: The unit's processes, in cgroup order.
        main: What systemd reports as the unit's MainPID.
    """
    for process in processes:
        directory = table / "proc" / str(process["pid"])
        directory.mkdir(parents=True)
        after = ["S", str(process["ppid"]), *(["0"] * 17), str(process["start_ticks"])]
        name = process["cmdline"].split()[0]
        (directory / "stat").write_bytes(f"{process['pid']} ({name}) {' '.join(after)}\n".encode())
        (directory / "cmdline").write_bytes(process["cmdline"].replace(" ", "\0").encode())
    (table / "proc" / "uptime").write_bytes(b"1000.00 4000.00\n")
    cgroup = table / "cgroup" / "runners.slice" / UNIT
    cgroup.mkdir(parents=True)
    (cgroup / "cgroup.procs").write_bytes(
        "".join(f"{process['pid']}\n" for process in processes).encode()
    )
    show = table / "show"
    show.mkdir()
    (show / f"{UNIT}.ControlGroup").write_bytes(f"/runners.slice/{UNIT}\n".encode())
    (show / f"{UNIT}.MainPID").write_bytes(f"{main}\n".encode())
    (table / "units").write_bytes(f"{UNIT} loaded active running GitHub Actions Runner\n".encode())
    (table / "unkillable").mkdir()


def _run(table: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the rendered reaper over the table.

    Args:
        table: The laid-out table.
        *arguments: The reaper's own arguments.

    Returns:
        The finished run.
    """
    script = runner_reaper_render.REAPER_SCRIPT
    for real, fake in (("cgroup_root=/sys/fs/cgroup", "cgroup"), ("proc_root=/proc", "proc")):
        assert f"\n{real}\n" in script
        script = script.replace(f"\n{real}\n", f'\n{real.split("=")[0]}="$TABLE/{fake}"\n')
    (table / "reaper.sh").write_bytes(script.encode())
    (table / "stand-ins.sh").write_bytes(STAND_INS.encode())
    return subprocess.run(
        [
            host_bash(),
            "-c",
            'TABLE="$PWD"; source stand-ins.sh; source reaper.sh "$@"',
            "x",
            *arguments,
        ],
        cwd=table,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def _killed(table: pathlib.Path) -> list[str]:
    """The pids the run killed, in order.

    Args:
        table: The laid-out table.

    Returns:
        The pids, empty when nothing was killed.
    """
    record = table / "killed"
    return record.read_text(encoding="utf-8").split() if record.exists() else []


class TestAReapPass:
    def test_it_kills_what_a_finished_job_left_and_nothing_of_the_unit_s_own(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN + LEFT)

        ran = _run(tmp_path)

        assert (ran.returncode, ran.stderr) == (0, "")
        assert ran.stdout == (
            f"fleet-runner-reaper: {UNIT}: killed 2 process(es) a finished job left behind\n"
        )
        assert _killed(tmp_path) == ["20", "21"]

    def test_a_busy_unit_loses_only_what_started_before_its_worker(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The Worker started at 90000 ticks: the forkserver at 10000 is a
        job before it, the worker process at 95000 is this job's own."""
        _lay_out(tmp_path, (*OWN, WORKER, *LEFT))

        ran = _run(tmp_path)

        assert (ran.returncode, ran.stderr) == (0, "")
        assert ran.stdout == (
            f"fleet-runner-reaper: {UNIT}: killed 1 process(es) a finished job left behind\n"
        )
        assert _killed(tmp_path) == ["20"]

    def test_a_busy_unit_keeps_a_process_whose_start_it_cannot_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER, *LEFT))
        (tmp_path / "proc" / "20" / "stat").unlink()

        ran = _run(tmp_path)

        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (0, "", [])

    def test_a_worker_whose_start_cannot_be_read_leaves_the_unit_idle(
        self, tmp_path: pathlib.Path
    ) -> None:
        """It exited after cgroup.procs listed it, so its job is over, and
        a kill of its pid fails as a kill of a gone process does."""
        _lay_out(tmp_path, (*OWN, WORKER, *LEFT))
        for entry in (tmp_path / "proc" / "13").iterdir():
            entry.unlink()
        (tmp_path / "proc" / "13").rmdir()

        ran = _run(tmp_path)

        assert ran.returncode == 0
        assert ran.stdout == (
            f"fleet-runner-reaper: {UNIT}: killed 2 process(es) a finished job left behind\n"
        )
        assert _killed(tmp_path) == ["20", "21"]

    def test_a_unit_mid_self_update_is_left_alone(self, tmp_path: pathlib.Path) -> None:
        """The Listener has exited to let _update.sh run outside the tree."""
        _lay_out(tmp_path, (*OWN[:2], *LEFT))

        ran = _run(tmp_path)

        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (0, "", [])

    def test_a_stopped_unit_is_skipped(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN + LEFT)
        (tmp_path / "show" / f"{UNIT}.ControlGroup").write_bytes(b"\n")

        ran = _run(tmp_path)

        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (0, "", [])

    def test_a_unit_with_no_main_process_is_skipped(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN + LEFT, main=0)

        ran = _run(tmp_path)

        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (0, "", [])

    def test_a_leftover_that_will_not_die_stops_the_pass_by_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, OWN + LEFT)
        (tmp_path / "unkillable" / "20").write_bytes(b"")

        ran = _run(tmp_path)

        assert ran.returncode == 1
        assert ran.stdout == (
            f"fleet-runner-reaper: {UNIT}: could not kill 20, a finished job's process\n"
        )

    def test_a_leftover_that_exited_before_its_kill_is_not_counted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Listed in cgroup.procs, gone from /proc by the time kill runs."""
        _lay_out(tmp_path, OWN + LEFT)
        (tmp_path / "unkillable" / "21").write_bytes(b"")
        for entry in (tmp_path / "proc" / "21").iterdir():
            entry.unlink()
        (tmp_path / "proc" / "21").rmdir()

        ran = _run(tmp_path)

        assert ran.returncode == 0
        assert ran.stdout == (
            f"fleet-runner-reaper: {UNIT}: killed 1 process(es) a finished job left behind\n"
        )
        assert _killed(tmp_path) == ["20"]


def _install() -> RunnerInstall:
    """The WSL install the render cases place.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/API",
        runner_name="lavender-wsl",
        side="wsl",
        service=UNIT,
        workdir="/home/gharunner/actions-runner-api-1/_work",
        labels=["lavender-wsl"],
        python_toolcache=[],
    )


class TestTheRender:
    def test_the_drop_in_ends_the_whole_group_and_is_written_only_when_it_differs(
        self,
    ) -> None:
        lines = runner_reaper_render.render_kill_mode_lines(_install())
        path = f"/etc/systemd/system/{UNIT}.d/fleet-kill.conf"
        assert runner_reaper_render.KILL_DROP_IN == "[Service]\nKillMode=control-group\n"
        assert lines == [
            f'if [ "$(cat {path} 2>/dev/null)" != '
            "\"$(printf '[Service]\\nKillMode=control-group\\n')\" ]; then",
            f"    mkdir -p /etc/systemd/system/{UNIT}.d",
            f"    printf '[Service]\\nKillMode=control-group\\n' > {path}",
            "    systemctl daemon-reload",
            f"    echo 'kill mode set for {UNIT}: a stop ends its whole cgroup'",
            "fi",
        ]

    def test_an_unscriptable_unit_name_is_refused(self) -> None:
        install = _install()
        install["service"] = "bad'unit.service"
        with pytest.raises(ValueError, match="service"):
            runner_reaper_render.render_kill_mode_lines(install)

    def test_the_reaper_and_its_units_are_written_whole_and_the_timer_started(self) -> None:
        lines = runner_reaper_render.render_reaper_lines()
        assert lines[0] == "cat > /usr/local/sbin/fleet-runner-reaper <<'REAPER_EOF'"
        assert lines[1] == runner_reaper_render.REAPER_SCRIPT.rstrip("\n")
        assert lines[2:4] == ["REAPER_EOF", "chmod +x /usr/local/sbin/fleet-runner-reaper"]
        assert runner_reaper_render.REAPER_SERVICE.rstrip("\n") in lines
        assert runner_reaper_render.REAPER_TIMER.rstrip("\n") in lines
        assert lines[-2:] == [
            "systemctl daemon-reload",
            "systemctl enable --now fleet-runner-reaper.timer",
        ]

    def test_the_timer_runs_the_service_every_thirty_seconds(self) -> None:
        assert "OnUnitActiveSec=30s\n" in runner_reaper_render.REAPER_TIMER
        assert "WantedBy=timers.target\n" in runner_reaper_render.REAPER_TIMER
        assert (
            "ExecStart=/usr/local/sbin/fleet-runner-reaper\n" in runner_reaper_render.REAPER_SERVICE
        )

    def test_the_provision_installs_the_reaper_before_any_runner(self) -> None:
        roster = load_runner_spec(str(pathlib.Path(__file__).parents[1] / "runners.json"))
        script = runner_render.render_provision(roster["hosts"][0])["linux_script"]
        assert script.index("enable --now fleet-runner-reaper.timer") < script.index(
            "# --- runner installs ---"
        )
        # Twice per WSL runner: the comparison and the write.
        assert script.count("fleet-kill.conf") == 2 * len(
            [i for i in roster["hosts"][0]["installs"] if i["side"] == "wsl"]
        )


class TestTheAuditMode:
    def test_it_counts_only_leftovers_older_than_the_bound_and_kills_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """900 s and 50 s old against a 60 s bound: one."""
        _lay_out(tmp_path, OWN + LEFT)

        ran = _run(tmp_path, "--audit", "60", UNIT)

        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (0, "1\n", [])

    def test_a_leftover_gone_from_proc_is_not_counted(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, OWN + LEFT)
        (tmp_path / "proc" / "20" / "stat").unlink()

        ran = _run(tmp_path, "--audit", "0", UNIT)

        assert (ran.returncode, ran.stdout) == (0, "1\n")

    def test_a_busy_unit_counts_only_what_started_before_its_worker(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*OWN, WORKER, *LEFT))

        ran = _run(tmp_path, "--audit", "0", UNIT)

        assert (ran.returncode, ran.stdout, _killed(tmp_path)) == (0, "1\n", [])
