"""Each job outside its runner's listener (MCPs board task 715b9ec5).

The helper is executed as rendered under bash, over a process table laid out
as files: ``/proc/<pid>/stat``, ``cmdline`` and ``cgroup``, ``/proc/uptime``
and the cgroups' ``cgroup.procs``. Only its roots and the pid it starts
from are pointed at that table; ``systemctl``, ``busctl`` and ``getconf`` are
shell functions defined before it runs. The ``busctl`` stand-in records its
call and, unless the case says otherwise, does what systemd 255 did on
lavender-wsl on 2026-10-17 when asked the same: the adopted pid's cgroup
becomes the new scope's, under ``runners.slice``. The table mirrors a job's
hook: ``runsvc.sh``, the node service, the Listener, its Worker, the hook's
bash, and sudo running the helper.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from typing_extensions import TypedDict

from fleet.contracts.runner_slice import CI_SLICE_NAME, LISTENER_SLICE_NAME
from fleet.contracts.runners import RunnerInstall
from fleet.core import (
    runner_job_scope_render,
    runner_render,
    runner_tool_caches,
    runner_workspace_unlock,
)
from fleet.core.runner_unit_files import render_drop_in_lines
from tests._audit_rows import run_rows
from tests._host_bash import host_bash
from tests._runner_fixtures import RESERVED_CORES, a_host, an_install

#: The unit every case runs.
UNIT = "actions.runner.wagner-austin-API.lavender.service"

#: The scope its Worker, pid 13, is adopted into.
SCOPE = "fleet-job-actions.runner.wagner-austin-API.lavender-13.scope"

#: The stand-ins.
STAND_INS = """
systemctl() {
    cat "$TABLE/show/$5.$3"
}
busctl() {
    echo "$*" >> "$TABLE/busctl.log"
    if [ -e "$TABLE/busctl-refuses" ]; then
        echo "Failed to start transient scope unit: Access denied" >&2
        return 1
    fi
    if [ -e "$TABLE/busctl-moves-late" ]; then
        printf '0::/runners.slice/%s\\n' "$8" > "$TABLE/moves-to"
        echo "${14}" > "$TABLE/moves-pid"
    elif [ ! -e "$TABLE/busctl-moves-nothing" ]; then
        printf '0::/runners.slice/%s\\n' "$8" > "$TABLE/proc/${14}/cgroup"
    fi
}
sleep() {
    echo "$1" >> "$TABLE/slept"
    if [ -e "$TABLE/moves-to" ]; then
        cp "$TABLE/moves-to" "$TABLE/proc/$(cat "$TABLE/moves-pid")/cgroup"
    fi
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
        cmdline: Its command line, split on spaces.
        start_ticks: Its start, in clock ticks since boot; the table's uptime
            is 1000 s at 100 ticks a second.
        cgroup: Its cgroup v2 path.
    """

    pid: int
    ppid: int
    cmdline: str
    start_ticks: int
    cgroup: str


def _moved(process: Process, *, cgroup: str, start_ticks: int | None = None) -> Process:
    """A process as it is in another cgroup, or started at another time.

    Args:
        process: The process.
        cgroup: The cgroup it is in.
        start_ticks: Its start, or None to keep it.

    Returns:
        The row.
    """
    return Process(
        pid=process["pid"],
        ppid=process["ppid"],
        cmdline=process["cmdline"],
        start_ticks=process["start_ticks"] if start_ticks is None else start_ticks,
        cgroup=cgroup,
    )


#: The unit's own cgroup under the listener slice.
OWN = f"/{LISTENER_SLICE_NAME}/{UNIT}"

#: The unit's own processes.
SERVICE: tuple[Process, ...] = (
    Process(pid=10, ppid=1, cmdline="/bin/bash ./runsvc.sh", start_ticks=1000, cgroup=OWN),
    Process(pid=11, ppid=10, cmdline="node ./bin/RunnerService.js", start_ticks=1000, cgroup=OWN),
    Process(pid=12, ppid=11, cmdline="./bin/Runner.Listener run", start_ticks=1000, cgroup=OWN),
)

#: A job's Worker, 10 s old, before its hook moved it.
WORKER = Process(
    pid=13, ppid=12, cmdline="./bin/Runner.Worker spawnclient", start_ticks=99000, cgroup=OWN
)

#: The hook's bash and the sudo running the helper, under the Worker.
HOOK: tuple[Process, ...] = (
    Process(
        pid=14,
        ppid=13,
        cmdline="/usr/bin/bash -e /usr/local/bin/fleet-runner-job-started.sh",
        start_ticks=99500,
        cgroup=OWN,
    ),
    Process(
        pid=15,
        ppid=14,
        cmdline="sudo -n /usr/local/sbin/fleet-runner-job-scope",
        start_ticks=99600,
        cgroup=OWN,
    ),
)


def _lay_out(
    table: pathlib.Path, processes: tuple[Process, ...], *, unit_cgroup: str = OWN
) -> None:
    """Write the process table, its cgroups and the unit's ControlGroup.

    Args:
        table: The directory to lay it out in.
        processes: Every process, each listed in its own cgroup.
        unit_cgroup: What systemd reports as the unit's ControlGroup.
    """
    members: dict[str, list[int]] = {unit_cgroup: []}
    for process in processes:
        directory = table / "proc" / str(process["pid"])
        directory.mkdir(parents=True)
        after = ["S", str(process["ppid"]), *(["0"] * 17), str(process["start_ticks"])]
        name = process["cmdline"].split()[0]
        (directory / "stat").write_bytes(f"{process['pid']} ({name}) {' '.join(after)}\n".encode())
        (directory / "cmdline").write_bytes(process["cmdline"].replace(" ", "\0").encode())
        (directory / "cgroup").write_bytes(f"0::{process['cgroup']}\n".encode())
        members.setdefault(process["cgroup"], []).append(process["pid"])
    (table / "proc").mkdir(exist_ok=True)
    (table / "proc" / "uptime").write_bytes(b"1000.00 4000.00\n")
    for cgroup, pids in members.items():
        directory = table / "cgroup" / cgroup.lstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cgroup.procs").write_bytes("".join(f"{pid}\n" for pid in pids).encode())
    show = table / "show"
    show.mkdir()
    (show / f"{UNIT}.ControlGroup").write_bytes(f"{unit_cgroup}\n".encode())


def _run(
    table: pathlib.Path, *arguments: str, caller: int = 15
) -> subprocess.CompletedProcess[str]:
    """Run the rendered helper over the table.

    Args:
        table: The laid-out table.
        *arguments: The helper's own arguments.
        caller: The pid it starts its walk from, its parent sudo's.

    Returns:
        The finished run.
    """
    script = runner_job_scope_render.JOB_SCOPE_SCRIPT
    for real, fake in (
        ("cgroup_root=/sys/fs/cgroup", 'cgroup_root="$TABLE/cgroup"'),
        ("proc_root=/proc", 'proc_root="$TABLE/proc"'),
        ("caller=$PPID", f"caller={caller}"),
    ):
        assert f"\n{real}\n" in script
        script = script.replace(f"\n{real}\n", f"\n{fake}\n")
    (table / "helper.sh").write_bytes(script.encode())
    (table / "stand-ins.sh").write_bytes(STAND_INS.encode())
    return subprocess.run(
        [
            host_bash(),
            "-c",
            'TABLE="$PWD"; source stand-ins.sh; source helper.sh "$@"',
            "x",
            *arguments,
        ],
        cwd=table,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def _busctl_calls(table: pathlib.Path) -> list[str]:
    """Every busctl call the run made.

    Args:
        table: The laid-out table.

    Returns:
        The calls' arguments, none when it was never called.
    """
    log = table / "busctl.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


class TestAdoption:
    def test_the_worker_is_adopted_into_a_scope_of_the_ci_slice_bound_to_its_unit(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*SERVICE, WORKER, *HOOK))
        ran = _run(tmp_path)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            f"fleet-runner-job-scope: Runner.Worker 13 of {UNIT} runs in {CI_SLICE_NAME}/{SCOPE}, "
            "outside its listener's cgroup\n"
        )
        assert _busctl_calls(tmp_path) == [
            "--quiet call org.freedesktop.systemd1 /org/freedesktop/systemd1 "
            "org.freedesktop.systemd1.Manager StartTransientUnit ssa(sv)a(sa(sv)) "
            f"{SCOPE} fail 5 PIDs au 1 13 Slice s runners.slice BindsTo as 1 {UNIT} "
            "CollectMode s inactive-or-failed "
            f"Description s GitHub Actions job of {UNIT}, outside its listener 0"
        ]

    def test_a_worker_an_updated_runner_started_is_adopted(self, tmp_path: pathlib.Path) -> None:
        """A runner that updated itself starts its Listener through ``bin`` and
        its Workers from ``bin.<version>``: lavender's ran from ``bin.2.338.0``
        from 2026-10-25T18:26Z, and the helper refused every job's hook until
        MCPs board task 0e8d4118."""
        root = "/home/gharunner/actions-runner-mcps-1"
        listener = Process(
            pid=12,
            ppid=11,
            cmdline=f"{root}/bin/Runner.Listener run --startuptype service",
            start_ticks=1000,
            cgroup=OWN,
        )
        worker = Process(
            pid=13,
            ppid=12,
            cmdline=f"{root}/bin.2.338.0/Runner.Worker spawnclient 132 135",
            start_ticks=99000,
            cgroup=OWN,
        )
        _lay_out(tmp_path, (*SERVICE[:2], listener, worker, *HOOK))
        ran = _run(tmp_path)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            f"fleet-runner-job-scope: Runner.Worker 13 of {UNIT} runs in {CI_SLICE_NAME}/{SCOPE}, "
            "outside its listener's cgroup\n"
        )

    def test_a_caller_with_no_worker_above_it_is_refused_and_nothing_is_adopted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A shell of the runner account's own, not a job's hook."""
        shell = Process(
            pid=40, ppid=1, cmdline="/bin/bash -l", start_ticks=5000, cgroup="/user.slice"
        )
        _lay_out(tmp_path, (*SERVICE, shell))
        ran = _run(tmp_path, caller=40)
        assert ran.returncode == 1
        assert ran.stderr == (
            "FLEET_RUNNER_JOB_SCOPE_REFUSED: no Runner.Worker among the ancestors of 40: "
            "only a job's hook may ask\n"
        )
        assert _busctl_calls(tmp_path) == []

    def test_a_worker_whose_parent_is_no_listener_is_refused(self, tmp_path: pathlib.Path) -> None:
        stray = Process(
            pid=13, ppid=1, cmdline="./bin/Runner.Worker spawnclient", start_ticks=99000, cgroup=OWN
        )
        _lay_out(tmp_path, (*SERVICE, stray, *HOOK))
        ran = _run(tmp_path)
        assert ran.returncode == 1
        assert ran.stderr == (
            "FLEET_RUNNER_JOB_SCOPE_REFUSED: the parent 1 of Runner.Worker 13 is no "
            "Runner.Listener\n"
        )
        assert _busctl_calls(tmp_path) == []

    def test_a_worker_of_a_unit_still_in_the_ci_slice_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A unit the provision has not yet moved, which runs its jobs where
        they always ran; moving its Worker would split the job from a listener
        that shares the job's budget anyway."""
        old = f"/{CI_SLICE_NAME}/{UNIT}"
        moved = tuple(_moved(row, cgroup=old) for row in (*SERVICE, WORKER, *HOOK))
        _lay_out(tmp_path, moved, unit_cgroup=old)
        ran = _run(tmp_path)
        assert ran.returncode == 1
        assert ran.stderr == (
            f"FLEET_RUNNER_JOB_SCOPE_REFUSED: Runner.Worker 13 is in {old}, not in a runner "
            f"unit of {LISTENER_SLICE_NAME}\n"
        )

    def test_a_scope_systemd_will_not_start_fails_the_job_by_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*SERVICE, WORKER, *HOOK))
        (tmp_path / "busctl-refuses").write_bytes(b"")
        ran = _run(tmp_path)
        assert ran.returncode == 1
        assert ran.stderr == (
            "Failed to start transient scope unit: Access denied\n"
            f"FLEET_RUNNER_JOB_SCOPE_REFUSED: systemd would not adopt Runner.Worker 13 of {UNIT} "
            f"into {CI_SLICE_NAME}/{SCOPE}\n"
        )

    def test_a_move_that_lands_after_the_call_answers_is_waited_for(
        self, tmp_path: pathlib.Path
    ) -> None:
        """systemd answers once the scope's start job is queued and moves the
        Worker when it runs: an immediate read failed every job on lavender-wsl
        on 2026-10-17."""
        _lay_out(tmp_path, (*SERVICE, WORKER, *HOOK))
        (tmp_path / "busctl-moves-late").write_bytes(b"")
        ran = _run(tmp_path)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.startswith("fleet-runner-job-scope: Runner.Worker 13 of ")
        assert (tmp_path / "slept").read_text(encoding="utf-8").split() == ["0.1"]

    def test_an_adoption_that_left_the_worker_where_it_was_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The cgroup is read back for the whole wait, so a call that answered
        and moved nothing is not taken for a move."""
        _lay_out(tmp_path, (*SERVICE, WORKER, *HOOK))
        (tmp_path / "busctl-moves-nothing").write_bytes(b"")
        ran = _run(tmp_path)
        assert ran.returncode == 1
        wait = runner_job_scope_render.MOVE_WAIT_SECONDS
        assert ran.stderr == (
            f"FLEET_RUNNER_JOB_SCOPE_REFUSED: systemd took Runner.Worker 13, yet after {wait} s "
            f"it is in {OWN}, not {CI_SLICE_NAME}/{SCOPE}\n"
        )
        assert (tmp_path / "slept").read_text(encoding="utf-8").split() == ["0.1"] * wait * 10

    @pytest.mark.parametrize("arguments", [("--adopt", "13"), ("--workers",), ("--audit",)])
    def test_any_argument_but_the_audit_s_or_the_hub_s_is_refused(
        self, tmp_path: pathlib.Path, arguments: tuple[str, ...]
    ) -> None:
        _lay_out(tmp_path, (*SERVICE, WORKER, *HOOK))
        ran = _run(tmp_path, *arguments)
        assert ran.returncode == 2
        assert ran.stderr == (
            "fleet-runner-job-scope: usage: fleet-runner-job-scope | "
            "fleet-runner-job-scope --audit UNIT | fleet-runner-job-scope --workers UNIT...\n"
        )
        assert _busctl_calls(tmp_path) == []


class TestTheAuditMode:
    def test_a_unit_in_the_listener_slice_with_its_worker_in_a_scope_is_placed(
        self, tmp_path: pathlib.Path
    ) -> None:
        scoped = _moved(WORKER, cgroup=f"/{CI_SLICE_NAME}/{SCOPE}", start_ticks=1000)
        _lay_out(tmp_path, (*SERVICE, scoped))
        ran = _run(tmp_path, "--audit", UNIT)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            "placed\n"
            f"{UNIT} runs in {LISTENER_SLICE_NAME}, and each of its 1 Runner.Worker(s) is in "
            f"{CI_SLICE_NAME} or under 300 s old\n"
        )

    def test_a_worker_the_hook_has_not_yet_moved_is_placed_inside_the_grace(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, (*SERVICE, WORKER))
        ran = _run(tmp_path, "--audit", UNIT)
        assert ran.stdout.splitlines()[0] == "placed"

    def test_a_worker_left_in_the_listener_s_cgroup_past_the_grace_is_misplaced(
        self, tmp_path: pathlib.Path
    ) -> None:
        old = _moved(WORKER, cgroup=OWN, start_ticks=1000)
        _lay_out(tmp_path, (*SERVICE, old))
        ran = _run(tmp_path, "--audit", UNIT)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            "misplaced\n"
            f"{UNIT} runs in {LISTENER_SLICE_NAME}, but 1 of its 1 Runner.Worker(s) ran over 300 s "
            f"in the listener's own cgroup, never moved to {CI_SLICE_NAME}\n"
        )

    def test_a_unit_still_in_the_ci_slice_is_misplaced(self, tmp_path: pathlib.Path) -> None:
        old = f"/{CI_SLICE_NAME}/{UNIT}"
        _lay_out(tmp_path, tuple(_moved(row, cgroup=old) for row in SERVICE), unit_cgroup=old)
        ran = _run(tmp_path, "--audit", UNIT)
        assert ran.stdout == (
            f"misplaced\n{UNIT} runs in {old}, not {LISTENER_SLICE_NAME}/{UNIT}\n"
        )

    def test_a_stopped_unit_is_misplaced_with_no_cgroup(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, (), unit_cgroup="")
        ran = _run(tmp_path, "--audit", UNIT)
        assert ran.stdout == (
            f"misplaced\n{UNIT} runs in no cgroup, not {LISTENER_SLICE_NAME}/{UNIT}\n"
        )


#: Another unit of the host, asked about beside :data:`UNIT`.
OTHER_UNIT = "actions.runner.wagner-austin-API.lavender-2.service"


class TestTheWorkersMode:
    """The hub's read for the dropped-job reading: each unit's Workers by pid."""

    def test_each_worker_in_the_unit_or_its_scopes_is_named_with_its_start_and_age(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A Worker 990 s old in its scope, started from the ``bin.<version>``
        an updated runner starts its Workers from, beside one 10 s old in the
        unit, and a step whose command line names a Worker but which is none."""
        scoped = Process(
            pid=20,
            ppid=12,
            cmdline="./bin.2.338.0/Runner.Worker spawnclient",
            start_ticks=1000,
            cgroup=f"/{CI_SLICE_NAME}/fleet-job-{UNIT.removesuffix('.service')}-20.scope",
        )
        step = Process(
            pid=21,
            ppid=20,
            cmdline="grep -c Runner.Worker",
            start_ticks=2000,
            cgroup=scoped["cgroup"],
        )
        _lay_out(tmp_path, (*SERVICE, WORKER, scoped, step))
        ran = _run(tmp_path, "--workers", UNIT)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == f"{UNIT} 13 99000 10\n{UNIT} 20 1000 990\n"

    def test_a_unit_with_no_worker_and_a_stopped_unit_each_read_none(
        self, tmp_path: pathlib.Path
    ) -> None:
        _lay_out(tmp_path, SERVICE)
        (tmp_path / "show" / f"{OTHER_UNIT}.ControlGroup").write_bytes(b"\n")
        ran = _run(tmp_path, "--workers", UNIT, OTHER_UNIT)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == f"{UNIT} -\n{OTHER_UNIT} -\n"

    def test_a_worker_whose_start_cannot_be_read_is_not_aged(self, tmp_path: pathlib.Path) -> None:
        _lay_out(tmp_path, (*SERVICE, WORKER))
        (tmp_path / "proc" / "13" / "stat").unlink()
        ran = _run(tmp_path, "--workers", UNIT)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == f"{UNIT} -\n"


def _install() -> RunnerInstall:
    """The unit's install.

    Returns:
        API's first install, whose unit is :data:`UNIT`.
    """
    install = an_install("wagner-austin/API")
    assert install["service"] == UNIT
    return install


def _placement_row(tmp_path: pathlib.Path, helper: str) -> str:
    """Run the placement row under the audit's own functions, the helper a stand-in.

    Args:
        tmp_path: The case's directory.
        helper: The stand-in helper's body.

    Returns:
        The row's ``CHECK`` line.
    """
    [row] = run_rows(
        tmp_path,
        runner_job_scope_render.render_placement_check_lines(_install()),
        {runner_job_scope_render.JOB_SCOPE_PATH: helper},
    )
    return row


class TestTheRenderedPieces:
    def test_the_hook_gives_the_job_its_cache_generation_then_hands_it_to_root(self) -> None:
        """MCPs board task 4b405239: the generation lines run first, under
        the hook's strict options, and root's move ends it."""
        lines = runner_job_scope_render.JOB_HOOK_SCRIPT.splitlines()
        assert runner_job_scope_render.JOB_HOOK_PATH.endswith(".sh")
        strict = lines.index("set -euo pipefail")
        assert lines[strict + 1 : -1] == [
            *runner_tool_caches.HOOK_LINES,
            *runner_workspace_unlock.UNLOCK_LINES,
        ]
        assert lines[-1] == "exec sudo -n /usr/local/sbin/fleet-runner-job-scope"

    def test_sudo_admits_the_helper_with_no_argument_only(self) -> None:
        assert runner_job_scope_render.SUDOERS_RULE == (
            'gharunner ALL=(root) NOPASSWD: /usr/local/sbin/fleet-runner-job-scope ""\n'
        )
        assert "." not in runner_job_scope_render.SUDOERS_PATH.rsplit("/", 1)[1]

    def test_the_rule_reaches_sudoers_d_only_once_visudo_accepts_it(self) -> None:
        lines = runner_job_scope_render.render_job_scope_lines()
        staged = "/etc/sudoers.d/fleet-runner-job-scope.new"
        check = lines.index(f"visudo -cf {staged}")
        assert lines[check - 1] == f"chmod 0440 {staged}"
        assert lines[check + 1] == f"mv {staged} /etc/sudoers.d/fleet-runner-job-scope"
        assert lines[check - 3 : check - 1] == [
            'gharunner ALL=(root) NOPASSWD: /usr/local/sbin/fleet-runner-job-scope ""',
            "SUDOERS_EOF",
        ]

    def test_each_unit_s_drop_in_names_the_hook_through_the_one_unit_writer(self) -> None:
        assert runner_job_scope_render.HOOK_DROP_IN == (
            "[Service]\nEnvironment=ACTIONS_RUNNER_HOOK_JOB_STARTED="
            "/usr/local/bin/fleet-runner-job-started.sh\n"
            "Environment=ACTIONS_RUNNER_HOOK_JOB_COMPLETED="
            "/usr/local/bin/fleet-runner-job-completed.sh\n"
        )
        assert runner_job_scope_render.render_job_hook_lines(_install()) == render_drop_in_lines(
            _install(),
            name="fleet-job-scope.conf",
            body=runner_job_scope_render.HOOK_DROP_IN,
            announcement=f"job hook set for {UNIT}: each job runs in a scope of {CI_SLICE_NAME}",
        )

    def test_the_provision_installs_the_helper_before_any_runner(self) -> None:
        rendered = runner_render.render_provision(a_host([_install()]), RESERVED_CORES)
        script = rendered["provision_script"]
        assert script.index("visudo -cf") < script.index("./svc.sh start")

    def test_the_placement_row_is_named_for_the_install(self) -> None:
        assert runner_job_scope_render.placement_check_id(_install()) == (
            "placement:wagner-austin/API:lavender"
        )

    def test_the_placement_row_passes_when_the_helper_reads_placed(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert _placement_row(tmp_path, 'echo placed; echo "$2 in listeners.slice"') == (
            "CHECK placement:wagner-austin/API:lavender OK"
        )

    @pytest.mark.parametrize(
        ("helper", "read"),
        [
            ('echo misplaced; echo "$2 runs in runners.slice"; exit 0', "misplaced"),
            ("echo placed; exit 1", "placed"),
        ],
    )
    def test_the_placement_row_drifts_with_the_helper_s_words_otherwise(
        self, tmp_path: pathlib.Path, helper: str, read: str
    ) -> None:
        row = _placement_row(tmp_path, helper)
        assert row.startswith(
            f"CHECK placement:wagner-austin/API:lavender DRIFT {UNIT} in {LISTENER_SLICE_NAME} "
            f"and its jobs in {CI_SLICE_NAME}; the helper read: {read}"
        )
