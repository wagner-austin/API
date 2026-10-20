"""A runner host laid out as files, for the action cache's adopt and report scripts (MCPs ce2b92e0).

Support module, not a test module: test_runner_action_cache_units runs the
adopt script and each install's drop-in lines over it, and
test_runner_cache_report the report script. Each script runs under a real
bash with the absolute roots it reads, ``/etc/systemd/system``, ``/proc``,
``/sys/fs/cgroup`` and the status file's directory, rewritten to the test's
own directories, and with ``systemctl``, ``sleep`` and ``ps`` stand-ins.
The ``systemctl`` stand-in keeps each unit's state the way systemd does:
``daemon-reload`` loads a unit's environment from the drop-in file the lines
wrote, and ``restart`` gives the running process whatever environment is
loaded. ``find`` is the host's own, reading the Worker logs' real mtimes.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

from fleet.contracts.action_cache import STATUS_PATH
from fleet.contracts.runners import RunnerInstall
from fleet.core import runner_action_cache, runner_cache_adopt
from tests._host_bash import host_bash
from tests._runner_fixtures import an_install

#: The setting a unit reads the cache by.
SETTING = f"{runner_action_cache.CACHE_VARIABLE}={runner_action_cache.CACHE_DIR}"

#: The install the drop-in cases run, the fourth of lavender's MCPs runners.
FOURTH: RunnerInstall = an_install("wagner-austin/MCPs", "lavender-4", slot=4)

#: Its unit.
UNIT = FOURTH["service"]

#: A second install, for the cases that run a script over more than one.
SECOND: RunnerInstall = an_install("wagner-austin/API")

#: Its unit.
OTHER = SECOND["service"]

#: The settle window every rendered adopt call passes.
SETTLE = runner_cache_adopt.ADOPT_SETTLE_SECONDS

#: ``systemctl`` over ``state/<unit>/``: ``main`` is the unit's main pid (0
#: when stopped), ``loaded`` the environment systemd holds for it, and its
#: working directory ``runner/<unit>``. ``sleep``, which passes no time: it
#: records each call, and on the first one after a case writes ``arrives``
#: (``<cgroup.procs path> <pid>``) puts that pid in that cgroup, the way a job
#: arrives inside a settle window. ``ps -o etimes= -p <pid>``, which answers
#: ``proc/<pid>/elapsed``, the seconds since that process started.
STAND_INS = (
    "sleep() {\n"
    '    echo "$*" >> sleeps.log\n'
    "    local procs pid\n"
    "    if [ -f arrives ]; then\n"
    "        read -r procs pid < arrives\n"
    '        echo "$pid" >> "$procs"\n'
    "        rm arrives\n"
    "    fi\n"
    "}\n"
    "ps() {\n"
    '    cat "proc/${!#}/elapsed"\n'
    "}\n"
    "systemctl() {\n"
    '    echo "$*" >> calls.log\n'
    "    local unit=${!#} u\n"
    '    case "$1 ${3:-}" in\n'
    '        "show MainPID") cat "state/$unit/main" ;;\n'
    '        "show ControlGroup") echo "/system.slice/$unit" ;;\n'
    '        "show Environment") cat "state/$unit/loaded" ;;\n'
    '        "show WorkingDirectory") echo "runner/$unit" ;;\n'
    "    esac\n"
    '    case "$1" in\n'
    "        daemon-reload)\n"
    "            for u in $(ls state); do\n"
    f'                if [ -f "units/$u.d/{runner_action_cache.CACHE_DROP_IN_NAME}" ]; then\n'
    f'                    echo "LANG=C.UTF-8 {SETTING}" > "state/$u/loaded"\n'
    "                fi\n"
    "            done ;;\n"
    "        restart)\n"
    '            u=$(cat "state/$unit/main")\n'
    '            printf "HOME=/home/gharunner\\0" > "proc/$u/environ"\n'
    f'            if grep -q "{SETTING}" "state/$unit/loaded"; then\n'
    f'                printf "{SETTING}\\0" >> "proc/$u/environ"\n'
    "            fi ;;\n"
    "    esac\n"
    "}\n"
)

#: Where the report's status file lands under the test's directory.
STATUS_LAID_OUT = "statusdir/status"


def laid_out(script: str) -> str:
    """A rendered script with its absolute roots moved into the test's directory.

    Args:
        script: The script as rendered.

    Returns:
        It, reading ``units``, ``proc`` and ``cgroup`` under its working
        directory, writing its status to :data:`STATUS_LAID_OUT`, and
        running the adopt script from there.
    """
    for real, moved in (
        ("/etc/systemd/system", "units"),
        ("/proc/", "proc/"),
        ("/sys/fs/cgroup", "cgroup"),
        (STATUS_PATH, STATUS_LAID_OUT),
        (runner_cache_adopt.ADOPT_PATH, '"$BASH" adopt.sh'),
    ):
        script = script.replace(real, moved)
    return script


class Host:
    """Runner units laid out as files: drop-ins, processes, cgroups, loaded state, Worker logs."""

    def __init__(self, root: pathlib.Path) -> None:
        """Write the adopt and report scripts, with the stand-ins, where they are called.

        Args:
            root: The test's directory.
        """
        self.root = root
        (root / "state").mkdir()
        for name, script in (
            ("adopt.sh", runner_cache_adopt.ADOPT_SCRIPT),
            ("report.sh", runner_cache_adopt.REPORT_SCRIPT),
        ):
            (root / name).write_bytes((STAND_INS + laid_out(script)).encode())

    def unit(
        self,
        name: str,
        pid: int,
        *,
        running: bool = True,
        loaded: bool = True,
        adopted: bool = False,
        busy: bool = False,
        cgroup_readable: bool = True,
        elapsed: int | None = 3600,
    ) -> None:
        """Lay out one unit.

        Args:
            name: The unit.
            pid: Its main process; a job's Worker, when busy, is ``pid + 1``.
            running: False for a stopped unit, whose main pid reads 0.
            loaded: Whether systemd holds the setting for it.
            adopted: Whether its running process already has the setting.
            busy: Whether a Worker runs in its cgroup.
            cgroup_readable: False for a cgroup whose process list is missing.
            elapsed: Seconds since its main process started, or None for a
                process that is gone before ``ps`` asks.
        """
        state = self.root / "state" / name
        state.mkdir()
        (state / "main").write_text(f"{pid if running else 0}\n", encoding="utf-8")
        configured = f"LANG=C.UTF-8 {SETTING}" if loaded else "LANG=C.UTF-8"
        (state / "loaded").write_text(configured + "\n", encoding="utf-8")
        process = self.root / "proc" / str(pid)
        process.mkdir(parents=True)
        environ = b"HOME=/home/gharunner\0LANG=C.UTF-8\0"
        (process / "environ").write_bytes(environ + (f"{SETTING}\0".encode() if adopted else b""))
        (process / "cmdline").write_bytes(b"/bin/bash\0runsvc.sh\0")
        if elapsed is not None:
            (process / "elapsed").write_text(f"  {elapsed}\n", encoding="utf-8")
        cgroup = self.root / "cgroup" / "system.slice" / name
        cgroup.mkdir(parents=True)
        procs = [pid]
        if busy:
            worker = self.root / "proc" / str(pid + 1)
            worker.mkdir()
            (worker / "cmdline").write_bytes(b"/home/gharunner/bin/Runner.Worker\0spawnclient\0")
            procs.append(pid + 1)
        if cgroup_readable:
            (cgroup / "cgroup.procs").write_text("".join(f"{p}\n" for p in procs), encoding="utf-8")
        (self.root / "runner" / name / "_diag").mkdir(parents=True)

    def worker_log(self, name: str, stamp: str, minutes_ago: int, lookups: int, hits: int) -> None:
        """Write a job's Worker log as the runner leaves it.

        Args:
            name: The unit whose runner ran the job.
            stamp: The log's name stamp, ``YYYYMMDD-HHMMSS``.
            minutes_ago: When it was last written.
            lookups: Actions the job looked up in the cache.
            hits: Those the cache held.
        """
        lines = [f"[{stamp}Z INFO Worker] Job started"]
        for index in range(lookups):
            action = f"actions/a{index}@{'0' * 40}"
            lines.append(
                f"[{stamp}Z INFO ActionManager] Check if action archive '{action}' already "
                "exists in cache directory '/var/cache/fleet-actions'"
            )
            if index < hits:
                lines.append(
                    f"[{stamp}Z INFO ActionManager] Found action archive "
                    f"'/var/cache/fleet-actions/a{index}.tar.gz' in cache directory "
                    "'/var/cache/fleet-actions'"
                )
        log = self.root / "runner" / name / "_diag" / f"Worker_{stamp}-utc.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written = time.time() - minutes_ago * 60
        os.utime(log, (written, written))

    def idle(self, name: str, pid: int) -> None:
        """End the job running in a unit.

        Args:
            name: The unit.
            pid: Its main process.
        """
        procs = self.root / "cgroup" / "system.slice" / name / "cgroup.procs"
        procs.write_text(f"{pid}\n", encoding="utf-8")

    def scoped_job(self, name: str, pid: int) -> pathlib.Path:
        """Run a job whose Worker the hook moved into its scope (MCPs board task 715b9ec5).

        Args:
            name: The unit.
            pid: Its main process; the job's Worker is ``pid + 1``.

        Returns:
            The scope's directory, which the job's end removes.
        """
        worker = self.root / "proc" / str(pid + 1)
        worker.mkdir()
        (worker / "cmdline").write_bytes(b"/home/gharunner/bin/Runner.Worker\0spawnclient\0")
        stem = name.removesuffix(".service")
        scope = self.root / "cgroup" / "runners.slice" / f"fleet-job-{stem}-{pid + 1}.scope"
        scope.mkdir(parents=True)
        (scope / "cgroup.procs").write_text(f"{pid + 1}\n", encoding="utf-8")
        return scope

    def job_arrives(self, name: str, pid: int) -> None:
        """Start a job in a unit at the first second of its settle window.

        Args:
            name: The unit.
            pid: Its main process; the job's Worker is ``pid + 1``.
        """
        worker = self.root / "proc" / str(pid + 1)
        worker.mkdir()
        (worker / "cmdline").write_bytes(b"/home/gharunner/bin/Runner.Worker\0spawnclient\0")
        procs = f"cgroup/system.slice/{name}/cgroup.procs"
        (self.root / "arrives").write_text(f"{procs} {pid + 1}\n", encoding="utf-8")

    def sleeps(self) -> int:
        """How many one-second samples the scripts waited between.

        Returns:
            The number of sleep calls, each asserted to be one second.
        """
        log = self.root / "sleeps.log"
        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        assert set(lines) <= {"1"}
        return len(lines)

    def run(self, script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run a script in the test's directory with exactly these arguments.

        Args:
            script: The script's name there.
            arguments: Its arguments.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), script, *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def provision(self, install: RunnerInstall) -> subprocess.CompletedProcess[str]:
        """Run one install's cache lines, the drop-in then the adopt, as provision does.

        Provision starts the unit between the two; the host :meth:`unit`
        laid out is already running, the shape of a unit an older provision
        started without the cache.

        Args:
            install: The install, laid out with :meth:`unit` under its unit's name.

        Returns:
            The finished process.
        """
        lines = [
            *runner_cache_adopt.render_cache_drop_in_lines(install),
            *runner_cache_adopt.render_cache_adopt_now_lines(install),
        ]
        body = "set -euo pipefail\n" + laid_out("\n".join(lines)) + "\n"
        (self.root / "provision.sh").write_bytes((STAND_INS + body).encode())
        return self.run("provision.sh")

    def adopt(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run the adopt script with exactly these arguments.

        Args:
            arguments: Its arguments.

        Returns:
            The finished process.
        """
        return self.run("adopt.sh", *arguments)

    def timer(self, *names: str) -> subprocess.CompletedProcess[str]:
        """Run the adopt script as its timer does, with the rendered settle window.

        Args:
            names: The units it is given.

        Returns:
            The finished process.
        """
        return self.adopt(str(SETTLE), *names)

    def report(self, *names: str) -> subprocess.CompletedProcess[str]:
        """Run the report script as the timer's service does.

        Args:
            names: The units it is given.

        Returns:
            The finished process.
        """
        return self.run("report.sh", *names)

    def status(self) -> str:
        """The status file the report wrote.

        Returns:
            Its text.
        """
        return (self.root / STATUS_LAID_OUT).read_text(encoding="utf-8")

    def calls(self) -> list[str]:
        """Every systemctl call, in order.

        Returns:
            The calls' arguments, none when it was never called.
        """
        log = self.root / "calls.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def environ(self, pid: int) -> bytes:
        """A process's environment.

        Args:
            pid: The process.

        Returns:
            Its bytes.
        """
        return (self.root / "proc" / str(pid) / "environ").read_bytes()
