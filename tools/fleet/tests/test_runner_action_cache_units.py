"""The action archive cache's units, and how a runner unit comes to read it (board task ce2b92e0).

The fill's service is held to the passes it hands the script, one per side
the host runs installs on. The adopt script and each install's drop-in lines
run under a real bash with ``systemctl`` a stand-in and the absolute roots
they read, ``/etc/systemd/system``, ``/proc`` and ``/sys/fs/cgroup``,
rewritten to the test's own directories. The stand-in keeps each unit's
state the way systemd does: ``daemon-reload`` loads a unit's environment
from the drop-in file the lines wrote, and ``restart`` gives the running
process whatever environment is loaded. The fill itself runs in
test_runner_action_cache.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core import runner_action_cache, runner_cache_adopt, runner_render
from tests._action_cache_fixtures import UNIT, a_host, an_install
from tests._host_bash import host_bash

#: The setting a unit reads the cache by.
SETTING = f"{runner_action_cache.CACHE_VARIABLE}={runner_action_cache.WSL_CACHE_DIR}"

#: A second unit, for the cases that run the script over more than one.
OTHER = "actions.runner.wagner-austin-API.lavender-wsl.service"

#: ``systemctl`` over ``state/<unit>/``: ``main`` is the unit's main pid (0
#: when stopped) and ``loaded`` the environment systemd holds for it.
SYSTEMCTL = (
    "systemctl() {\n"
    '    echo "$*" >> calls.log\n'
    "    local unit=${!#} u\n"
    '    case "$1 ${3:-}" in\n'
    '        "show MainPID") cat "state/$unit/main" ;;\n'
    '        "show ControlGroup") echo "/system.slice/$unit" ;;\n'
    '        "show Environment") cat "state/$unit/loaded" ;;\n'
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


def _laid_out(script: str) -> str:
    """A rendered script with its absolute roots moved into the test's directory.

    Args:
        script: The script as rendered.

    Returns:
        It, reading ``units``, ``proc`` and ``cgroup`` under its working
        directory and running the adopt script from there.
    """
    for real, laid_out in (
        ("/etc/systemd/system", "units"),
        ("/proc/", "proc/"),
        ("/sys/fs/cgroup", "cgroup"),
        (runner_cache_adopt.ADOPT_PATH, '"$BASH" adopt.sh'),
    ):
        script = script.replace(real, laid_out)
    return script


class Host:
    """Runner units laid out as files: drop-ins, running processes, cgroups, loaded state."""

    def __init__(self, root: pathlib.Path) -> None:
        """Write the adopt script, with the stand-in, where the lines call it.

        Args:
            root: The test's directory.
        """
        self.root = root
        (root / "state").mkdir()
        (root / "adopt.sh").write_bytes(
            (SYSTEMCTL + _laid_out(runner_cache_adopt.ADOPT_SCRIPT)).encode()
        )

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

    def idle(self, name: str, pid: int) -> None:
        """End the job running in a unit.

        Args:
            name: The unit.
            pid: Its main process.
        """
        procs = self.root / "cgroup" / "system.slice" / name / "cgroup.procs"
        procs.write_text(f"{pid}\n", encoding="utf-8")

    def _run(self, script: str) -> subprocess.CompletedProcess[str]:
        """Run a script in the test's directory.

        Args:
            script: The script's name there.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), script], cwd=self.root, capture_output=True, text=True, check=False
        )

    def provision(self, name: str) -> subprocess.CompletedProcess[str]:
        """Run one install's drop-in lines, as provision does after it starts the unit.

        Args:
            name: The unit.

        Returns:
            The finished process.
        """
        lines = runner_cache_adopt.render_cache_environment_lines(an_install(name))
        body = "set -euo pipefail\n" + _laid_out("\n".join(lines)) + "\n"
        (self.root / "provision.sh").write_bytes((SYSTEMCTL + body).encode())
        return self._run("provision.sh")

    def adopt(self, *names: str) -> subprocess.CompletedProcess[str]:
        """Run the adopt script, as its timer does.

        Args:
            names: The units it is given.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), "adopt.sh", *names],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

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


def _restarted(unit: str) -> str:
    """The adopt script's line for a unit it restarted.

    Args:
        unit: The unit.

    Returns:
        The line, with its newline.
    """
    return f"fleet-action-cache-adopt: {unit} restarted while idle, to read the action cache\n"


def _left_busy(unit: str) -> str:
    """The adopt script's line for a unit running a job.

    Args:
        unit: The unit.

    Returns:
        The line, with its newline.
    """
    return (
        f"fleet-action-cache-adopt: {unit} is running a job; "
        "fleet-action-cache-adopt.timer restarts it the first minute it is idle\n"
    )


class TestTheFillUnits:
    """The service passes each side's workdirs to its own cache."""

    def test_a_host_with_both_sides_runs_one_pass_per_side(self) -> None:
        service = runner_action_cache.render_fill_service(
            a_host(
                [
                    an_install(UNIT),
                    an_install(OTHER, workdir="/home/gharunner/a/_work"),
                    an_install(
                        "actions.runner.wagner-austin-MCPs.lavender",
                        "windows",
                        "C:/actions-runner/_work",
                    ),
                ]
            )
        )
        assert [line for line in service.splitlines() if line.startswith("ExecStart=")] == [
            "ExecStart=/usr/local/sbin/fleet-action-cache-fill tar.gz /var/cache/fleet-actions "
            "/home/gharunner/r/_work /home/gharunner/a/_work -- "
            "zip /mnt/c/fleet/action-cache /mnt/c/actions-runner/_work",
        ]

    def test_a_host_with_one_side_runs_one_pass(self) -> None:
        service = runner_action_cache.render_fill_service(a_host([an_install(UNIT)]))
        assert [line for line in service.splitlines() if line.startswith("ExecStart=")] == [
            "ExecStart=/usr/local/sbin/fleet-action-cache-fill tar.gz /var/cache/fleet-actions "
            "/home/gharunner/r/_work"
        ]

    def test_a_windows_workdir_without_a_drive_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a forward-slashed drive path"):
            runner_action_cache.render_fill_service(
                a_host([an_install(UNIT, "windows", "actions-runner/_work")])
            )

    def test_the_lines_install_the_script_both_units_and_both_caches_then_start_a_pass(
        self,
    ) -> None:
        lines = runner_action_cache.render_fill_lines(a_host([an_install(UNIT)]))
        assert lines[:3] == [
            "cat > /usr/local/sbin/fleet-action-cache-fill <<'FILL_EOF'",
            runner_action_cache.FILL_SCRIPT.rstrip("\n"),
            "FILL_EOF",
        ]
        assert lines[-4:] == [
            "install -d -m 0755 /var/cache/fleet-actions /mnt/c/fleet/action-cache",
            "systemctl daemon-reload",
            "systemctl enable --now fleet-action-cache-fill.timer",
            "systemctl start --no-block fleet-action-cache-fill.service",
        ]
        assert "OnUnitActiveSec=15min" in runner_action_cache.FILL_TIMER


class TestTheAdoptUnits:
    """The timer passes every WSL unit, every minute, and is installed before any unit starts."""

    def test_the_service_names_every_wsl_unit_and_the_timer_fires_every_minute(self) -> None:
        lines = runner_cache_adopt.render_adopt_lines(
            a_host(
                [
                    an_install(UNIT),
                    an_install(OTHER),
                    an_install("actions.runner.wagner-austin-MCPs.lavender", "windows", "C:/w"),
                ]
            )
        )
        assert lines[:4] == [
            "cat > /usr/local/sbin/fleet-action-cache-adopt <<'ADOPT_EOF'",
            runner_cache_adopt.ADOPT_SCRIPT.rstrip("\n"),
            "ADOPT_EOF",
            "chmod +x /usr/local/sbin/fleet-action-cache-adopt",
        ]
        assert f"ExecStart=/usr/local/sbin/fleet-action-cache-adopt {UNIT} {OTHER}" in lines[5]
        assert lines[-2:] == [
            "systemctl daemon-reload",
            "systemctl enable --now fleet-action-cache-adopt.timer",
        ]
        assert "OnUnitActiveSec=1min" in runner_cache_adopt.ADOPT_TIMER

    def test_a_host_with_no_wsl_unit_installs_nothing(self) -> None:
        windows = an_install("actions.runner.wagner-austin-MCPs.lavender", "windows", "C:/w")
        assert runner_cache_adopt.render_adopt_lines(a_host([windows])) == []

    def test_provision_installs_the_script_before_the_installs_that_call_it(self) -> None:
        script = runner_render.render_provision(a_host([an_install(UNIT)]))["linux_script"]
        installed = script.index(f"cat > {runner_cache_adopt.ADOPT_PATH} <<'ADOPT_EOF'")
        called = script.index(f"\n{runner_cache_adopt.ADOPT_PATH} {UNIT}\n")
        assert installed < script.index("# --- runner installs ---") < called


class TestProvisionGivesAUnitTheCache:
    """The drop-in lands once, and the unit reads it after one restart while idle."""

    def test_an_idle_unit_is_given_the_drop_in_and_restarted_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, loaded=False)
        first = host.provision(UNIT)
        assert first.returncode == 0, first.stderr
        assert first.stdout == _restarted(UNIT)
        drop_in = tmp_path / "units" / f"{UNIT}.d" / runner_action_cache.CACHE_DROP_IN_NAME
        assert drop_in.read_bytes() == runner_action_cache.CACHE_DROP_IN.encode()
        assert SETTING.encode() in host.environ(100)
        second = host.provision(UNIT)
        assert second.returncode == 0, second.stderr
        assert second.stdout == ""
        assert host.calls() == [
            "daemon-reload",
            f"show -p MainPID --value {UNIT}",
            f"show -p Environment --value {UNIT}",
            f"show -p ControlGroup --value {UNIT}",
            f"restart {UNIT}",
            f"show -p MainPID --value {UNIT}",
            f"show -p Environment --value {UNIT}",
        ]

    def test_a_unit_running_a_job_is_left_then_adopted_by_the_timer_once_idle(
        self, tmp_path: pathlib.Path
    ) -> None:
        # lavender, 2026-10-15: three provision runs found the four MCPs
        # WSL runners busy; the timer is what catches each between jobs.
        host = Host(tmp_path)
        host.unit(UNIT, 100, loaded=False, busy=True)
        provisioned = host.provision(UNIT)
        assert provisioned.returncode == 0, provisioned.stderr
        assert provisioned.stdout == _left_busy(UNIT)
        assert f"restart {UNIT}" not in host.calls()
        still_busy = host.adopt(UNIT)
        assert still_busy.stdout == _left_busy(UNIT)
        host.idle(UNIT, 100)
        adopted = host.adopt(UNIT)
        assert adopted.returncode == 0, adopted.stderr
        assert adopted.stdout == _restarted(UNIT)
        assert SETTING.encode() in host.environ(100)
        assert host.adopt(UNIT).stdout == ""

    def test_an_unscriptable_unit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be embedded in a rendered script verbatim"):
            runner_cache_adopt.render_cache_environment_lines(an_install("bad'unit"))


class TestTheAdoptScriptRestartsOnlyAnIdleConfiguredUnit:
    """Every other unit is left as it is, and the next unit is still examined."""

    def test_a_stopped_unit_is_never_started_and_the_next_is_still_adopted(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, running=False)
        host.unit(OTHER, 200)
        ran = host.adopt(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _restarted(OTHER)
        assert f"restart {UNIT}" not in host.calls()

    def test_a_unit_not_configured_with_the_cache_is_not_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Without this, a timer that runs before the drop-in exists would
        # restart an idle unit every minute and never give it the cache.
        host = Host(tmp_path)
        host.unit(UNIT, 100, loaded=False)
        host.unit(OTHER, 200)
        ran = host.adopt(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _restarted(OTHER)
        assert f"restart {UNIT}" not in host.calls()

    def test_a_unit_already_reading_the_cache_is_not_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, adopted=True)
        host.unit(OTHER, 200)
        ran = host.adopt(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _restarted(OTHER)
        assert f"restart {UNIT}" not in host.calls()

    def test_a_unit_whose_cgroup_cannot_be_read_is_left_running(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, cgroup_readable=False)
        host.unit(OTHER, 200)
        ran = host.adopt(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        # The harness moves /sys/fs/cgroup to its own cgroup directory.
        assert ran.stdout == (
            f"fleet-action-cache-adopt: {UNIT} left running: its cgroup "
            f"cgroup/system.slice/{UNIT} cannot be read, so no job can be ruled out\n"
            + _restarted(OTHER)
        )
        assert f"restart {UNIT}" not in host.calls()

    def test_a_call_naming_no_unit_is_refused_with_its_usage(self, tmp_path: pathlib.Path) -> None:
        ran = Host(tmp_path).adopt()
        assert ran.returncode == 2
        assert ran.stderr == "fleet-action-cache-adopt: usage: fleet-action-cache-adopt UNIT...\n"
