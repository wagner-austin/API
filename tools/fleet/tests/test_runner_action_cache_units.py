"""The action archive cache's units and runner drop-in (MCPs board task ce2b92e0).

The fill's service is held to the passes it hands the script, one per side
the host runs installs on. The drop-in lines run under a real bash with
``systemctl`` a recording stand-in and the three absolute roots they read,
``/etc/systemd/system``, ``/proc`` and ``/sys/fs/cgroup``, rewritten to the
test's own directories: the write, the reload, the read of the running
process's environment and the busy check all run as rendered. The fill
itself runs in test_runner_action_cache.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core import runner_action_cache
from tests._action_cache_fixtures import UNIT, a_host, an_install
from tests._host_bash import host_bash


class TestTheFillUnits:
    """The service passes each side's workdirs to its own cache."""

    def test_a_host_with_both_sides_runs_one_pass_per_side(self) -> None:
        service = runner_action_cache.render_fill_service(
            a_host(
                [
                    an_install(UNIT),
                    an_install(
                        "actions.runner.wagner-austin-API.lavender-wsl.service",
                        workdir="/home/gharunner/a/_work",
                    ),
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


class Unit:
    """A runner unit laid out as files: its drop-ins, its main process, its cgroup."""

    def __init__(self, root: pathlib.Path) -> None:
        """Lay out a running unit whose main process lacks the variable.

        Args:
            root: The test's directory.
        """
        self.root = root
        (root / "proc" / "100").mkdir(parents=True)
        (root / "proc" / "100" / "environ").write_bytes(b"HOME=/home/gharunner\0LANG=C.UTF-8\0")
        (root / "proc" / "100" / "cmdline").write_bytes(b"/bin/bash\0runsvc.sh\0")
        self.cgroup = root / "cgroup" / "system.slice" / UNIT
        self.cgroup.mkdir(parents=True)
        (self.cgroup / "cgroup.procs").write_text("100\n", encoding="utf-8")
        body = "\n".join(runner_action_cache.render_cache_environment_lines(an_install(UNIT)))
        for real, laid_out in (
            ("/etc/systemd/system", "units"),
            ("/proc/", "proc/"),
            ("/sys/fs/cgroup", "cgroup"),
        ):
            body = body.replace(real, laid_out)
        # restart gives the unit's main process the drop-in's environment, as
        # systemd does when it starts the unit again.
        stand_ins = (
            "systemctl() {\n"
            '    echo "$*" >> calls.log\n'
            '    case "$1 $3" in\n'
            '        "show MainPID") echo 100 ;;\n'
            f'        "show ControlGroup") echo /system.slice/{UNIT} ;;\n'
            "    esac\n"
            '    if [ "$1" = restart ]; then\n'
            "        printf 'HOME=/home/gharunner\\0"
            f"{runner_action_cache.CACHE_VARIABLE}={runner_action_cache.WSL_CACHE_DIR}\\0' "
            "> proc/100/environ\n"
            "    fi\n"
            "}\n"
        )
        (root / "cache.sh").write_bytes((stand_ins + body + "\n").encode())

    def busy(self) -> None:
        """Put a job's Worker in the unit's cgroup."""
        (self.root / "proc" / "200").mkdir()
        (self.root / "proc" / "200" / "cmdline").write_bytes(
            b"/home/gharunner/bin/Runner.Worker\0spawnclient\0"
        )
        (self.cgroup / "cgroup.procs").write_text("100\n200\n", encoding="utf-8")

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the lines once.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), "cache.sh"], cwd=self.root, capture_output=True, text=True, check=False
        )

    def calls(self) -> list[str]:
        """Every systemctl call, in order.

        Returns:
            The calls' arguments.
        """
        return (self.root / "calls.log").read_text(encoding="utf-8").splitlines()


class TestTheDropInRunsForReal:
    """The variable lands once, and reaches the unit only by a restart of an idle unit."""

    def test_an_idle_unit_is_given_the_variable_and_restarted_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        unit = Unit(tmp_path)
        first = unit.run()
        assert first.returncode == 0, first.stderr
        assert first.stdout == f"action cache set for {UNIT}: restarted while idle\n"
        drop_in = tmp_path / "units" / f"{UNIT}.d" / runner_action_cache.CACHE_DROP_IN_NAME
        assert drop_in.read_bytes() == runner_action_cache.CACHE_DROP_IN.encode()
        second = unit.run()
        assert second.returncode == 0, second.stderr
        assert second.stdout == ""
        assert unit.calls() == [
            "daemon-reload",
            f"show -p MainPID --value {UNIT}",
            f"show -p ControlGroup --value {UNIT}",
            f"restart {UNIT}",
            f"show -p MainPID --value {UNIT}",
        ]

    def test_a_unit_running_a_job_is_left_for_the_next_run(self, tmp_path: pathlib.Path) -> None:
        unit = Unit(tmp_path)
        unit.busy()
        ran = unit.run()
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            f"action cache set for {UNIT} at its next restart: it is running a job; "
            "run provision again to restart it\n"
        )
        assert f"restart {UNIT}" not in unit.calls()

    def test_an_unscriptable_unit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be embedded in a rendered script verbatim"):
            runner_action_cache.render_cache_environment_lines(an_install("bad'unit"))
