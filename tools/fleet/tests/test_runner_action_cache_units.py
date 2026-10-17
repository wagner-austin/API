"""The action archive cache's units, and how a runner unit comes to read it (board task ce2b92e0).

The fill's service is held to the passes it hands the script, one per side
the host runs installs on. The adopt script and each install's drop-in lines
run under a real bash over a host laid out as files (tests/_cache_host.py
says how). The fill itself runs in test_runner_action_cache, the report in
test_runner_cache_report.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.core import runner_action_cache, runner_cache_adopt, runner_render
from tests._action_cache_fixtures import UNIT, a_host, an_install
from tests._cache_host import OTHER, SETTING, SETTLE, Host


def _restarted(unit: str) -> str:
    """The adopt script's line for a unit it restarted.

    Args:
        unit: The unit.

    Returns:
        The line, with its newline.
    """
    return (
        f"fleet-action-cache-adopt: {unit} restarted after {SETTLE} s idle, "
        "to read the action cache\n"
    )


def _left_busy(unit: str) -> str:
    """The adopt script's line for a unit running a job.

    Args:
        unit: The unit.

    Returns:
        The line, with its newline.
    """
    return (
        f"fleet-action-cache-adopt: {unit} is running a job; "
        f"fleet-action-cache-adopt.timer restarts it once it is idle for {SETTLE} s\n"
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
        assert lines[:8] == [
            "cat > /usr/local/sbin/fleet-action-cache-adopt <<'ADOPT_EOF'",
            runner_cache_adopt.ADOPT_SCRIPT.rstrip("\n"),
            "ADOPT_EOF",
            "chmod +x /usr/local/sbin/fleet-action-cache-adopt",
            "cat > /usr/local/sbin/fleet-action-cache-report <<'REPORT_EOF'",
            runner_cache_adopt.REPORT_SCRIPT.rstrip("\n"),
            "REPORT_EOF",
            "chmod +x /usr/local/sbin/fleet-action-cache-report",
        ]
        # The report runs after the adopt, over the same units, so the counts
        # it publishes follow any restart the adopt made.
        assert [line for line in lines[9].splitlines() if line.startswith("ExecStart=")] == [
            f"ExecStart=/usr/local/sbin/fleet-action-cache-adopt {SETTLE} {UNIT} {OTHER}",
            f"ExecStart=/usr/local/sbin/fleet-action-cache-report {UNIT} {OTHER}",
        ]
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
        called = script.index(f"\n{runner_cache_adopt.ADOPT_PATH} {SETTLE} {UNIT}\n")
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
        # Idle at every one of SETTLE + 1 samples, a second apart.
        assert host.sleeps() == SETTLE
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
        # lavender, 2026-10-15: four provision runs found the four MCPs
        # WSL runners busy; the timer is what restarts each once idle.
        host = Host(tmp_path)
        host.unit(UNIT, 100, loaded=False, busy=True)
        provisioned = host.provision(UNIT)
        assert provisioned.returncode == 0, provisioned.stderr
        assert provisioned.stdout == _left_busy(UNIT)
        assert f"restart {UNIT}" not in host.calls()
        still_busy = host.timer(UNIT)
        assert still_busy.stdout == _left_busy(UNIT)
        host.idle(UNIT, 100)
        adopted = host.timer(UNIT)
        assert adopted.returncode == 0, adopted.stderr
        assert adopted.stdout == _restarted(UNIT)
        assert SETTING.encode() in host.environ(100)
        assert host.timer(UNIT).stdout == ""

    def test_a_job_in_its_scope_keeps_its_unit_from_being_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """MCPs board task 715b9ec5: the Worker runs outside the unit's own
        cgroup, which then holds no Worker, and a restart would still end the
        job through the scope's binding to the unit."""
        host = Host(tmp_path)
        host.unit(UNIT, 100)
        scope = host.scoped_job(UNIT, 100)
        busy = host.timer(UNIT)
        assert busy.returncode == 0, busy.stderr
        assert busy.stdout == _left_busy(UNIT)
        assert f"restart {UNIT}" not in host.calls()
        (scope / "cgroup.procs").unlink()
        scope.rmdir()
        assert host.timer(UNIT).stdout == _restarted(UNIT)

    def test_a_job_taken_inside_the_settle_window_stops_the_restart(
        self, tmp_path: pathlib.Path
    ) -> None:
        # lavender-wsl-2, 2026-10-15: no Worker from 13:41:12Z, when one job
        # ended, to 13:41:14Z, when the listener's next job started.
        host = Host(tmp_path)
        host.unit(UNIT, 100)
        host.job_arrives(UNIT, 100)
        ran = host.timer(UNIT)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _left_busy(UNIT)
        assert f"restart {UNIT}" not in host.calls()
        assert host.sleeps() == 1

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
        ran = host.timer(UNIT, OTHER)
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
        ran = host.timer(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _restarted(OTHER)
        assert f"restart {UNIT}" not in host.calls()

    def test_a_unit_already_reading_the_cache_is_not_restarted(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, adopted=True)
        host.unit(OTHER, 200)
        ran = host.timer(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _restarted(OTHER)
        assert f"restart {UNIT}" not in host.calls()

    def test_a_unit_whose_cgroup_cannot_be_read_is_left_running(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100, cgroup_readable=False)
        host.unit(OTHER, 200)
        ran = host.timer(UNIT, OTHER)
        assert ran.returncode == 0, ran.stderr
        # The harness moves /sys/fs/cgroup to its own cgroup directory.
        assert ran.stdout == (
            f"fleet-action-cache-adopt: {UNIT} left running: its cgroup "
            f"cgroup/system.slice/{UNIT} cannot be read, so no job can be ruled out\n"
            + _restarted(OTHER)
        )
        assert f"restart {UNIT}" not in host.calls()

    @pytest.mark.parametrize("arguments", [(), ("15",), (UNIT,), ("15s", UNIT)])
    def test_a_call_without_a_settle_window_and_a_unit_is_refused_with_its_usage(
        self, tmp_path: pathlib.Path, arguments: tuple[str, ...]
    ) -> None:
        host = Host(tmp_path)
        host.unit(UNIT, 100)
        ran = host.adopt(*arguments)
        assert ran.returncode == 2
        assert ran.stderr == (
            "fleet-action-cache-adopt: usage: fleet-action-cache-adopt SETTLE_SECONDS UNIT...\n"
        )
        assert host.calls() == []
