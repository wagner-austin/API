"""The audit row for what a finished job left behind (MCPs board task 53528106, A2 and A4).

The row's PowerShell is executed by the Pester suite over the committed
render (tests/pester/rendered-audit.Tests.ps1); these pin what each side
asks and how it names its row, so the two halves of a host agree.
"""

from __future__ import annotations

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_orphan_check
from tests._runner_fixtures import a_base, a_ci_slice


def _install(side: str, workdir: str, service: str) -> RunnerInstall:
    """One install on either side.

    Args:
        side: ``wsl`` or ``windows``.
        workdir: Its ``_work`` tree.
        service: Its service name.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender-wsl" if side == "wsl" else "lavender",
        side="wsl" if side == "wsl" else "windows",
        service=service,
        workdir=workdir,
        labels=["lavender"],
        python_toolcache=[],
    )


WSL = _install(
    "wsl",
    "/home/gharunner/actions-runner/_work",
    "actions.runner.wagner-austin-MCPs.lavender-wsl.service",
)
WINDOWS = _install(
    "windows", "C:/actions-runner/_work", "actions.runner.wagner-austin-MCPs.lavender"
)


def _host(minutes: int) -> HostRunnerSpec:
    """A host holding both installs, bounded at ``minutes``.

    Args:
        minutes: Its job timeout.

    Returns:
        The host.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=None,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        job_timeout_minutes=minutes,
        installs=[WSL, WINDOWS],
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


class TestTheRowIds:
    def test_each_side_names_its_repository_side_and_runner(self) -> None:
        assert (
            runner_orphan_check.orphan_check_id(WSL)
            == "orphans:wagner-austin/MCPs:wsl:lavender-wsl"
        )
        assert (
            runner_orphan_check.orphan_check_id(WINDOWS)
            == "orphans:wagner-austin/MCPs:windows:lavender"
        )

    def test_a_windows_root_ends_in_a_separator_so_a_sibling_runner_never_matches(self) -> None:
        assert runner_orphan_check.runner_root(WINDOWS) == "C:\\actions-runner\\"


class TestTheWslRow:
    def test_it_asks_the_reaper_s_audit_mode_for_the_unit_at_the_host_s_bound(self) -> None:
        lines = runner_orphan_check.render_orphan_check_lines(_host(60), WSL)
        assert lines[0] == (
            '$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper '
            "--audit 3600 'actions.runner.wagner-austin-MCPs.lavender-wsl.service'\""
        )
        assert lines[2] == (
            "Write-Check 'orphans:wagner-austin/MCPs:wsl:lavender-wsl' "
            "($Probe.Exit -eq 0 -and $Leftover -eq '0') "
            "('processes older than 60 minutes that a finished job left in "
            "actions.runner.wagner-austin-MCPs.lavender-wsl.service, "
            "or under a Worker past that job timeout; the reaper counted: ' + $Probe.Text)"
        )


class TestTheWindowsRow:
    def test_it_walks_the_service_s_tree_under_its_own_root_at_the_host_s_bound(self) -> None:
        lines = runner_orphan_check.render_orphan_check_lines(_host(360), WINDOWS)
        assert lines[:5] == [
            "$ServicePid = 0",
            "foreach ($Row in $Service) {",
            "    $ServicePid = [int]$Row.ProcessId",
            "}",
            "$Leftover = @(Get-RunnerLeftover @(& $GetProcesses) $ServicePid "
            "'C:\\actions-runner\\' 21600)",
        ]
        assert lines[5].startswith(
            "Write-Check 'orphans:wagner-austin/MCPs:windows:lavender' ($Leftover.Count -eq 0) "
        )

    def test_an_unscriptable_directory_is_refused(self) -> None:
        install = _install("windows", "C:/it's/_work", "actions.runner.x.lavender")
        with pytest.raises(ValueError, match="workdir"):
            runner_orphan_check.render_orphan_check_lines(_host(60), install)
