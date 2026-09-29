"""The Windows runner service account: the names, and the audit row.

The provision's convergence and starting lines run inside the Windows
provision's install loop and are executed there, by
tests/pester/rendered-provision-windows.Tests.ps1: a service under another
account rebound with sc.exe's exact arguments and its orphaned tree removed,
a SYSTEM one left alone, a missing one and a refused rebind each refused by
name, and a stopped one started.
"""

from __future__ import annotations

import pathlib

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_account, runner_audit
from tests._runner_fixtures import a_base, a_ci_slice

#: The service every case names.
_SERVICE = "actions.runner.wagner-austin-MCPs.lavender"


def _install(workdir: pathlib.Path) -> RunnerInstall:
    """A windows-side install whose work tree is the test's own.

    Args:
        workdir: The ``_work`` directory.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender",
        side="windows",
        service=_SERVICE,
        workdir=workdir.as_posix(),
        labels=["lavender"],
        python_toolcache=[],
    )


class TestRendered:
    """The account names both tools use, and the lines over the loop's variables."""

    def test_config_cmd_and_the_service_manager_name_the_same_account(self) -> None:
        assert runner_account.WINDOWS_SERVICE_ACCOUNT == "NT AUTHORITY\\SYSTEM"
        assert runner_account.WINDOWS_SERVICE_START_NAME == "LocalSystem"

    def test_the_rebind_compares_the_start_name_and_acts_through_the_parameters(self) -> None:
        assert runner_account.render_service_account_lines() == [
            "$Rows = @(& $GetService $ServiceName)",
            "if ($Rows.Count -eq 0) {",
            '    throw "FLEET_RUNNER_SERVICE_MISSING: service $ServiceName is not installed"',
            "}",
            "$StartName = [string]$Rows[0].StartName",
            "if ($StartName -ne 'LocalSystem') {",
            "    & $StopService $ServiceName",
            "    & $Sc config $ServiceName obj= LocalSystem | Out-Null",
            "    if ($LASTEXITCODE -ne 0) {",
            '        throw "FLEET_RUNNER_REBIND_REFUSED: sc.exe config $ServiceName exited '
            '$LASTEXITCODE"',
            "    }",
            "    if (Test-Path -LiteralPath $Workdir) {",
            "        Remove-Item -LiteralPath $Workdir -Recurse -Force",
            "    }",
            "    & $StartService $ServiceName",
            "    Write-Output ('rebound ' + $ServiceName + ' from ' + $StartName + "
            "' to LocalSystem and removed its work tree')",
            "}",
        ]

    def test_a_stopped_service_is_started_through_the_parameter(self) -> None:
        assert runner_account.render_service_running_lines()[2] == (
            "    & $StartService $ServiceName"
        )

    def test_an_unscriptable_service_name_is_refused(self, tmp_path: pathlib.Path) -> None:
        install = _install(tmp_path / "_work")
        install["service"] = "bad'name"
        with pytest.raises(ValueError) as refused:
            runner_account.render_service_account_check_lines(install)
        assert str(refused.value) == (
            'service "bad\'name" contains "\'", which cannot be embedded in a rendered '
            "script verbatim; rename the item rather than escaping it"
        )


class TestTheAccountRow:
    """The audit row. It is executed by the Pester suite over the committed
    audit render (tests/pester/rendered-audit.Tests.ps1): SYSTEM passes,
    another account and an absent service drift naming what was read."""

    def test_the_row_joins_the_service_rows_the_driver_already_read(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert runner_account.render_service_account_check_lines(_install(tmp_path / "_work")) == [
            "$Account = (@($Service | ForEach-Object { [string]$_.StartName }) -join '')",
            f"Write-Check 'account:windows:{_SERVICE}:LocalSystem' "
            "($Account -eq 'LocalSystem') ('Win32_Service StartName: ' + $Account)",
        ]

    def test_the_row_follows_its_services_rows_in_the_roster(self, tmp_path: pathlib.Path) -> None:
        workdir = tmp_path / "w" / "_work"
        expected = runner_audit.expected_checks(
            HostRunnerSpec(
                name="lavender",
                host="lavender",
                wsl_distro="Ubuntu",
                keepalive_task=None,
                wslconfig_min_memory_gb=None,
                scratch_dir=tmp_path.as_posix(),
                gpu_required=False,
                systemd_timers=[],
                installs=[_install(workdir)],
                assets=[],
                base=a_base(),
                ci_slice=a_ci_slice(),
            )
        )
        assert [check["check_id"] for check in expected[-3:]] == [
            f"service:windows:{_SERVICE}",
            "workdir:wagner-austin/MCPs:windows:lavender",
            f"account:windows:{_SERVICE}:LocalSystem",
        ]
        assert expected[-1]["reason"] == runner_account.SERVICE_ACCOUNT_REASON
