"""The Windows runner service account: rendered, and the lines run for real.

The convergence lines are executed under Windows PowerShell 5.1 with
functions defined ahead of them standing in for the four commands that would
change this machine's services (``Get-CimInstance``, ``Stop-Service``,
``Start-Service``, ``sc.exe``). PowerShell resolves a function before a
cmdlet or an executable of the same name, ``sc.exe`` included, so the work
tree is tested and removed by the real ``Test-Path`` and ``Remove-Item``
against a real directory.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_account, runner_audit
from fleet.core.dialect_windows import POWERSHELL_INVOCATION
from tests._runner_fixtures import a_base

#: The service every case converges.
_SERVICE = "actions.runner.wagner-austin-MCPs.lavender"

#: Stand-ins that record each service act as a CALL line on stdout.
_FAKES = """function Get-CimInstance {
    param([string]$ClassName, [string]$Filter)
    Write-Output ('CALL Get-CimInstance ' + $ClassName + ' ' + $Filter) | Out-Host
    if ($null -ne $script:StartName) { [pscustomobject]@{ StartName = $script:StartName } }
}
function Stop-Service { param([string]$Name) Write-Output ('CALL Stop-Service ' + $Name) }
function Start-Service { param([string]$Name) Write-Output ('CALL Start-Service ' + $Name) }
function sc.exe {
    Write-Output ('CALL sc.exe ' + ($args -join ' '))
    $global:LASTEXITCODE = $script:ScExit
}
"""


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


def _converge(
    tmp_path: pathlib.Path, start_name: str | None, sc_exit: int = 0
) -> subprocess.CompletedProcess[str]:
    """Run the convergence lines for an install whose tree holds one file.

    Args:
        tmp_path: The test's directory; ``_work`` is created under it.
        start_name: What ``Win32_Service.StartName`` reads, or ``None`` for a
            service that is not installed.
        sc_exit: The exit code ``sc.exe config`` reports.

    Returns:
        The completed process.
    """
    workdir = tmp_path / "_work"
    workdir.mkdir()
    (workdir / "checkout.txt").write_text("owned by the old account", encoding="utf-8")
    start_value = "$null" if start_name is None else "'" + start_name + "'"
    prelude = (
        f"$script:StartName = {start_value}\n"
        f"$script:ScExit = {sc_exit}\n"
        "$ErrorActionPreference = 'Stop'\n"
    )
    script = tmp_path / "converge.ps1"
    lines = runner_account.render_service_account_lines(_install(workdir))
    script.write_text(prelude + _FAKES + "\n".join(lines) + "\n", encoding="utf-8")
    return subprocess.run(
        [*POWERSHELL_INVOCATION, str(script)], capture_output=True, text=True, check=False
    )


def _thrown(ran: subprocess.CompletedProcess[str]) -> str:
    """The message a failed run threw, joined across PowerShell's wrapping.

    Args:
        ran: A run that exited nonzero.

    Returns:
        The stderr text up to PowerShell's ``At <script>`` locator.
    """
    joined = " ".join(line.strip() for line in ran.stderr.splitlines())
    return joined.split(" At ", 1)[0]


class TestRendered:
    """The account names both tools use."""

    def test_config_cmd_and_the_service_manager_name_the_same_account(self) -> None:
        assert runner_account.WINDOWS_SERVICE_ACCOUNT == "NT AUTHORITY\\SYSTEM"
        assert runner_account.WINDOWS_SERVICE_START_NAME == "LocalSystem"

    def test_an_unscriptable_service_name_is_refused(self, tmp_path: pathlib.Path) -> None:
        install = _install(tmp_path / "_work")
        install["service"] = "bad'name"
        with pytest.raises(ValueError) as refused:
            runner_account.render_service_account_lines(install)
        assert str(refused.value) == (
            'service "bad\'name" contains "\'", which cannot be embedded in a rendered '
            "script verbatim; rename the item rather than escaping it"
        )


@pytest.mark.skipif(sys.platform != "win32", reason="the lines are PowerShell; run them here")
class TestConvergenceRunsForReal:
    """The lines executed, not matched."""

    def test_a_network_service_install_is_rebound_and_its_tree_removed(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The reinstalled lavender's state: stopped, rebound with sc.exe's
        exact arguments, the orphaned tree gone, started again."""
        ran = _converge(tmp_path, "NT AUTHORITY\\NetworkService")
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.splitlines() == [
            f"CALL Get-CimInstance Win32_Service Name='{_SERVICE}'",
            f"CALL Stop-Service {_SERVICE}",
            f"CALL sc.exe config {_SERVICE} obj= LocalSystem",
            f"CALL Start-Service {_SERVICE}",
            f"rebound {_SERVICE} from NT AUTHORITY\\NetworkService to LocalSystem "
            "and removed its work tree",
        ]
        assert not (tmp_path / "_work").exists()

    def test_a_system_install_is_left_alone_with_its_tree(self, tmp_path: pathlib.Path) -> None:
        ran = _converge(tmp_path, "LocalSystem")
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.splitlines() == [f"CALL Get-CimInstance Win32_Service Name='{_SERVICE}'"]
        assert (tmp_path / "_work" / "checkout.txt").read_text(encoding="utf-8") == (
            "owned by the old account"
        )

    def test_a_missing_service_throws_before_touching_anything(
        self, tmp_path: pathlib.Path
    ) -> None:
        ran = _converge(tmp_path, None)
        assert ran.returncode == 1
        assert _thrown(ran) == f"service {_SERVICE} is not installed"
        assert (tmp_path / "_work" / "checkout.txt").exists()

    def test_a_failed_rebind_throws_and_keeps_the_tree_and_the_service_stopped(
        self, tmp_path: pathlib.Path
    ) -> None:
        ran = _converge(tmp_path, "NT AUTHORITY\\NetworkService", sc_exit=1060)
        assert ran.returncode == 1
        assert _thrown(ran) == f"sc.exe config {_SERVICE} exited 1060"
        assert ran.stdout.splitlines() == [
            f"CALL Get-CimInstance Win32_Service Name='{_SERVICE}'",
            f"CALL Stop-Service {_SERVICE}",
            f"CALL sc.exe config {_SERVICE} obj= LocalSystem",
        ]
        assert (tmp_path / "_work" / "checkout.txt").exists()


#: The audit's other host-level reads, answered as a laid host answers them.
_AUDIT_FAKES = """function wsl { $global:LASTEXITCODE = 0; return @('Used', '  46G') }
function Get-ExecutionPolicy { param([string]$Scope) return 'RemoteSigned' }
function Get-ItemProperty { param([string]$LiteralPath) [pscustomobject]@{ LongPathsEnabled = 1 } }
function git { return 'true' }
function Get-Service {
    param([string]$Name, [string]$ErrorAction)
    [pscustomobject]@{ Status = 'Running' }
}
"""


def _audit(tmp_path: pathlib.Path, start_name: str | None) -> list[str]:
    """Run the whole rendered audit for a host with one Windows-side install.

    Args:
        tmp_path: The test's directory; the install's real ``_work`` is here.
        start_name: What ``Win32_Service.StartName`` reads, or ``None`` for
            a service that is not installed.

    Returns:
        The CHECK lines the script emitted.
    """
    workdir = tmp_path / "_work"
    workdir.mkdir()
    spec = HostRunnerSpec(
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
    )
    start_value = "$null" if start_name is None else "'" + start_name + "'"
    script = tmp_path / "audit.ps1"
    script.write_text(
        f"$script:StartName = {start_value}\n"
        + _FAKES
        + _AUDIT_FAKES
        + runner_audit.render_audit_script(spec),
        encoding="utf-8",
    )
    ran = subprocess.run(
        [*POWERSHELL_INVOCATION, str(script)], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
    return [line for line in ran.stdout.splitlines() if line.startswith("CHECK ")]


@pytest.mark.skipif(sys.platform != "win32", reason="the audit is PowerShell; run it here")
class TestTheAccountRowRunsForReal:
    """The audit row executed: one line per Windows-side service, after its
    workdir row, whatever the service reports."""

    @pytest.mark.parametrize(
        ("start_name", "verdict"),
        [
            ("LocalSystem", "OK"),
            (
                "NT AUTHORITY\\NetworkService",
                "DRIFT Win32_Service StartName: NT AUTHORITY\\NetworkService",
            ),
            (None, "DRIFT Win32_Service StartName: "),
        ],
    )
    def test_the_row_passes_system_and_drifts_anything_else(
        self, tmp_path: pathlib.Path, start_name: str | None, verdict: str
    ) -> None:
        lines = _audit(tmp_path, start_name)
        assert lines == [
            "CHECK disk:/:ceiling-150gb:baseline-46gb@2026-09-26 OK",
            "CHECK execution-policy:LocalMachine:RemoteSigned OK",
            f"CHECK {runner_audit.LONG_PATHS_CHECK_ID} OK",
            f"CHECK service:windows:{_SERVICE} OK",
            "CHECK workdir:wagner-austin/MCPs:windows:lavender OK",
            f"CHECK account:windows:{_SERVICE}:LocalSystem {verdict}",
        ]

    def test_the_rows_the_script_emits_are_the_rows_the_roster_expects(
        self, tmp_path: pathlib.Path
    ) -> None:
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
            )
        )
        assert expected[-1] == runner_audit.ExpectedCheck(
            check_id=f"account:windows:{_SERVICE}:LocalSystem",
            reason=runner_account.SERVICE_ACCOUNT_REASON,
        )
        assert [f"CHECK {check['check_id']} OK" for check in expected] == _audit(
            tmp_path, "LocalSystem"
        )
