"""The WSL keepalive's registration, run for real.

The lines are executed under Windows PowerShell 5.1 with functions standing
in for the scheduler cmdlets and ``whoami``, each recording what it was
handed, so the test reads the principal, trigger, action and settings the
real ``Register-ScheduledTask`` would receive, and nothing is registered on
this machine.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_keepalive
from fleet.core.dialect_windows import POWERSHELL_INVOCATION
from tests._runner_fixtures import a_base

#: Stand-ins that print what each cmdlet received.
_FAKES = """function whoami { 'lavender\\test' }
function New-ScheduledTaskAction {
    param([string]$Execute, [string]$Argument)
    "ACTION $Execute | $Argument"
}
function New-ScheduledTaskTrigger { param([switch]$AtStartup) "TRIGGER AtStartup=$AtStartup" }
function New-ScheduledTaskPrincipal {
    param([string]$UserId, [string]$LogonType, [string]$RunLevel)
    "PRINCIPAL $UserId $LogonType $RunLevel"
}
function New-ScheduledTaskSettingsSet {
    param([switch]$AllowStartIfOnBatteries, [switch]$DontStopIfGoingOnBatteries,
          [TimeSpan]$ExecutionTimeLimit, [int]$RestartCount, [TimeSpan]$RestartInterval)
    "SETTINGS limit=$ExecutionTimeLimit restarts=$RestartCount every=$RestartInterval"
}
function Register-ScheduledTask {
    param([string]$TaskName, $Action, $Trigger, $Principal, $Settings, [switch]$Force)
    Write-Host "REGISTER $TaskName Force=$Force"
    Write-Host $Action
    Write-Host $Trigger
    Write-Host $Principal
    Write-Host $Settings
}
function Start-ScheduledTask { param([string]$TaskName) "START $TaskName" }
"""


def _host(keepalive: str | None) -> HostRunnerSpec:
    """A host whose only Windows-side declaration is the keepalive.

    Args:
        keepalive: The keepalive task's name, or ``None``.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task=keepalive,
        wslconfig_min_memory_gb=None,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=[],
        installs=[],
        assets=[],
        base=a_base(),
    )


class TestRendered:
    """What renders without running."""

    def test_no_declared_task_renders_nothing(self) -> None:
        assert runner_keepalive.render_keepalive_lines(_host(None)) == []

    def test_a_task_name_the_script_cannot_carry_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            runner_keepalive.render_keepalive_lines(_host("wsl'keepalive"))
        assert str(refused.value) == (
            'keepalive_task "wsl\'keepalive" contains "\'", which cannot be embedded in a '
            "rendered script verbatim; rename the item rather than escaping it"
        )


@pytest.mark.skipif(sys.platform != "win32", reason="the lines are PowerShell; run them here")
class TestRegistrationRunsForReal:
    """The lines executed against recording stand-ins."""

    def test_the_task_is_s4u_at_startup_restarting_and_started(
        self, tmp_path: pathlib.Path
    ) -> None:
        """S4U for the provisioning user, never Interactive, so it runs from
        boot with nobody logged on; forced, so a re-run replaces the
        Interactive task the first recipe left."""
        script = tmp_path / "keepalive.ps1"
        lines = runner_keepalive.render_keepalive_lines(_host("wsl-keepalive"))
        script.write_text(_FAKES + "\n".join(lines) + "\n", encoding="utf-8")
        ran = subprocess.run(
            [*POWERSHELL_INVOCATION, str(script)], capture_output=True, text=True, check=False
        )
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.splitlines() == [
            "REGISTER wsl-keepalive Force=True",
            "ACTION C:\\Windows\\System32\\wsl.exe | -d Ubuntu --exec /usr/bin/sleep infinity",
            "TRIGGER AtStartup=True",
            "PRINCIPAL lavender\\test S4U Highest",
            "SETTINGS limit=00:00:00 restarts=999 every=00:01:00",
            "START wsl-keepalive",
            "keepalive task wsl-keepalive registered through S4U and started",
        ]
