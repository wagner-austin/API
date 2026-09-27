"""The scheduled task that holds a runner host's WSL VM open.

A WSL2 distro shuts down when idle, taking systemd and every WSL runner with
it, and a job whose labels match only offline self-hosted runners queues
forever without failing. The keepalive is a scheduled task running
``sleep infinity`` inside the distro from boot.

Lifted from MCPs' ``scripts/ops/provision-wsl-runner.ps1``, which registered
it as the provisioning user through S4U with an at-startup trigger. The
first rebuild recipe used ``schtasks /create`` with no account instead,
which registers an Interactive task that runs only while that user is
logged on: after the 2026-09-26 19:37Z reboot of lavender it sat at Ready,
and nothing held the distro open. S4U runs it with no logon; SYSTEM cannot
run it at all (WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED).
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec
from fleet.core.powershell_text import system32_parameter
from fleet.core.script_values import scriptable


def render_keepalive_parameters(spec: HostRunnerSpec) -> list[str]:
    """The Windows provision's param-block lines for the keepalive.

    The task's name, the distro it holds open and wsl.exe are parameters so
    the Pester suite over the committed provision registers a task under a
    minted name whose action is a stand-in, and deletes it afterwards.

    Args:
        spec: The host's roster entry.

    Returns:
        ``$KeepaliveTask``, ``$Distro`` and ``$Wsl``, each ending in a comma;
        none when the roster declares no keepalive task.

    Raises:
        ValueError: When the task or distro name cannot be embedded
            verbatim; see :func:`fleet.core.script_values.scriptable`.
    """
    raw_task = spec["keepalive_task"]
    if raw_task is None:
        return []
    task = scriptable(raw_task, label="keepalive_task")
    distro = scriptable(spec["wsl_distro"], label="wsl_distro")
    return [
        f"    [string]$KeepaliveTask = '{task}',",
        f"    [string]$Distro = '{distro}',",
        "    " + system32_parameter("Wsl", "wsl.exe") + ",",
    ]


def render_keepalive_lines(spec: HostRunnerSpec) -> list[str]:
    """Windows provision lines that register and start the host's WSL keepalive.

    Args:
        spec: The host's roster entry.

    Returns:
        The lines over :func:`render_keepalive_parameters`' parameters, none
        when the roster declares no keepalive task. The registration is
        ``-Force``, so a re-run replaces the Interactive task the first
        recipe left, and the principal is the running account, read from
        its token rather than from whoami.exe.
    """
    if spec["keepalive_task"] is None:
        return []
    return [
        "$KeepaliveAction = New-ScheduledTaskAction -Execute $Wsl "
        "-Argument ('-d ' + $Distro + ' --exec /usr/bin/sleep infinity')",
        "$KeepaliveTrigger = New-ScheduledTaskTrigger -AtStartup",
        "$KeepaliveUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name",
        "$KeepalivePrincipal = New-ScheduledTaskPrincipal -UserId $KeepaliveUser -LogonType S4U "
        "-RunLevel Highest",
        "$KeepaliveSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 "
        "-RestartInterval (New-TimeSpan -Minutes 1)",
        "Register-ScheduledTask -TaskName $KeepaliveTask -Action $KeepaliveAction "
        "-Trigger $KeepaliveTrigger -Principal $KeepalivePrincipal "
        "-Settings $KeepaliveSettings -Force | Out-Null",
        "Start-ScheduledTask -TaskName $KeepaliveTask",
        "Write-Output ('keepalive task ' + $KeepaliveTask + ' registered through S4U and started')",
    ]


__all__ = ["render_keepalive_lines", "render_keepalive_parameters"]
