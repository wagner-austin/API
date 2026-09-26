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
from fleet.core.script_values import scriptable


def render_keepalive_lines(spec: HostRunnerSpec) -> list[str]:
    """PowerShell lines that register and start the host's WSL keepalive.

    Args:
        spec: The host's roster entry.

    Returns:
        The lines, none when the roster declares no keepalive task. The
        registration is ``-Force``, so a re-run replaces the Interactive
        task the first recipe left.

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
        "$KeepaliveAction = New-ScheduledTaskAction -Execute 'C:\\Windows\\System32\\wsl.exe' "
        f"-Argument '-d {distro} --exec /usr/bin/sleep infinity'",
        "$KeepaliveTrigger = New-ScheduledTaskTrigger -AtStartup",
        "$KeepalivePrincipal = New-ScheduledTaskPrincipal -UserId (whoami) -LogonType S4U "
        "-RunLevel Highest",
        "$KeepaliveSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 "
        "-RestartInterval (New-TimeSpan -Minutes 1)",
        f"Register-ScheduledTask -TaskName '{task}' -Action $KeepaliveAction "
        "-Trigger $KeepaliveTrigger -Principal $KeepalivePrincipal "
        "-Settings $KeepaliveSettings -Force | Out-Null",
        f"Start-ScheduledTask -TaskName '{task}'",
        f"Write-Output 'keepalive task {task} registered through S4U and started'",
    ]


__all__ = ["render_keepalive_lines"]
