"""Restart-on-failure for every runner service, on both service managers.

MCPs' fleet-mcp boot standard (``fleet-mcp/src/standard-boot.ts``) holds
each declared unit to two rows: it is up after boot, and its manager
restarts it when it fails. After the 2026-09-26 19:37Z reboot of lavender
every runner read up, and all twelve failed the second row: the Windows
services carried config.cmd's default actions with restart-on-non-crash
failure off, so a runner that stops itself with an error stays stopped, and
the units svc.sh writes carry ``Restart=no``. These lines give both the same
policy the fleet's own sshd and Tailscale services carry.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall
from fleet.core.script_values import scriptable

#: The Windows failure actions: three restarts five seconds apart, the last
#: repeating for every later failure, the count reset after a quiet day.
WINDOWS_FAILURE_ACTIONS = "restart/5000/restart/5000/restart/5000"

#: Seconds without a failure after which the SCM resets its failure count.
WINDOWS_FAILURE_RESET_SECONDS = 86400

#: The systemd drop-in's file name inside ``<unit>.d``.
SYSTEMD_DROP_IN_NAME = "fleet-restart.conf"

#: The drop-in's body. ``always`` rather than ``on-failure``: the runner's
#: own wrapper exits cleanly on some losses of connection, and an explicit
#: ``systemctl stop`` is still honoured.
SYSTEMD_DROP_IN = "[Service]\nRestart=always\nRestartSec=15\n"


def render_windows_recovery_lines(install: RunnerInstall) -> list[str]:
    """PowerShell lines that make the SCM restart a runner service on failure.

    Args:
        install: A windows-side install.

    Returns:
        The lines. Both settings are written on every run: writing a value
        the service already holds changes nothing, and a failed write
        throws with sc.exe's exit code.

    Raises:
        ValueError: When the service name cannot be embedded verbatim; see
            :func:`fleet.core.script_values.scriptable`.
    """
    service = scriptable(install["service"], label="service")
    return [
        f"& sc.exe failure '{service}' reset= {WINDOWS_FAILURE_RESET_SECONDS} "
        f"actions= {WINDOWS_FAILURE_ACTIONS} | Out-Null",
        f"if ($LASTEXITCODE -ne 0) {{ throw 'sc.exe failure {service} exited ' + $LASTEXITCODE }}",
        f"& sc.exe failureflag '{service}' 1 | Out-Null",
        "if ($LASTEXITCODE -ne 0) { "
        f"throw 'sc.exe failureflag {service} exited ' + $LASTEXITCODE }}",
    ]


def render_wsl_recovery_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that give a runner's systemd unit a restart policy.

    Run after ``svc.sh install`` has written the unit and before it starts.

    Args:
        install: A wsl-side install; its ``service`` is the unit's name.

    Returns:
        The lines. The drop-in is written, and systemd reloaded, only when
        the file does not already hold exactly :data:`SYSTEMD_DROP_IN`.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    directory = f"/etc/systemd/system/{unit}.d"
    path = f"{directory}/{SYSTEMD_DROP_IN_NAME}"
    body = SYSTEMD_DROP_IN.replace("\n", "\\n")
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(printf \'{body}\')" ]; then',
        f"    mkdir -p {directory}",
        f"    printf '{body}' > {path}",
        "    systemctl daemon-reload",
        f"    echo 'restart policy set for {unit}'",
        "fi",
    ]


__all__ = [
    "SYSTEMD_DROP_IN",
    "SYSTEMD_DROP_IN_NAME",
    "WINDOWS_FAILURE_ACTIONS",
    "WINDOWS_FAILURE_RESET_SECONDS",
    "render_windows_recovery_lines",
    "render_wsl_recovery_lines",
]
