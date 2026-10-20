"""Restart-on-failure for every runner service.

MCPs' fleet-mcp boot standard (``fleet-mcp/src/standard-boot.ts``) holds
each declared unit to two rows: it is up after boot, and its manager
restarts it when it fails. After the 2026-09-26 19:37Z reboot of lavender
every runner read up, and all of them failed the second row: the units
svc.sh writes carry ``Restart=no``, so a runner that stops itself with an
error stays stopped. These lines give each unit the same policy the fleet's
own sshd and Tailscale services carry.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall
from fleet.core.runner_unit_files import render_drop_in_lines

#: The systemd drop-in's file name inside ``<unit>.d``.
SYSTEMD_DROP_IN_NAME = "fleet-restart.conf"

#: The drop-in's body. ``always`` rather than ``on-failure``: the runner's
#: own wrapper exits cleanly on some losses of connection, and an explicit
#: ``systemctl stop`` is still honoured.
SYSTEMD_DROP_IN = "[Service]\nRestart=always\nRestartSec=15\n"


def render_recovery_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that give a runner's systemd unit a restart policy.

    Run after ``svc.sh install`` has written the unit and before it starts.

    Args:
        install: The install; its ``service`` is the unit's name.

    Returns:
        The lines. The drop-in is written, and systemd reloaded, only when
        the file does not already hold exactly :data:`SYSTEMD_DROP_IN`.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    return render_drop_in_lines(
        install,
        name=SYSTEMD_DROP_IN_NAME,
        body=SYSTEMD_DROP_IN,
        announcement=f"restart policy set for {install['service']}",
    )


__all__ = ["SYSTEMD_DROP_IN", "SYSTEMD_DROP_IN_NAME", "render_recovery_lines"]
