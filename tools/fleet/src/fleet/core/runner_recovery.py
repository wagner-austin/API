"""Restart-on-failure for every runner service.

MCPs' fleet-mcp boot standard (``fleet-mcp/src/standard-boot.ts``) holds
each declared unit to two rows: it is up after boot, and its manager
restarts it when it fails. After the 2026-09-26 19:37Z reboot of lavender
every runner read up, and all of them failed the second row: the units
svc.sh writes carry ``Restart=no``, so a runner that stops itself with an
error stays stopped. This drop-in gives each unit the same policy the
fleet's own sshd and Tailscale services carry; the runner host's NixOS
generation carries it beside each runner unit
(:mod:`fleet.core.runner_host_document`).
"""

from __future__ import annotations

#: The systemd drop-in's file name inside ``<unit>.d``.
SYSTEMD_DROP_IN_NAME = "fleet-restart.conf"

#: The drop-in's body. ``always`` rather than ``on-failure``: the runner's
#: own wrapper exits cleanly on some losses of connection, and an explicit
#: ``systemctl stop`` is still honoured.
SYSTEMD_DROP_IN = "[Service]\nRestart=always\nRestartSec=15\n"


__all__ = ["SYSTEMD_DROP_IN", "SYSTEMD_DROP_IN_NAME"]
