"""Where a Linux check runs: a system unit inside dispatch's share (MCPs board task bc8c445c).

UNTIL THIS, every Linux check was a transient unit of the login account's
own systemd manager (``systemd-run --user``), under ``user.slice``. Each
roster machine splits its cores and its disks between the owner's
``system-corvis.slice`` and dispatch's ``system-dispatch.slice`` (MCPs
``scripts/host/lib/cpu-shares.sh``, board tasks 355fba60 and bc8c445c), and
neither slice's quota or io weight reached a unit under ``user.slice``:
io.cost weighs a cgroup against its siblings, and a user unit's siblings are
the user manager's other units. Measured on diphtheria on 2026-10-23, a
doc-extract-api check beside a 17 GB docker load spent 70% of its test phase
stalled on io and ran 311 s against its 300 s budget.

NOW the check starts as a SYSTEM unit through :data:`CHECK_UNIT_HELPER`,
which MCPs' ``step_cpu_shares`` installs on every fleet node with a sudoers
rule letting the ``docker`` group run it (that group already reaches
``/var/run/docker.sock``, which is root). The helper runs nothing as root:
every unit it starts carries ``User=`` and the group of the account sudo
names, that account's ``XDG_RUNTIME_DIR`` and session bus, so ``systemctl
--user`` still answers inside a check, and it runs in
``system-dispatch-checks.slice``, a child of ``system-dispatch.slice``, so
inside dispatch's quota and weight, carrying systemd-oomd's
``ManagedOOMMemoryPressure=kill`` at 50% as ``user@.service`` did (fleet job
859e7ab3, :mod:`fleet.core.linux_unit_end`). Its verbs, each through
``sudo -n``:

- ``start <unit> <directory> <build> <ended>``: the build as a unit named
  ``<unit>``, its working directory, and ``<ended>`` as its
  ``ExecStopPost=``; both files must sit directly in the directory and,
  with it, belong to the caller.
- ``stop <unit>``: stop the unit if it is active and runs as the caller.
- ``run <program> [<argument>...]``: run one command in the same place and
  wait for it, its output and exit status passed back, which the toolchain
  probe asks its docker questions through (:mod:`fleet.core.
  linux_toolchain_probe`).
- ``loopback <unit> <program> [<argument>...]``: run one command of the
  caller's running check ``<unit>`` as ``<unit>-loopback.service`` with
  ``PrivateNetwork=yes``, in the directory it was called from, bound to the
  check's unit so whatever ends the check ends it, and wait for it, its
  output and exit status passed back. A project that declares
  ``loopback_only`` runs its ``make check`` so (:mod:`fleet.core.
  linux_loopback`, MCPs board task 649fcc7a).

A unit is ``fleet-<run id>`` (:func:`fleet.core.names.task_name`), with or
without ``.service``; the helper refuses any other name, and every refusal
exits 2 with a ``FLEET_CHECK_UNIT_REFUSED`` line. A node without the helper
refuses every launch, so a node is provisioned with it before a fleet
runner that launches through it is rolled.
"""

from __future__ import annotations

from typing import Final

#: The helper, as MCPs ``scripts/host/lib/cpu-shares.sh`` installs it.
CHECK_UNIT_HELPER: Final = "/usr/local/sbin/fleet-check-unit"

#: How every script calls it: through sudo, never asking for a password.
CHECK_UNIT: Final = f"sudo -n {CHECK_UNIT_HELPER}"

__all__ = [
    "CHECK_UNIT",
    "CHECK_UNIT_HELPER",
]
