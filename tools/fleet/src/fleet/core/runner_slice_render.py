"""The units that hold a host's runners to their CI budget.

MCPs board task 45a4f22b; the budget and the incident behind it are
:mod:`fleet.contracts.runner_slice`. Two pieces, which the runner host's
NixOS generation carries (:mod:`fleet.core.runner_host_document`, MCPs board
task 9aba868d):

1. The two slice units: the CI slice, carrying the host's ``MemoryHigh``,
   ``MemoryMax``, ``MemorySwapMax``, ``CPUWeight`` and root-disk read bounds
   (board task cb264851), and the listener slice, carrying none (board task
   715b9ec5). systemd re-applies a loaded slice's cgroup limits on reload, so
   a changed budget reaches jobs already inside it without restarting any of
   them.
2. A per-runner drop-in naming the listener slice, beside the restart
   policy of :mod:`fleet.core.runner_recovery`. The unit runs there and its
   jobs in scopes of the CI slice (:mod:`fleet.core.runner_job_scope_render`),
   so the budget throttles the jobs and never the listener holding their
   lease with GitHub. A generation declares the drop-in before the unit's
   first start, so no runner ever starts outside the slice.
"""

from __future__ import annotations

from fleet.contracts.runner_slice import (
    CI_SLICE_NAME,
    IO_BOUND_PATH,
    LISTENER_SLICE_NAME,
    CiSlice,
)

#: The runner drop-in's file name inside ``<unit>.d``.
SLICE_DROP_IN_NAME = "fleet-slice.conf"

#: The drop-in's body.
SLICE_DROP_IN = f"[Service]\nSlice={LISTENER_SLICE_NAME}\n"

#: Where the CI slice unit lives.
SLICE_UNIT_PATH = f"/etc/systemd/system/{CI_SLICE_NAME}"

#: Where the listener slice unit lives.
LISTENER_SLICE_UNIT_PATH = f"/etc/systemd/system/{LISTENER_SLICE_NAME}"

#: The listener slice unit: accounted, so ``systemctl show`` and the audit
#: read what the listeners hold, and bounded by nothing.
LISTENER_SLICE_UNIT = (
    "[Unit]\n"
    "Description=GitHub Actions runners' listeners, outside their jobs' budget, rendered by "
    "fleet-runners (MCPs board task 715b9ec5)\n"
    "\n"
    "[Slice]\n"
    "MemoryAccounting=yes\n"
    "CPUAccounting=yes\n"
    "IOAccounting=yes\n"
)


#: The controllers the slice accounts, so each bound below is enforced and
#: ``systemctl show`` reports the slice's use against it.
SLICE_ACCOUNTING = ("MemoryAccounting=yes", "CPUAccounting=yes", "IOAccounting=yes")


def render_slice_bounds(budget: CiSlice) -> list[str]:
    """The slice's bound directives for one host's budget.

    Args:
        budget: The host's CI budget.

    Returns:
        One ``Name=value`` directive per bound: memory, swap, CPU weight and
        the root disk's read bandwidth and read operations, in that order.
        systemd reads ``G`` as 2^30 bytes and a bandwidth's ``M`` as 10^6.
    """
    return [
        f"MemoryHigh={budget['memory_high_gb']}G",
        f"MemoryMax={budget['memory_max_gb']}G",
        f"MemorySwapMax={budget['swap_max_gb']}G",
        f"CPUWeight={budget['cpu_weight']}",
        f"IOReadBandwidthMax={IO_BOUND_PATH} {budget['io_read_mb_per_s']}M",
        f"IOReadIOPSMax={IO_BOUND_PATH} {budget['io_read_iops']}",
    ]


def render_slice_unit(budget: CiSlice) -> str:
    """The slice unit's text for one host's budget.

    Args:
        budget: The host's CI budget.

    Returns:
        The unit file, newline-terminated.
    """
    directives = [*SLICE_ACCOUNTING, *render_slice_bounds(budget)]
    return (
        "[Unit]\n"
        "Description=GitHub Actions runners' share of this host, rendered by fleet-runners "
        "(MCPs board tasks 45a4f22b, cb264851)\n"
        "\n"
        "[Slice]\n" + "".join(f"{directive}\n" for directive in directives)
    )


__all__ = [
    "LISTENER_SLICE_UNIT",
    "LISTENER_SLICE_UNIT_PATH",
    "SLICE_ACCOUNTING",
    "SLICE_DROP_IN",
    "SLICE_DROP_IN_NAME",
    "SLICE_UNIT_PATH",
    "render_slice_bounds",
    "render_slice_unit",
]
