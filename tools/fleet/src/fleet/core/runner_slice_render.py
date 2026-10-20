"""The bash that holds a host's runners to their CI budget.

MCPs board task 45a4f22b; the budget and the incident behind it are
:mod:`fleet.contracts.runner_slice`. Three pieces, all idempotent:

1. The two slice units: the CI slice, carrying the host's ``MemoryHigh``,
   ``MemoryMax``, ``MemorySwapMax``, ``CPUWeight`` and root-disk read bounds
   (board task cb264851), and the listener slice, carrying none (board task
   715b9ec5). Each is written and reloaded only when its text differs.
   systemd re-applies a loaded slice's cgroup limits on reload, so a changed
   budget reaches jobs already inside it without restarting any of them.
2. A per-runner drop-in naming the listener slice, beside the restart
   policy of :mod:`fleet.core.runner_recovery`, written before the unit
   first starts. The unit runs there and its jobs in scopes of the CI slice
   (:mod:`fleet.core.runner_job_scope_render`), so the budget throttles the
   jobs and never the listener holding their lease with GitHub.
3. The move. ``Slice=`` takes effect only when a unit starts, so a runner
   still in its old cgroup is restarted, and only while it runs no job: a
   job is a ``Runner.Worker`` process in the unit's cgroup, which is where
   every Worker of a unit outside the listener slice runs, and a restart
   under one would kill the build. A busy runner is named and left where it
   is, and the reaper's 30 s pass restarts it at its first moment between
   jobs (:mod:`fleet.core.runner_reaper_render`), since lavender's runners go
   from one job straight into the next and a provision seldom finds one idle.
"""

from __future__ import annotations

from fleet.contracts.runner_slice import (
    CI_SLICE_NAME,
    IO_BOUND_PATH,
    LISTENER_SLICE_NAME,
    CiSlice,
)
from fleet.contracts.runners import RunnerInstall
from fleet.core.runner_unit_files import render_drop_in_lines, render_unit_file_lines
from fleet.core.script_values import scriptable

#: The runner drop-in's file name inside ``<unit>.d``.
SLICE_DROP_IN_NAME = "fleet-slice.conf"

#: The drop-in's body.
SLICE_DROP_IN = f"[Service]\nSlice={LISTENER_SLICE_NAME}\n"

#: Where the CI slice unit is written.
SLICE_UNIT_PATH = f"/etc/systemd/system/{CI_SLICE_NAME}"

#: Where the listener slice unit is written.
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


def render_slice_unit_lines(budget: CiSlice) -> list[str]:
    """Bash lines that write both slice units when they differ, and start them.

    Args:
        budget: The host's CI budget.

    Returns:
        The lines; they print what changed.
    """
    return [
        *render_unit_file_lines(
            SLICE_UNIT_PATH,
            render_slice_unit(budget),
            announcement=f"ci budget set: {CI_SLICE_NAME} {' '.join(render_slice_bounds(budget))}",
        ),
        f"systemctl start {CI_SLICE_NAME}",
        *render_unit_file_lines(
            LISTENER_SLICE_UNIT_PATH,
            LISTENER_SLICE_UNIT,
            announcement=f"listener slice set: {LISTENER_SLICE_NAME}, bounded by nothing",
        ),
        f"systemctl start {LISTENER_SLICE_NAME}",
    ]


def render_runner_slice_drop_in_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that name the listener slice in one runner unit's drop-in.

    Run after ``svc.sh install`` and BEFORE ``svc.sh start``, so a new unit
    starts in the listener slice: a fresh runner takes a queued job within
    seconds of its first start, and on lavender's rebuild of 2026-10-20 the
    corvis-stick runner took one at 02:53:32Z, four seconds after starting
    in ``system.slice``, so that job ran outside the CI budget.

    Args:
        install: The install; its ``service`` is the unit's name.

    Returns:
        The lines. The drop-in is written, and systemd reloaded, only when
        it differs.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    return render_drop_in_lines(
        install,
        name=SLICE_DROP_IN_NAME,
        body=SLICE_DROP_IN,
        announcement=f"listener slice named for {unit}",
    )


def render_runner_slice_move_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that move a running unit still outside the listener slice into it.

    Run after ``svc.sh start``, so the move sees the unit's live cgroup. A
    unit that first started with :func:`render_runner_slice_drop_in_lines`
    in place is already there and is left alone; the move is for a rebuild
    over a unit an older provision started elsewhere.

    Args:
        install: The install; its ``service`` is the unit's name.

    Returns:
        The lines. The unit is restarted only when it is outside the slice
        and no job runs in it.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    return [
        f"cgroup=$(systemctl show -p ControlGroup --value {unit})",
        'case "$cgroup" in',
        f"    /{LISTENER_SLICE_NAME}/*) ;;",
        "    *)",
        # /dev/null first: a unit with no processes lists none, and grep
        # given no file would read this script's stdin instead.
        "        if grep -qs Runner.Worker /dev/null "
        '$(sed "s#.*#/proc/&/cmdline#" "/sys/fs/cgroup$cgroup/cgroup.procs"); then',
        f"            echo 'listener placement pending: {unit} is running a job; "
        "the reaper restarts it into the listener slice between jobs'",
        "        else",
        f"            systemctl restart {unit}",
        f"            echo 'listener placed: {unit} moved into {LISTENER_SLICE_NAME}, "
        f"its jobs into {CI_SLICE_NAME}'",
        "        fi",
        "        ;;",
        "esac",
    ]


__all__ = [
    "LISTENER_SLICE_UNIT",
    "LISTENER_SLICE_UNIT_PATH",
    "SLICE_ACCOUNTING",
    "SLICE_DROP_IN",
    "SLICE_DROP_IN_NAME",
    "SLICE_UNIT_PATH",
    "render_runner_slice_drop_in_lines",
    "render_runner_slice_move_lines",
    "render_slice_bounds",
    "render_slice_unit",
    "render_slice_unit_lines",
]
