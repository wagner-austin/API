"""The bash that holds a host's WSL runners to their CI budget.

MCPs board task 45a4f22b; the budget and the incident behind it are
:mod:`fleet.contracts.runner_slice`. Three pieces, all idempotent:

1. The two slice units: the CI slice, carrying the host's ``MemoryHigh``,
   ``MemoryMax``, ``MemorySwapMax``, ``CPUWeight`` and root-disk read bounds
   (board task cb264851), and the listener slice, carrying none (board task
   715b9ec5). Each is written and reloaded only when its text differs.
   systemd re-applies a loaded slice's cgroup limits on reload, so a changed
   budget reaches jobs already inside it without restarting any of them.
2. A per-runner drop-in naming the listener slice, beside the restart
   policy of :mod:`fleet.core.runner_recovery`. The unit runs there and its
   jobs in scopes of the CI slice (:mod:`fleet.core.runner_job_scope_render`),
   so the budget throttles the jobs and never the listener holding their
   lease with GitHub.
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
        "Description=GitHub Actions runners' share of this VM, rendered by fleet-runners "
        "(MCPs board tasks 45a4f22b, cb264851)\n"
        "\n"
        "[Slice]\n" + "".join(f"{directive}\n" for directive in directives)
    )


def _render_unit_file_lines(path: str, body: str, said: str) -> list[str]:
    """Bash lines that write one unit file only when its text differs.

    Args:
        path: Where the unit is written.
        body: Its text.
        said: What the lines print when they write it.

    Returns:
        The lines.
    """
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(cat <<\'SLICE_EOF\'',
        body.rstrip("\n"),
        "SLICE_EOF",
        ')" ]; then',
        f"    cat > {path} <<'SLICE_EOF'",
        body.rstrip("\n"),
        "SLICE_EOF",
        "    systemctl daemon-reload",
        f"    echo '{said}'",
        "fi",
    ]


def render_slice_unit_lines(budget: CiSlice) -> list[str]:
    """Bash lines that write both slice units when they differ, and start them.

    Args:
        budget: The host's CI budget.

    Returns:
        The lines; they print what changed.
    """
    return [
        *_render_unit_file_lines(
            SLICE_UNIT_PATH,
            render_slice_unit(budget),
            f"ci budget set: {CI_SLICE_NAME} {' '.join(render_slice_bounds(budget))}",
        ),
        f"systemctl start {CI_SLICE_NAME}",
        *_render_unit_file_lines(
            LISTENER_SLICE_UNIT_PATH,
            LISTENER_SLICE_UNIT,
            f"listener slice set: {LISTENER_SLICE_NAME}, bounded by nothing",
        ),
        f"systemctl start {LISTENER_SLICE_NAME}",
    ]


def render_runner_slice_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that put one WSL runner unit in the listener slice.

    Run after the unit exists and after ``svc.sh start``, so the move sees
    the unit's live cgroup.

    Args:
        install: A wsl-side install; its ``service`` is the unit's name.

    Returns:
        The lines. The drop-in is written, and systemd reloaded, only when
        it differs; the unit is restarted only when it is outside the slice
        and no job runs in it.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    directory = f"/etc/systemd/system/{unit}.d"
    path = f"{directory}/{SLICE_DROP_IN_NAME}"
    body = SLICE_DROP_IN.replace("\n", "\\n")
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(printf \'{body}\')" ]; then',
        f"    mkdir -p {directory}",
        f"    printf '{body}' > {path}",
        "    systemctl daemon-reload",
        "fi",
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
    "render_runner_slice_lines",
    "render_slice_bounds",
    "render_slice_unit",
    "render_slice_unit_lines",
]
