"""The bash that holds a host's WSL runners to their CI budget.

MCPs board task 45a4f22b; the budget and the incident behind it are
:mod:`fleet.contracts.runner_slice`. Three pieces, all idempotent:

1. The slice unit, carrying the host's ``MemoryHigh``, ``MemoryMax`` and
   ``CPUWeight``, written and reloaded only when its text differs. systemd
   re-applies a loaded slice's cgroup limits on reload, so a changed budget
   reaches runners already inside it without restarting any of them.
2. A per-runner drop-in naming the slice, beside the restart policy of
   :mod:`fleet.core.runner_recovery`.
3. The move. ``Slice=`` takes effect only when a unit starts, so a runner
   still in its old cgroup is restarted, and only while it runs no job: a
   job is a ``Runner.Worker`` process in the unit's cgroup, and a restart
   under one would kill the build. A busy runner is named and left where it
   is; the next provision moves it.
"""

from __future__ import annotations

from fleet.contracts.runner_slice import CI_SLICE_NAME, CiSlice
from fleet.contracts.runners import RunnerInstall
from fleet.core.script_values import scriptable

#: The runner drop-in's file name inside ``<unit>.d``.
SLICE_DROP_IN_NAME = "fleet-slice.conf"

#: The drop-in's body.
SLICE_DROP_IN = f"[Service]\nSlice={CI_SLICE_NAME}\n"

#: Where the slice unit is written.
SLICE_UNIT_PATH = f"/etc/systemd/system/{CI_SLICE_NAME}"


def render_slice_unit(budget: CiSlice) -> str:
    """The slice unit's text for one host's budget.

    Args:
        budget: The host's CI budget.

    Returns:
        The unit file, newline-terminated.
    """
    return (
        "[Unit]\n"
        "Description=GitHub Actions runners' share of this VM, rendered by fleet-runners "
        "(MCPs board task 45a4f22b)\n"
        "\n"
        "[Slice]\n"
        "MemoryAccounting=yes\n"
        f"MemoryHigh={budget['memory_high_gb']}G\n"
        f"MemoryMax={budget['memory_max_gb']}G\n"
        "CPUAccounting=yes\n"
        f"CPUWeight={budget['cpu_weight']}\n"
    )


def render_slice_unit_lines(budget: CiSlice) -> list[str]:
    """Bash lines that write the slice unit when it differs, and start it.

    Args:
        budget: The host's CI budget.

    Returns:
        The lines; they print what changed.
    """
    body = render_slice_unit(budget)
    return [
        f'if [ "$(cat {SLICE_UNIT_PATH} 2>/dev/null)" != "$(cat <<\'SLICE_EOF\'',
        body.rstrip("\n"),
        "SLICE_EOF",
        ')" ]; then',
        f"    cat > {SLICE_UNIT_PATH} <<'SLICE_EOF'",
        body.rstrip("\n"),
        "SLICE_EOF",
        "    systemctl daemon-reload",
        f"    echo 'ci budget set: {CI_SLICE_NAME} MemoryHigh={budget['memory_high_gb']}G "
        f"MemoryMax={budget['memory_max_gb']}G CPUWeight={budget['cpu_weight']}'",
        "fi",
        f"systemctl start {CI_SLICE_NAME}",
    ]


def render_runner_slice_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that put one WSL runner in the CI slice.

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
        f"    /{CI_SLICE_NAME}/*) ;;",
        "    *)",
        # /dev/null first: a unit with no processes lists none, and grep
        # given no file would read this script's stdin instead.
        "        if grep -qs Runner.Worker /dev/null "
        '$(sed "s#.*#/proc/&/cmdline#" "/sys/fs/cgroup$cgroup/cgroup.procs"); then',
        f"            echo 'ci budget pending: {unit} is running a job; "
        "the next provision moves it'",
        "        else",
        f"            systemctl restart {unit}",
        f"            echo 'ci budget applied: {unit} moved into {CI_SLICE_NAME}'",
        "        fi",
        "        ;;",
        "esac",
    ]


__all__ = [
    "SLICE_DROP_IN",
    "SLICE_DROP_IN_NAME",
    "SLICE_UNIT_PATH",
    "render_runner_slice_lines",
    "render_slice_unit",
    "render_slice_unit_lines",
]
