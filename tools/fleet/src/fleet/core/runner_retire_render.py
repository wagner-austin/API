"""Retiring a runner unit the roster no longer declares.

The provision installs and starts every declared install, and every step of
it adds: until board task 49d1a916 nothing removed a unit an earlier
provision had installed once the roster dropped it. That task halves MCPs'
runners on lavender from four to two, because four concurrent MCPs check
jobs oversubscribe the CI slice's eight threads (MCPs board task 349e25ee)
and ran mcp-shared at 359 s and packages/db at 382 s against a 300 s
budget. Left alone, the two dropped units would keep taking jobs, and the
placement check (``tests/test_runner_placement_host.py``) holds the host's
loaded runner units equal to the declared ones.

So the provision ends by retiring each loaded ``actions.runner.*`` unit no
install of the host declares. It waits, up to the host's job timeout, until
no job of that unit runs: a job runs in its own scope,
``<prefix><unit without .service>-<Worker pid>.scope`` in the CI slice
(:data:`~fleet.core.runner_proc_bash.JOB_SCOPE_PREFIX`), and the unit is read
exactly, so the scopes of ``...lavender-2`` are never taken for
``...lavender``'s. It then stops and uninstalls the unit through the
``svc.sh`` in its working directory, removes its drop-ins and its install
directory, and reloads systemd. A job the listener accepts between the last
read and the stop is cancelled by the stop, as any stop of a runner cancels
it. Its GitHub registration is left for ``fleet-runners --prune yes``
(:mod:`fleet.core.runner_registrations`), which deletes a registration no
roster host declares once GitHub reads it offline.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.contracts.runners import HostRunnerSpec
from fleet.core.runner_proc_bash import JOB_SCOPE_PREFIX
from fleet.core.script_values import scriptable

#: Where the unified cgroup hierarchy is mounted on a runner host.
CGROUP_ROOT: Final[str] = "/sys/fs/cgroup"

#: Seconds between two reads of a retiring unit's job scopes.
RETIRE_POLL_SECONDS: Final[int] = 10


def render_retire_lines(spec: HostRunnerSpec) -> list[str]:
    """Bash lines that retire every loaded runner unit the host does not declare.

    Args:
        spec: The host's roster entry; its installs are the units kept, and
            its ``job_timeout_minutes`` bounds the wait for a running job.

    Returns:
        The lines. A unit still running a job when the timeout passes, or
        whose working directory holds no ``svc.sh``, is refused on stderr
        and the script exits 1, leaving that unit installed.

    Raises:
        ValueError: When a declared unit name cannot be embedded verbatim.
    """
    declared = " ".join(
        scriptable(install["service"], label="service") for install in spec["installs"]
    )
    timeout = spec["job_timeout_minutes"] * 60
    prefix = f"{CGROUP_ROOT}/{CI_SLICE_NAME}/{JOB_SCOPE_PREFIX}"
    return [
        f"declared=' {declared} '",
        "jobs_of() {",
        "    local scope pid",
        f'    for scope in "{prefix}$1"-*.scope; do',
        "        pid=${scope##*-}",
        "        pid=${pid%.scope}",
        f'        if [ "${{scope%-*}}" = "{prefix}$1" ] && [[ $pid =~ ^[0-9]+$ ]]; then',
        '            echo "$scope"',
        "        fi",
        "    done",
        "}",
        "for unit in $(systemctl list-units 'actions.runner.*' --all --no-legend --plain "
        "| awk '{print $1}'); do",
        '    case "$declared" in *" $unit "*) continue ;; esac',
        "    waited=0",
        '    while [ -n "$(jobs_of "${unit%.service}")" ]; do',
        f'        if [ "$waited" -ge {timeout} ]; then',
        f'            echo "runner retire refused: $unit still runs a job after {timeout} s" >&2',
        "            exit 1",
        "        fi",
        f"        sleep {RETIRE_POLL_SECONDS}",
        f"        waited=$((waited + {RETIRE_POLL_SECONDS}))",
        "    done",
        '    root=$(systemctl show -p WorkingDirectory --value "$unit")',
        '    if [ ! -x "$root/svc.sh" ]; then',
        '        echo "runner retire refused: $unit has no svc.sh in its working directory '
        "'$root'\" >&2",
        "        exit 1",
        "    fi",
        '    (cd "$root" && ./svc.sh stop && ./svc.sh uninstall)',
        '    rm -rf "/etc/systemd/system/$unit.d" "$root"',
        "    systemctl daemon-reload",
        '    echo "runner retired: $unit and its install $root; '
        'fleet-runners --prune yes deletes its registration"',
        "done",
    ]


__all__ = ["CGROUP_ROOT", "RETIRE_POLL_SECONDS", "render_retire_lines"]
