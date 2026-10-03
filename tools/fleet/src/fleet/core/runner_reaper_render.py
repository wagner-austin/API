"""The bash that keeps a finished GitHub Actions job from holding a runner's memory.

MCPs board task 53528106. On 2026-10-02 lavender's runners.slice held
17,180,516,352 bytes of its 19,327,352,832 ceiling while no job ran: 145
processes in one runner unit and 13 in another, pytorch data workers,
forkserver workers, vitest and npm ci, up to 6,611 seconds old, left by
Check runs GitHub had cancelled. Every push cancels the previous Check, and
the runner kills only its worker, so whatever the job had daemonized stayed
in the unit's cgroup. The load average reached 125, and GitHub read all
eight runners offline although systemd read each one running.

Two pieces, both idempotent:

1. A per-runner drop-in, ``KillMode=control-group``. ``svc.sh`` writes the
   unit with ``KillMode=process``, under which a ``systemctl restart`` left
   all 137 processes in place (measured 04:55Z that day); with the whole
   group, a stop or restart ends everything the unit holds. THE TRADEOFF,
   written down as the task asks: a stop now ends a running job's whole
   tree, which a stop already did to the job itself. The runner's
   self-update is unaffected, because it never stops the unit: the
   Listener exits and ``runsvc.sh``, the unit's main process, starts the
   new one.
2. A reaper, run by a timer every 30 seconds as root, because a job ends
   inside a running unit and no ``KillMode`` applies then. In each runner
   unit with a ``Runner.Listener`` alive, a process in the unit's cgroup
   that does not descend from its main process is one a job daemonized
   and init adopted. With no ``Runner.Worker`` alive every such process
   outlived its job and is killed. With one alive, only the orphan trees
   whose root started before the earliest live Worker are: a job's
   processes all start after its Worker, so an older one belongs to a job
   before it, and so does whatever that old process forks later, which is
   why the root's start is read rather than the process's own
   (gpt6-idle-1001's rule, measured as a forkserver's late worker). That
   second rule is measured, not assumed: across 124 snapshots of lavender
   on 2026-10-02 every busy runner went from one job straight into the
   next, so a reaper waiting for an idle unit would never have run there.
   A unit with no Listener is mid self-update, whose ``_update.sh`` is
   outside the tree by design, and is left alone that pass.
3. A Worker older than the host's ``job_timeout_minutes`` is no live
   Worker. GitHub ends every job by then, and a Worker still alive is one
   whose cancel never arrived: at 10:09Z on 2026-10-02 an API runner GitHub
   read offline still held a Worker and its pytest 3 h 22 min after the
   Worker's log stopped, and because a live Worker spared the unit, 49
   forkservers and 12,678,709,248 bytes stayed with it. Such a Worker does
   not shield the unit's orphans, and it and every process under it are
   killed with them.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.script_values import scriptable

#: The runner drop-in's file name inside ``<unit>.d``.
KILL_DROP_IN_NAME = "fleet-kill.conf"

#: The drop-in's body.
KILL_DROP_IN = "[Service]\nKillMode=control-group\n"

#: Where the reaper is installed.
REAPER_PATH = "/usr/local/sbin/fleet-runner-reaper"

#: The reaper's units, by file name under ``/etc/systemd/system``.
REAPER_SERVICE_NAME = "fleet-runner-reaper.service"
REAPER_TIMER_NAME = "fleet-runner-reaper.timer"

#: How often the reaper runs. Half the one minute the task allows from a
#: cancelled job to an empty cgroup.
REAPER_INTERVAL_SECONDS = 30

#: The reaper. ``SECONDS`` is the host's job timeout: past it a Worker is
#: stale. ``--audit SECONDS UNIT`` prints, for one unit, how many of the
#: processes it would kill are older than SECONDS, and kills nothing: the
#: audit's check (:mod:`fleet.core.runner_orphan_check`) asks the same
#: question the reaper acts on.
REAPER_SCRIPT = """#!/usr/bin/env bash
# fleet-runner-reaper -- rendered by fleet-runners (MCPs board task 53528106).
# Kills what a finished GitHub Actions job left in its runner unit's cgroup.
# Usage: fleet-runner-reaper SECONDS | fleet-runner-reaper --audit SECONDS UNIT
set -euo pipefail
cgroup_root=/sys/fs/cgroup
proc_root=/proc

parent_of() {
    local stat
    stat=$(cat "$proc_root/$1/stat" 2>/dev/null) || return 1
    stat=${stat##*) }
    set -- $stat
    echo "$2"
}

descends_from() {
    local pid=$1
    while [ "$pid" -gt 1 ]; do
        [ "$pid" = "$2" ] && return 0
        pid=$(parent_of "$pid") || return 1
    done
    return 1
}

runs() {
    local pid
    for pid in $2; do
        grep -qs "$1" "$proc_root/$pid/cmdline" && return 0
    done
    return 1
}

start_of() {
    local stat
    stat=$(cat "$proc_root/$1/stat" 2>/dev/null) || return 1
    stat=${stat##*) }
    set -- $stat
    echo "${20}"
}

age_of() {
    local ticks uptime
    ticks=$(start_of "$1") || return 1
    uptime=$(cut -d. -f1 "$proc_root/uptime")
    echo $(( uptime - ticks / $(getconf CLK_TCK) ))
}

tree_start_of() {
    local pid=$1 parent
    while :; do
        parent=$(parent_of "$pid") || return 1
        [ "$parent" -gt 1 ] || break
        pid=$parent
    done
    start_of "$pid"
}

leftovers() {
    local unit=$1 bound=$2 cgroup main pids pid started age since='' stale='' worker
    cgroup=$(systemctl show -p ControlGroup --value "$unit")
    [ -n "$cgroup" ] && [ -r "$cgroup_root$cgroup/cgroup.procs" ] || return 0
    main=$(systemctl show -p MainPID --value "$unit")
    [ "$main" != 0 ] || return 0
    pids=$(cat "$cgroup_root$cgroup/cgroup.procs")
    runs Runner.Listener "$pids" || return 0
    for pid in $pids; do
        grep -qs Runner.Worker "$proc_root/$pid/cmdline" || continue
        started=$(start_of "$pid") || continue
        age=$(age_of "$pid") || continue
        if [ "$age" -gt "$bound" ]; then
            stale="$stale $pid"
        elif [ -z "$since" ] || [ "$started" -lt "$since" ]; then
            since=$started
        fi
    done
    for pid in $pids; do
        if descends_from "$pid" "$main"; then
            for worker in $stale; do
                if descends_from "$pid" "$worker"; then
                    echo "$pid"
                    break
                fi
            done
            continue
        fi
        if [ -n "$since" ]; then
            started=$(tree_start_of "$pid") || continue
            [ "$started" -lt "$since" ] || continue
        fi
        echo "$pid"
    done
}

if [ "${1:-}" = --audit ]; then
    count=0
    for pid in $(leftovers "$3" "$2"); do
        age=$(age_of "$pid") || continue
        [ "$age" -le "$2" ] || count=$((count + 1))
    done
    echo "$count"
    exit 0
fi

if [ $# -ne 1 ]; then
    echo "fleet-runner-reaper: usage: fleet-runner-reaper SECONDS, the host job timeout" >&2
    exit 2
fi
bound=$1
for unit in $(systemctl list-units 'actions.runner.*' --no-legend --plain | cut -d' ' -f1); do
    killed=0
    for pid in $(leftovers "$unit" "$bound"); do
        if kill -KILL "$pid" 2>/dev/null; then
            killed=$((killed + 1))
        elif [ -e "$proc_root/$pid" ]; then
            echo "fleet-runner-reaper: $unit: could not kill $pid, a finished job's process"
            exit 1
        fi
    done
    if [ "$killed" -gt 0 ]; then
        echo "fleet-runner-reaper: $unit: killed $killed process(es) a finished job left behind"
    fi
done
"""

#: The reaper's timer.
REAPER_TIMER = f"""[Unit]
Description=Run fleet-runner-reaper every {REAPER_INTERVAL_SECONDS} seconds

[Timer]
OnBootSec=1min
OnUnitActiveSec={REAPER_INTERVAL_SECONDS}s
AccuracySec=5s

[Install]
WantedBy=timers.target
"""


def render_kill_mode_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that make a runner unit's stop end its whole cgroup.

    Run after ``svc.sh install`` has written the unit. A running unit takes
    the new ``KillMode`` at the reload, so its next stop or restart honours it.

    Args:
        install: A wsl-side install; its ``service`` is the unit's name.

    Returns:
        The lines. The drop-in is written, and systemd reloaded, only when
        the file does not already hold exactly :data:`KILL_DROP_IN`.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    directory = f"/etc/systemd/system/{unit}.d"
    path = f"{directory}/{KILL_DROP_IN_NAME}"
    body = KILL_DROP_IN.replace("\n", "\\n")
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(printf \'{body}\')" ]; then',
        f"    mkdir -p {directory}",
        f"    printf '{body}' > {path}",
        "    systemctl daemon-reload",
        f"    echo 'kill mode set for {unit}: a stop ends its whole cgroup'",
        "fi",
    ]


def render_reaper_service(spec: HostRunnerSpec) -> str:
    """The reaper's service, which passes the host's job timeout.

    Args:
        spec: The host; its ``job_timeout_minutes`` is the age past which a
            Worker is stale.

    Returns:
        The unit file's text.
    """
    seconds = spec["job_timeout_minutes"] * 60
    return (
        "[Unit]\n"
        "Description=Kill what finished GitHub Actions jobs left in their runner units\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart={REAPER_PATH} {seconds}\n"
    )


def render_reaper_lines(spec: HostRunnerSpec) -> list[str]:
    """Bash lines that install the reaper and start its timer.

    Args:
        spec: The host, whose job timeout the service passes the reaper.

    Returns:
        The lines: the script and both units written whole on every run, as
        ``ci-clean``'s are, then a reload and the timer enabled and started.
    """
    return [
        f"cat > {REAPER_PATH} <<'REAPER_EOF'",
        REAPER_SCRIPT.rstrip("\n"),
        "REAPER_EOF",
        f"chmod +x {REAPER_PATH}",
        f"cat > /etc/systemd/system/{REAPER_SERVICE_NAME} <<'UNIT_EOF'",
        render_reaper_service(spec).rstrip("\n"),
        "UNIT_EOF",
        f"cat > /etc/systemd/system/{REAPER_TIMER_NAME} <<'TIMER_EOF'",
        REAPER_TIMER.rstrip("\n"),
        "TIMER_EOF",
        "systemctl daemon-reload",
        f"systemctl enable --now {REAPER_TIMER_NAME}",
    ]


__all__ = [
    "KILL_DROP_IN",
    "KILL_DROP_IN_NAME",
    "REAPER_INTERVAL_SECONDS",
    "REAPER_PATH",
    "REAPER_SCRIPT",
    "REAPER_SERVICE_NAME",
    "REAPER_TIMER",
    "REAPER_TIMER_NAME",
    "render_kill_mode_lines",
    "render_reaper_lines",
    "render_reaper_service",
]
