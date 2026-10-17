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

SINCE MCPs BOARD TASK 715b9ec5 A JOB RUNS OUTSIDE ITS UNIT'S CGROUP, in a
scope of the CI slice bound to the unit (:mod:`fleet.core.runner_job_scope_render`),
so what it daemonizes is left in that scope. Every rule above reads a unit's
processes as ``unit_procs`` lists them, its own cgroup's and its job
scopes', and a stop of the unit still ends the job: the scope is bound to it.

THE SAME TASK'S A2: ``--dropped UNIT PID START`` ends one Worker GitHub has
dropped, which rule 3 would spare until the job timeout. On 2026-10-16 six
such Workers held lavender-wsl's CI slice for 1 h 45 min against a six-hour
timeout. The host cannot tell such a Worker from a long job, so the hub names
it (:mod:`fleet.core.runner_reap_dropped`), by pid and by its start in clock
ticks since boot, and the reaper ends it only while that pid is still that
process, a Runner.Worker of that unit. It writes each pid's argv before any
signal, then kills the Worker and every process of the unit descending from
it, and never the listener above it. A Worker already gone is said and is no
fault, since a job may end between the hub's reading and this, while a
process that outlives SIGKILL is a refusal naming it.

THE SAME TASK'S LISTENER MOVE FINISHES HERE. The provision writes each
unit's ``Slice=`` drop-in for the listener slice
(:mod:`fleet.core.runner_slice_render`) but restarts only a unit no job runs
in, and lavender's runners go from one job straight into the next: at
2026-10-17 18:11Z all six API and MCPs units were still pending 28 minutes
after the first provision, each reading ``Slice=`` the listener slice while
its cgroup stayed in the CI slice. So each pass restarts a unit in exactly
that state the moment its Listener runs and it holds no Worker, in its
cgroup or its job scopes; a unit with no Listener is mid self-update, as
rule 2 says, and is left for a later pass.
"""

from __future__ import annotations

from fleet.contracts.runner_slice import LISTENER_SLICE_NAME
from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.runner_proc_bash import PROCESS_FUNCTIONS, UNIT_PROCS_FUNCTION
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

#: Seconds the ``--dropped`` mode waits, a second at a time, for every pid it
#: sent SIGKILL to be gone. The kernel ends a killed process at once unless it
#: sleeps uninterruptibly, which a slice over its memory high causes: ten
#: seconds outlasts such a sleep and still ends a hub pass promptly.
DROPPED_KILL_WAIT_SECONDS = 10

#: The reaper. ``SECONDS`` is the host's job timeout: past it a Worker is
#: stale. ``--audit SECONDS UNIT`` prints, for one unit, how many of the
#: processes it would kill are older than SECONDS, and kills nothing: the
#: audit's check (:mod:`fleet.core.runner_orphan_check`) asks the same
#: question the reaper acts on. ``--dropped UNIT PID START`` ends the tree of
#: the one Worker the hub names, as the module docstring says.
REAPER_SCRIPT = (
    (
        """#!/usr/bin/env bash
# fleet-runner-reaper -- rendered by fleet-runners (MCPs board task 53528106).
# Kills what a finished GitHub Actions job left in its runner unit's cgroup
# and in the job scopes its Workers were moved into, and, when the hub names
# one, a Runner.Worker whose job GitHub dropped (MCPs board task 715b9ec5).
# Usage: fleet-runner-reaper SECONDS | fleet-runner-reaper --audit SECONDS UNIT
#        | fleet-runner-reaper --dropped UNIT PID START
set -euo pipefail
cgroup_root=/sys/fs/cgroup
proc_root=/proc
kill_wait=$KILL_WAIT

"""
        + PROCESS_FUNCTIONS
        + "\n"
        + UNIT_PROCS_FUNCTION
        + """
refuse() {
    echo "FLEET_RUNNER_REAPER_REFUSED: $1" >&2
    exit 1
}

state_of() {
    local stat
    stat=$(cat "$proc_root/$1/stat" 2>/dev/null) || return 1
    stat=${stat##*) }
    set -- $stat
    echo "$1"
}

alive() {
    local state
    state=$(state_of "$1") || return 1
    [ "$state" != Z ]
}

held() {
    local pid
    for pid in $2; do
        [ "$pid" = "$1" ] && return 0
    done
    return 1
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
    pids=$(unit_procs "$cgroup" "$unit")
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

dropped() {
    local unit=$1 worker=$2 started=$3 cgroup pids pid tree='' count=0 survivors waited=0
    cgroup=$(systemctl show -p ControlGroup --value "$unit")
    if [ -z "$cgroup" ]; then
        echo "fleet-runner-reaper: $unit runs in no cgroup, so Runner.Worker $worker ended with it"
        return 0
    fi
    pids=$(unit_procs "$cgroup" "$unit") || refuse "$unit's cgroup $cgroup cannot be read"
    if ! held "$worker" "$pids" || [ "$(start_of "$worker" || true)" != "$started" ]; then
        echo "fleet-runner-reaper: $unit: Runner.Worker $worker, started at $started, has ended"
        return 0
    fi
    case "$(argv0_of "$worker" || true)" in
        */bin/Runner.Worker) ;;
        *) refuse "$unit's pid $worker, started at $started, is no Runner.Worker" ;;
    esac
    for pid in $pids; do
        if descends_from "$pid" "$worker"; then
            tree="$tree $pid"
            count=$((count + 1))
        fi
    done
    for pid in $tree; do
        echo "fleet-runner-reaper: $unit: ending pid $pid:" \\
            "$(tr '\\0' ' ' < "$proc_root/$pid/cmdline" 2>/dev/null || true)"
    done
    for pid in $tree; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    while :; do
        survivors=''
        for pid in $tree; do
            if alive "$pid"; then
                survivors="$survivors $pid"
            fi
        done
        [ -n "$survivors" ] || break
        [ "$waited" -lt "$kill_wait" ] ||
            refuse "$unit: pid(s)$survivors of Runner.Worker $worker's tree outlived SIGKILL"
        sleep 1
        waited=$((waited + 1))
    done
    echo "fleet-runner-reaper: $unit: ended Runner.Worker $worker and its tree, $count" \\
        "process(es), whose job GitHub dropped; its listener runs on"
}

place() {
    local unit=$1 cgroup pids
    [ "$(systemctl show -p Slice --value "$unit")" = $LISTENER_SLICE ] || return 0
    cgroup=$(systemctl show -p ControlGroup --value "$unit")
    case "$cgroup" in
        ''|/$LISTENER_SLICE/*) return 0 ;;
    esac
    pids=$(unit_procs "$cgroup" "$unit") || return 0
    # No Listener is a self-update under way, which a restart would cut short.
    runs Runner.Listener "$pids" || return 0
    runs Runner.Worker "$pids" && return 0
    systemctl restart "$unit"
    echo "fleet-runner-reaper: $unit: restarted between jobs into $LISTENER_SLICE, as its" \\
        "drop-in says"
}

if [ "${1:-}" = --dropped ]; then
    if [ $# -ne 4 ] || [ -z "$3" ] || [ -z "$4" ] || [ -n "${3//[0-9]/}${4//[0-9]/}" ]; then
        echo "fleet-runner-reaper: usage: fleet-runner-reaper --dropped UNIT PID START" >&2
        exit 2
    fi
    dropped "$2" "$3" "$4"
    exit 0
fi

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
    place "$unit"
done
"""
    )
    .replace("$KILL_WAIT", str(DROPPED_KILL_WAIT_SECONDS))
    .replace("$LISTENER_SLICE", LISTENER_SLICE_NAME)
)

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
    "DROPPED_KILL_WAIT_SECONDS",
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
