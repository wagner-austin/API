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
   unit with a ``Runner.Listener`` alive and no ``Runner.Worker``, every
   process in the unit's cgroup that does not descend from its main
   process is killed: a job's processes descend from the Worker, so once
   none runs, a process outside that tree is one a finished job reparented
   to init. A unit with no Listener is mid self-update, whose ``_update.sh``
   is outside the tree by design, and is left alone that pass.
"""

from __future__ import annotations

from fleet.contracts.runners import RunnerInstall
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

#: The reaper. ``--audit SECONDS UNIT`` prints, for one unit, how many of
#: its processes outside the main process's tree are older than SECONDS
#: while no Worker runs, and kills nothing: the audit's check
#: (:mod:`fleet.core.runner_orphan_check`) asks the same question the reaper
#: acts on.
REAPER_SCRIPT = """#!/usr/bin/env bash
# fleet-runner-reaper -- rendered by fleet-runners (MCPs board task 53528106).
# Kills what a finished GitHub Actions job left in its runner unit's cgroup.
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

age_of() {
    local stat ticks uptime
    stat=$(cat "$proc_root/$1/stat" 2>/dev/null) || return 1
    stat=${stat##*) }
    set -- $stat
    ticks=${20}
    uptime=$(cut -d. -f1 "$proc_root/uptime")
    echo $(( uptime - ticks / $(getconf CLK_TCK) ))
}

leftovers() {
    local unit=$1 cgroup main pids pid
    cgroup=$(systemctl show -p ControlGroup --value "$unit")
    [ -n "$cgroup" ] && [ -r "$cgroup_root$cgroup/cgroup.procs" ] || return 0
    main=$(systemctl show -p MainPID --value "$unit")
    [ "$main" != 0 ] || return 0
    pids=$(cat "$cgroup_root$cgroup/cgroup.procs")
    runs Runner.Listener "$pids" || return 0
    ! runs Runner.Worker "$pids" || return 0
    for pid in $pids; do
        descends_from "$pid" "$main" || echo "$pid"
    done
}

if [ "${1:-}" = --audit ]; then
    count=0
    for pid in $(leftovers "$3"); do
        age=$(age_of "$pid") || continue
        [ "$age" -le "$2" ] || count=$((count + 1))
    done
    echo "$count"
    exit 0
fi

for unit in $(systemctl list-units 'actions.runner.*' --no-legend --plain | cut -d' ' -f1); do
    killed=0
    for pid in $(leftovers "$unit"); do
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

#: The reaper's service.
REAPER_SERVICE = f"""[Unit]
Description=Kill what finished GitHub Actions jobs left in their runner units

[Service]
Type=oneshot
ExecStart={REAPER_PATH}
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


def render_reaper_lines() -> list[str]:
    """Bash lines that install the reaper and start its timer.

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
        REAPER_SERVICE.rstrip("\n"),
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
    "REAPER_SERVICE",
    "REAPER_SERVICE_NAME",
    "REAPER_TIMER",
    "REAPER_TIMER_NAME",
    "render_kill_mode_lines",
    "render_reaper_lines",
]
