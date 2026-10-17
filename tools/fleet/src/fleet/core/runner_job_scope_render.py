"""The bash that runs each GitHub Actions job outside its runner's listener.

MCPs board task 715b9ec5. Until it, every WSL runner unit, its
``Runner.Listener`` included, sat in the CI slice its jobs fill, and the
listener is what holds a job's lease with GitHub. On 2026-10-02 and again on
2026-10-16 the slice sat at its ``MemoryHigh`` and every listener's calls to
GitHub timed out, GitHub dropped the jobs, and their Workers ran on unseen
for 1 h 45 min. ``MemoryLow`` cannot protect a listener there: the kernel
puts every task that charges memory beneath a cgroup over its high to sleep,
whatever the task's own protection (measured, :data:`~fleet.contracts.
runner_slice.LISTENER_SLICE_NAME`), and the slice's read bounds and CPU
weight reach the listener the same way.

So the units run in :data:`~fleet.contracts.runner_slice.LISTENER_SLICE_NAME`,
which carries no bound, and each job is moved into a scope of
:data:`~fleet.contracts.runner_slice.CI_SLICE_NAME` before its first step:

1. A per-runner drop-in sets ``ACTIONS_RUNNER_HOOK_JOB_STARTED`` to
   :data:`JOB_HOOK_PATH`, which the runner runs, before the job's steps, as
   a descendant of the job's ``Runner.Worker``.
2. The hook runs :data:`JOB_SCOPE_PATH` as root through a sudoers rule that
   permits that one command with no argument.
3. The helper walks its own ancestors to the ``Runner.Worker``, refuses one
   whose parent is no ``Runner.Listener`` or which is not in a runner unit of
   the listener slice, and asks systemd to adopt it into
   ``<CI slice>/fleet-job-<unit without .service>-<pid>.scope``, bound to the
   unit: a stop of the unit stops the scope, so the unit's
   ``KillMode=control-group`` (:mod:`fleet.core.runner_reaper_render`) still
   ends its job. A Worker keeps its parent, so the listener still sees it
   end; everything the job then forks starts in the scope.

A refusal fails the job by name, before its first step, rather than running
it inside the listener's cgroup. The helper's ``--audit UNIT`` mode is the
audit's placement row (:mod:`fleet.core.runner_audit`), and its ``--workers
UNIT...`` mode the host's half of the dropped-job row
(:mod:`fleet.core.runner_dropped_check`).
"""

from __future__ import annotations

from fleet.contracts.runner_slice import CI_SLICE_NAME, LISTENER_SLICE_NAME
from fleet.contracts.runners import RunnerInstall
from fleet.core.runner_proc_bash import JOB_SCOPE_PREFIX, PROCESS_FUNCTIONS, UNIT_PROCS_FUNCTION
from fleet.core.script_values import scriptable

#: Where the helper is installed.
JOB_SCOPE_PATH = "/usr/local/sbin/fleet-runner-job-scope"

#: Where the hook is installed. The runner picks a hook's shell by its
#: extension, so it ends ``.sh``.
JOB_HOOK_PATH = "/usr/local/bin/fleet-runner-job-started.sh"

#: The runner's job-started hook variable.
HOOK_VARIABLE = "ACTIONS_RUNNER_HOOK_JOB_STARTED"

#: The runner drop-in's file name inside ``<unit>.d``.
HOOK_DROP_IN_NAME = "fleet-job-scope.conf"

#: The drop-in's body.
HOOK_DROP_IN = f"[Service]\nEnvironment={HOOK_VARIABLE}={JOB_HOOK_PATH}\n"

#: The sudoers file. No dot in the name: sudo skips such files in sudoers.d.
SUDOERS_PATH = "/etc/sudoers.d/fleet-runner-job-scope"

#: The rule. ``""`` permits the command with no argument and nothing else.
SUDOERS_RULE = f'gharunner ALL=(root) NOPASSWD: {JOB_SCOPE_PATH} ""\n'

#: Seconds a ``Runner.Worker`` may run in its unit's own cgroup before the
#: placement row counts it. The hook runs once the job is set up, which the
#: host's action cache makes a matter of seconds; five minutes leaves a slow
#: setup room and still names a job the hook never moved within one audit.
ADOPTION_GRACE_SECONDS = 300

#: The hook.
JOB_HOOK_SCRIPT = f"""#!/usr/bin/env bash
# fleet-runner-job-started.sh -- rendered by fleet-runners (MCPs board task 715b9ec5).
# The runners' job-started hook ({HOOK_VARIABLE}): before the job's first
# step, root moves this job's Runner.Worker out of its listener's cgroup into
# a scope of {CI_SLICE_NAME}. A refusal fails the job by name.
set -euo pipefail
exec sudo -n {JOB_SCOPE_PATH}
"""

#: The helper.
JOB_SCOPE_SCRIPT = (
    (
        r"""#!/usr/bin/env bash
# fleet-runner-job-scope -- rendered by fleet-runners (MCPs board task 715b9ec5).
# Moves the calling job's Runner.Worker out of its runner unit's cgroup in
# $LISTENER_SLICE into a scope of $JOB_SLICE bound to the unit, so the CI
# budget throttles the job and never the listener holding its lease.
# Usage: fleet-runner-job-scope                as root, from the job-started hook
#        fleet-runner-job-scope --audit UNIT    prints placed or misplaced, then why
#        fleet-runner-job-scope --workers UNIT... prints each unit and the age in
#                                               seconds of its oldest Runner.Worker, or -
set -euo pipefail
cgroup_root=/sys/fs/cgroup
proc_root=/proc
caller=$PPID
grace=$GRACE

refuse() {
    echo "FLEET_RUNNER_JOB_SCOPE_REFUSED: $1" >&2
    exit 1
}

"""
        + PROCESS_FUNCTIONS
        + "\n"
        + UNIT_PROCS_FUNCTION
        + r"""
adopt() {
    local pid=$caller worker='' step listener cgroup unit scope moved
    for step in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
        case "$(argv0_of "$pid")" in
            */bin/Runner.Worker) worker=$pid; break ;;
        esac
        pid=$(parent_of "$pid") || break
        [ "$pid" -gt 1 ] || break
    done
    [ -n "$worker" ] ||
        refuse "no Runner.Worker among the ancestors of $caller: only a job's hook may ask"
    listener=$(parent_of "$worker") || refuse "Runner.Worker $worker ended before it was moved"
    case "$(argv0_of "$listener")" in
        */bin/Runner.Listener) ;;
        *) refuse "the parent $listener of Runner.Worker $worker is no Runner.Listener" ;;
    esac
    cgroup=$(cgroup_of "$worker") || refuse "Runner.Worker $worker ended before it was moved"
    case "$cgroup" in
        /$LISTENER_SLICE/actions.runner.*.service) unit=${cgroup##*/} ;;
        *) refuse "Runner.Worker $worker is in $cgroup, not in a runner unit of $LISTENER_SLICE" ;;
    esac
    scope="$JOB_PREFIX${unit%.service}-$worker.scope"
    busctl --quiet call org.freedesktop.systemd1 /org/freedesktop/systemd1 \
        org.freedesktop.systemd1.Manager StartTransientUnit 'ssa(sv)a(sa(sv))' \
        "$scope" fail 5 \
        PIDs au 1 "$worker" \
        Slice s "$JOB_SLICE" \
        BindsTo as 1 "$unit" \
        CollectMode s inactive-or-failed \
        Description s "GitHub Actions job of $unit, outside its listener" 0 ||
        refuse "systemd would not adopt Runner.Worker $worker of $unit into $JOB_SLICE/$scope"
    moved=$(cgroup_of "$worker") || refuse "Runner.Worker $worker ended as it was moved"
    [ "$moved" = "/$JOB_SLICE/$scope" ] ||
        refuse "systemd took Runner.Worker $worker, yet it is in $moved, not $JOB_SLICE/$scope"
    echo "fleet-runner-job-scope: Runner.Worker $worker of $unit runs in $JOB_SLICE/$scope," \
        "outside its listener's cgroup"
}

audit() {
    local unit=$1 cgroup pid age workers=0 kept=0
    cgroup=$(systemctl show -p ControlGroup --value "$unit")
    if [ "$cgroup" != "/$LISTENER_SLICE/$unit" ]; then
        echo misplaced
        echo "$unit runs in ${cgroup:-no cgroup}, not $LISTENER_SLICE/$unit"
        return 0
    fi
    for pid in $(unit_procs "$cgroup" "$unit"); do
        grep -qs Runner.Worker "$proc_root/$pid/cmdline" || continue
        workers=$((workers + 1))
    done
    for pid in $(cat "$cgroup_root$cgroup/cgroup.procs"); do
        grep -qs Runner.Worker "$proc_root/$pid/cmdline" || continue
        age=$(age_of "$pid") || continue
        if [ "$age" -gt "$grace" ]; then
            kept=$((kept + 1))
        fi
    done
    if [ "$kept" -gt 0 ]; then
        echo misplaced
        echo "$unit runs in $LISTENER_SLICE, but $kept of its $workers Runner.Worker(s)" \
            "ran over $grace s in the listener's own cgroup, never moved to $JOB_SLICE"
        return 0
    fi
    echo placed
    echo "$unit runs in $LISTENER_SLICE, and each of its $workers Runner.Worker(s)" \
        "is in $JOB_SLICE or under $grace s old"
}

workers() {
    local unit cgroup pid age oldest
    for unit in "$@"; do
        oldest=-
        cgroup=$(systemctl show -p ControlGroup --value "$unit")
        if [ -n "$cgroup" ]; then
            for pid in $(unit_procs "$cgroup" "$unit"); do
                grep -qs Runner.Worker "$proc_root/$pid/cmdline" || continue
                age=$(age_of "$pid") || continue
                if [ "$oldest" = - ] || [ "$age" -gt "$oldest" ]; then
                    oldest=$age
                fi
            done
        fi
        echo "$unit $oldest"
    done
}

if [ $# -eq 0 ]; then
    adopt
elif [ $# -eq 2 ] && [ "$1" = --audit ]; then
    audit "$2"
elif [ $# -ge 2 ] && [ "$1" = --workers ]; then
    shift
    workers "$@"
else
    echo "fleet-runner-job-scope: usage: fleet-runner-job-scope |" \
        "fleet-runner-job-scope --audit UNIT | fleet-runner-job-scope --workers UNIT..." >&2
    exit 2
fi
"""
    )
    .replace("$LISTENER_SLICE", LISTENER_SLICE_NAME)
    .replace("$JOB_SLICE", CI_SLICE_NAME)
    .replace("$JOB_PREFIX", JOB_SCOPE_PREFIX)
    .replace("$GRACE", str(ADOPTION_GRACE_SECONDS))
)


def render_job_scope_lines() -> list[str]:
    """Bash lines that install the helper, the hook and the sudoers rule.

    Returns:
        The lines. The helper and the hook are written whole on every run, as
        the reaper is. The rule is written beside its final name, checked with
        ``visudo -cf`` and only then moved over it, so a rule sudo would
        refuse never reaches sudoers.d and fails the provision instead.
    """
    staged = f"{SUDOERS_PATH}.new"
    return [
        f"cat > {JOB_SCOPE_PATH} <<'JOB_SCOPE_EOF'",
        JOB_SCOPE_SCRIPT.rstrip("\n"),
        "JOB_SCOPE_EOF",
        f"chmod 0755 {JOB_SCOPE_PATH}",
        f"cat > {JOB_HOOK_PATH} <<'JOB_HOOK_EOF'",
        JOB_HOOK_SCRIPT.rstrip("\n"),
        "JOB_HOOK_EOF",
        f"chmod 0755 {JOB_HOOK_PATH}",
        f"cat > {staged} <<'SUDOERS_EOF'",
        SUDOERS_RULE.rstrip("\n"),
        "SUDOERS_EOF",
        f"chmod 0440 {staged}",
        f"visudo -cf {staged}",
        f"mv {staged} {SUDOERS_PATH}",
    ]


def render_job_hook_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that give one WSL runner unit the job-started hook.

    Run before the unit starts or is restarted into the listener slice
    (:func:`fleet.core.runner_slice_render.render_runner_slice_lines`), which
    is when its runner reads its environment.

    Args:
        install: A wsl-side install; its ``service`` is the unit's name.

    Returns:
        The lines. The drop-in is written, and systemd reloaded, only when the
        file does not already hold exactly :data:`HOOK_DROP_IN`.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    directory = f"/etc/systemd/system/{unit}.d"
    path = f"{directory}/{HOOK_DROP_IN_NAME}"
    body = HOOK_DROP_IN.replace("\n", "\\n")
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(printf \'{body}\')" ]; then',
        f"    mkdir -p {directory}",
        f"    printf '{body}' > {path}",
        "    systemctl daemon-reload",
        f"    echo 'job hook set for {unit}: each job runs in a scope of {CI_SLICE_NAME}'",
        "fi",
    ]


#: Why the placement row matters, printed beside a drift line.
PLACEMENT_REASON = (
    f"a listener inside {CI_SLICE_NAME} is throttled with the jobs filling it, and on "
    "2026-10-02 and 2026-10-16 every listener's calls to GitHub timed out there, GitHub "
    "dropped their jobs, and the jobs ran on unseen for 1 h 45 min"
)


def placement_check_id(install: RunnerInstall) -> str:
    """The placement row's id for one WSL install.

    Args:
        install: A wsl-side install.

    Returns:
        E.g. ``placement:wagner-austin/MCPs:wsl:lavender-wsl``.
    """
    return f"placement:{install['repo']}:{install['side']}:{install['runner_name']}"


def render_placement_check_lines(install: RunnerInstall) -> list[str]:
    """Audit lines for one WSL install's placement row.

    Args:
        install: A wsl-side install.

    Returns:
        The lines. The row passes only when the helper's ``--audit`` exits 0
        and its first line reads ``placed``: the unit runs in the listener
        slice, and no ``Runner.Worker`` older than
        :data:`ADOPTION_GRACE_SECONDS` runs in the unit's own cgroup. The
        helper's whole answer is the row's detail either way.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    service = scriptable(install["service"], label="service")
    check_id = placement_check_id(install)
    return [
        f"$Probe = Invoke-InDistro $Cmd $Wsl $Distro \"{JOB_SCOPE_PATH} --audit '{service}'\"",
        "$Placed = (@($Probe.Lines | Select-Object -First 1) -join '')",
        f"Write-Check '{check_id}' ($Probe.Exit -eq 0 -and $Placed -eq 'placed') "
        f"('{service} in {LISTENER_SLICE_NAME} and its jobs in {CI_SLICE_NAME}; the helper read: ' "
        "+ $Probe.Text)",
    ]


__all__ = [
    "ADOPTION_GRACE_SECONDS",
    "HOOK_DROP_IN",
    "HOOK_DROP_IN_NAME",
    "HOOK_VARIABLE",
    "JOB_HOOK_PATH",
    "JOB_HOOK_SCRIPT",
    "JOB_SCOPE_PATH",
    "JOB_SCOPE_SCRIPT",
    "PLACEMENT_REASON",
    "SUDOERS_PATH",
    "SUDOERS_RULE",
    "placement_check_id",
    "render_job_hook_lines",
    "render_job_scope_lines",
    "render_placement_check_lines",
]
