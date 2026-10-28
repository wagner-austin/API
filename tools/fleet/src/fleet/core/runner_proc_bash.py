"""The bash functions the runner host's root scripts share, defined once.

The reaper (:mod:`fleet.core.runner_reaper_render`), the job scope helper
(:mod:`fleet.core.runner_job_scope_render`) and the action cache's adopt
script (:mod:`fleet.core.runner_cache_adopt`) each read a process table and a
runner unit's cgroups. Each script is rendered whole onto the host, so the
functions are text embedded in each; this module is the one place that text
is written. Every function reads through ``$proc_root`` and ``$cgroup_root``,
which each script sets to ``/proc`` and ``/sys/fs/cgroup`` and its tests point
at a laid-out table. What names a ``Runner.Worker`` is here too, for those
scripts and for the Python that reads a host's processes
(:mod:`fleet.core.runner_load`).
"""

from __future__ import annotations

import fnmatch

from fleet.contracts.runner_slice import CI_SLICE_NAME

#: Where an install's ``Runner.Worker`` runs from, below the install's root:
#: ``bin`` as the runner's tarball unpacks, and ``bin.<version>`` once the
#: runner has updated itself, which it does unasked when GitHub publishes a
#: release. ``runsvc.sh`` still starts the Listener through ``bin``, then a
#: link to the newest ``bin.<version>``, and the Listener starts each Worker
#: by the path the link resolves to: on lavender, from its update to 2.338.0
#: at 2026-10-25T18:26Z, every Worker ran as ``bin.2.338.0/Runner.Worker``,
#: and the job scope helper, which knew only ``bin``, refused every job's
#: hook (MCPs board task 0e8d4118).
WORKER_PATHS = ("bin/Runner.Worker", "bin.[0-9]*/Runner.Worker")

#: :data:`WORKER_PATHS` under any install's root, as one ``case`` pattern over
#: a process's first argument.
WORKER_CASE_PATTERN = "|".join(f"*/{path}" for path in WORKER_PATHS)


def is_runner_worker(argv0: str, install_root: str) -> bool:
    """Whether a process's first argument is a ``Runner.Worker`` of an install.

    Args:
        argv0: The process's first argument.
        install_root: The install's root, a glob: ``*`` for any install's.

    Returns:
        True when it is the Worker of :data:`WORKER_PATHS` under that root.
    """
    return any(fnmatch.fnmatchcase(argv0, f"{install_root}/{path}") for path in WORKER_PATHS)


#: What names a job scope: ``<prefix><unit without .service>-<Worker pid>.scope``
#: in :data:`~fleet.contracts.runner_slice.CI_SLICE_NAME` (MCPs board task
#: 715b9ec5). The pid is last and has no dash, so the unit is everything
#: before the last dash, read exactly: a glob alone would give the scopes of
#: ``...lavender-2`` to ``...lavender`` too.
JOB_SCOPE_PREFIX = "fleet-job-"

#: A process's parent, its start in clock ticks since boot, its age in
#: seconds, its first argument, and its cgroup v2 path. Each prints nothing
#: and fails when the process has gone.
PROCESS_FUNCTIONS = r"""parent_of() {
    local stat
    stat=$(cat "$proc_root/$1/stat" 2>/dev/null) || return 1
    stat=${stat##*) }
    set -- $stat
    echo "$2"
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

argv0_of() {
    local argv=''
    [ -r "$proc_root/$1/cmdline" ] || return 1
    IFS= read -r -d '' argv < "$proc_root/$1/cmdline" || [ -n "$argv" ] || return 1
    echo "$argv"
}

cgroup_of() {
    local line
    line=$(cat "$proc_root/$1/cgroup" 2>/dev/null) || return 1
    echo "${line#0::}"
}
"""

#: Every process runner unit ``$2`` holds, given its cgroup ``$1``: those of
#: the unit's own cgroup, the listener's, and those of each job scope the job
#: scope helper adopted one of its Workers into. A job's own steps and what
#: it daemonizes start under its Worker, so they are in its scope. Fails,
#: listing nothing more, when the unit's own cgroup cannot be read, so no
#: caller reads an unreadable unit as one that runs no job.
UNIT_PROCS_FUNCTION = r"""unit_procs() {
    local scope name rest
    cat "$cgroup_root$1/cgroup.procs" 2>/dev/null || return 1
    for scope in "$cgroup_root/$JOB_SLICE/$JOB_PREFIX"*.scope; do
        name=${scope##*/}
        rest=${name#$JOB_PREFIX}
        rest=${rest%.scope}
        [ "${rest%-*}" = "${2%.service}" ] || continue
        # systemd removes a scope, file and all, once its last process exits;
        # one that goes between the glob and this read holds nothing to list.
        cat "$scope/cgroup.procs" 2>/dev/null || true
    done
}
""".replace("$JOB_SLICE", CI_SLICE_NAME).replace("$JOB_PREFIX", JOB_SCOPE_PREFIX)

__all__ = [
    "JOB_SCOPE_PREFIX",
    "PROCESS_FUNCTIONS",
    "UNIT_PROCS_FUNCTION",
    "WORKER_CASE_PATTERN",
    "WORKER_PATHS",
    "is_runner_worker",
]
