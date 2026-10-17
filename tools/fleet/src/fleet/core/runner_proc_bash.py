"""The bash functions the runner host's root scripts share, defined once.

The reaper (:mod:`fleet.core.runner_reaper_render`), the job scope helper
(:mod:`fleet.core.runner_job_scope_render`) and the action cache's adopt
script (:mod:`fleet.core.runner_cache_adopt`) each read a process table and a
runner unit's cgroups. Each script is rendered whole onto the host, so the
functions are text embedded in each; this module is the one place that text
is written. Every function reads through ``$proc_root`` and ``$cgroup_root``,
which each script sets to ``/proc`` and ``/sys/fs/cgroup`` and its tests point
at a laid-out table.
"""

from __future__ import annotations

from fleet.contracts.runner_slice import CI_SLICE_NAME

#: What names a job scope: ``<prefix><unit without .service>-<Worker pid>.scope``
#: in :data:`~fleet.contracts.runner_slice.CI_SLICE_NAME` (MCPs board task
#: 715b9ec5). The pid is last and has no dash, so the unit is everything
#: before the last dash, read exactly: a glob alone would give the scopes of
#: ``...lavender-wsl-2`` to ``...lavender-wsl`` too.
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

__all__ = ["JOB_SCOPE_PREFIX", "PROCESS_FUNCTIONS", "UNIT_PROCS_FUNCTION"]
