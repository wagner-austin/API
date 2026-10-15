"""Every action a runner's jobs use, kept on the host instead of fetched per job.

MCPs board task ce2b92e0. Each job's "Set up job" downloads every action it
uses from codeload.github.com, and lavender's eight WSL runners start 60 to
67 jobs per MCPs Check run, all from one address. codeload answered 429 Too
Many Requests three times in a row and the job failed before running a step:
run 38582170660 on lavender-wsl-3 (2026-10-14T14:38Z), runs 38699479621 and
38709479412 on lavender-wsl-4 (2026-10-15 08:59Z and 09:14Z), each on
``actions/checkout@v7`` (commit 3d3c42e5aac5ba805825da76410c181273ba90b1).

THE RUNNER'S OWN CACHE (actions/runner #2857, 2023; read in
``src/Runner.Worker/ActionManager.cs`` at v2.338.0). A runner whose
environment names ``ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE`` looks for
``<cache>/<owner>_<repo>/<sha>.tar.gz`` on Linux, and ``.zip`` on Windows,
keyed by the commit the job resolved the action's ref to, and copies that
file instead of downloading. An archive it does not find it downloads as it
always did, so the cache can only remove downloads, and every job's log says
which it did: ``Action archive cache usage: <owner>/<repo>@<sha> use cache
True``.

THREE PIECES, each idempotent:

1. Each WSL runner unit gets a drop-in naming the cache. A unit takes an
   environment only when it starts, so one whose running process lacks the
   variable is restarted, and only while none of its processes is a
   ``Runner.Worker``: a restart ends a running job (the unit's ``KillMode``
   is ``control-group``), so a busy unit is named and left for the next
   provision run.
2. Each Windows runner service gets the same variable in its service
   environment (:mod:`fleet.core.runner_windows_provision`).
3. A root timer in the distro, ``fleet-action-cache-fill``, reads the
   ``uses:`` lines of the workflows and composite actions in the runners'
   own checkouts (the roster's ``workdir`` of every install), resolves each
   ref with ``git ls-remote`` the way the runner resolved it (measured: v7
   of actions/checkout read 3d3c42e5aac5 both ways; an annotated tag is
   asked for as ``<tag>^{}``, because ls-remote prints the peeled commit for
   a pattern only when that ref is named, measured on git's v2.40.0), and
   fetches each
   commit's archive once, the WSL installs' as ``.tar.gz`` into
   :data:`WSL_CACHE_DIR` and the Windows installs' as ``.zip`` into
   :data:`WINDOWS_CACHE_DIR` through ``/mnt``. codeload then sees one
   download per action version, not one per job. An archive no workflow has
   named for :data:`PRUNE_DAYS` days is removed, so a window in which the
   checkouts are gone never empties the cache.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.runner_distro import windows_to_wsl_path
from fleet.core.script_values import scriptable

#: The variable the runner reads (``Constants.Variables.Agent``).
CACHE_VARIABLE = "ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE"

#: The WSL runners' cache, readable by the runner account, written by root.
WSL_CACHE_DIR = "/var/cache/fleet-actions"

#: The Windows runners' cache, as Windows names it, forward-slashed.
WINDOWS_CACHE_DIR = "C:/fleet/action-cache"

#: The runner unit drop-in's file name inside ``<unit>.d``.
CACHE_DROP_IN_NAME = "fleet-action-cache.conf"

#: The drop-in's body.
CACHE_DROP_IN = f"[Service]\nEnvironment={CACHE_VARIABLE}={WSL_CACHE_DIR}\n"

#: Where the fill is installed.
FILL_PATH = "/usr/local/sbin/fleet-action-cache-fill"

#: The fill's units, by file name under ``/etc/systemd/system``.
FILL_SERVICE_NAME = "fleet-action-cache-fill.service"
FILL_TIMER_NAME = "fleet-action-cache-fill.timer"

#: Days an archive no workflow names is kept before the fill removes it.
PRUNE_DAYS = 30

#: The fill. Each pass, separated from the next by ``--``, names a FORMAT,
#: the archive the runners on that side read (tar.gz on Linux, zip on
#: Windows, and also the codeload path segment that serves it), a CACHE and
#: the WORKDIRs whose checkouts it reads. Every pass runs whatever an earlier
#: one did: on lavender's first run, 2026-10-15T12:41Z, codeload answered 429
#: for actions/checkout@v7, the WSL pass exited 1, and a second ExecStart
#: for the Windows pass never ran.
FILL_SCRIPT = (
    r"""#!/usr/bin/env bash
# fleet-action-cache-fill -- rendered by fleet-runners (MCPs board task ce2b92e0).
# Keeps one archive of each action the runners' checked-out workflows use,
# at the commit its ref names now, where ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE
# points the runners; removes an archive no workflow has named for PRUNE_DAYS days.
# Usage: fleet-action-cache-fill tar.gz|zip CACHE WORKDIR... [-- tar.gz|zip CACHE WORKDIR...]...
set -euo pipefail
github=https://github.com
codeload=https://codeload.github.com
"""
    + f"prune_days={PRUNE_DAYS}\n"
    + r"""
# A "uses:" line's action (1), its path inside the action's repository (2)
# and its ref (3).
uses="^[[:space:]-]*uses:[[:space:]]*['\"]?"
uses+="([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)(\/[^@'\"[:space:]]*)?"
uses+="@([^'\"[:space:]#]+).*"

# Every "uses: owner/repo[/path]@ref" in the workflows and composite actions
# the checkouts under each WORKDIR hold, as owner/repo@ref, once each.
references() {
    local workdir file
    for workdir in "$@"; do
        for file in "$workdir"/*/*/.github/workflows/*.yml \
            "$workdir"/*/*/.github/workflows/*.yaml \
            "$workdir"/*/*/.github/actions/*/action.yml \
            "$workdir"/*/*/.github/actions/*/action.yaml; do
            if [ -f "$file" ]; then
                sed -nE "s/$uses/\1@\3/p" "$file"
            fi
        done
    done | sort -u
}

# The commit owner/repo's ref names: a full sha as itself, else an annotated
# tag's commit, a lightweight tag's, or a branch's, in that order. The peeled
# ref is asked for by name: ls-remote prints it for a pattern only then.
resolve() {
    local listing name sha
    if [[ $2 =~ ^[0-9a-f]{40}$ ]]; then
        echo "$2"
        return 0
    fi
    listing=$(git ls-remote "$github/$1" \
        "refs/tags/$2" "refs/tags/$2^{}" "refs/heads/$2") || return 1
    for name in "refs/tags/$2^{}" "refs/tags/$2" "refs/heads/$2"; do
        sha=$(awk -v name="$name" '$2 == name { print $1 }' <<<"$listing")
        if [ -n "$sha" ]; then
            echo "$sha"
            return 0
        fi
    done
    return 1
}

# One pass: FORMAT CACHE WORKDIR... Its last line counts what it did, and it
# fails when any reference resolved to nothing or any archive was not fetched.
fill() {
    local format=$1 cache=$2
    shift 2
    local -A named=()
    local examined=0 cached=0 held=0 failed=0 pruned=0
    local reference repo ref sha archive
    for reference in $(references "$@"); do
        examined=$((examined + 1))
        repo=${reference%@*}
        ref=${reference#*@}
        if ! sha=$(resolve "$repo" "$ref"); then
            echo "fleet-action-cache-fill: $repo@$ref names no tag, branch or commit" \
                "at $github/$repo"
            failed=$((failed + 1))
            continue
        fi
        archive="$cache/${repo/\//_}/$sha.$format"
        named[$archive]=1
        if [ -f "$archive" ]; then
            touch "$archive"
            held=$((held + 1))
            continue
        fi
        mkdir -p "${archive%/*}"
        if curl -fsSL -o "$archive.part" "$codeload/$repo/$format/$sha"; then
            mv "$archive.part" "$archive"
            echo "fleet-action-cache-fill: cached $repo@$ref at $sha"
            cached=$((cached + 1))
        else
            rm -f "$archive.part"
            echo "fleet-action-cache-fill: could not fetch $repo@$ref at $sha from $codeload"
            failed=$((failed + 1))
        fi
    done
    for archive in "$cache"/*/*."$format"; do
        if [ -f "$archive" ] && [ -z "${named[$archive]:-}" ] \
            && [ -n "$(find "$archive" -mtime +"$prune_days")" ]; then
            rm -f "$archive"
            pruned=$((pruned + 1))
        fi
    done
    echo "fleet-action-cache-fill: $format into $cache: $examined action reference(s)" \
        "in the checkouts of $# workdir(s): $cached cached now, $held already held," \
        "$failed failed; $pruned unnamed for $prune_days days removed"
    [ "$failed" -eq 0 ]
}

usage() {
    echo "fleet-action-cache-fill: usage: fleet-action-cache-fill tar.gz|zip CACHE WORKDIR..." \
        "[-- tar.gz|zip CACHE WORKDIR...]..." >&2
    exit 2
}

# Every pass is read before any runs, so a malformed call changes nothing.
passes=()
pass=()
for argument in "$@" --; do
    if [ "$argument" != -- ]; then
        pass+=("$argument")
        continue
    fi
    if [ ${#pass[@]} -lt 3 ] || { [ "${pass[0]}" != tar.gz ] && [ "${pass[0]}" != zip ]; }; then
        usage
    fi
    passes+=("${pass[*]}")
    pass=()
done

# Each pass runs in its own subshell under set -e, and runs whatever the
# passes before it did; the fill fails when any of them did.
status=0
for each in "${passes[@]}"; do
    read -r -a pass <<<"$each"
    set +e
    (set -e; fill "${pass[@]}")
    passed=$?
    set -e
    if [ "$passed" -ne 0 ]; then
        status=1
    fi
done
exit "$status"
"""
)

#: How often the fill runs, and how soon after boot. Fifteen minutes, not an
#: hour: a pass that holds every archive asks ls-remote once per action and
#: downloads nothing, and an archive codeload refused is asked for again
#: before most of an hour's jobs have fetched it themselves.
FILL_TIMER = """[Unit]
Description=Run fleet-action-cache-fill every 15 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
"""


def wsl_workdirs(spec: HostRunnerSpec) -> list[str]:
    """Every WSL install's work tree, where its checkouts are.

    Args:
        spec: The host's roster entry.

    Returns:
        The ``workdir`` of each wsl-side install, in roster order.
    """
    return [i["workdir"] for i in spec["installs"] if i["side"] == "wsl"]


def windows_workdirs(spec: HostRunnerSpec) -> list[str]:
    """Every Windows install's work tree, as the distro reaches it.

    Args:
        spec: The host's roster entry.

    Returns:
        The ``/mnt`` form of each windows-side install's ``workdir``.

    Raises:
        ValueError: When a workdir is not a forward-slashed drive path.
    """
    return [windows_to_wsl_path(i["workdir"]) for i in spec["installs"] if i["side"] == "windows"]


def render_fill_service(spec: HostRunnerSpec) -> str:
    """The fill's service: one ``ExecStart``, one pass per side the host runs installs on.

    Args:
        spec: The host's roster entry.

    Returns:
        The unit file's text, a oneshot whose one ``ExecStart`` runs a pass
        for the WSL installs' ``.tar.gz`` cache and, after ``--``, one for
        the Windows installs' ``.zip`` cache, each only when the host has
        installs on that side. One command rather than one ``ExecStart``
        per pass, because systemd runs no ``ExecStart`` after one fails.

    Raises:
        ValueError: When a path cannot be embedded verbatim.
    """
    passes = [
        ("tar.gz", WSL_CACHE_DIR, wsl_workdirs(spec)),
        ("zip", windows_to_wsl_path(WINDOWS_CACHE_DIR), windows_workdirs(spec)),
    ]
    arguments = [
        f"{archive_format} {cache} " + " ".join(scriptable(w, label="workdir") for w in workdirs)
        for archive_format, cache, workdirs in passes
        if workdirs
    ]
    return (
        "\n".join(
            [
                "[Unit]",
                "Description=Keep an archive of every action the runners' workflows use",
                "Wants=network-online.target",
                "After=network-online.target",
                "",
                "[Service]",
                "Type=oneshot",
                f"ExecStart={FILL_PATH} " + " -- ".join(arguments),
            ]
        )
        + "\n"
    )


def render_fill_lines(spec: HostRunnerSpec) -> list[str]:
    """Bash lines that install the fill, make both caches, and start its timer.

    Args:
        spec: The host's roster entry.

    Returns:
        The lines: the script and both units written whole on every run, as
        the reaper's are, both cache directories made readable by every
        account, a reload, the timer enabled and started, and one pass
        started without waiting, so a provision run does not wait on the
        network and a failed pass does not fail the provision.
    """
    return [
        f"cat > {FILL_PATH} <<'FILL_EOF'",
        FILL_SCRIPT.rstrip("\n"),
        "FILL_EOF",
        f"chmod +x {FILL_PATH}",
        f"cat > /etc/systemd/system/{FILL_SERVICE_NAME} <<'UNIT_EOF'",
        render_fill_service(spec).rstrip("\n"),
        "UNIT_EOF",
        f"cat > /etc/systemd/system/{FILL_TIMER_NAME} <<'TIMER_EOF'",
        FILL_TIMER.rstrip("\n"),
        "TIMER_EOF",
        f"install -d -m 0755 {WSL_CACHE_DIR} {windows_to_wsl_path(WINDOWS_CACHE_DIR)}",
        "systemctl daemon-reload",
        f"systemctl enable --now {FILL_TIMER_NAME}",
        f"systemctl start --no-block {FILL_SERVICE_NAME}",
    ]


def render_cache_environment_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that give a runner unit the cache variable, restarting it only while idle.

    Run after the unit has been started. The drop-in is written, and systemd
    reloaded, only when the file does not already hold exactly
    :data:`CACHE_DROP_IN`; the unit is restarted only when its running main
    process lacks the variable and none of its processes is a
    ``Runner.Worker``.

    Args:
        install: A wsl-side install; its ``service`` is the unit's name.

    Returns:
        The lines.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    unit = scriptable(install["service"], label="service")
    directory = f"/etc/systemd/system/{unit}.d"
    path = f"{directory}/{CACHE_DROP_IN_NAME}"
    body = CACHE_DROP_IN.replace("\n", "\\n")
    setting = f"{CACHE_VARIABLE}={WSL_CACHE_DIR}"
    return [
        f'if [ "$(cat {path} 2>/dev/null)" != "$(printf \'{body}\')" ]; then',
        f"    mkdir -p {directory}",
        f"    printf '{body}' > {path}",
        "    systemctl daemon-reload",
        "fi",
        f"main=$(systemctl show -p MainPID --value {unit})",
        f'if ! tr "\\0" "\\n" < "/proc/$main/environ" 2>/dev/null | grep -qxF "{setting}"; then',
        "    busy=no",
        f'    for pid in $(cat "/sys/fs/cgroup$(systemctl show -p ControlGroup --value {unit})'
        '/cgroup.procs" 2>/dev/null); do',
        '        if grep -qs Runner.Worker "/proc/$pid/cmdline"; then',
        "            busy=yes",
        "        fi",
        "    done",
        '    if [ "$busy" = no ]; then',
        f"        systemctl restart {unit}",
        f"        echo 'action cache set for {unit}: restarted while idle'",
        "    else",
        f"        echo 'action cache set for {unit} at its next restart: it is running a job;"
        " run provision again to restart it'",
        "    fi",
        "fi",
    ]


def render_windows_cache_environment_lines() -> list[str]:
    """Windows provision lines that give a runner service the cache variable while idle.

    A service takes its ``Environment`` value only when it starts, and a
    service process's environment cannot be read back the way a unit's
    ``/proc/<pid>/environ`` can. So the value is written only while none of
    the install's ``Runner.Worker`` processes runs, and the service is
    restarted at once: a service whose value holds the setting is one that
    has been started with it. A busy service is named and left unwritten,
    for the next provision run.

    Returns:
        The lines, over the provision loop's ``$ServiceName`` and
        ``$Directory`` and its ``$ServicesKey``, ``$GetWorkers``,
        ``$StopService`` and ``$StartService`` parameters
        (:mod:`fleet.core.runner_windows_provision`). Any other entry the
        value holds is kept; an older setting of the same variable is
        replaced.
    """
    windows_dir = WINDOWS_CACHE_DIR.replace("/", "\\")
    setting = f"{CACHE_VARIABLE}={windows_dir}"
    return [
        "$ServiceKey = Join-Path $ServicesKey $ServiceName",
        "$Held = @()",
        "$Value = (Get-ItemProperty -LiteralPath $ServiceKey).PSObject.Properties['Environment']",
        "if ($null -ne $Value) {",
        "    $Held = @($Value.Value)",
        "}",
        f"if ($Held -notcontains '{setting}') {{",
        "    if (@(& $GetWorkers $Directory).Count -eq 0) {",
        f"        $Kept = @($Held | Where-Object {{ $_ -notlike '{CACHE_VARIABLE}=*' }})",
        "        New-ItemProperty -LiteralPath $ServiceKey -Name Environment "
        "-PropertyType MultiString "
        f"-Value ([string[]]($Kept + '{setting}')) -Force | Out-Null",
        "        & $StopService $ServiceName",
        "        & $StartService $ServiceName",
        "        Write-Output ('action cache set for ' + $ServiceName + ': restarted while idle')",
        "    } else {",
        "        Write-Output ('action cache not yet set for ' + $ServiceName + "
        "': it is running a job; run provision again to set it')",
        "    }",
        "}",
    ]


__all__ = [
    "CACHE_DROP_IN",
    "CACHE_DROP_IN_NAME",
    "CACHE_VARIABLE",
    "FILL_PATH",
    "FILL_SCRIPT",
    "FILL_SERVICE_NAME",
    "FILL_TIMER",
    "FILL_TIMER_NAME",
    "PRUNE_DAYS",
    "WINDOWS_CACHE_DIR",
    "WSL_CACHE_DIR",
    "render_cache_environment_lines",
    "render_fill_lines",
    "render_fill_service",
    "render_windows_cache_environment_lines",
    "windows_workdirs",
    "wsl_workdirs",
]
