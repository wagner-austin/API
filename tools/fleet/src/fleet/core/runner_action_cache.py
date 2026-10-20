"""Every action a runner's jobs use, kept on the host instead of fetched per job.

MCPs board task ce2b92e0. Each job's "Set up job" downloads every action it
uses from codeload.github.com, and lavender's eight runners start 60 to
67 jobs per MCPs Check run, all from one address. codeload answered 429 Too
Many Requests three times in a row and the job failed before running a step:
run 38582170660 on lavender-wsl-3 (2026-10-14T14:38Z), runs 38699479621 and
38709479412 on lavender-wsl-4 (2026-10-15 08:59Z and 09:14Z), each on
``actions/checkout@v7`` (commit 3d3c42e5aac5ba805825da76410c181273ba90b1).

THE RUNNER'S OWN CACHE (actions/runner #2857, 2023; read in
``src/Runner.Worker/ActionManager.cs`` at v2.338.0). A runner whose
environment names ``ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE`` looks for
``<cache>/<owner>_<repo>/<sha>.tar.gz`` on Linux, keyed by the commit the
job resolved the action's ref to, and copies that file instead of
downloading. An archive it does not find it downloads as it always did, so
the cache can only remove downloads, and every job's log says which it did:
``Action archive cache usage: <owner>/<repo>@<sha> use cache True``.

TWO PIECES, each idempotent:

1. Each runner unit gets a drop-in naming the cache, and is restarted to
   read it the first minute it is idle (:mod:`fleet.core.runner_cache_adopt`).
2. A root timer, ``fleet-action-cache-fill``, reads the ``uses:`` lines of
   the workflows and composite actions in the runners' own checkouts (the
   roster's ``workdir`` of every install), resolves each ref with
   ``git ls-remote`` the way the runner resolved it (measured: v7 of
   actions/checkout read 3d3c42e5aac5 both ways; an annotated tag is asked
   for as ``<tag>^{}``, because ls-remote prints the peeled commit for a
   pattern only when that ref is named, measured on git's v2.40.0), and
   builds each commit's archive once into :data:`CACHE_DIR`. An archive no
   workflow has named for :data:`PRUNE_DAYS` days is removed, so a window in
   which the checkouts are gone never empties the cache.

THE ARCHIVE IS BUILT WITH GIT, NOT DOWNLOADED FROM CODELOAD. The fill first
fetched codeload's own archive, and on lavender at 12:41Z and again at
12:53Z on 2026-10-15 codeload answered 429 for exactly the URL the
uncached runners were still asking for, ``actions/checkout/tar.gz/3d3c42e``,
while the same commit's zip came at 12:53Z. So the fill fetches the one
commit from github.com over git, the transport ``ls-remote`` already uses,
and writes it with ``git archive`` under one top directory, which is all
the runner asks of an archive: it unwraps whatever single directory it
finds (``ActionManager.cs``: "repository archive from github always
contains a nested folder"). Measured on the hub: ``git fetch --depth 1``
of actions/checkout at 3d3c42e then ``git archive --format=tar.gz``
wrote 480,054 bytes under one directory.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec
from fleet.core.runner_unit_files import render_unit_file_lines
from fleet.core.script_values import scriptable

#: The variable the runner reads (``Constants.Variables.Agent``).
CACHE_VARIABLE = "ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE"

#: The runners' cache, readable by the runner account, written by root.
CACHE_DIR = "/var/cache/fleet-actions"

#: The runner unit drop-in's file name inside ``<unit>.d``.
CACHE_DROP_IN_NAME = "fleet-action-cache.conf"

#: The drop-in's body.
CACHE_DROP_IN = f"[Service]\nEnvironment={CACHE_VARIABLE}={CACHE_DIR}\n"

#: Where the fill is installed.
FILL_PATH = "/usr/local/sbin/fleet-action-cache-fill"

#: The fill's units, by file name under ``/etc/systemd/system``.
FILL_SERVICE_NAME = "fleet-action-cache-fill.service"
FILL_TIMER_NAME = "fleet-action-cache-fill.timer"

#: Days an archive no workflow names is kept before the fill removes it.
PRUNE_DAYS = 30

#: The fill. It names the CACHE the runners read and the WORKDIRs whose
#: checkouts it reads, and builds each archive as ``.tar.gz``, the runner's
#: Linux format (``git archive --format``). It fails when any reference
#: resolved to nothing or any archive was not built, after building every
#: other one: on lavender's first run, 2026-10-15T12:41Z, codeload answered
#: 429 for actions/checkout@v7 alone.
FILL_SCRIPT = (
    r"""#!/usr/bin/env bash
# fleet-action-cache-fill -- rendered by fleet-runners (MCPs board task ce2b92e0).
# Keeps one archive of each action the runners' checked-out workflows use,
# at the commit its ref names now, where ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE
# points the runners; removes an archive no workflow has named for PRUNE_DAYS days.
# Usage: fleet-action-cache-fill CACHE WORKDIR...
set -euo pipefail
github=https://github.com
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

# Writes owner/repo at SHA as a .tar.gz archive to OUT, under one top
# directory, from a shallow fetch of that one commit into a scratch
# repository, named with --git-dir so OUT is read from where the fill runs.
# The bytes are the commit's own: core.autocrlf is off for the archive, as a
# host's own setting would convert line endings codeload never does.
build() {
    local scratch status
    scratch=$(mktemp -d)
    if git init -q --bare "$scratch" \
        && git --git-dir="$scratch" fetch -q --depth 1 "$github/$1" "$2" \
        && git --git-dir="$scratch" -c core.autocrlf=false archive --format=tar.gz \
            --prefix="${1/\//-}-${2:0:12}/" -o "$3" "$2"; then
        status=0
    else
        status=1
    fi
    rm -rf "$scratch"
    return "$status"
}

if [ $# -lt 2 ]; then
    echo "fleet-action-cache-fill: usage: fleet-action-cache-fill CACHE WORKDIR..." >&2
    exit 2
fi
cache=$1
shift

# Its last line counts what it did, and it fails when any reference resolved
# to nothing or any archive was not built.
declare -A named=()
examined=0 cached=0 held=0 failed=0 pruned=0
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
    archive="$cache/${repo/\//_}/$sha.tar.gz"
    named[$archive]=1
    if [ -f "$archive" ]; then
        touch "$archive"
        held=$((held + 1))
        continue
    fi
    mkdir -p "${archive%/*}"
    if build "$repo" "$sha" "$archive.part"; then
        mv "$archive.part" "$archive"
        echo "fleet-action-cache-fill: cached $repo@$ref at $sha"
        cached=$((cached + 1))
    else
        rm -f "$archive.part"
        echo "fleet-action-cache-fill: could not fetch $repo@$ref at $sha from $github/$repo"
        failed=$((failed + 1))
    fi
done
for archive in "$cache"/*/*.tar.gz; do
    if [ -f "$archive" ] && [ -z "${named[$archive]:-}" ] \
        && [ -n "$(find "$archive" -mtime +"$prune_days")" ]; then
        rm -f "$archive"
        pruned=$((pruned + 1))
    fi
done
echo "fleet-action-cache-fill: into $cache: $examined action reference(s)" \
    "in the checkouts of $# workdir(s): $cached cached now, $held already held," \
    "$failed failed; $pruned unnamed for $prune_days days removed"
[ "$failed" -eq 0 ]
"""
)

#: How often the fill runs, and how soon after boot. Fifteen minutes, not an
#: hour: a pass that holds every archive asks ls-remote once per action and
#: fetches nothing, and an archive a pass could not build is built again
#: before most of an hour's jobs have fetched the action themselves.
FILL_TIMER = """[Unit]
Description=Run fleet-action-cache-fill every 15 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
"""


def render_fill_service(spec: HostRunnerSpec) -> str:
    """The fill's service: one oneshot over every install's work tree.

    Args:
        spec: The host's roster entry.

    Returns:
        The unit file's text.

    Raises:
        ValueError: When a workdir cannot be embedded verbatim.
    """
    workdirs = " ".join(scriptable(i["workdir"], label="workdir") for i in spec["installs"])
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
                f"ExecStart={FILL_PATH} {CACHE_DIR} {workdirs}",
            ]
        )
        + "\n"
    )


def render_fill_lines(spec: HostRunnerSpec) -> list[str]:
    """Bash lines that install the fill, make the cache, and start its timer.

    Args:
        spec: The host's roster entry.

    Returns:
        The lines: the script written whole on every run, as the reaper's
        is, both units written when they differ, the cache directory made
        readable by every account, the timer enabled and started, and one
        pass started without waiting, so a provision run does not wait on
        the network and a failed pass does not fail the provision.

    Raises:
        ValueError: When a workdir cannot be embedded verbatim.
    """
    return [
        f"cat > {FILL_PATH} <<'FILL_EOF'",
        FILL_SCRIPT.rstrip("\n"),
        "FILL_EOF",
        f"chmod +x {FILL_PATH}",
        *render_unit_file_lines(
            f"/etc/systemd/system/{FILL_SERVICE_NAME}",
            render_fill_service(spec),
            announcement=f"{FILL_SERVICE_NAME} written",
        ),
        *render_unit_file_lines(
            f"/etc/systemd/system/{FILL_TIMER_NAME}",
            FILL_TIMER,
            announcement=f"{FILL_TIMER_NAME} written",
        ),
        f"install -d -m 0755 {CACHE_DIR}",
        f"systemctl enable --now {FILL_TIMER_NAME}",
        f"systemctl start --no-block {FILL_SERVICE_NAME}",
    ]


__all__ = [
    "CACHE_DIR",
    "CACHE_DROP_IN",
    "CACHE_DROP_IN_NAME",
    "CACHE_VARIABLE",
    "FILL_PATH",
    "FILL_SCRIPT",
    "FILL_SERVICE_NAME",
    "FILL_TIMER",
    "FILL_TIMER_NAME",
    "PRUNE_DAYS",
    "render_fill_lines",
    "render_fill_service",
]
