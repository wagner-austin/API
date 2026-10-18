"""ci-clean, the daily hygiene a CI runner host's provision installs.

:mod:`fleet.core.runner_render` writes these three payloads into the rendered
``provision.sh``: the script to ``/usr/local/bin/ci-clean``, and its service
and timer beside it. They live here, apart from the provision's other lines,
because they are a program in their own right that the tests execute.

DOCKER FIRST. The Actions runner removes a job's service containers but never
their anonymous volumes, and by 2026-09-25 lavender held 8,571 of them, 536
GB, which filled its disk and remounted every runner's root read-only (board
task 9e5c7a17). The workflows now mount a tmpfs over each declared volume;
this is the backstop for whatever image or job leaks next. Docker counts its
own references, so nothing a running container holds is pruned.

THE BUILD CACHE, BOUNDED TWICE (MCPs board task 46c7b11b). Every job that runs
docker build fills BuildKit's cache and no other pass reaches it: on
2026-10-18T07:32Z lavender's daemon held 103.7 GB of build cache, 98.55 GB of
it reclaimable, while the three prunes before it reclaimed 0 B a day. So the
cache no build has used for a week goes, and then the reclaimable cache above
60 GB, least recently used first, the two passes and the bound diphtheria's
corvis-docker-prune runs on its stack's daemon (MCPs
scripts/host/diphtheria/docker-prune.sh). A record a running build holds is in
use, so neither pass removes it.

THE PYTHON CACHES, UNGATED TOO (MCPs board task ecb13d3d). Until then this half
refused to run while any Runner.Worker executed, and on lavender one always
did: its journal for the 30 days to 2026-10-18 reads 'skipping this pass' on
every run and never the line that reports a reclaim. So each pass now carries
its own reason a running job is safe from it:

- a virtualenv goes only when every source path its ``.pth`` files record is
  gone, and a running job's checkout is not (:func:`orphan_sweep_lines`); one
  recording no path is kept, since poetry may still be creating it.
- poetry's artifacts and repository cache and pip's cache lose only the files
  nothing has read for :data:`CACHE_UNREAD_DAYS` days, then the directories
  that left empty and unchanged as long. A job reading a file moves its atime,
  at least daily on a ``relatime`` mount (lavender's root, 2026-10-18), so a
  file a job uses is never that old; and a download fragment a killed job left
  goes after a day, since a live one is newer.

On 2026-10-18T10:12Z lavender's poetry cache held 31 GB of virtualenvs, eight,
each read within two days and none an orphan; 8.8 GB of artifacts, 8 files of
405 unread for 14 days; and 8.9 GB of repository cache, 2837 files of 4106 and
8.97 GB unread for 14 days.

WHAT IT DELIBERATELY KEEPS: hashed poetry venvs whose source project still
exists (per-package working state the runners share, not duplicates),
everything under any _work tree (in-project venvs persist there by design),
named docker volumes, and every tagged image, which is the pull cache.
"""

from __future__ import annotations

from typing import Final

#: How long a cache file may go unread before the daily pass removes it.
CACHE_UNREAD_DAYS: Final = 14


def orphan_sweep_lines(venvs: str) -> tuple[str, ...]:
    """The orphan rule over a virtualenvs directory, as POSIX sh lines.

    Each ``.pth`` line is read through ``while read``, so a path holding a
    space is still one path, and an absolute one is a recorded source. A
    ``.pth`` file the pass cannot read stops it, naming the file, under the
    script's ``set -e``: read as recording nothing, it would keep its
    virtualenv with no word said (MCPs board task 346abc45, A2).

    Args:
        venvs: A shell word naming the virtualenvs directory, quoted by the
            caller.

    Returns:
        The lines. They leave ``removed``, ``total`` and ``kb`` set and print
        ``venv-sweep: removed <n> of <total> (<mb> MB)``.
    """
    return (
        f"venvs={venvs}",
        "total=0",
        "removed=0",
        "kb=0",
        'if [ -d "$venvs" ]; then',
        '    for venv in "$venvs"/*/; do',
        '        [ -d "$venv" ] || continue',
        "        total=$((total + 1))",
        "        recorded=0",
        "        alive=0",
        '        for pth in "$venv"lib/python*/site-packages/*.pth; do',
        '            [ -e "$pth" ] || continue',
        '            if [ ! -r "$pth" ]; then',
        '                echo "venv-sweep: cannot read $pth" >&2',
        "                exit 1",
        "            fi",
        '            while IFS= read -r source || [ -n "$source" ]; do',
        '                case "$source" in /*) ;; *) continue ;; esac',
        "                recorded=1",
        '                if [ -e "$source" ]; then alive=1; break; fi',
        '            done < "$pth"',
        '            [ "$alive" -eq 0 ] || break',
        "        done",
        '        [ "$recorded" -eq 1 ] || continue',
        '        [ "$alive" -eq 0 ] || continue',
        '        size=$(du -sk "$venv" | cut -f1)',
        '        rm -rf "$venv"',
        "        removed=$((removed + 1))",
        "        kb=$((kb + size))",
        "    done",
        "fi",
        'echo "venv-sweep: removed $removed of $total ($((kb / 1024)) MB)"',
    )


#: The orphan rule over the runner account's poetry virtualenvs.
ORPHAN_SWEEP: Final = "\n".join(orphan_sweep_lines('"$POETRY/virtualenvs"'))

#: The Python half of :data:`CI_CLEAN_SCRIPT`, separate so a test executes it
#: against a home of its own without a docker daemon.
CI_CLEAN_CACHES: Final = f"""POETRY="$HOME/.cache/pypoetry"
UNREAD_DAYS={CACHE_UNREAD_DAYS}
reclaimed_kb=0

# Removes the files under $1 that nothing has read for UNREAD_DAYS days, then
# the directories that left empty and unchanged as long, and counts the space.
# -ignore_readdir_race: a running job may remove a file between the listing
# and the test, and that file is not this pass's to report.
prune_unread() {{
    [ -d "$1" ] || return 0
    local freed_kb
    freed_kb=$(find "$1" -ignore_readdir_race -type f -atime +"$UNREAD_DAYS" \\
        -printf '%k\\n' -delete | awk '{{ s += $1 }} END {{ print s + 0 }}')
    find "$1" -ignore_readdir_race -mindepth 1 -type d -empty \\
        -mtime +"$UNREAD_DAYS" -delete
    reclaimed_kb=$((reclaimed_kb + freed_kb))
    echo "ci-clean: $1: ${{freed_kb}}KB unread for $UNREAD_DAYS days"
}}

# The orphan rule; a virtualenv recording no source path stays
# (unknown provenance: keep).
{ORPHAN_SWEEP}
reclaimed_kb=$((reclaimed_kb + kb))

if [ -d "$POETRY/artifacts" ]; then
    fragments_kb=$(find "$POETRY/artifacts" -ignore_readdir_race -type f -name 'tmp*' \\
        -mmin +1440 -printf '%k\\n' -delete | awk '{{ s += $1 }} END {{ print s + 0 }}')
    reclaimed_kb=$((reclaimed_kb + fragments_kb))
fi
prune_unread "$POETRY/artifacts"
prune_unread "$POETRY/cache"
prune_unread "$HOME/.cache/pip"

echo "ci-clean: reclaimed $((reclaimed_kb / 1024))MB this pass."
"""

#: The daily hygiene script, installed to /usr/local/bin/ci-clean.
CI_CLEAN_SCRIPT: Final = f"""#!/usr/bin/env bash
# ci-clean: daily hygiene for a CI runner host. Rendered from
# fleet.core.ci_clean -- edit THERE, never here.
set -euo pipefail

# Docker: it never prunes what a running container or build references.
# `volume prune` takes anonymous volumes only; named ones need --all.
docker container prune -f --filter until=24h
docker volume prune -f
docker image prune -f
docker builder prune -f --filter until=168h
docker builder prune -f --max-used-space 60gb

{CI_CLEAN_CACHES}"""

#: systemd unit for the hygiene pass.
CI_CLEAN_SERVICE: Final = """[Unit]
Description=CI runner host hygiene (docker leftovers, poetry orphans, unread caches)

[Service]
Type=oneshot
User=gharunner
ExecStart=/usr/local/bin/ci-clean
"""

#: systemd timer for the hygiene pass: daily, because a week of a leak
#: bounded only by the disk is what filled lavender.
CI_CLEAN_TIMER: Final = """[Unit]
Description=Daily CI hygiene pass

[Timer]
OnCalendar=*-*-* 04:00
Persistent=true

[Install]
WantedBy=timers.target
"""

__all__ = [
    "CACHE_UNREAD_DAYS",
    "CI_CLEAN_CACHES",
    "CI_CLEAN_SCRIPT",
    "CI_CLEAN_SERVICE",
    "CI_CLEAN_TIMER",
    "ORPHAN_SWEEP",
    "orphan_sweep_lines",
]
