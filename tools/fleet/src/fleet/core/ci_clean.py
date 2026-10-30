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
:data:`~fleet.contracts.runner_base.BUILD_CACHE_BOUND_GIB`, least recently
used first, the two passes diphtheria's corvis-docker-prune runs on its
stack's daemon (MCPs scripts/host/diphtheria/docker-prune.sh). The bound was
that script's 60 GB until board task 621902c3 held it to 20, and then to
10, inside the runner host's disk ceiling. A record a running build holds is
in use, so neither pass removes it.

THE PYTHON CACHES, UNGATED TOO (MCPs board task ecb13d3d). Until then this half
refused to run while any Runner.Worker executed, and on lavender one always
did: its journal for the 30 days to 2026-10-18 reads 'skipping this pass' on
every run and never the line that reports a reclaim. So each pass now carries
its own reason a running job is safe from it:

- a virtualenv goes only when every source path its ``.pth`` files record is
  gone, and a running job's checkout is not (:func:`orphan_sweep_lines`); one
  recording no path is kept, since poetry may still be creating it.
- poetry's, pip's and npm's download caches go only a whole generation at a
  time, one no running job reads
  (:data:`fleet.core.tool_caches.SH_REMOVE_UNUSED`, MCPs board task
  4b405239): each job reads the generation its job-started hook chose
  (:mod:`fleet.core.runner_tool_caches`). Until then this pass removed single
  files nothing had read for 14 days, which could part a poetry metadata
  file from its body and fail every later lock with ``IncompleteRead``.
  Which jobs run is read from the process table, never from the markers
  alone (:data:`RUNNER_GENERATIONS_IN_USE`, MCPs board task 9ab83f4b).

EVERY RUN REPORTS ITSELF (MCPs board tasks 46c7b11b and ecb13d3d, A2). A
service's last result says nothing of which copy of the script ran, and the
build cache is bounded only right after the prune: lavender's read 68.73 GB
Private at 2026-10-18T14:28Z, ten hours of builds after its 04:00 pass. So
each run writes :data:`CI_CLEAN_REPORT` in the state directory systemd makes
for the service: everything the passes printed, the instant it started, the
sha256 of the script that ran, and the daemon's build cache records as its
engine API counts them, in bytes, before the prunes and after them. The runner host's hygiene
case reads it against the render and the bound, and reads
the jobs' use of the generations from it (:data:`GENERATION_READERS`), since
the runner account's home is closed to the fleet's user.

WHAT IT DELIBERATELY KEEPS: hashed poetry venvs whose source project still
exists (per-package working state the runners share, not duplicates),
everything under any _work tree (in-project venvs persist there by design),
named docker volumes, and every tagged image, which is the pull cache.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.runner_base import BUILD_CACHE_BOUND_GIB
from fleet.core import names
from fleet.core.runner_install import RUNNER_ACCOUNT
from fleet.core.runner_proc_bash import PROCESS_FUNCTIONS, WORKER_CASE_PATTERN
from fleet.core.runner_tool_caches import RUNNER_MARKERS, RUNNER_TOOLS, RUNNER_VIRTUALENVS
from fleet.core.tool_caches import SH_REMOVE_UNUSED

#: The state directory systemd makes for the service, owned by its user.
STATE_DIRECTORY_NAME: Final = "ci-clean"

#: The report each run leaves, readable by every account on the host.
CI_CLEAN_REPORT: Final = f"/var/lib/{STATE_DIRECTORY_NAME}/last-run"

#: How the report's own lines open, after what the passes printed.
REPORT_STARTED: Final = "started: "
REPORT_DIGEST: Final = "script-sha256: "
REPORT_BUILD_CACHE: Final = "build-cache: "
REPORT_BUILD_CACHE_BEFORE: Final = "build-cache-before: "


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
ORPHAN_SWEEP: Final = "\n".join(orphan_sweep_lines(f'"{RUNNER_VIRTUALENVS}"'))

#: The runner host's ``generations_in_use MARKERS``: the marker of each
#: runner install a ``Runner.Worker`` runs from now, so the generation each
#: running job reads.
#:
#: A marker alone names the generation of its install's LAST job, running or
#: long finished, since the job-started hook writes it and nothing clears it.
#: On lavender on 2026-10-30T19:46Z markers kept generation 1792891106, 18
#: GB, for three installs that had run no job since it stopped being current
#: and for two (``actions-runner-mcps-3`` and ``-4``) that board task 49d1a916
#: had retired, which would have kept it forever (MCPs board task 9ab83f4b).
#: The hook that writes a marker runs under its job's Worker, so a job whose
#: marker this pass skips started after the pass read the processes, and
#: reads the current generation or one inside the removal's grace.
RUNNER_GENERATIONS_IN_USE: Final = (
    "proc_root=/proc",
    PROCESS_FUNCTIONS.rstrip("\n"),
    "generations_in_use() {",
    "    local process argv0 install marker",
    '    for process in "$proc_root"/[0-9]*/cmdline; do',
    "        process=${process%/cmdline}",
    '        argv0=$(argv0_of "${process##*/}") || continue',
    '        case "$argv0" in',
    f"            {WORKER_CASE_PATTERN}) ;;",
    "            *) continue ;;",
    "        esac",
    "        install=${argv0%/*/Runner.Worker}",
    f'        marker="$1/${{install##*/}}/{names.GENERATION_MARKER_NAME}"',
    '        if [ -f "$marker" ]; then cat "$marker"; fi',
    "    done",
    "}",
)

#: The removal rule over the runner account's generations, against the
#: markers of the runner installs running a job.
GENERATIONS_SWEEP: Final = "\n".join(
    (
        *SH_REMOVE_UNUSED,
        *RUNNER_GENERATIONS_IN_USE,
        "as=",
        f'remove_unused "{RUNNER_TOOLS}" "{RUNNER_MARKERS}"',
    )
)

#: What each runner install's last job read and what the current generation
#: holds, the jobs' own use of the generations, which the runner host's
#: hygiene case reads from the report: a job's tools write into a generation
#: only when the hook named it to them. npm's caches are listed one per
#: install, the others whole.
GENERATION_READERS: Final = "\n".join(
    (
        f'if [ -f "{RUNNER_TOOLS}/{names.CURRENT_GENERATION_NAME}" ]; then',
        f'    current=$(cat "{RUNNER_TOOLS}/{names.CURRENT_GENERATION_NAME}")',
        f'    for marker in "{RUNNER_MARKERS}"/*/{names.GENERATION_MARKER_NAME}; do',
        '        read_generation=$(cat "$marker")',
        "        install=${marker%/*}",
        '        echo "cache-reader: ${install##*/} read generation $read_generation"',
        "    done",
        f'    held_root="{RUNNER_TOOLS}/$current"',
        '    held=$(find "$held_root" -mindepth 1 -maxdepth 2 -type d'
        ' \\( -path "$held_root/npm/*" -o ! -path "$held_root/*/*" \\) -printf \'%P\\n\' | sort)',
        "    for tool in $held; do",
        "        size=$(find \"$held_root/$tool\" -type f -printf '%s\\n'"
        " | awk '{ total += $1 } END { print total + 0 }')",
        '        echo "cache-tool: generation $current holds $tool ($size bytes of files)"',
        "    done",
        "else",
        '    echo "cache-reader: no generation yet"',
        "fi",
    )
)

#: The Python half of :data:`CI_CLEAN_SCRIPT`, separate so a test executes it
#: against a home of its own without a docker daemon.
CI_CLEAN_CACHES: Final = f"""# The orphan rule; a virtualenv recording no source path stays
# (unknown provenance: keep).
{ORPHAN_SWEEP}

# npm's, pip's and poetry's download caches, a whole generation at a time.
{GENERATIONS_SWEEP}

# Which generation the jobs read, and what they wrote into the current one.
{GENERATION_READERS}
"""

#: The daily hygiene script, installed to /usr/local/bin/ci-clean.
CI_CLEAN_SCRIPT: Final = f"""#!/usr/bin/env bash
# ci-clean: daily hygiene for a CI runner host. Rendered from
# fleet.core.ci_clean -- edit THERE, never here.
set -euo pipefail

# The service's StateDirectory= names where each run's report goes.
report="${{STATE_DIRECTORY:?ci-clean runs as {STATE_DIRECTORY_NAME}.service}}/last-run"
started=$(date +%s)
digest=$(sha256sum "$0")
# The build cache as the daemon's engine API counts it, in bytes, before the
# prunes and again after them: one line of JSON each, too long for the
# journal, so they go to the report alone.
build_cache_url='http://localhost/system/df?type=build-cache'
build_cache_before=$(curl -sf --unix-socket /var/run/docker.sock "$build_cache_url")
{{
# Docker: it never prunes what a running container or build references.
# `volume prune` takes anonymous volumes only; named ones need --all.
docker container prune -f --filter until=24h
docker volume prune -f
docker image prune -f
docker builder prune -f --filter until=168h
docker builder prune -f --max-used-space {BUILD_CACHE_BOUND_GIB}gb

{CI_CLEAN_CACHES}}} | tee "$report.pending"

build_cache=$(curl -sf --unix-socket /var/run/docker.sock "$build_cache_url")
printf '%s\\n' "{REPORT_STARTED}$started" "{REPORT_DIGEST}${{digest%% *}}" \\
    | tee -a "$report.pending"
printf '%s\\n' "{REPORT_BUILD_CACHE_BEFORE}$build_cache_before" >> "$report.pending"
printf '%s\\n' "{REPORT_BUILD_CACHE}$build_cache" >> "$report.pending"
mv -f "$report.pending" "$report"
echo "ci-clean: reported to $report"
"""

#: systemd unit for the hygiene pass.
CI_CLEAN_SERVICE: Final = f"""[Unit]
Description=CI runner host hygiene (docker leftovers, poetry orphans, unread cache generations)

[Service]
Type=oneshot
User={RUNNER_ACCOUNT}
StateDirectory={STATE_DIRECTORY_NAME}
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
    "CI_CLEAN_CACHES",
    "CI_CLEAN_REPORT",
    "CI_CLEAN_SCRIPT",
    "CI_CLEAN_SERVICE",
    "CI_CLEAN_TIMER",
    "GENERATIONS_SWEEP",
    "GENERATION_READERS",
    "ORPHAN_SWEEP",
    "REPORT_BUILD_CACHE",
    "REPORT_BUILD_CACHE_BEFORE",
    "REPORT_DIGEST",
    "REPORT_STARTED",
    "RUNNER_GENERATIONS_IN_USE",
    "STATE_DIRECTORY_NAME",
    "orphan_sweep_lines",
]
