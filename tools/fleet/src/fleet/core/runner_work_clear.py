"""A CI runner's job trees are emptied when each job ends (MCPs board task 621902c3).

Measured on lavender on 2026-10-30 from 20:28Z with ``du -xs -BM``, the
seven installs' ``_work`` trees held 29437 MB. 12095 MB of it was two venvs
the last job of each API install built and left: ``actions-runner-api-1``
kept ``services/handwriting-ai/.venv``, 5706 MB, and ``-api-2``
``services/Model-Trainer/.venv``, 6389 MB, torch venvs the workflow keeps out
of the Actions cache by design (API ``.github/workflows/packages.yml``). Each
MCPs install kept, beside its checkout, the sixteen sibling clones its jobs
make one level above the workspace, and 4774 MB, 682 MB in each install, was
the runner's own self-update of 2026-10-25 under ``_work/_update``, which
ci-clean removes (:data:`fleet.core.ci_clean.WORK_TREES`).

None of it is read again. Every workflow of every repository a runner here
serves checks out with actions/checkout's default ``clean: true``, which
removes every untracked and ignored file before the next job fetches, and
MCPs' Check removes each sibling clone and clones it again rather than fetch
one (``.github/workflows/check.yml``, 'A SELF-HOSTED RUNNER KEEPS ITS
WORKSPACE'). The state stays only until the next job on the same install, so
an install that runs nothing for a week holds its last job's gigabytes for a
week, and ci-clean, which runs beside the jobs, cannot remove a tree without
racing the job that may start in it.

So the runners' job-completed hook, ``ACTIONS_RUNNER_HOOK_JOB_COMPLETED``,
runs :data:`JOB_COMPLETED_SCRIPT` after the job's last step, in the job,
where no other job of its runner can start: it empties the job's workspace
and removes everything beside it under ``_work/<repository>``, after leaving
every directory in it writable (:mod:`fleet.core.runner_workspace_unlock`
says why a sealed one would fail the removal). It refuses a
``GITHUB_WORKSPACE`` that is not ``_work/<repository>/<name>`` of the job's
own install, so a workspace a workflow moved elsewhere is never emptied. It
then records the instant and what it removed in the install's marker
directory, where ci-clean reads it (:data:`fleet.core.ci_clean.WORK_TREES`),
since the runner account's home is closed to the fleet's user.

The workspace directory itself stays: the runner made it when the job was
set up, and a later job of the same repository runs its steps in it.
"""

from __future__ import annotations

from typing import Final

from fleet.core.runner_tool_caches import INSTALL_LINES

#: The runner's job-completed hook variable.
COMPLETED_HOOK_VARIABLE: Final = "ACTIONS_RUNNER_HOOK_JOB_COMPLETED"

#: Where the hook is installed. The runner picks a hook's shell by its
#: extension, so it ends ``.sh``.
JOB_COMPLETED_HOOK_PATH: Final = "/usr/local/bin/fleet-runner-job-completed.sh"

#: The record the hook leaves in its install's marker directory: the instant,
#: the entries it removed and their size in MB, on one line.
WORK_CLEARED_NAME: Final = "work-cleared"

#: The hook's lines after :data:`fleet.core.runner_tool_caches.INSTALL_LINES`
#: has named ``install_root`` and ``marker_dir``.
CLEAR_LINES: Final = (
    'work="$install_root/_work"',
    'job="${GITHUB_WORKSPACE:?the runner sets GITHUB_WORKSPACE}"',
    'job="${job#"$work"/}"',
    'case "$job" in',
    '    "$GITHUB_WORKSPACE" | */*/* | [._]* | */[._]*) job= ;;',
    "    ?*/?*) ;;",
    "    *) job= ;;",
    "esac",
    'if [ -z "$job" ]; then',
    '    echo "FLEET_RUNNER_WORK_REFUSED: GITHUB_WORKSPACE $GITHUB_WORKSPACE is no'
    ' _work/<repository>/<name> of $install_root" >&2',
    "    exit 1",
    "fi",
    'tree="$work/${job%%/*}"',
    'find "$tree" -xdev -type d -user "$(id -u)" ! -perm -u+w -exec chmod u+w {} +',
    'size=$(($(du -xsk "$tree" | cut -f1) / 1024))',
    'entries=$(find "$tree" -xdev -mindepth 1 ! -path "$GITHUB_WORKSPACE" | wc -l)',
    'find "$tree" -mindepth 1 -maxdepth 1 ! -path "$GITHUB_WORKSPACE"'
    " -exec rm -rf --one-file-system {} +",
    'find "$GITHUB_WORKSPACE" -mindepth 1 -maxdepth 1 -exec rm -rf --one-file-system {} +',
    f"echo \"$(date +%s) $entries $size\" > \"$marker_dir/{WORK_CLEARED_NAME}\"",
    'echo "fleet-runner-job-completed: removed $entries entries ($size MB) under $tree"',
)

#: The hook.
JOB_COMPLETED_SCRIPT: Final = (
    f"""#!/usr/bin/env bash
# fleet-runner-job-completed.sh -- rendered by fleet-runners (MCPs board task 621902c3).
# The runners' job-completed hook ({COMPLETED_HOOK_VARIABLE}): after the job's
# last step, its workspace is emptied and every clone beside it under
# _work/<repository> removed (fleet.core.runner_work_clear), and the removal is
# recorded where ci-clean reports it.
set -euo pipefail
"""
    + "".join(f"{line}\n" for line in (*INSTALL_LINES, *CLEAR_LINES))
)

__all__ = [
    "CLEAR_LINES",
    "COMPLETED_HOOK_VARIABLE",
    "JOB_COMPLETED_HOOK_PATH",
    "JOB_COMPLETED_SCRIPT",
    "WORK_CLEARED_NAME",
]
