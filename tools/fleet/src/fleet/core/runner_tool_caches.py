"""A CI runner host's package caches, one generation at a time (MCPs board task 4b405239).

The fleet nodes' rule (:mod:`fleet.core.tool_caches`) on the GitHub runners'
own account. Until it, poetry, pip and npm read ``~/.cache`` and ci-clean
removed single files from those caches that nothing had read for 14 days
(MCPs board task ecb13d3d). Poetry keeps each repository cache entry as two
files, ``<key>`` and ``<key>.body``; measured 2026-10-18 with poetry 2.4.2, a
metadata file left without its body failed every later ``poetry lock`` with
``IncompleteRead``, and under ``relatime`` a 304 revalidation that rewrites
the metadata alone parts the two files' access times by up to a day, so the
per-file rule could split an entry.

So the runners' job-started hook (:mod:`fleet.core.runner_job_scope_render`)
runs :data:`HOOK_LINES` before the job's first step: it reads, or starts, the
generation under :data:`RUNNER_TOOLS`, writes the name into its runner
install's own marker under :data:`RUNNER_MARKERS`, and hands the job the
generation's paths through ``GITHUB_ENV``, which the runner reads back into
every later step's environment. Measured on lavender-wsl on 2026-10-18 with a
probe hook, the hook's environment carries ``GITHUB_ENV`` and ``RUNNER_TEMP``,
and ``RUNNER_NAME`` reads ``lavender-wsl`` for the API and the MCPs runners
alike, so the marker is named by the runner install's directory, which
``RUNNER_TEMP`` lies under, never by ``RUNNER_NAME``. A runner runs one job at
a time, so its marker names the generation its job reads, and ci-clean
removes a generation only whole, when no runner's marker names it
(:data:`fleet.core.tool_caches.SH_REMOVE_UNUSED`).

Poetry's virtualenvs stay where they were, :data:`RUNNER_VIRTUALENVS`: a
runner's checkout keeps its path from job to job, so its virtualenv is
reused, and ci-clean's orphan rule removes one whose sources are gone.
"""

from __future__ import annotations

from typing import Final

from fleet.core import names
from fleet.core.tool_caches import sh_build_lines

#: Where the runner account's generations live.
RUNNER_TOOLS: Final = "$HOME/.cache/fleet-tools"

#: Where each runner install's marker lives, one directory per install.
RUNNER_MARKERS: Final = "$HOME/.cache/fleet-jobs"

#: Where the runner account's poetry virtualenvs live, outside the generations.
RUNNER_VIRTUALENVS: Final = "$HOME/.cache/pypoetry/virtualenvs"

#: The hook's lines, before it moves the job into its scope. A ``RUNNER_TEMP``
#: outside a runner install's ``_work/_temp`` fails the job by name.
HOOK_LINES: Final = (
    'case "$RUNNER_TEMP" in',
    "    */_work/_temp) ;;",
    '    *) echo "FLEET_RUNNER_CACHES_REFUSED: RUNNER_TEMP $RUNNER_TEMP is no runner'
    " install's _work/_temp\" >&2; exit 1 ;;",
    "esac",
    'install_root="${RUNNER_TEMP%/_work/_temp}"',
    f'marker_dir="{RUNNER_MARKERS}/${{install_root##*/}}"',
    'mkdir -p "$marker_dir"',
    *sh_build_lines(
        tools=f'"{RUNNER_TOOLS}"',
        marker=f'"$marker_dir/{names.GENERATION_MARKER_NAME}"',
        run_as="",
    ),
    "{",
    "    printf 'POETRY_CACHE_DIR=%s\\n' \"$tool_cache/pypoetry\"",
    f"    printf 'POETRY_VIRTUALENVS_PATH=%s\\n' \"{RUNNER_VIRTUALENVS}\"",
    "    printf 'PIP_CACHE_DIR=%s\\n' \"$tool_cache/pip\"",
    "    printf 'npm_config_cache=%s\\n' \"$tool_cache/npm\"",
    '} >> "$GITHUB_ENV"',
    'echo "fleet-runner-job-started: npm, pip and poetry read cache generation $generation"',
)

__all__ = ["HOOK_LINES", "RUNNER_MARKERS", "RUNNER_TOOLS", "RUNNER_VIRTUALENVS"]
