"""The package managers' caches on a fleet node, one generation at a time.

WHY GENERATIONS (MCPs board task 346abc45). npm's and poetry's caches are
shared by every build on a node and nothing bounded them: measured
2026-10-18, poetry's held 10 to 23 GB on each node, up to 4.6 GB of it unread
for 14 days, and neither tool evicts anything by age. Removing single files
from them is not safe: measured the same day, an npm index line left without
its content file made the next install fail with ENOENT, and a poetry
metadata file left without its body made every later lock fail with
``IncompleteRead``. So the fleet never edits a cache's files; it owns each
cache as a whole directory. A build reads the generation
:data:`~fleet.core.names.CURRENT_GENERATION_NAME` names under
``<cache root>/tools``, starting a new one, named by the epoch second it
starts, once that one is :data:`GENERATION_DAYS` days old, and writes the
name it read into its own run directory
(:data:`~fleet.core.names.GENERATION_MARKER_NAME`), from which
:mod:`fleet.core.cache_generations` knows which generations a live run still
reads and removes the rest whole.

A cache is then at most two generations, each a tool's own complete cache.
The cost is one cold install per project per node every
:data:`GENERATION_DAYS` days, in the install phase before a check's clock
starts.

Playwright's browsers stay outside the generations: Playwright removes, at
each ``playwright install``, every browser no live link names. Poetry's
virtualenvs are in the project (:data:`~fleet.core.names.POETRY_IN_PROJECT`),
so the retire removes them with the run.
"""

from __future__ import annotations

from typing import Final

from fleet.core import names

#: How long a generation is the one builds start in.
GENERATION_DAYS: Final = 14

#: :data:`GENERATION_DAYS` in seconds, as the scripts compare it.
GENERATION_SECONDS: Final = GENERATION_DAYS * 86400

#: How long a generation that stopped being current is kept even with no
#: marker naming it: a build reads the current name and writes its marker
#: one line later, so an hour covers any build that read it.
REMOVAL_GRACE_SECONDS: Final = 3600

#: The removal rule as a POSIX sh function, ``remove_unused TOOLS MARKERS``:
#: every generation under TOOLS that is not current, that no
#: :data:`~fleet.core.names.GENERATION_MARKER_NAME` one directory below
#: MARKERS names, and that stopped being current over
#: :data:`REMOVAL_GRACE_SECONDS` ago goes whole. Every file operation takes
#: the caller's ``$as`` prefix, empty or ``sudo -n``. The fleet nodes run it
#: after each retire (:mod:`fleet.core.cache_generations`) and ci-clean daily
#: on a CI runner host (:mod:`fleet.core.ci_clean`); it prints one line.
SH_REMOVE_UNUSED: Final = (
    "remove_unused() {",
    "    tools=$1",
    "    stage=$2",
    f'    if ! $as test -f "$tools/{names.CURRENT_GENERATION_NAME}"; then',
    '        echo "cache-generations: $tools: none yet"',
    "        return 0",
    "    fi",
    f'    current=$($as cat "$tools/{names.CURRENT_GENERATION_NAME}")',
    "    now=$(date +%s)",
    '    used=$($as find "$stage" -mindepth 2 -maxdepth 2 '
    f"-name {names.GENERATION_MARKER_NAME} -exec cat {{}} +)",
    "    total=0",
    "    removed=0",
    "    kb=0",
    '    for entry in $($as ls "$tools"); do',
    f'        [ "$entry" != {names.CURRENT_GENERATION_NAME} ] || continue',
    '        case "$entry" in',
    "            *[!0-9]*)",
    '                echo "cache-generations: $tools/$entry is no generation" >&2',
    "                exit 1",
    "                ;;",
    "        esac",
    "        total=$((total + 1))",
    '        [ "$entry" != "$current" ] || continue',
    '        if printf \'%s\\n\' "$used" | grep -qxF "$entry"; then continue; fi',
    "        newest=$((entry > current ? entry : current))",
    f"        [ $((now - newest)) -gt {REMOVAL_GRACE_SECONDS} ] || continue",
    '        size=$($as du -sk "$tools/$entry" | cut -f1)',
    '        $as rm -rf "$tools/$entry"',
    "        removed=$((removed + 1))",
    "        kb=$((kb + size))",
    "    done",
    '    echo "cache-generations: $tools: generation $current current, '
    'removed $removed of $total ($((kb / 1024)) MB)"',
    "}",
)


def tool_caches_directory(cache_root: str) -> str:
    """Where a cache root keeps its generations.

    Args:
        cache_root: The cache root, a node's (:func:`fleet.core.names.cache_root`)
            or execdocker's.

    Returns:
        ``<cache_root>/tools``.
    """
    return f"{cache_root}/{names.TOOL_CACHES_DIRECTORY}"


def sh_build_lines(*, tools: str, marker: str, run_as: str) -> tuple[str, ...]:
    """The lines a Linux build runs before its first install step.

    Args:
        tools: A shell word naming the generations directory, quoted by the
            caller.
        marker: A shell word naming the run directory's marker file.
        run_as: The prefix every file operation takes, ``""`` for the
            build's own user or ``"sudo -n "`` for execdocker's caches.

    Returns:
        The lines. They leave ``$tool_cache`` naming the generation the
        build reads.
    """
    current = f'"$tools/{names.CURRENT_GENERATION_NAME}"'
    return (
        f"tools={tools}",
        f'{run_as}mkdir -p "$tools"',
        "now=$(date +%s)",
        f"if ! {run_as}test -f {current} || "
        f"[ $((now - $({run_as}cat {current}))) -ge {GENERATION_SECONDS} ]; then",
        f'    {run_as}mkdir -p "$tools/$now"',
        f'    printf \'%s\\n\' "$now" | {run_as}tee "$tools/$now/.pending" > /dev/null',
        f'    {run_as}mv -f "$tools/$now/.pending" {current}',
        "fi",
        f"generation=$({run_as}cat {current})",
        f"printf '%s\\n' \"$generation\" | {run_as}tee {marker} > /dev/null",
        'tool_cache="$tools/$generation"',
    )


def powershell_build_lines() -> tuple[str, ...]:
    """The lines a Windows build runs before its first install step.

    Returns:
        The lines, over the build script's ``$CacheRoot`` and ``$Target``.
        They leave ``$toolCache`` naming the generation the build reads.
    """
    return (
        f'$tools = "$CacheRoot/{names.TOOL_CACHES_DIRECTORY}"',
        f'$currentFile = "$tools/{names.CURRENT_GENERATION_NAME}"',
        "[void][System.IO.Directory]::CreateDirectory($tools)",
        "$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()",
        "if (-not (Test-Path -LiteralPath $currentFile) -or "
        "($now - [long](Get-Content -LiteralPath $currentFile -Raw).Trim()) "
        f"-ge {GENERATION_SECONDS}) {{",
        '    [void][System.IO.Directory]::CreateDirectory("$tools/$now")',
        '    [System.IO.File]::WriteAllText("$tools/$now/.pending", "$now")',
        '    Move-Item -LiteralPath "$tools/$now/.pending" -Destination $currentFile -Force',
        "}",
        "$generation = (Get-Content -LiteralPath $currentFile -Raw).Trim()",
        f'[System.IO.File]::WriteAllText("$Target/{names.GENERATION_MARKER_NAME}", $generation)',
        '$toolCache = "$tools/$generation"',
    )


__all__ = [
    "GENERATION_DAYS",
    "GENERATION_SECONDS",
    "REMOVAL_GRACE_SECONDS",
    "SH_REMOVE_UNUSED",
    "powershell_build_lines",
    "sh_build_lines",
    "tool_caches_directory",
]
