"""Removing the fleet cache's poetry virtualenvs whose projects are gone (MCPs board task 7b07c5d2).

Every build on a node shares ``<stage_root>/cache/pypoetry``
(:func:`fleet.core.names.cache_root`), and poetry names a virtualenv after the
path of the project it serves. A fleet check stages into a fresh
``<stage_root>/<run_id>`` each time, so every check of a Python project made a
new virtualenv, and the retire that removes the run's tree
(:mod:`fleet.core.retire`) left it behind. Measured 2026-09-29: lavender-wsl
held seven, 24 GB of its 33 GB poetry cache, three of them doc-extract-api at
7.8 GB each, and every one recorded only source paths under retired runs;
serendipity held 104 at 13.7 GB and sedona 34 at 3.8 GB.

THE RULE IS CI-CLEAN'S (:data:`fleet.core.runner_render.CI_CLEAN_SCRIPT`): a
virtualenv whose recorded source paths, the absolute lines of its ``.pth``
files, are all gone is an orphan and is removed. One with no recorded path
is kept, since its provenance is unknown and a virtualenv poetry is still
creating has none yet, and one serving a build still running is kept
because that build's tree still exists. The runner sends it right after
each retire, the moment a virtualenv becomes an orphan, so the cache holds
at most the virtualenvs of the runs still on the node.
"""

from __future__ import annotations

import shlex
import threading
from typing import Final

from platform_core.logging import get_logger

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import dialect, names, remote
from fleet.core.dialect_linux import PROLOGUE
from fleet.core.powershell_text import STRICT_HEADER
from fleet.core.script_values import scriptable

_log = get_logger(__name__)

#: Where poetry keeps its virtualenvs inside the node's cache root.
VIRTUALENVS: Final = "pypoetry/virtualenvs"

#: The sweep script's name under the stage root. One name for every run, so
#: no two sweeps of one process run at once (:data:`SWEEPING`).
SWEEP_STEM: Final = "fleet-venv-sweep"

#: Held across each sweep. A serving node runner settles several runs at once
#: (:mod:`fleet.cli.run_locks`, MCPs board task 8993c306), and two sweeps of
#: one node would write the one script path above together and remove the
#: same orphaned virtualenvs twice.
SWEEPING: Final = threading.Lock()


def virtualenvs_directory(stage_root: str) -> str:
    """Where a node's fleet cache keeps poetry's virtualenvs.

    Args:
        stage_root: The node's declared stage root.

    Returns:
        ``<stage_root>/cache/pypoetry/virtualenvs``.
    """
    return f"{names.cache_root(stage_root)}/{VIRTUALENVS}"


def linux_script(venvs: str) -> str:
    """The sweep as a POSIX sh script.

    Args:
        venvs: The virtualenvs directory.

    Returns:
        The script's text. It prints one line, ``venv-sweep: removed <n> of
        <total> (<mb> MB)``. A virtualenv's source paths are read through a
        ``while read`` loop, so a path holding a space is still one path.
    """
    lines = (
        f"venvs={shlex.quote(venvs)}",
        "total=0",
        "removed=0",
        "kb=0",
        'if [ -d "$venvs" ]; then',
        '    for venv in "$venvs"/*/; do',
        '        [ -d "$venv" ] || continue',
        "        total=$((total + 1))",
        '        sources=$(cat "$venv"lib/python*/site-packages/*.pth 2>/dev/null '
        "| grep '^/' | sort -u)",
        '        [ -n "$sources" ] || continue',
        "        alive=$(printf '%s\\n' \"$sources\" | while IFS= read -r source; do "
        'if [ -e "$source" ]; then echo 1; break; fi; done)',
        '        [ -z "$alive" ] || continue',
        '        size=$(du -sk "$venv" | cut -f1)',
        '        rm -rf "$venv"',
        "        removed=$((removed + 1))",
        "        kb=$((kb + size))",
        "    done",
        "fi",
        'echo "venv-sweep: removed $removed of $total ($((kb / 1024)) MB)"',
    )
    return PROLOGUE + "".join(f"{line}\n" for line in lines)


def windows_script(venvs: str) -> str:
    """The sweep as a PowerShell script, its directory a parameter.

    The directory defaults to the rendered path, so a node runs it with no
    arguments and the Pester suite over its committed render points it at a
    directory it laid out.

    Args:
        venvs: The virtualenvs directory.

    Returns:
        The script's text, printing the same one line as :func:`linux_script`.
        A Windows virtualenv keeps its packages under ``Lib\\site-packages``
        and a ``.pth`` line is absolute when it starts with a drive letter.

    Raises:
        ValueError: When the directory cannot be embedded verbatim.
    """
    body = (
        "param(",
        f"    [string]$Venvs = '{scriptable(venvs, label='virtualenvs directory')}'",
        ")",
        *STRICT_HEADER,
        "$total = 0",
        "$removed = 0",
        "$bytes = [long]0",
        "if (Test-Path -LiteralPath $Venvs) {",
        "    foreach ($venv in @(Get-ChildItem -LiteralPath $Venvs -Directory -Force)) {",
        "        $total++",
        "        $site = Join-Path $venv.FullName 'Lib\\site-packages'",
        "        $sources = @()",
        "        if (Test-Path -LiteralPath $site) {",
        "            foreach ($pth in @(Get-ChildItem -LiteralPath $site -Filter '*.pth' "
        "-File -Force)) {",
        "                $sources += @(Get-Content -LiteralPath $pth.FullName | "
        "Where-Object { $_ -match '^[A-Za-z]:[\\\\/]' })",
        "            }",
        "        }",
        "        if ($sources.Count -eq 0) {",
        "            continue",
        "        }",
        "        if (@($sources | Where-Object { Test-Path -LiteralPath $_ }).Count -gt 0) {",
        "            continue",
        "        }",
        "        $size = @(Get-ChildItem -LiteralPath $venv.FullName -Recurse -File -Force) | "
        "Measure-Object -Property Length -Sum",
        "        Remove-Item -Recurse -Force -LiteralPath $venv.FullName",
        "        $removed++",
        "        $bytes += [long]$size.Sum",
        "    }",
        "}",
        "'venv-sweep: removed {0} of {1} ({2} MB)' -f $removed, $total, "
        "[math]::Floor($bytes / 1MB)",
    )
    return "\n".join(body) + "\n"


def script_for(platform: NodePlatform, *, stage_root: str) -> str:
    """The sweep script a node is sent.

    Args:
        platform: The node's declared platform.
        stage_root: The node's declared stage root.

    Returns:
        The script's text.
    """
    venvs = virtualenvs_directory(stage_root)
    if platform is NodePlatform.WINDOWS:
        return windows_script(venvs)
    return linux_script(venvs)


def sweep_on_node(node: NodeConfig) -> str:
    """Remove the orphaned virtualenvs in one node's fleet cache.

    Args:
        node: The node.

    Returns:
        The node's one-line report, which is also logged.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, the latter carrying the node's own error when a
            virtualenv could not be read or removed.
    """
    stage_root = node["stage_root"]
    with SWEEPING:
        report = remote.run_script(
            node["host"],
            dialect.for_platform(node["platform"]).script_path(stage_root, SWEEP_STEM),
            script_for(node["platform"], stage_root=stage_root),
            platform=node["platform"],
        ).strip()
    _log.info("%s on %s", report, node["host"])
    return report


__all__ = [
    "SWEEPING",
    "SWEEP_STEM",
    "VIRTUALENVS",
    "linux_script",
    "script_for",
    "sweep_on_node",
    "virtualenvs_directory",
    "windows_script",
]
