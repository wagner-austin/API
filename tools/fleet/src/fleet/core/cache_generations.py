"""Removing the cache generations no run on a node can still read.

The generations are :mod:`fleet.core.tool_caches`'s (MCPs board task
346abc45). After each retire the runner sends the node :func:`script_for`,
which removes every generation under the node's cache root, and under
execdocker's on a node that runs docker projects' builds as that user
(:func:`fleet.core.retire.runs_isolated`), that is not current, that
no run directory still on the node names in its marker, and that has not
been current for :data:`REMOVAL_GRACE_SECONDS`. The grace covers a build
that read the current name just before it changed: it writes its marker in
the next line. Each generation goes whole, so no tool ever finds half of one
of its own entries.
"""

from __future__ import annotations

import shlex
import threading
from typing import Final

from platform_core.logging import get_logger

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import dialect, names, remote, retire
from fleet.core.dialect_linux import PROLOGUE
from fleet.core.linux_isolated_build import EXEC_CACHE, EXEC_HOME_LINE, EXEC_STAGE
from fleet.core.powershell_text import STRICT_HEADER
from fleet.core.script_values import scriptable
from fleet.core.tool_caches import tool_caches_directory

_log = get_logger(__name__)

#: How long a generation that stopped being current is kept even with no
#: marker naming it: a build reads the current name and writes its marker
#: one line later, so an hour covers any build that read it.
REMOVAL_GRACE_SECONDS: Final = 3600

#: The removal script's name under the stage root. One name for every run,
#: so no two removals of one process run at once (:data:`REMOVING`).
REMOVAL_STEM: Final = "fleet-cache-generations"

#: Held across each removal. A serving node runner settles several runs at
#: once (:mod:`fleet.cli.run_locks`, MCPs board task 8993c306), and two
#: removals of one node would write the one script path above together.
REMOVING: Final = threading.Lock()


def linux_script(stage_root: str, *, isolated: bool) -> str:
    """The removal as a POSIX sh script, over the node's caches and execdocker's.

    Args:
        stage_root: The node's declared stage root.
        isolated: Whether the node runs docker projects' builds as
            execdocker (:func:`fleet.core.retire.runs_isolated`), whose
            caches are then judged too, as root, against that user's copies.

    Returns:
        The script's text, printing one line per cache root it judged.
    """
    tools = tool_caches_directory(names.cache_root(stage_root))
    current = names.CURRENT_GENERATION_NAME
    lines = (
        "remove_unused() {",
        "    tools=$1",
        "    stage=$2",
        f'    if ! $as test -f "$tools/{current}"; then',
        '        echo "cache-generations: $tools: none yet"',
        "        return 0",
        "    fi",
        f'    current=$($as cat "$tools/{current}")',
        "    now=$(date +%s)",
        '    used=$($as find "$stage" -mindepth 2 -maxdepth 2 '
        f"-name {names.GENERATION_MARKER_NAME} -exec cat {{}} +)",
        "    total=0",
        "    removed=0",
        "    kb=0",
        '    for entry in $($as ls "$tools"); do',
        f'        [ "$entry" != {current} ] || continue',
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
        "as=",
        f"remove_unused {shlex.quote(tools)} {shlex.quote(stage_root)}",
    )
    isolated_lines = (
        EXEC_HOME_LINE,
        "as='sudo -n'",
        f'remove_unused "{tool_caches_directory(EXEC_CACHE)}" "{EXEC_STAGE}"',
    )
    return PROLOGUE + "".join(f"{line}\n" for line in lines + (isolated_lines if isolated else ()))


def windows_script(stage_root: str) -> str:
    """The removal as a PowerShell script, its stage root a parameter.

    The root defaults to the rendered one, so a node runs it with no
    arguments and the Pester suite over its committed render points it at a
    directory it laid out.

    Args:
        stage_root: The node's declared stage root.

    Returns:
        The script's text, printing the same line as :func:`linux_script`.

    Raises:
        ValueError: When the stage root cannot be embedded verbatim.
    """
    body = (
        "param(",
        f"    [string]$StageRoot = '{scriptable(stage_root, label='stage root')}'",
        ")",
        *STRICT_HEADER,
        f'$tools = "$StageRoot/{names.CACHE_DIRECTORY}/{names.TOOL_CACHES_DIRECTORY}"',
        f'$currentFile = "$tools/{names.CURRENT_GENERATION_NAME}"',
        "if (-not (Test-Path -LiteralPath $currentFile)) {",
        '    "cache-generations: ${tools}: none yet"',
        "    return",
        "}",
        "$current = (Get-Content -LiteralPath $currentFile -Raw).Trim()",
        "$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()",
        "$used = @(foreach ($run in @(Get-ChildItem -LiteralPath $StageRoot -Directory -Force)) {",
        f"    $marker = Join-Path $run.FullName '{names.GENERATION_MARKER_NAME}'",
        "    if (Test-Path -LiteralPath $marker) {",
        "        (Get-Content -LiteralPath $marker -Raw).Trim()",
        "    }",
        "})",
        "$total = 0",
        "$removed = 0",
        "$bytes = [long]0",
        "foreach ($entry in @(Get-ChildItem -LiteralPath $tools -Force)) {",
        f"    if ($entry.Name -eq '{names.CURRENT_GENERATION_NAME}') {{",
        "        continue",
        "    }",
        "    if (-not $entry.PSIsContainer -or $entry.Name -notmatch '^[0-9]+$') {",
        '        throw "cache-generations: $($entry.FullName) is no generation"',
        "    }",
        "    $total++",
        "    if ($entry.Name -eq $current -or $used -contains $entry.Name) {",
        "        continue",
        "    }",
        "    $newest = [math]::Max([long]$entry.Name, [long]$current)",
        f"    if ($now - $newest -le {REMOVAL_GRACE_SECONDS}) {{",
        "        continue",
        "    }",
        "    $size = @(Get-ChildItem -LiteralPath $entry.FullName -Recurse -File -Force) | "
        "Measure-Object -Property Length -Sum",
        "    Remove-Item -Recurse -Force -LiteralPath $entry.FullName",
        "    $removed++",
        "    $bytes += [long]$size.Sum",
        "}",
        "'cache-generations: {0}: generation {1} current, removed {2} of {3} ({4} MB)' -f "
        "$tools, $current, $removed, $total, [math]::Floor($bytes / 1MB)",
    )
    return "\n".join(body) + "\n"


def script_for(platform: NodePlatform, *, stage_root: str, isolated: bool) -> str:
    """The removal script a node is sent.

    Args:
        platform: The node's declared platform.
        stage_root: The node's declared stage root.
        isolated: Whether the node runs docker projects' builds as execdocker.

    Returns:
        The script's text.

    Raises:
        ValueError: When a Windows node is said to run isolated builds,
            which none does.
    """
    if platform is NodePlatform.LINUX:
        return linux_script(stage_root, isolated=isolated)
    if isolated:
        raise ValueError(
            f"{stage_root} is a Windows node's stage root, and no Windows node runs a build "
            "as execdocker"
        )
    return windows_script(stage_root)


def remove_unused_on_node(node: NodeConfig) -> str:
    """Remove the cache generations no run on one node can still read.

    A NODE THAT DOES NOT ANSWER IS LEFT TO THE NEXT REMOVAL (MCPs board task
    c1d48330). Every caller removes after its row is closed, and a removal
    judges every generation, not one run's, so one the node missed leaves
    nothing the next one does not remove. Raising there ended lavender-wsl's
    serve at 04:05Z on 2026-10-07, after the row it settled had closed.

    Args:
        node: The node.

    Returns:
        The node's report, or the line saying it did not answer; either is
        also logged.

    Raises:
        AppError: With ``DISPATCH_FAILED`` carrying the node's own error
            when it answered and a generation could not be read or removed.
    """
    stage_root = node["stage_root"]
    with REMOVING:
        read = remote.read_script(
            node["host"],
            dialect.for_platform(node["platform"]).script_path(stage_root, REMOVAL_STEM),
            script_for(
                node["platform"], stage_root=stage_root, isolated=retire.runs_isolated(node)
            ),
            platform=node["platform"],
        )
    output = read["output"]
    if output is None:
        missed = (
            f"cache-generations: {node['host']} did not answer; the next removal judges its "
            f"generations: {read['unreachable']}"
        )
        _log.info("%s", missed)
        return missed
    report = output.strip()
    _log.info("%s on %s", report, node["host"])
    return report


__all__ = [
    "REMOVAL_GRACE_SECONDS",
    "REMOVAL_STEM",
    "REMOVING",
    "linux_script",
    "remove_unused_on_node",
    "script_for",
    "windows_script",
]
