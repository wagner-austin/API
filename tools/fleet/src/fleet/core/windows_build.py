"""The script a Windows node runs as one dispatch's build.

Split out of :mod:`fleet.core.dialect_windows` (MCPs board task d69786fa)
when the build took parameters and the strict header, as
:mod:`fleet.core.windows_task` did for the launch and the stop. The script is
committed as a render under ``rendered/`` and executed by
``tests/pester/rendered-dialect-build.Tests.ps1`` against stand-in tools
that pass, fail and write to both streams.

WHY cmd.exe CARRIES EVERY NATIVE RUN. The build ran under
``$ErrorActionPreference = 'Continue'`` and appended each tool's output with
``*>>``, because in Windows PowerShell 5.1 a native program's stderr line
redirected by PowerShell becomes an error record, and under ``Stop`` the
first one ends the script: ``make`` writes to stderr on a passing run. So the
build could not take the strict header, and every cmdlet failure in it passed
silently. Here the redirection is cmd.exe's (``>> "<log>" 2>&1``), so a
tool's stderr goes to the transcript without PowerShell seeing it, and the
script runs under ``Stop`` like every other render. The transcript is the
tools' own bytes, UTF-8 in practice, which
:mod:`fleet.core.windows_log_tail` reads; ``*>>`` wrote UTF-16.
"""

from __future__ import annotations

from fleet.contracts.project import MAKE_TARGET
from fleet.contracts.source import InstallStep
from fleet.core import names
from fleet.core.agent_label import AGENT_LABEL_VARIABLE, require_agent_label
from fleet.core.names import CACHE_VARIABLE
from fleet.core.phase_markers import PHASE_MARKER
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.script_values import scriptable


def _array(values: tuple[str, ...], *, label: str) -> str:
    """A PowerShell array literal of values held to the scriptable grammar.

    Args:
        values: The values, each embedded in single quotes.
        label: What they are, for a refusal.

    Returns:
        ``@('a', 'b')``.
    """
    return "@(" + ", ".join(f"'{scriptable(value, label=label)}'" for value in values) + ")"


def build_script(
    *,
    target: str,
    path: str,
    workers: int,
    install: tuple[InstallStep, ...],
    cache_root: str,
    elevated: bool,
    agent: str,
) -> str:
    """Ready the tree, run the recipe in the project, write its status last.

    THE CACHES ARE THE NODE'S, NOT THE RUN'S. A clean export carries no
    ``node_modules``, no ``.venv`` and no browsers, and installing them from
    the network on every run would make a twelve-minute suite a thirty-minute
    one. The three package managers each honour one environment variable
    naming their cache, so the build points all three at
    ``<stage_root>/cache`` (:func:`fleet.core.names.cache_root`), which every
    run on the node shares and no run directory contains.

    THE INSTALL STEPS RUN AT THE EXPORT ROOT, before the recipe, each
    appending to the same transcript after a ``$ <command>`` line; the first
    that exits non-zero is the build's status and the recipe does not run,
    so a dependency that will not install reads as an install failure in the
    log and never as the suite's fault. Each token was held to the source
    grammar at decode (:mod:`fleet.contracts.source`) and is refused here
    too if it could not be embedded verbatim.

    ``$LASTEXITCODE`` is read in the statement after each cmd.exe run, never
    ``$?``, since the status is the tool's exit code.

    THE BUILD SAYS WHICH LANE LAUNCHED IT, as ``CORVIS_FLEET_ELEVATED``
    (``1`` or ``0``, MCPs board task a98d7083). The launch alone decides the
    token (:mod:`fleet.core.windows_task`), and a suite cannot tell from its
    token whether it was meant to have that one; with the lane in its
    environment, MCPs' execution suite asserts that the token it holds is the
    token the fleet launched it with, in both directions.

    THE BUILD SAYS WHO ASKED FOR IT, as ``BOARD_AGENT_LABEL``: the job's
    submitter, held to the board's grammar (:mod:`fleet.core.agent_label`),
    so a fleet hold the suite takes is attributed to that session.

    ``bash`` IS GIT FOR WINDOWS' BASH, its ``bin`` put first on the PATH
    (``$GitBin``, MCPs board task daae17f2). Every testdb project's install
    step is ``bash scripts/testdb-setup.sh``, a script written for the bash
    the hub and the Linux nodes run, and a Windows PATH names
    ``System32\\bash.exe`` first, which is WSL's launcher: on sedona it
    entered Docker Desktop's ``docker-desktop`` distro. Git's ``bin`` holds
    only ``bash``, ``sh`` and ``git``, so nothing else a recipe calls
    resolves differently.

    EACH STEP, AND THE RECIPE AS THE PHASE ``check``, IS TIMED (MCPs board
    task 74b13c20): ``Invoke-Phase`` writes the ``fleet-phase`` lines
    :mod:`fleet.core.phase_markers` defines around it, the stamps read from
    ``[DateTime]::UtcNow``. ``$InstallPhases`` runs parallel to ``$Install``,
    one phase per step.

    Args:
        target: Absolute remote directory holding the export, its root.
        path: The project's directory inside the export, ``""`` for the root.
        workers: Test workers the capacity check granted.
        install: The project's declared install steps, each with its phase.
        cache_root: The node's cache directory.
        elevated: Whether the build was launched at RunLevel Highest.
        agent: The job's submitting agent label.

    Returns:
        The script's text. Its first statement after the header records its
        own process id in :data:`~fleet.core.names.PID_NAME`, for the stop,
        and it then writes the status to the result file, whose absence is
        how a run is known to be unfinished. It exits 0 having written it:
        the status is the file's, and the task's own result says only that
        the build ran to the end.

    Raises:
        ValueError: When a location or an install token cannot be embedded
            verbatim, or ``agent`` is outside the board's label grammar.
    """
    label = require_agent_label(agent)
    commands = tuple(" ".join(step["argv"]) for step in install)
    lines = [
        "param(",
        f"    [string]$Target = '{scriptable(target, label='target')}',",
        "    [string]$Recipe = "
        f"'{scriptable(names.recipe_directory(target, path), label='recipe directory')}',",
        f"    [int]$Workers = {workers},",
        f"    [string]$CacheRoot = '{scriptable(cache_root, label='cache root')}',",
        f"    [string[]]$Install = {_array(commands, label='install token')},",
        "    [string[]]$InstallPhases = "
        f"{_array(tuple(step['phase'] for step in install), label='install phase')},",
        "    [string]$Make = 'make',",
        '    [string]$GitBin = "$env:ProgramFiles\\Git\\bin",',
        "    " + system32_parameter("Cmd", "cmd.exe"),
        ")",
        *STRICT_HEADER,
        f'$PID | Set-Content -LiteralPath "$Target/{names.PID_NAME}"',
        f'$log = "{names.log_path("$Target")}"',
        f'$result = "$Target/{names.RESULT_NAME}"',
        '$env:npm_config_cache = "$CacheRoot/npm"',
        '$env:POETRY_CACHE_DIR = "$CacheRoot/pypoetry"',
        '$env:PLAYWRIGHT_BROWSERS_PATH = "$CacheRoot/ms-playwright"',
        '$env:PYTEST_XDIST_AUTO_NUM_WORKERS = "$Workers"',
        f"$env:CORVIS_FLEET_ELEVATED = '{1 if elevated else 0}'",
        f"$env:{AGENT_LABEL_VARIABLE} = '{label}'",
        f"$env:{CACHE_VARIABLE} = $CacheRoot",
        '$env:PATH = "$GitBin;$env:PATH"',
        "function Invoke-Logged {",
        "    param([string]$Shell, [string]$Command)",
        '    & $Shell /d /s /c "$Command >> `"$log`" 2>&1"',
        "    return $LASTEXITCODE",
        "}",
        "function Get-PhaseStamp {",
        "    return [DateTime]::UtcNow.ToString(\"yyyy-MM-dd'T'HH:mm:ss'Z'\")",
        "}",
        "function Invoke-Phase {",
        "    param([string]$Shell, [string]$Name, [string]$Command)",
        "    $started = [DateTime]::UtcNow",
        f'    $opening = "{PHASE_MARKER} $Name started $(Get-PhaseStamp)"',
        '    [System.IO.File]::AppendAllText($log, "$opening`r`n")',
        "    $code = Invoke-Logged $Shell $Command",
        "    $seconds = [int][Math]::Floor(([DateTime]::UtcNow - $started).TotalSeconds)",
        (
            f'    $closing = "{PHASE_MARKER} $Name ended $(Get-PhaseStamp) '
            'after $seconds s, exit $code"'
        ),
        '    [System.IO.File]::AppendAllText($log, "$closing`r`n")',
        "    return $code",
        "}",
        "Set-Location -LiteralPath $Target",
        "$status = 0",
        "for ($index = 0; $index -lt $Install.Count; $index++) {",
        "    if ($status -eq 0) {",
        '        [System.IO.File]::AppendAllText($log, "`$ $($Install[$index])`r`n")',
        (
            "        $status = Invoke-Phase -Shell $Cmd -Name $InstallPhases[$index] "
            "-Command $Install[$index]"
        ),
        "    }",
        "}",
        "if ($status -eq 0) {",
        "    Set-Location -LiteralPath $Recipe",
        (
            f"    $status = Invoke-Phase -Shell $Cmd -Name '{MAKE_TARGET}' "
            f'-Command "`"$Make`" {MAKE_TARGET}"'
        ),
        "}",
        "$status | Set-Content -LiteralPath $result",
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["build_script"]
