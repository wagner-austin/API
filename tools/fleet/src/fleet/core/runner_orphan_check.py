"""The audit row that fails a runner holding what a finished job left behind.

MCPs board task 53528106, A2 and A4. A cancelled GitHub Actions job kills
its worker, and what the job daemonized lives on: on 2026-10-02 lavender's
WSL runners held 158 such processes and 17 GB with no job running, and
nothing in the audit could see it. This row asks each runner, on both
sides, how many of its processes lie outside its service's own tree, are
older than the host's ``job_timeout_minutes``, and live while no
``Runner.Worker`` runs: any such process outlived every job that could own
it.

A WSL runner is asked through the reaper's own ``--audit`` mode
(:mod:`fleet.core.runner_reaper_render`), so the audit measures exactly what
the reaper acts on, and a host without the reaper drifts on that row as
well as on its timer's. A Windows runner is read natively: its service's
process, every descendant of it, and every process whose command line or
executable sits under the runner's directory but outside that tree.
"""

from __future__ import annotations

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core.runner_reaper_render import REAPER_PATH
from fleet.core.script_values import scriptable

#: Why the row matters, printed beside a drift line.
ORPHAN_REASON = (
    "a finished job's processes hold the runner's memory until something kills them; on "
    "2026-10-02 they held 17 GB of lavender's CI budget and GitHub read every WSL runner offline"
)

#: The audit's parameter that reads every process, a script block so the
#: Pester suite can hand it a process table.
PROCESS_PARAMETER = "[scriptblock]$GetProcesses = { @(Get-CimInstance Win32_Process) }"

#: The PowerShell function the Windows rows call: the processes naming a
#: runner directory outside its service's tree, older than the bound, when
#: no Runner.Worker runs in the tree. A stopped service has no tree, so
#: every such process counts.
LEFTOVER_FUNCTION = [
    "function Get-RunnerLeftover {",
    "    param([object[]]$Processes, [int]$ServicePid, [string]$Root, [int]$OlderThanSeconds)",
    "    $children = @{}",
    "    foreach ($Process in $Processes) {",
    "        $Parent = [int]$Process.ParentProcessId",
    "        if (-not $children.ContainsKey($Parent)) {",
    "            $children[$Parent] = [System.Collections.Generic.List[object]]::new()",
    "        }",
    "        $children[$Parent].Add($Process)",
    "    }",
    "    $tree = @{}",
    "    $pending = [System.Collections.Generic.Queue[int]]::new()",
    "    if ($ServicePid -ne 0) {",
    "        $pending.Enqueue($ServicePid)",
    "    }",
    "    while ($pending.Count -gt 0) {",
    "        $Id = $pending.Dequeue()",
    "        $tree[$Id] = $true",
    "        if ($children.ContainsKey($Id)) {",
    "            foreach ($Child in $children[$Id]) {",
    "                if (-not $tree.ContainsKey([int]$Child.ProcessId)) {",
    "                    $pending.Enqueue([int]$Child.ProcessId)",
    "                }",
    "            }",
    "        }",
    "    }",
    "    $working = @($Processes | Where-Object { $tree.ContainsKey([int]$_.ProcessId) "
    "-and [string]$_.Name -eq 'Runner.Worker.exe' })",
    "    if ($working.Count -gt 0) {",
    "        return @()",
    "    }",
    "    $cutoff = (Get-Date).AddSeconds(-$OlderThanSeconds)",
    "    return @($Processes | Where-Object {",
    "        -not $tree.ContainsKey([int]$_.ProcessId) -and $null -ne $_.CreationDate -and",
    "        $_.CreationDate -lt $cutoff -and (",
    "            ([string]$_.CommandLine).Replace('/', '\\').IndexOf($Root, "
    "[StringComparison]::OrdinalIgnoreCase) -ge 0 -or",
    "            ([string]$_.ExecutablePath).StartsWith($Root, "
    "[StringComparison]::OrdinalIgnoreCase))",
    "    })",
    "}",
]


def orphan_check_id(install: RunnerInstall) -> str:
    """The row's id for one install, unique across repositories.

    Args:
        install: The install.

    Returns:
        E.g. ``orphans:wagner-austin/MCPs:wsl:lavender-wsl``.
    """
    return f"orphans:{install['repo']}:{install['side']}:{install['runner_name']}"


def runner_root(install: RunnerInstall) -> str:
    """The runner's own directory, the ``_work`` parent, in Windows form.

    Args:
        install: A windows-side install.

    Returns:
        The directory with backslashes and a trailing one, so that
        ``C:\\actions-runner\\`` never matches ``C:\\actions-runner-chat``.
    """
    return install["workdir"].rsplit("/", 1)[0].replace("/", "\\") + "\\"


def render_orphan_check_lines(spec: HostRunnerSpec, install: RunnerInstall) -> list[str]:
    """Script lines for one install's row.

    Args:
        spec: The host, for its job timeout.
        install: The install.

    Returns:
        The lines. A WSL row passes only when the reaper exits 0 and counts
        nothing; a Windows row only when no leftover is found, and names the
        first five by process name and id when one is. A Windows row reads
        the ``$Service`` rows its service row read, so it follows that row.

    Raises:
        ValueError: When the unit name or the directory cannot be embedded
            verbatim.
    """
    seconds = spec["job_timeout_minutes"] * 60
    minutes = spec["job_timeout_minutes"]
    check_id = orphan_check_id(install)
    service = scriptable(install["service"], label="service")
    if install["side"] == "wsl":
        return [
            f'$Probe = Invoke-InDistro $Cmd $Wsl $Distro "{REAPER_PATH} --audit {seconds} '
            f"'{service}'\"",
            "$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')",
            f"Write-Check '{check_id}' ($Probe.Exit -eq 0 -and $Leftover -eq '0') "
            f"('processes older than {minutes} minutes outside {service} with no job "
            "running; the reaper counted: ' + $Probe.Text)",
        ]
    root = scriptable(runner_root(install), label="workdir")
    # The $Service rows the driver's service row read just before, as the
    # account row reads them: one Win32_Service read per runner.
    return [
        "$ServicePid = 0",
        "foreach ($Row in $Service) {",
        "    $ServicePid = [int]$Row.ProcessId",
        "}",
        f"$Leftover = @(Get-RunnerLeftover @(& $GetProcesses) $ServicePid '{root}' {seconds})",
        f"Write-Check '{check_id}' ($Leftover.Count -eq 0) "
        f"([string]$Leftover.Count + ' process(es) under {root} outside {service}, older than "
        f"{minutes} minutes with no job running: ' + "
        "((@($Leftover | Select-Object -First 5 | ForEach-Object { [string]$_.Name + ' pid ' + "
        "[string]$_.ProcessId })) -join ', '))",
    ]


__all__ = [
    "LEFTOVER_FUNCTION",
    "ORPHAN_REASON",
    "PROCESS_PARAMETER",
    "orphan_check_id",
    "render_orphan_check_lines",
    "runner_root",
]
