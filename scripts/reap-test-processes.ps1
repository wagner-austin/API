# Reap pytest/xdist processes that a test run left behind.
#
# WHY THIS EXISTS
# ---------------
# `pytest -n auto` spawns one execnet worker per core, and in the torch projects
# every worker imports torch, which RESERVES ~1.1 GB of address space on import.
# Working set stays tiny (~1-15 MB), so Task Manager makes a wedged run look
# harmless while it holds tens of GB of COMMIT.
#
# The wedge: pytest-timeout has no SIGALRM on Windows, so it uses the `thread`
# method, whose expiry path is literally `os._exit(1)` (pytest_timeout.py,
# timeout_timer). That kills a worker with no cleanup and no protocol shutdown, so
# the xdist controller is left waiting on a channel that will never answer. It
# hangs rather than tearing the session down, and the remaining workers block
# forever in `sys.stdin.readline()` -- which is their entire command line:
#     python -u -c "import sys;exec(eval(sys.stdin.readline()))"
#
# xdist makes it worse before it makes it better: get_default_max_worker_restart
# is `numprocesses * 4`, so a crash loop clones up to ~124 replacement workers,
# each re-importing torch. The crash loop manufactures the memory pressure that
# causes more crashes.
#
# 2026-08-19 incident: three `make check` runs on 08-18 (14:52, 18:10, 18:52) left
# 101 live processes holding ~112 GB of commit for 23 hours. The box threw an
# "Out of Virtual Memory" popup at 19:00 that night, and a 22,906-feature SIRIUS
# annotation run died at 67% at 03:58 the next morning.
#
# TWO MODES
# ---------
#   -RootPid <n>    Kill descendants of that pid. They are ours by construction,
#                   so no age or idle gate is applied.
#
#   -SweepStale     Standalone cleanup of a PREVIOUS run's wreckage. Gated twice:
#                   by age, and by an idle check (below). Use -WhatIf first.
#
# THE IDLE GATE, AND WHY IT IS AGGREGATE
# --------------------------------------
# Age alone cannot distinguish a 90-minute suite from a 90-minute wedge. So sweep
# mode samples cumulative CPU across the candidate set twice and reaps only if the
# WHOLE SET is idle. Aggregate, not per-process, because a live run legitimately
# contains idle workers waiting for work -- killing those would break it. If any
# candidate is burning CPU, the run is alive and the sweep aborts entirely.
# Measured separation on a controlled pair: 4.984 s (busy) vs 0.000 s (blocked).

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    # Kill descendants of this pid (launcher mode).
    [int]$RootPid = 0,

    # Standalone mode: sweep a previous run's stale processes.
    [switch]$SweepStale,

    # Sweep mode: only processes whose ancestry mentions this path are eligible.
    # Defaults to the current directory, i.e. the project being built.
    [string]$ProjectPath = (Get-Location).Path,

    # Sweep mode: minimum age before a process is considered stale.
    [int]$OlderThanMinutes = 60,

    # Sweep mode: seconds between the two CPU samples.
    [int]$IdleSampleSeconds = 5,

    # Sweep mode: total CPU-seconds across the candidate set that still counts as
    # idle. Anything above this means a live run; the sweep aborts.
    [double]$IdleThresholdSeconds = 0.10
)

$ErrorActionPreference = 'Stop'

if ($RootPid -eq 0 -and -not $SweepStale) {
    Write-Error 'Specify either -RootPid <n> or -SweepStale.'
    exit 2
}

function Get-ProcSnapshot { Get-CimInstance Win32_Process }

$all = Get-ProcSnapshot
$byId = @{}
foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }

# Matches the three shapes a run leaves behind: the poetry launcher, the pytest
# controller, and the execnet workers.
function Test-IsTestProcess {
    param($Proc)
    if ($Proc.Name -notin @('python.exe', 'pytest.exe', 'poetry.exe')) { return $false }
    if (-not $Proc.CommandLine) { return $false }
    return ($Proc.CommandLine -like '*pytest*' -or $Proc.CommandLine -like '*exec(eval*')
}

# Deepest-first: killing a parent first can leave a child reparented and missed.
function Get-Descendants {
    param([int]$Id, [int]$Depth = 0)
    if ($Depth -gt 15) { return @() }
    $out = @()
    foreach ($k in ($all | Where-Object { $_.ParentProcessId -eq $Id -and $_.ProcessId -ne $Id })) {
        $out += Get-Descendants -Id $k.ProcessId -Depth ($Depth + 1)
        $out += $k
    }
    return $out
}

function Get-AncestorDepth {
    param($Proc)
    $d = 0; $cur = $Proc
    while ($cur -and $d -lt 15) { $cur = $byId[[int]$cur.ParentProcessId]; $d++ }
    return $d
}

# An execnet worker's own command line does NOT name the project -- it is the
# generic stdin bootstrap. Resolve membership through the ancestry instead.
function Test-InProject {
    param($Proc, [string]$Needle)
    $cur = $Proc
    for ($d = 0; $d -lt 15 -and $cur; $d++) {
        if ($cur.CommandLine -and $cur.CommandLine -like "*$Needle*") { return $true }
        if ($cur.ExecutablePath -and $cur.ExecutablePath -like "*$Needle*") { return $true }
        $cur = $byId[[int]$cur.ParentProcessId]
    }
    return $false
}

function Get-CpuTicks {
    param($Proc)
    return [int64]$Proc.UserModeTime + [int64]$Proc.KernelModeTime
}

$targets = @()

if ($RootPid -ne 0) {
    Write-Host "reap: descendants of pid $RootPid"
    $targets = @(Get-Descendants -Id $RootPid | Where-Object { Test-IsTestProcess $_ })
} else {
    # Match on the leaf directory name: full paths vary in case and separator
    # between the Makefile's cwd, poetry's argv and the venv's ExecutablePath.
    $needle = Split-Path -Leaf $ProjectPath
    Write-Host "reap: sweep stale (project '$needle', older than $OlderThanMinutes min)"

    $cutoff = (Get-Date).AddMinutes(-$OlderThanMinutes)
    $candidates = @($all | Where-Object {
        (Test-IsTestProcess $_) -and $_.CreationDate -lt $cutoff -and (Test-InProject $_ $needle)
    })

    if ($candidates.Count -eq 0) {
        Write-Host 'reap: nothing stale.'
        exit 0
    }

    # Aggregate idle gate.
    $before = @{}
    foreach ($c in $candidates) { $before[[int]$c.ProcessId] = Get-CpuTicks $c }
    Start-Sleep -Seconds $IdleSampleSeconds
    $after = Get-ProcSnapshot

    $deltaTicks = [int64]0
    foreach ($a in $after) {
        $id = [int]$a.ProcessId
        if ($before.ContainsKey($id)) { $deltaTicks += ((Get-CpuTicks $a) - $before[$id]) }
    }
    $deltaSec = [math]::Round($deltaTicks / 10000000, 3)
    Write-Host ("reap: {0} candidate(s), CPU delta over {1}s = {2}s (idle threshold {3}s)" -f $candidates.Count, $IdleSampleSeconds, $deltaSec, $IdleThresholdSeconds)

    if ($deltaSec -gt $IdleThresholdSeconds) {
        Write-Host 'reap: ABORTING - these processes are doing work, so this is a LIVE run, not wreckage.'
        exit 0
    }

    $targets = @($candidates | Sort-Object -Property @{ Expression = { Get-AncestorDepth $_ } } -Descending)
}

if ($targets.Count -eq 0) {
    Write-Host 'reap: nothing to reap.'
    exit 0
}

$commitMb = [math]::Round((($targets | Measure-Object PageFileUsage -Sum).Sum) / 1KB)
Write-Host ("reap: {0} process(es) holding {1:N0} MB commit" -f $targets.Count, $commitMb)

$killed = 0
$failed = 0
foreach ($t in $targets) {
    $age = [math]::Round(((Get-Date) - $t.CreationDate).TotalMinutes)
    $label = "pid=$($t.ProcessId) $($t.Name) age=${age}min"
    if ($PSCmdlet.ShouldProcess($label, 'Stop-Process')) {
        try {
            Stop-Process -Id $t.ProcessId -Force -ErrorAction Stop
            $killed++
        } catch {
            # Already gone (a parent's death took it) is success, not failure.
            # A single stubborn pid must not strand the rest of the sweep.
            if (Get-Process -Id $t.ProcessId -ErrorAction SilentlyContinue) {
                Write-Warning "reap: could not kill $label -- $($_.Exception.Message)"
                $failed++
            } else {
                $killed++
            }
        }
    }
}

Write-Host ("reap: killed {0}, failed {1}, ~{2:N0} MB commit reclaimed" -f $killed, $failed, $commitMb)
if ($failed -gt 0) { exit 1 }
exit 0
