# Shared pytest launcher for every project in this monorepo.
#
# Each project's Makefile used to inline this recipe, and 35 of the 36 copies were
# byte-identical. The duplication is why a lifecycle bug had to be fixed 36 times
# instead of once, so the recipe now lives here and the Makefiles call it.
#
# WHAT THIS ADDS OVER THE OLD INLINE RECIPE
# -----------------------------------------
# 1. JOB OBJECT. The launcher assigns ITSELF to a Windows job object with
#    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Every descendant -- poetry, pytest, and
#    all execnet workers -- inherits membership, so when this process dies for ANY
#    reason the OS tears the whole tree down.
#
#    This is the part a `finally` block cannot do. On 2026-08-18 three `make check`
#    runs left 101 live processes holding ~112 GB of commit for 23 hours; the
#    launching shell was already dead while make, poetry, pytest and 92 workers
#    stayed alive, so a finally-block reaper would never have run. Verified: with
#    self-assignment, force-killing the launcher killed grandchildren two levels
#    down, 3 of 3, with no finally involved.
#
#    Assigning SELF rather than the child is deliberate -- it removes the window
#    between spawning a child and assigning it, during which a grandchild could
#    escape the job.
#
# 2. PRE-RUN SWEEP. Reaps a previous run's wreckage before starting, so a wedged
#    run cannot silently starve the next one. Gated on age AND on an aggregate CPU
#    idle check so it can never kill a live run (see reap-test-processes.ps1).
#
# 3. --max-worker-restart=0. xdist otherwise defaults to numprocesses * 4 (~124 on
#    this box) and CLONES a replacement for every crashed worker, each re-importing
#    torch and reserving ~1.1 GB of commit -- the crash loop manufactures the
#    memory pressure that causes more crashes. Zero turns the first hard-exited
#    worker into an immediate, loud failure instead of a silent 124-clone spiral.
#
# Everything else is preserved exactly: the coverage argument construction, the
# per-run COVERAGE_FILE GUID under runs/, the cleanup of runs/.coverage-*, and the
# pytest exit code.

[CmdletBinding()]
param(
    # Skip the pre-run sweep. For debugging only.
    [switch]$NoSweep,

    # Extra arguments appended to the pytest invocation.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @()
)

# Native tools write to stderr routinely; 'Stop' would turn that into a thrown
# NativeCommandError under PS 5.1. Exit codes are checked explicitly instead.
$ErrorActionPreference = 'Continue'

$projectRoot = (Get-Location).Path
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$reaper = Join-Path $scriptDir 'reap-test-processes.ps1'

# ---------------------------------------------------------------------------
# 1. Join a kill-on-close job object.
# ---------------------------------------------------------------------------

# Guarded: Add-Type throws if the type already exists, which would happen if this
# script ran twice in one PowerShell process. Each project currently gets a fresh
# shell from make, so it cannot fire today -- but that is an accident of the
# caller, not a property of this script, and the failure would be an obscure
# "type already exists" rather than anything pointing here.
if (-not ('CorvisBuild.Job' -as [type])) {
    Add-Type -Namespace CorvisBuild -Name Job -MemberDefinition @'
[DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
public static extern IntPtr CreateJobObject(IntPtr a, string lpName);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool SetInformationJobObject(IntPtr hJob, int infoClass, IntPtr lpInfo, uint cbInfo);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr GetCurrentProcess();
'@
}

function Join-KillOnCloseJob {
    $job = [CorvisBuild.Job]::CreateJobObject([IntPtr]::Zero, $null)
    if ($job -eq [IntPtr]::Zero) { return 'CreateJobObject returned NULL' }

    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION is info class 9. Its size is
    # pointer-width dependent and SetInformationJobObject rejects any other
    # length with ERROR_BAD_LENGTH (24):
    #
    #   BASIC_LIMIT_INFORMATION  x64 64 / x86 44   (two LARGE_INTEGERs, then
    #                                               DWORDs and SIZE_T/ULONG_PTR)
    #   IO_COUNTERS              48 both           (six ULONGLONG)
    #   four trailing SIZE_T     x64 32 / x86 16
    #   tail padding to 8        x64  0 / x86  4   (LARGE_INTEGER forces 8-align)
    #   ------------------------------------------------------------------
    #   total                    x64 144 / x86 112
    #
    # This is NOT hypothetical on Windows: GNU make is a 32-bit binary, so
    # WOW64 redirection resolves `powershell.exe` in a Makefile recipe to the
    # 32-bit PowerShell. A hardcoded 144 therefore failed on every `make check`
    # while succeeding in any hand-run 64-bit shell -- and the failure path
    # still assigned the process to a limit-less job, so it looked protected.
    #
    # LimitFlags sits at offset 16 in both layouts; 0x2000 is
    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    $size = if ([IntPtr]::Size -eq 8) { 144 } else { 112 }
    $ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    try {
        for ($i = 0; $i -lt $size; $i += 4) {
            [Runtime.InteropServices.Marshal]::WriteInt32($ptr, $i, 0)
        }
        [Runtime.InteropServices.Marshal]::WriteInt32($ptr, 16, 0x2000)

        if (-not [CorvisBuild.Job]::SetInformationJobObject($job, 9, $ptr, $size)) {
            $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            return "SetInformationJobObject failed (win32 $code) at cbInfo=$size for a $([IntPtr]::Size * 8)-bit process"
        }
    } finally {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
    }

    if (-not [CorvisBuild.Job]::AssignProcessToJobObject($job, [CorvisBuild.Job]::GetCurrentProcess())) {
        return "AssignProcessToJobObject failed (win32 $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
    }
    return $null
}

$jobError = Join-KillOnCloseJob
if ($jobError) {
    # Fatal, deliberately. There is no degraded mode here: the post-run reap
    # cannot cover a killed shell, which is the exact case the job object
    # exists for, so continuing would run the suite with the protection
    # silently absent. That is how this failure went unnoticed -- a warning
    # scrolls past, a non-zero exit does not.
    Write-Error "run-tests: could not join a kill-on-close job object -- $jobError"
    exit 1
}
Write-Host 'run-tests: joined kill-on-close job object (tree dies with this process)'

# ---------------------------------------------------------------------------
# 2. Sweep a previous run's wreckage.
# ---------------------------------------------------------------------------

if (-not $NoSweep) {
    if (Test-Path $reaper) {
        & $reaper -SweepStale -ProjectPath $projectRoot -Confirm:$false
    } else {
        Write-Warning "run-tests: reaper not found at $reaper; skipping pre-run sweep"
    }
}

# ---------------------------------------------------------------------------
# 3. Run pytest, preserving the original recipe's behaviour exactly.
# ---------------------------------------------------------------------------

$covArgs = @('--cov-branch', '--cov-report=term-missing')
foreach ($c in @('src', 'scripts')) {
    if (Test-Path (Join-Path '.' $c)) { $covArgs += "--cov=$c" }
}

if (-not (Test-Path 'runs')) { New-Item -ItemType Directory -Path 'runs' | Out-Null }
$env:COVERAGE_FILE = (Resolve-Path -LiteralPath '.').Path + '\runs\.coverage-' + [System.Guid]::NewGuid().ToString('N').Substring(0, 8)

$code = 0
try {
    poetry run pytest -n auto -v --max-worker-restart=0 @covArgs @PytestArgs
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 1 }
} finally {
    Get-ChildItem -Path 'runs' -Filter '.coverage-*' -Force -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Remove-Item Env:\COVERAGE_FILE -ErrorAction SilentlyContinue

    # Belt and braces. The job object already covers the case where this process
    # is killed; this covers the ordinary path, and leaves the machine clean even
    # if job assignment failed above.
    if (Test-Path $reaper) { & $reaper -RootPid $PID -Confirm:$false }
}

exit $code
