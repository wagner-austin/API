param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$Recipe = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/packages/maketools',
    [int]$Workers = 4,
    [string]$CacheRoot = 'C:/fleet/stage/cache',
    [string[]]$Install = @('npm ci'),
    [string[]]$InstallPhases = @('install'),
    [string]$Make = 'make',
    [string]$GitBin = "$env:ProgramFiles\Git\bin",
    [string]$Cmd = "$env:SystemRoot\System32\cmd.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PID | Set-Content -LiteralPath "$Target/build.pid"
if ($null -eq ('FleetNode.KillOnCloseJob' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace FleetNode {
    public static class KillOnCloseJob {
        [StructLayout(LayoutKind.Sequential)]
        struct BasicLimits {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IoCounters {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct ExtendedLimits {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        const int ExtendedLimitInformation = 9;
        const uint KillOnJobClose = 0x2000;

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetInformationJobObject(
            IntPtr job, int infoClass, ref ExtendedLimits info, uint length);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll")]
        static extern IntPtr GetCurrentProcess();

        static IntPtr held = IntPtr.Zero;

        static void Fail(string code) {
            throw new Win32Exception(Marshal.GetLastWin32Error(), code);
        }

        // The handle is kept and never closed: the kernel closes it when this
        // process ends, and that close is what ends the job's processes.
        public static IntPtr Enter() {
            if (held != IntPtr.Zero) {
                return held;
            }
            IntPtr job = CreateJobObjectW(IntPtr.Zero, null);
            if (job == IntPtr.Zero) {
                Fail("FLEET_BUILD_JOB_CREATE_FAILED");
            }
            ExtendedLimits limits = new ExtendedLimits();
            limits.Basic.LimitFlags = KillOnJobClose;
            uint size = (uint)Marshal.SizeOf(typeof(ExtendedLimits));
            if (!SetInformationJobObject(job, ExtendedLimitInformation, ref limits, size)) {
                Fail("FLEET_BUILD_JOB_LIMIT_FAILED");
            }
            if (!AssignProcessToJobObject(job, GetCurrentProcess())) {
                Fail("FLEET_BUILD_JOB_ASSIGN_FAILED");
            }
            held = job;
            return held;
        }
    }
}
'@
}
[void][FleetNode.KillOnCloseJob]::Enter()
$log = "$Target/result.txt.log"
$result = "$Target/result.txt"
$env:npm_config_cache = "$CacheRoot/npm"
$env:POETRY_CACHE_DIR = "$CacheRoot/pypoetry"
$env:PLAYWRIGHT_BROWSERS_PATH = "$CacheRoot/ms-playwright"
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = "$Workers"
$env:CORVIS_FLEET_ELEVATED = '0'
$env:BOARD_AGENT_LABEL = 'opus-example-0929'
$env:CORVIS_FLEET_CACHE = $CacheRoot
$env:PATH = "$GitBin;$env:PATH"
function Invoke-Logged {
    param([string]$Shell, [string]$Command)
    & $Shell /d /s /c "$Command >> `"$log`" 2>&1"
    return $LASTEXITCODE
}
function Get-PhaseStamp {
    return [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
}
function Invoke-Phase {
    param([string]$Shell, [string]$Name, [string]$Command)
    $started = [DateTime]::UtcNow
    $opening = "fleet-phase $Name started $(Get-PhaseStamp)"
    [System.IO.File]::AppendAllText($log, "$opening`r`n")
    $code = Invoke-Logged $Shell $Command
    $seconds = [int][Math]::Floor(([DateTime]::UtcNow - $started).TotalSeconds)
    $closing = "fleet-phase $Name ended $(Get-PhaseStamp) after $seconds s, exit $code"
    [System.IO.File]::AppendAllText($log, "$closing`r`n")
    return $code
}
Set-Location -LiteralPath $Target
$status = 0
for ($index = 0; $index -lt $Install.Count; $index++) {
    if ($status -eq 0) {
        [System.IO.File]::AppendAllText($log, "`$ $($Install[$index])`r`n")
        $status = Invoke-Phase -Shell $Cmd -Name $InstallPhases[$index] -Command $Install[$index]
    }
}
if ($status -eq 0) {
    Set-Location -LiteralPath $Recipe
    $status = Invoke-Phase -Shell $Cmd -Name 'check' -Command "`"$Make`" check"
}
$status | Set-Content -LiteralPath $result
exit 0
