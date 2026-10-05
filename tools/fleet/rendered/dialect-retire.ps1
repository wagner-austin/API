param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$Staging = 'C:/fleet/stage/MCPs-packages-maketools-1790000000.stage',
    [string]$Log = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/result.txt.log',
    [string]$Retained = 'C:/fleet/stage/logs/MCPs-packages-maketools-1790000000.log',
    [string]$TaskName = 'fleet-MCPs-packages-maketools-1790000000',
    [string]$Script0 = 'C:/fleet/stage/mkdir-MCPs-packages-maketools-1790000000.ps1',
    [string]$Script1 = 'C:/fleet/stage/mkdir-MCPs-packages-maketools-1790000000.stage.ps1',
    [string]$Script2 = 'C:/fleet/stage/stop-MCPs-packages-maketools-1790000000.ps1',
    [string]$Script3 = 'C:/fleet/stage/retire-MCPs-packages-maketools-1790000000.ps1',
    [string]$Cmd = "$env:SystemRoot\System32\cmd.exe",
    [int[]]$EndableTypes = @(0, 1, 2, 5)
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null
if (Test-Path -LiteralPath $Log) {
    if ($null -eq ('FleetNode.FileHolders' -as [type])) {
        Add-Type -TypeDefinition @'
    using System;
    using System.ComponentModel;
    using System.Runtime.InteropServices;
    using System.Text;

    namespace FleetNode {
        public class FileHolder {
            public int Pid;
            public long StartTime;
            public int AppType;
        }

        public static class FileHolders {
            // FILETIME is two 4-byte halves, so the struct is 12 bytes; a long
            // here would be 8-aligned and shift every field after it.
            [StructLayout(LayoutKind.Sequential)]
            struct UniqueProcess {
                public int ProcessId;
                public uint StartTimeLow;
                public uint StartTimeHigh;
            }

            [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
            struct ProcessInfo {
                public UniqueProcess Process;
                [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
                public string AppName;
                [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
                public string ServiceShortName;
                public int ApplicationType;
                public uint AppStatus;
                public uint TSSessionId;
                [MarshalAs(UnmanagedType.Bool)]
                public bool Restartable;
            }

            const int MoreData = 234;
            const int InvalidParameter = 87;
            const uint QueryLimitedInformation = 0x1000;

            [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
            static extern int RmStartSession(out uint session, int flags, StringBuilder key);

            [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
            static extern int RmRegisterResources(
                uint session, uint fileCount, string[] files,
                uint processCount, UniqueProcess[] processes,
                uint serviceCount, string[] services);

            [DllImport("rstrtmgr.dll")]
            static extern int RmGetList(
                uint session, out uint needed, ref uint count,
                [In, Out] ProcessInfo[] found, ref uint rebootReasons);

            [DllImport("rstrtmgr.dll")]
            static extern int RmEndSession(uint session);

            [DllImport("kernel32.dll", SetLastError = true)]
            static extern IntPtr OpenProcess(uint access, bool inherit, int pid);

            [DllImport("kernel32.dll", SetLastError = true)]
            static extern bool GetProcessTimes(
                IntPtr process, out long creation, out long exit, out long kernel, out long user);

            [DllImport("kernel32.dll")]
            static extern bool CloseHandle(IntPtr handle);

            public static FileHolder[] List(string path) {
                uint session;
                int code = RmStartSession(out session, 0, new StringBuilder(33));
                if (code != 0) {
                    throw new Win32Exception(code, "FLEET_HOLDERS_SESSION_FAILED");
                }
                try {
                    code = RmRegisterResources(session, 1, new string[] { path }, 0, null, 0, null);
                    if (code != 0) {
                        throw new Win32Exception(code, "FLEET_HOLDERS_REGISTER_FAILED");
                    }
                    ProcessInfo[] found = new ProcessInfo[0];
                    uint count = 0;
                    uint needed;
                    uint reasons = 0;
                    code = RmGetList(session, out needed, ref count, found, ref reasons);
                    while (code == MoreData) {
                        found = new ProcessInfo[needed];
                        count = needed;
                        code = RmGetList(session, out needed, ref count, found, ref reasons);
                    }
                    if (code != 0) {
                        throw new Win32Exception(code, "FLEET_HOLDERS_LIST_FAILED");
                    }
                    FileHolder[] holders = new FileHolder[count];
                    for (int index = 0; index < count; index++) {
                        holders[index] = new FileHolder();
                        UniqueProcess process = found[index].Process;
                        holders[index].Pid = process.ProcessId;
                        holders[index].StartTime =
                            ((long)process.StartTimeHigh << 32) | process.StartTimeLow;
                        holders[index].AppType = found[index].ApplicationType;
                    }
                    return holders;
                } finally {
                    RmEndSession(session);
                }
            }

            public static long StartTimeOf(int pid) {
                IntPtr process = OpenProcess(QueryLimitedInformation, false, pid);
                if (process == IntPtr.Zero) {
                    int error = Marshal.GetLastWin32Error();
                    if (error == InvalidParameter) {
                        return 0;
                    }
                    throw new Win32Exception(error, "FLEET_HOLDERS_OPEN_FAILED");
                }
                try {
                    long creation, exit, kernel, user;
                    if (!GetProcessTimes(process, out creation, out exit, out kernel, out user)) {
                        int error = Marshal.GetLastWin32Error();
                        throw new Win32Exception(error, "FLEET_HOLDERS_TIMES_FAILED");
                    }
                    return creation;
                } finally {
                    CloseHandle(process);
                }
            }
        }
    }
'@
    }
    $holders = @([FleetNode.FileHolders]::List($Log) | Where-Object { [FleetNode.FileHolders]::StartTimeOf($_.Pid) -eq $_.StartTime })
    foreach ($holder in $holders) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($holder.Pid)"
        $named = "pid $($holder.Pid) $($process.Name) ($($process.CommandLine))"
        if ($EndableTypes -notcontains $holder.AppType) {
            throw "FLEET_RETIRE_HOLDER_PROTECTED: $named, of Restart Manager type $($holder.AppType), holds $Log"
        }
        Write-Output "FLEET_RETIRE_HOLDER_ENDED: $named held $Log"
        $running = Get-Process -Id $holder.Pid
        Stop-Process -InputObject $running -Force
        $running.WaitForExit()
    }
    Move-Item -Force -LiteralPath $Log -Destination $Retained
}
foreach ($directory in @($Target, $Staging)) {
    if (Test-Path -LiteralPath $directory) {
        $verbatim = '\\?\' + [IO.Path]::GetFullPath($directory)
        & $Cmd /d /c rd /s /q $verbatim
        if (Test-Path -LiteralPath $directory) {
            throw "FLEET_RETIRE_INCOMPLETE: rd exited $LASTEXITCODE and left $directory"
        }
    }
}
foreach ($script in @($Script0, $Script1, $Script2, $Script3)) {
    if (Test-Path -LiteralPath $script) {
        Remove-Item -Force -LiteralPath $script
    }
}
$scheduler = New-Object -ComObject Schedule.Service
$scheduler.Connect()
$root = $scheduler.GetFolder('\')
if (@($root.GetTasks(1) | Where-Object { $_.Name -eq $TaskName }).Count -gt 0) {
    $root.DeleteTask($TaskName, 0)
}
