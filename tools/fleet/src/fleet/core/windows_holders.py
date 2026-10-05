"""Who holds a settled run's transcript open, asked of Restart Manager (MCPs board task e40bca34).

WHY THE RETIRE ASKS. The retire moves the transcript to the stage root's
``logs`` directory before anything else (:mod:`fleet.core.windows_retire`),
and Windows refuses to move a file another process holds open. Until this
module that refusal was an ``IOException`` out of ``Move-Item``, which
named no process, and it ended the node's whole tick: from 2026-10-04T02:57Z
every tick of sedona's runner exited 1 on run
``MCPs-mcp-proxy-sedona-1791080670``, whose orphaned ``bash.exe`` pair held
the file, and the node launched nothing for 26 hours.

The transcript is created by the build's own cmd.exe redirection and
reaches no process but the build's descendants, by handle inheritance, so a
process holding it after the build has settled is a leftover of that run.
The retire ends each one by its process id, after reading that the process
holding the id now started when Restart Manager said the holder did, so an
id the holder has released and Windows has reused is never killed; it says
which by pid, image and command line, and only then moves the file. A
holder Restart Manager calls a service or a critical process is not the
run's to end: the retire refuses by name, naming it, instead of killing it.

Restart Manager (``rstrtmgr.dll``) is how Windows itself answers "who has
this file open" for an installer, without a handle-enumeration tool on the
node. Its session is ended in a ``finally`` so a failed query leaks none.
"""

from __future__ import annotations

from typing import Final

from fleet.core.powershell_text import add_type_lines

#: The compiled type's name, which the render's guard looks up.
HOLDERS_TYPE: Final = "FleetNode.FileHolders"

#: ``RM_APP_TYPE`` values the retire may end: unknown, a main window, another
#: window and a console process. Absent are a service (3), Explorer (4) and a
#: critical process (1000).
ENDABLE_APP_TYPES: Final = (0, 1, 2, 5)

#: The code the render throws for a holder it may not end.
HOLDER_PROTECTED: Final = "FLEET_RETIRE_HOLDER_PROTECTED"

#: The prefix of the line the render prints for each holder it ends.
HOLDER_ENDED: Final = "FLEET_RETIRE_HOLDER_ENDED"

#: The C# the render compiles. ``List`` answers every process holding the
#: file, with the start time Restart Manager identifies it by; ``StartTimeOf``
#: reads a running process's start time the same way, 0 for an id no process
#: holds. Each other Win32 failure is a ``Win32Exception`` with a named code.
HOLDERS_SOURCE: Final = """\
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
"""


def end_holders_lines(*, path_variable: str) -> tuple[str, ...]:
    """The lines that end every leftover process holding one file open.

    A holder whose id no longer names a process started when Restart
    Manager said it did (``StartTimeOf`` answers 0 for an id nothing holds)
    has already let go and is filtered out; each one left is looked up in
    ``Win32_Process`` for its image and command line. ``$EndableTypes`` is a
    parameter of the render, defaulting to :data:`ENDABLE_APP_TYPES`, so the
    Pester suite can make an ordinary holder read as protected.

    Args:
        path_variable: The render's variable naming the file, ``$`` and all.

    Returns:
        The guarded compile, then the loop. Each ended holder prints one
        :data:`HOLDER_ENDED` line before it is stopped, and the loop waits
        for it to exit, since its handles close only then.

    Raises:
        ValueError: When ``path_variable`` is not a plain PowerShell variable.
    """
    if not path_variable.startswith("$") or not path_variable[1:].isalnum():
        raise ValueError(f"a holder's file must be a plain variable, not {path_variable!r}")
    return (
        *add_type_lines(HOLDERS_TYPE, HOLDERS_SOURCE),
        f"$holders = @([{HOLDERS_TYPE}]::List({path_variable}) | Where-Object {{ "
        f"[{HOLDERS_TYPE}]::StartTimeOf($_.Pid) -eq $_.StartTime }})",
        "foreach ($holder in $holders) {",
        '    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($holder.Pid)"',
        '    $named = "pid $($holder.Pid) $($process.Name) ($($process.CommandLine))"',
        "    if ($EndableTypes -notcontains $holder.AppType) {",
        f'        throw "{HOLDER_PROTECTED}: $named, of Restart Manager type '
        f'$($holder.AppType), holds {path_variable}"',
        "    }",
        f'    Write-Output "{HOLDER_ENDED}: $named held {path_variable}"',
        "    $running = Get-Process -Id $holder.Pid",
        "    Stop-Process -InputObject $running -Force",
        "    $running.WaitForExit()",
        "}",
    )


__all__ = [
    "ENDABLE_APP_TYPES",
    "HOLDERS_SOURCE",
    "HOLDERS_TYPE",
    "HOLDER_ENDED",
    "HOLDER_PROTECTED",
    "end_holders_lines",
]
