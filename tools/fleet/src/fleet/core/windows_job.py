"""The kill-on-close job object a Windows build runs inside (MCPs board task e40bca34).

WHY THE BUILD CONTAINS ITSELF. The stop ends a build with ``taskkill /PID
<build.ps1> /T /F`` (:func:`fleet.core.windows_task.stop_script`), and
``/T`` walks parent links: a process whose parent has already exited has no
link back to the build and is never reached. Measured on sedona: run
``MCPs-mcp-proxy-sedona-1791080670`` hung in its test-database install step,
the runner stopped it past its lease at 2026-10-04T02:57Z, and two
``bash.exe`` processes of its ``ci-bootstrap-testdb.sh`` (pids 8728 and 14792,
their parent 19568 gone) lived on holding the transcript that cmd.exe's
``>>`` had handed them. Every retire after that stopped on the held file,
and every tick of sedona's runner exited 1 for 26 hours until the two were
ended by hand at 2026-10-05T04:44Z. ``make check`` itself runs inside
maketools' own kill-on-close job (``tools/maketools/src/maketools/job.py``),
which is why only an install step, run before it and outside it, leaked.

So the build's first act is to put its own process in a new job object
with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Every process it starts after
that, and every process those start, is created inside the job whatever
becomes of its parent, and the job's only handle is held by the build: when
the build ends, by finishing or by the stop's ``taskkill``, the kernel
closes the handle and ends every process still in the job. The handle is
not inheritable, so no child holds the job open. Breakaway is not allowed,
as maketools' job does not allow it.

ONE JOB PER PROCESS. ``Enter`` keeps the handle in a static field and
answers it again on a second call, so the Pester suite, which runs the
build render once per case in its own process, nests no second job; a node
runs each build in a fresh ``powershell.exe`` and enters once.

The type mirrors MCPs' ``scripts/lib/fleet-audit-run.ps1`` (board task
ffe0c3f1), which a render shipped to a node cannot load from another
repository.
"""

from __future__ import annotations

from typing import Final

from fleet.core.powershell_text import add_type_lines

#: The compiled type's name, which the render's guard looks up.
JOB_TYPE: Final = "FleetNode.KillOnCloseJob"

#: The C# the render compiles. Each Win32 failure is thrown as a
#: ``Win32Exception`` carrying a named code, so a build that could not be
#: contained stops before it starts anything.
JOB_SOURCE: Final = """\
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
"""


def enter_job_lines() -> tuple[str, ...]:
    """The lines that compile the job's type and put this process in its job.

    Returns:
        The guarded compile, then the call, its handle discarded because the
        static field already keeps it.
    """
    return (*add_type_lines(JOB_TYPE, JOB_SOURCE), f"[void][{JOB_TYPE}]::Enter()")


__all__ = ["JOB_SOURCE", "JOB_TYPE", "enter_job_lines"]
