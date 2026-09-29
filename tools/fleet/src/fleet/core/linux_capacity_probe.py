"""A Linux node's capacity probe body: free memory net of what capped containers are promised.

WHY FREE MEMORY IS NOT ENOUGH ON THE STACK HOST (MCPs board task a282400d).
``MemAvailable`` is the kernel's estimate of what a new process can take, so
it already excludes everything the corvis stack holds. diphtheria's budget
reserved 16 GB on top of that, set on 2026-09-20 (78e6e8d2e, repo API) when the
stack's backends were not yet there and the node read 26.4 GB free; once they
moved in, the reservation counted the stack a second time and the node refused
every dispatch (179 ``NODE_OWNER_RESERVED`` ticks by 18:30Z on 2026-09-29).

WHY A SMALLER CONSTANT IS NOT THE FIX EITHER. Four of the stack's containers
are memory-capped (the transcriber worker, both doc-extract workers and the
restore test, 4 GiB each, the second layer of the fleet's OOM containment),
and each may grow to its cap at any moment. Measured 2026-09-29 22:1xZ they
held 1.3, 2.0, 2.8 and 0.02 GB, so about 10.3 GB of growth was outstanding;
idle, about 16. A constant sized to admit a job on a busy afternoon admits it
while the workers sleep, and the workers then meet the job instead of their
caps. What stays put is the memory the caps have not yet been given, which is
what this probe subtracts.

WHAT A CAPPED CONTAINER IS STILL PROMISED is ``memory.max`` minus the
``anon`` line of its ``memory.stat``, not minus ``memory.current``:
``current`` includes the container's page cache, which ``MemAvailable``
already counts as reclaimable, so growing its anonymous memory to the cap
takes exactly ``max - anon`` out of what is available. A container whose
``memory.max`` reads ``max`` promises nothing; neither does one that exits
between the listing and the read, whose files are gone by then.

ONLY THE ROOTFUL DAEMON'S CONTAINERS COUNT, those under
``system.slice/docker-*.scope``. A node's execdocker user runs its rootless
containers under its own user slice, and those are the fleet's own jobs, which
the reservation must not charge twice. A node with no capped container, which
is every node but the stack host today, reads exactly the ``MemAvailable`` it
read before.

THE ROOTS ARE VARIABLES ON THEIR OWN LINES so a test can point the script at a
fake ``meminfo`` and cgroup tree and run it for real under ``sh``; nothing is
substituted into it in production. Arithmetic is shell integer arithmetic on
bytes, which ``dash`` carries in 64 bits, and the division to gibibytes with
three decimals happens once in ``awk``, so the parser sees the shape the
Windows dialect prints.
"""

from __future__ import annotations

#: Where the probe reads memory from; the test replaces these two lines.
MEMINFO_LINE = "meminfo=/proc/meminfo\n"
CGROUPS_LINE = "cgroups=/sys/fs/cgroup/system.slice\n"

#: The probe after the dialect's prologue, verbatim.
CAPACITY_PROBE_BODY = (
    MEMINFO_LINE
    + CGROUPS_LINE
    + 'available=$(awk \'/^MemAvailable:/ { printf "%.0f", $2 * 1024 }\' "$meminfo")\n'
    "promised=0\n"
    'for cgroup in "$cgroups"/docker-*.scope; do\n'
    '  if limit=$(cat "$cgroup/memory.max" 2>/dev/null) &&'
    ' anon=$(awk \'$1 == "anon" { print $2 }\' "$cgroup/memory.stat" 2>/dev/null); then\n'
    '    if [ "$limit" != max ] && [ -n "$anon" ]; then\n'
    "      promised=$((promised + limit - anon))\n"
    "    fi\n"
    "  fi\n"
    "done\n"
    'awk -v available="$available" -v promised="$promised" \'BEGIN {'
    ' printf "free_ram_gb=%.3f\\npromised_ram_gb=%.3f\\n",'
    " (available - promised) / 1073741824, promised / 1073741824 }'\n"
    "df -kP / | awk 'NR == 2 { printf \"free_disk_gb=%.3f\\n\", $4 / 1048576 }'\n"
    "printf 'logical_cores=%s\\n' \"$(nproc)\"\n"
)

__all__ = ["CAPACITY_PROBE_BODY", "CGROUPS_LINE", "MEMINFO_LINE"]
