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
``anon`` and ``shmem`` lines of its ``memory.stat``, not minus
``memory.current``: ``current`` includes the container's page cache, which
``MemAvailable`` already counts as reclaimable, so growing what it cannot
give back to the cap takes exactly ``max - anon - shmem`` out of what is
available. Shared memory is subtracted with the anonymous memory because it
is resident and swap-backed, so ``MemAvailable`` already leaves it out, and
charging it again counted it twice. Measured on colossus on 2026-10-14
04:33Z (MCPs board task 4c33062e): the public search's copy, capped at
8 GiB, held 0.09 GB anonymous and 2.04 GB of shared buffers, and the old
``max - anon`` charged it 7.9 GB, which with the other caps left the node
3.2 GB against its 6 GB reservation, so it claimed no job. A container
whose ``memory.max`` reads ``max`` promises nothing; neither does one that
exits between the listing and the read, whose files are gone by then, nor
one whose ``memory.stat`` has no ``anon`` line.

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

THE NODE'S CPU PRESSURE IS READ, NOT INFERRED FROM ITS LOAD (MCPs board task
2ce5dc13). ``cpu_pressure_pct`` is the ``some avg60`` of
``/proc/pressure/cpu``: the share of the last minute in which at least one
runnable task waited for a CPU, which is the slowdown a suite placed there
meets, where a load average counts tasks without saying whether they waited.
A kernel without the file, or a file without a ``some`` line, does not read
as a calm node: the first ends the probe under ``set -e`` with ``awk``'s own
message, the second prints the field empty, which :mod:`fleet.core.probe`
refuses as not a number. Measured 2026-10-16 every linux node carries it:
diphtheria ``some avg60=9.33``, colossus ``0.01``, lavender-wsl ``2.99``.

ITS I/O PRESSURE IS READ THE SAME WAY (MCPs board task 814c4555).
``io_pressure_pct`` is the ``some avg60`` of ``/proc/pressure/io``, the share
of the last minute in which at least one task stalled on the disk, which the
CPU reading does not see: on 2026-10-18 diphtheria read 14 percent CPU and 69
percent I/O while a database dump and a VM install shared its disk, and a
check placed there ran five times its usual time. It is printed after the
CPU reading and fails the same ways.

THE NODE'S RESERVATIONS ARE CHARGED LAST (MCPs board task 84a2b690,
:mod:`fleet.core.reservations` for the file and why). Each live
``.reservation`` file's ``cores`` are summed into ``reserved_cores``, which
the capacity check takes off the node's cores; an expired one counts
nothing. A file the probe cannot read prints ``reservation_unreadable=`` with
its path and ends the probe, so the node's reading is refused by name rather
than read as holding nothing. A carriage return a Windows-written file
carries is stripped before the fields are read.
"""

from __future__ import annotations

from fleet.core import reservations

#: Where the probe reads memory from; the test replaces these two lines.
MEMINFO_LINE = "meminfo=/proc/meminfo\n"
CGROUPS_LINE = "cgroups=/sys/fs/cgroup/system.slice\n"

#: Where the probe reads CPU pressure from; the test replaces it.
PRESSURE_LINE = "pressure=/proc/pressure/cpu\n"

#: Where the probe reads I/O pressure from; the test replaces it.
IO_PRESSURE_LINE = "io_pressure=/proc/pressure/io\n"


def _some_avg60(path_variable: str, field: str) -> str:
    """The lines that print one pressure file's ``some avg60`` as ``field``.

    Args:
        path_variable: The shell variable holding the file's path.
        field: The field the reading is printed as.

    Returns:
        An ``awk`` reading of the ``some`` line's ``avg60=`` and its
        ``printf``, empty when the file has no such line.
    """
    return (
        f'{field}=$(awk \'$1 == "some" {{ for (i = 2; i <= NF; i++)'
        + f' if (sub(/^avg60=/, "", $i)) print $i }}\' "${path_variable}")\n'
        + f"printf '{field}=%s\\n' \"${field}\"\n"
    )


#: The CPU then the I/O ``some avg60`` readings, printed after the cores
#: they are weighed beside.
PRESSURE_PART = (
    PRESSURE_LINE
    + _some_avg60("pressure", "cpu_pressure_pct")
    + IO_PRESSURE_LINE
    + _some_avg60("io_pressure", "io_pressure_pct")
)

#: Where the probe reads the node's reservations from; the test replaces it.
RESERVATIONS_LINE = f'reservations="$HOME/{reservations.DIRECTORY}"\n'

#: The reservations' sum, after the body's other fields, so an unreadable
#: file ends the probe with them already printed; the dialect's CI-slice
#: lines come after the body and are not printed then, which costs nothing,
#: since the node's reading is refused.
RESERVATIONS_PART = (
    RESERVATIONS_LINE
    + "now=$(date +%s)\n"
    + "reserved=0\n"
    + f'for reservation in "$reservations"/*{reservations.SUFFIX}; do\n'
    + '  [ -e "$reservation" ] || continue\n'
    + '  if ! held=$(awk -F= -v now="$now" \'{ sub(/\\r$/, "") }'
    + ' $1 == "cores" { cores = $2 } $1 == "until_unix" { ends = $2 }'
    + " END { if (cores !~ /^[0-9]+$/ || ends !~ /^[0-9]+$/) exit 1;"
    + ' if (ends + 0 > now + 0) print cores + 0; else print 0 }\' "$reservation"); then\n'
    + f"    printf '{reservations.UNREADABLE_FIELD}=%s\\n' \"$reservation\"\n"
    + "    exit 0\n"
    + "  fi\n"
    + "  reserved=$((reserved + held))\n"
    + "done\n"
    + f"printf '{reservations.RESERVED_CORES_FIELD}=%s\\n' \"$reserved\"\n"
)

#: The probe after the dialect's prologue, verbatim.
CAPACITY_PROBE_BODY = (
    MEMINFO_LINE
    + CGROUPS_LINE
    + 'available=$(awk \'/^MemAvailable:/ { printf "%.0f", $2 * 1024 }\' "$meminfo")\n'
    "promised=0\n"
    'for cgroup in "$cgroups"/docker-*.scope; do\n'
    '  if limit=$(cat "$cgroup/memory.max" 2>/dev/null) &&'
    ' resident=$(awk \'$1 == "anon" { anon = $2; seen = 1 } $1 == "shmem" { shmem = $2 }'
    ' END { if (seen) printf "%.0f", anon + shmem }\' "$cgroup/memory.stat" 2>/dev/null); then\n'
    '    if [ "$limit" != max ] && [ -n "$resident" ]; then\n'
    "      promised=$((promised + limit - resident))\n"
    "    fi\n"
    "  fi\n"
    "done\n"
    'awk -v available="$available" -v promised="$promised" \'BEGIN {'
    ' printf "free_ram_gb=%.3f\\npromised_ram_gb=%.3f\\n",'
    " (available - promised) / 1073741824, promised / 1073741824 }'\n"
    "df -kP / | awk 'NR == 2 { printf \"free_disk_gb=%.3f\\n\", $4 / 1048576 }'\n"
    "printf 'logical_cores=%s\\n' \"$(nproc)\"\n" + PRESSURE_PART + RESERVATIONS_PART
)

__all__ = [
    "CAPACITY_PROBE_BODY",
    "CGROUPS_LINE",
    "IO_PRESSURE_LINE",
    "MEMINFO_LINE",
    "PRESSURE_LINE",
    "PRESSURE_PART",
    "RESERVATIONS_LINE",
    "RESERVATIONS_PART",
]
