"""A Linux node's capacity probe body: its free memory, cores, pressures and reservations.

FREE MEMORY IS ``MemAvailable`` (MCPs board tasks a282400d and d111690a), the
kernel's estimate of what a new process can take, which already excludes
everything the corvis stack holds. A constant reservation for the stack on top
of it counted the stack twice: diphtheria's 16 GB, set on 2026-09-20
(78e6e8d2e, repo API) before the stack's backends moved in, refused every
dispatch once they had (179 ``NODE_OWNER_RESERVED`` ticks by 18:30Z on
2026-09-29). The node's own ``budget.reserved_ram_gb`` is the only memory the
capacity check holds back, for whoever uses the machine.

WHY NO CAPPED CONTAINER'S HEADROOM IS CHARGED. From 2026-09-29 the probe also
took off what each memory-capped stack container could still grow into
(``memory.max`` minus its ``anon`` and ``shmem``), so a job would never meet a
worker growing to its cap. On colossus that held back 17.13 GiB of an idle
host's 19.59 available (8d3e4b16, measured 2026-10-15), leaving fleet jobs
about 2.5 GiB of a 31 GB machine for memory nothing was using. The kernel now
decides who yields instead: every check runs at ``OOMScoreAdjust=1000``
(MCPs ``scripts/host/lib/cpu-shares.sh``), so when a stack container grows
into a job the job is ended first, and its runner gives it back to the queue
(:func:`fleet.core.linux_unit_end.killed_for_memory`); the compose caps stay
as runaway containment. The charge had in fact stopped on 2026-10-22, when
MCPs 96bea26be moved every container under ``system-corvis.slice`` or
``system-dispatch.slice`` and its ``system.slice/docker-*.scope`` glob matched
none: measured on colossus on 2026-10-25, it read 0 scopes there and 8 under
``system-corvis.slice``, and charged 0.000 GiB.

THE ROOTS ARE VARIABLES ON THEIR OWN LINES so a test can point the script at a
fake ``meminfo``, pressure files and reservations and run it for real under
``sh``; nothing is substituted into it in production. The division to
gibibytes with three decimals happens in ``awk``, so the parser sees the
shape the Windows dialect prints.

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

#: Where the probe reads memory from; the test replaces it.
MEMINFO_LINE = "meminfo=/proc/meminfo\n"

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
    + 'awk \'/^MemAvailable:/ { printf "free_ram_gb=%.3f\\n", $2 / 1048576 }\' "$meminfo"\n'
    "df -kP / | awk 'NR == 2 { printf \"free_disk_gb=%.3f\\n\", $4 / 1048576 }'\n"
    "printf 'logical_cores=%s\\n' \"$(nproc)\"\n" + PRESSURE_PART + RESERVATIONS_PART
)

__all__ = [
    "CAPACITY_PROBE_BODY",
    "IO_PRESSURE_LINE",
    "MEMINFO_LINE",
    "PRESSURE_LINE",
    "PRESSURE_PART",
    "RESERVATIONS_LINE",
    "RESERVATIONS_PART",
]
