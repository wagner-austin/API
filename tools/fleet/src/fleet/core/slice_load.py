"""How much CPU a node's measured slices took while a fleet check ran there.

MCPs board task 349e25ee. MCPs/execution-deploy ran 308 s of its 300 s
budget on lavender (fleet job 5d96ec2e) while the host's GitHub runners ran
CI jobs beside it, and the queue row said only ``budget=over:308s/300s``:
whether CI was running, and how hard, was in a session's note and nowhere a
query could read. The audit of the task's closure (2026-10-30T06:52Z) found
the same rows printing the same totals whatever the cause.

So a check's build reads each measured slice's ``cpu.stat`` when the recipe
starts and when it ends, and prints the difference as one line in the
transcript per slice, which the verdict (:func:`read_slice_load`, in
:func:`fleet.core.verdict.judge`) copies onto the queue row and the thread as
``<field>=<cpu>s/<wall>s``: the CPU-seconds the slice used in the seconds the
check ran. On a host without the slice the line says so and the field reads
``none``. Every ``cpu.stat`` here is one any account may read, so the
runner's own build and execdocker's read them alike.

Two slices are measured (:data:`MEASURED_SLICES`):

- ``ci``, :data:`~fleet.contracts.runner_slice.CI_SLICE_NAME`: what every
  runner's CI job took beside the check.
- ``dispatch``, :data:`~fleet.contracts.check_unit.DISPATCH_SLICE_NAME`:
  the roster's dispatch share, which holds the check's own unit and, on a
  NixOS node, the nix daemon that runs its builds (``Slice=`` of
  ``nix-daemon.service``). MCPs board task 51e3eac5: MCPs/fleet-nixos is
  charged the cores that share's builds fill, and the audit of that closure
  (a6179a34) found the CPU-seconds behind the number read by a session's
  sampler and written into no record. This field is that figure, for every
  run.

WHAT THE CHECK ITSELF WAITED ON (MCPs board task 2e2df92a). A slice's CPU
says how busy the node was, not whether the check was held back. On
2026-11-15 packages/maketools ran 416 s of its 300 s on lavender (fleet job
86f3aa91) beside a docker-lane Gradle build, its suite's median case at its
idle time and its process-starting cases 4 to 15 times it, while the
dispatch slice used 7.3 of its 8 cores: CPU, disk or memory, the transcript
could not say. So the build also reads its own cgroup's pressure stall
information (``cpu.pressure``, ``io.pressure``, ``memory.pressure``, the
``some`` line's ``total``) at the start and the end, and prints the seconds
in which at least one of its tasks stalled on each
(:func:`pressure_line`); a kernel without PSI prints
:func:`pressure_none_line`.

A docker project's build reads the docker lane's cgroup instead
(:mod:`fleet.core.linux_isolated_build` passes its own
:class:`PressureCgroup`): its containers run under execdocker's user manager,
not in the check's unit, whose cgroup holds only the shell that started
them. Measured on lavender on 2026-11-15, osrs-lab/observer's unit used 1 CPU-s
of its run while ``user@1001.service`` used 8 cores.
"""

from __future__ import annotations

import re
from typing import Final

from typing_extensions import TypedDict

from fleet.contracts.check_unit import DISPATCH_SLICE_NAME
from fleet.contracts.runner_slice import CI_SLICE_NAME

#: Every transcript line's opening token.
SLICE_LOAD_MARKER: Final = "FLEET_SLICE_LOAD"


class MeasuredSlice(TypedDict):
    """One slice a check's build measures.

    Attributes:
        name: The slice's unit name, which its transcript line names.
        cpu_stat: Its ``cpu.stat``, whose ``usage_usec`` counts its time.
        field: The verdict field it is copied onto, and the stem of the
            ``sh`` variables its lines keep.
    """

    name: str
    cpu_stat: str
    field: str


#: The CI slice every runner's jobs run in.
CI_SLICE: Final = MeasuredSlice(
    name=CI_SLICE_NAME, cpu_stat=f"/sys/fs/cgroup/{CI_SLICE_NAME}/cpu.stat", field="ci"
)

#: The dispatch share, which holds the check and a NixOS node's nix daemon.
DISPATCH_SLICE: Final = MeasuredSlice(
    name=DISPATCH_SLICE_NAME,
    cpu_stat=f"/sys/fs/cgroup/system.slice/{DISPATCH_SLICE_NAME}/cpu.stat",
    field="dispatch",
)

#: What every build measures, in the order its lines are printed.
MEASURED_SLICES: Final[tuple[MeasuredSlice, ...]] = (CI_SLICE, DISPATCH_SLICE)

#: The shell variable holding the clock when the recipe started.
CLOCK_STARTED: Final = "slice_clock_started"

#: The pressure line's opening token.
PRESSURE_MARKER: Final = "FLEET_RUN_PRESSURE"

#: Each resource whose pressure file the build reads, in the order the line names them.
PRESSURES: Final[tuple[str, ...]] = ("cpu", "io", "memory")

#: The cgroup v2 hierarchy the build's own cgroup sits in.
CGROUP_ROOT: Final = "/sys/fs/cgroup"

#: Where the build reads its own cgroup's path: cgroup v2's one ``0::<path>`` line.
OWN_CGROUP: Final = "/proc/self/cgroup"

#: The shell variable holding the measured cgroup's directory.
PRESSURE_CGROUP: Final = "pressure_cgroup"


class PressureCgroup(TypedDict):
    """The cgroup whose pressure stalls a build reads.

    Attributes:
        field: Its one-word name on the verdict's ``stalls`` field.
        subject: What the transcript line calls it. The line is printed inside
            single quotes in ``sh``, so it holds no apostrophe.
        locate: The ``sh`` that prints its path below :data:`CGROUP_ROOT`.
    """

    field: str
    subject: str
    locate: str


#: The build's own cgroup: the check's unit, read off cgroup v2's one ``0::<path>`` line.
CHECK_CGROUP: Final = PressureCgroup(
    field="check",
    subject="the cgroup of the check",
    locate=f"sed -n 's/^0:://p' '{OWN_CGROUP}'",
)


def pressure_line(measured: PressureCgroup, stalls: tuple[str, str, str], wall: str) -> str:
    """The line naming how long the measured cgroup's tasks stalled on each resource.

    Args:
        measured: The cgroup.
        stalls: The ``some`` seconds on cpu, io and memory, or ``printf`` placeholders.
        wall: The check's seconds, or a ``printf`` placeholder.

    Returns:
        The line, without its newline.
    """
    cpu, io, memory = stalls
    return (
        f"{PRESSURE_MARKER}: {measured['subject']} stalled {cpu} s on cpu, {io} s on io and "
        f"{memory} s on memory in the {wall} s the check ran"
    )


def pressure_none_line(measured: PressureCgroup) -> str:
    """The line on a kernel or cgroup that keeps no pressure stall information.

    Args:
        measured: The cgroup.

    Returns:
        The line, without its newline.
    """
    return f"{PRESSURE_MARKER}: {measured['subject']} keeps no pressure stall information"


def _stall(resource: str) -> str:
    """The ``sh`` that prints the ``some`` total, in microseconds, of the build's cgroup.

    Args:
        resource: One of :data:`PRESSURES`.

    Returns:
        The command.
    """
    path = f'"${PRESSURE_CGROUP}/{resource}.pressure"'
    return f'awk \'$1 == "some" {{ sub("total=", "", $5); print $5 }}\' {path}'


def _pressure_start_lines(cgroup_root: str, measured: PressureCgroup) -> list[str]:
    """The ``sh`` lines that note the measured cgroup and its stall totals.

    Each read sits in an ``if`` condition, so it cannot end a build under
    ``set -e``; a file that cannot be read leaves its total empty.

    Args:
        cgroup_root: :data:`CGROUP_ROOT` on a node.
        measured: The cgroup.

    Returns:
        The lines.
    """
    lines = [
        f'if {PRESSURE_CGROUP}="{cgroup_root}$({measured["locate"]})"; then :; '
        f"else {PRESSURE_CGROUP}=''; fi"
    ]
    for resource in PRESSURES:
        started = f"pressure_{resource}_started"
        lines.append(
            f"if {started}=\"$({_stall(resource)} 2>/dev/null)\"; then :; else {started}=''; fi"
        )
    return lines


def _pressure_end_lines(log: str, measured: PressureCgroup) -> list[str]:
    """The ``sh`` lines that print the measured cgroup's stall seconds into the transcript.

    Args:
        log: The transcript's path, appended to.
        measured: The cgroup.

    Every read sits in the ``if`` condition, so a total that cannot be read
    at either end prints :func:`pressure_none_line` rather than ending the
    build under ``set -e``.

    Returns:
        The lines, run after the recipe's phase.
    """
    reads = " && ".join(
        f'[ -n "$pressure_{resource}_started" ] && '
        f'pressure_{resource}_ended="$({_stall(resource)} 2>/dev/null)" && '
        f'[ -n "$pressure_{resource}_ended" ]'
        for resource in PRESSURES
    )
    lines = [f"if {reads}; then"]
    seconds = " ".join(
        f'"$(( (pressure_{resource}_ended - pressure_{resource}_started) / 1000000 ))"'
        for resource in PRESSURES
    )
    line = pressure_line(measured, ("%s", "%s", "%s"), "%s")
    lines.extend(
        [
            f"  printf '{line}\\n' {seconds} \"$(( $(date +%s) - {CLOCK_STARTED} ))\" >> '{log}'",
            "else",
            f"  printf '%s\\n' '{pressure_none_line(measured)}' >> '{log}'",
            "fi",
        ]
    )
    return lines


def slice_used_line(measured: MeasuredSlice, cpu: str, throttled: str, wall: str) -> str:
    """The line on a host whose slice ran through the whole check.

    Args:
        measured: The slice.
        cpu: Its CPU-seconds over the check, or a ``printf`` placeholder.
        throttled: The seconds its CPU quota held it back over the check,
            ``cpu.stat``'s ``throttled_usec``, or a ``printf`` placeholder.
        wall: The check's seconds, or a ``printf`` placeholder.

    Returns:
        The line, without its newline.
    """
    return (
        f"{SLICE_LOAD_MARKER}: {measured['name']} used {cpu} CPU-seconds and was throttled "
        f"for {throttled} s in the {wall} s the check ran"
    )


def slice_none_line(measured: MeasuredSlice) -> str:
    """The line on a host without the slice.

    Args:
        measured: The slice.

    Returns:
        The line, without its newline.
    """
    return f"{SLICE_LOAD_MARKER}: no {measured['name']} on this node"


def slice_gone_line(measured: MeasuredSlice) -> str:
    """The line when the slice was there at the start and gone at the end.

    Args:
        measured: The slice.

    Returns:
        The line, without its newline.
    """
    return f"{SLICE_LOAD_MARKER}: {measured['name']} ended before the check did"


#: The two ``cpu.stat`` counters each slice's lines read, as the ``sh``
#: variables they are kept in name them: its CPU time and the time its quota
#: held it back. A slice systemd gives a CPU weight or quota has the cpu
#: controller, so its ``cpu.stat`` carries both (read on lavender, sedona and
#: serendipity on 2026-11-15).
SLICE_COUNTERS: Final[tuple[tuple[str, str], ...]] = (
    ("usage", "usage_usec"),
    ("throttled", "throttled_usec"),
)


def _counter(cpu_stat: str, key: str) -> str:
    """The ``sh`` that prints one of a slice's ``cpu.stat`` counters.

    Args:
        cpu_stat: The slice's ``cpu.stat``.
        key: The counter, one of :data:`SLICE_COUNTERS`' keys.

    Returns:
        The command.
    """
    return f"awk '$1 == \"{key}\" {{ print $2 }}' '{cpu_stat}'"


def sh_slice_load_start_lines(
    slices: tuple[MeasuredSlice, ...], cgroup_root: str, pressure_of: PressureCgroup
) -> list[str]:
    """The ``sh`` lines that note each slice's CPU time, the measured stalls and the clock.

    Each read sits in an ``if`` condition, so it cannot end a build under
    ``set -e``.

    Args:
        slices: The slices, :data:`MEASURED_SLICES` on a node.
        cgroup_root: :data:`CGROUP_ROOT` on a node.
        pressure_of: The cgroup whose stalls are read, :data:`CHECK_CGROUP`
            for a build whose work runs in its own unit.

    Returns:
        The lines; ``<field>_<counter>_started`` is empty on a host without the slice.
    """
    lines = _pressure_start_lines(cgroup_root, pressure_of)
    for measured in slices:
        stat = measured["cpu_stat"]
        for name, key in SLICE_COUNTERS:
            started = f"{measured['field']}_{name}_started"
            lines.append(
                f"if [ -r '{stat}' ] && {started}=\"$({_counter(stat, key)})\"; then :; "
                f"else {started}=''; fi"
            )
    lines.append(f'{CLOCK_STARTED}="$(date +%s)"')
    return lines


def sh_slice_load_end_lines(
    slices: tuple[MeasuredSlice, ...], log: str, pressure_of: PressureCgroup
) -> list[str]:
    """The ``sh`` lines that print each slice's CPU time over the recipe into the transcript.

    Every read sits in an ``if`` condition, so a slice that vanished during
    the check prints :func:`slice_gone_line` rather than ending the build
    under ``set -e`` before it writes its status.

    Args:
        slices: The slices, :data:`MEASURED_SLICES` on a node.
        log: The transcript's path, appended to.
        pressure_of: The cgroup the start lines read, so the line names it.

    Returns:
        The lines, run after the recipe's phase and before the status is written.
    """
    lines: list[str] = []
    for measured in slices:
        stat = measured["cpu_stat"]
        reads = " && ".join(
            f'[ -n "${measured["field"]}_{name}_started" ] && '
            f'{measured["field"]}_{name}_ended="$({_counter(stat, key)})" && '
            f'[ -n "${measured["field"]}_{name}_ended" ]'
            for name, key in SLICE_COUNTERS
        )
        seconds = " ".join(
            f'"$(( ({measured["field"]}_{name}_ended - {measured["field"]}_{name}_started) '
            f'/ 1000000 ))"'
            for name, _ in SLICE_COUNTERS
        )
        used = slice_used_line(measured, "%s", "%s", "%s")
        started = f"{measured['field']}_usage_started"
        lines.extend(
            [
                f"if {reads}; then",
                f"  printf '{used}\\n' {seconds} "
                f"\"$(( $(date +%s) - {CLOCK_STARTED} ))\" >> '{log}'",
                f'elif [ -n "${started}" ]; then',
                f"  printf '%s\\n' '{slice_gone_line(measured)}' >> '{log}'",
                "else",
                f"  printf '%s\\n' '{slice_none_line(measured)}' >> '{log}'",
                "fi",
            ]
        )
    return [*lines, *_pressure_end_lines(log, pressure_of)]


def read_slice_load(tail: str, measured: MeasuredSlice) -> str:
    """Read one slice's load over a check off its transcript tail.

    Args:
        tail: The transcript's last lines.
        measured: The slice.

    Returns:
        ``<cpu>s/<wall>s/throttled:<seconds>s`` from the slice's last
        :func:`slice_used_line`, ``gone`` from :func:`slice_gone_line`,
        ``none`` from :func:`slice_none_line`, else ``unread`` (a build from
        before the line, or one stopped before it).
    """
    parts = slice_used_line(measured, "\n", "\n", "\n").split("\n")
    pattern = re.compile("^" + r"(\d+)".join(re.escape(part) for part in parts) + "$", re.MULTILINE)
    used: re.Match[str] | None = None
    for match in pattern.finditer(tail):
        used = match
    if used is not None:
        cpu, throttled, wall = used.groups()
        return f"{cpu}s/{wall}s/throttled:{throttled}s"
    if slice_gone_line(measured) in tail:
        return "gone"
    if slice_none_line(measured) in tail:
        return "none"
    return "unread"


def read_run_pressure(tail: str, cgroups: tuple[PressureCgroup, ...]) -> str:
    """Read the measured cgroup's stalls over a check off its transcript tail.

    Args:
        tail: The transcript's last lines.
        cgroups: Every cgroup a build may measure, so the line's subject
            names which one it read.

    Returns:
        ``<field>:<cpu>s-cpu/<io>s-io/<memory>s-memory`` from the tail's last
        :data:`PRESSURE_MARKER` line when it is a :func:`pressure_line`,
        ``<field>:none`` when it is a :func:`pressure_none_line`, else
        ``unread`` (no such line: a build from before it, or one stopped
        before it; or a subject none of ``cgroups`` writes).
    """
    printed = [line for line in tail.splitlines() if line.startswith(f"{PRESSURE_MARKER}: ")]
    if not printed:
        return "unread"
    last = printed[-1]
    for measured in cgroups:
        if last == pressure_none_line(measured):
            return f"{measured['field']}:none"
        parts = pressure_line(measured, ("\n", "\n", "\n"), "\n").split("\n")
        stalled = re.fullmatch(r"(\d+)".join(re.escape(part) for part in parts), last)
        if stalled is not None:
            cpu, io, memory, _ = stalled.groups()
            return f"{measured['field']}:{cpu}s-cpu/{io}s-io/{memory}s-memory"
    return "unread"


__all__ = [
    "CGROUP_ROOT",
    "CHECK_CGROUP",
    "CI_SLICE",
    "CLOCK_STARTED",
    "DISPATCH_SLICE",
    "MEASURED_SLICES",
    "OWN_CGROUP",
    "PRESSURES",
    "PRESSURE_CGROUP",
    "PRESSURE_MARKER",
    "SLICE_COUNTERS",
    "SLICE_LOAD_MARKER",
    "MeasuredSlice",
    "PressureCgroup",
    "pressure_line",
    "pressure_none_line",
    "read_run_pressure",
    "read_slice_load",
    "sh_slice_load_end_lines",
    "sh_slice_load_start_lines",
    "slice_gone_line",
    "slice_none_line",
    "slice_used_line",
]
