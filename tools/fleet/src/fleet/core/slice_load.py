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


def slice_used_line(measured: MeasuredSlice, cpu: str, wall: str) -> str:
    """The line on a host whose slice ran through the whole check.

    Args:
        measured: The slice.
        cpu: Its CPU-seconds over the check, or a ``printf`` placeholder.
        wall: The check's seconds, or a ``printf`` placeholder.

    Returns:
        The line, without its newline.
    """
    return (
        f"{SLICE_LOAD_MARKER}: {measured['name']} used {cpu} CPU-seconds "
        f"in the {wall} s the check ran"
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


def _usage(cpu_stat: str) -> str:
    """The ``sh`` that prints a slice's ``usage_usec``.

    Args:
        cpu_stat: The slice's ``cpu.stat``.

    Returns:
        The command.
    """
    return f"awk '$1 == \"usage_usec\" {{ print $2 }}' '{cpu_stat}'"


def sh_slice_load_start_lines(slices: tuple[MeasuredSlice, ...]) -> list[str]:
    """The ``sh`` lines that note each slice's CPU time and the clock before the recipe.

    Each read sits in an ``if`` condition, so it cannot end a build under
    ``set -e``.

    Args:
        slices: The slices, :data:`MEASURED_SLICES` on a node.

    Returns:
        The lines; ``<field>_usage_started`` is empty on a host without the slice.
    """
    lines: list[str] = []
    for measured in slices:
        started = f"{measured['field']}_usage_started"
        stat = measured["cpu_stat"]
        lines.append(
            f"if [ -r '{stat}' ] && {started}=\"$({_usage(stat)})\"; then :; else {started}=''; fi"
        )
    lines.append(f'{CLOCK_STARTED}="$(date +%s)"')
    return lines


def sh_slice_load_end_lines(slices: tuple[MeasuredSlice, ...], log: str) -> list[str]:
    """The ``sh`` lines that print each slice's CPU time over the recipe into the transcript.

    Every read sits in an ``if`` condition, so a slice that vanished during
    the check prints :func:`slice_gone_line` rather than ending the build
    under ``set -e`` before it writes its status.

    Args:
        slices: The slices, :data:`MEASURED_SLICES` on a node.
        log: The transcript's path, appended to.

    Returns:
        The lines, run after the recipe's phase and before the status is written.
    """
    lines: list[str] = []
    for measured in slices:
        started = f"{measured['field']}_usage_started"
        ended = f"{measured['field']}_usage_ended"
        used = slice_used_line(measured, "%s", "%s")
        lines.extend(
            [
                f'if [ -n "${started}" ] && {ended}="$({_usage(measured["cpu_stat"])})" '
                f'&& [ -n "${ended}" ]; then',
                f"  printf '{used}\\n' \"$(( ({ended} - {started}) / 1000000 ))\" "
                f"\"$(( $(date +%s) - {CLOCK_STARTED} ))\" >> '{log}'",
                f'elif [ -n "${started}" ]; then',
                f"  printf '%s\\n' '{slice_gone_line(measured)}' >> '{log}'",
                "else",
                f"  printf '%s\\n' '{slice_none_line(measured)}' >> '{log}'",
                "fi",
            ]
        )
    return lines


def read_slice_load(tail: str, measured: MeasuredSlice) -> str:
    """Read one slice's load over a check off its transcript tail.

    Args:
        tail: The transcript's last lines.
        measured: The slice.

    Returns:
        ``<cpu>s/<wall>s`` from the slice's last :func:`slice_used_line`,
        ``gone`` from :func:`slice_gone_line`, ``none`` from
        :func:`slice_none_line`, else ``unread`` (a build from before the
        line, or one stopped before it).
    """
    before, between, after = slice_used_line(measured, "\n", "\n").split("\n")
    pattern = re.compile(
        rf"^{re.escape(before)}(\d+){re.escape(between)}(\d+){re.escape(after)}$", re.MULTILINE
    )
    used: re.Match[str] | None = None
    for match in pattern.finditer(tail):
        used = match
    if used is not None:
        return f"{used.group(1)}s/{used.group(2)}s"
    if slice_gone_line(measured) in tail:
        return "gone"
    if slice_none_line(measured) in tail:
        return "none"
    return "unread"


__all__ = [
    "CI_SLICE",
    "CLOCK_STARTED",
    "DISPATCH_SLICE",
    "MEASURED_SLICES",
    "SLICE_LOAD_MARKER",
    "MeasuredSlice",
    "read_slice_load",
    "sh_slice_load_end_lines",
    "sh_slice_load_start_lines",
    "slice_gone_line",
    "slice_none_line",
    "slice_used_line",
]
