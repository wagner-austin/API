"""How much CPU a runner host's CI took while a fleet check ran there.

MCPs board task 349e25ee. MCPs/execution-deploy ran 308 s of its 300 s
budget on lavender (fleet job 5d96ec2e) while the host's GitHub runners ran
CI jobs beside it, and the queue row said only ``budget=over:308s/300s``:
whether CI was running, and how hard, was in a session's note and nowhere a
query could read. The audit of the task's closure (2026-10-30T06:52Z) found
the same rows printing the same totals whatever the cause.

So a check's build reads the CI slice's ``cpu.stat`` when the recipe starts
and when it ends, and prints the difference as one line in the transcript,
which the verdict (:func:`read_ci_load`, in :func:`fleet.core.verdict.judge`) copies onto the queue
row and the thread as ``ci=<cpu>s/<wall>s``: the CPU-seconds every runner's
job used in the seconds the check ran. On a host with no CI slice the line
says so and the field reads ``none``. The slice is
:data:`~fleet.contracts.runner_slice.CI_SLICE_NAME`, whose ``cpu.stat`` any
account may read, so the runner's own build and execdocker's read it alike.
"""

from __future__ import annotations

import re
from typing import Final

from fleet.contracts.runner_slice import CI_SLICE_NAME

#: The transcript line's opening token.
CI_LOAD_MARKER: Final = "FLEET_CI_LOAD"

#: The CI slice's CPU accounting, whose ``usage_usec`` counts every job's time.
CI_CPU_STAT: Final = f"/sys/fs/cgroup/{CI_SLICE_NAME}/cpu.stat"

#: The line on a host whose CI slice ran through the whole check.
CI_LOAD_USED: Final[re.Pattern[str]] = re.compile(
    rf"^{CI_LOAD_MARKER}: {CI_SLICE_NAME} used (\d+) CPU-seconds in the (\d+) s the check ran$",
    re.MULTILINE,
)

#: The line on a host with no CI slice.
CI_LOAD_NONE: Final = f"{CI_LOAD_MARKER}: no {CI_SLICE_NAME} on this node"

#: The line when the slice was there at the start and gone at the end.
CI_LOAD_GONE: Final = f"{CI_LOAD_MARKER}: {CI_SLICE_NAME} ended before the check did"


def _usage(cpu_stat: str) -> str:
    """The ``sh`` that prints a slice's ``usage_usec``.

    Args:
        cpu_stat: The slice's ``cpu.stat``.

    Returns:
        The command.
    """
    return f"awk '$1 == \"usage_usec\" {{ print $2 }}' '{cpu_stat}'"


def sh_ci_load_start_lines(cpu_stat: str) -> list[str]:
    """The ``sh`` lines that note the CI slice's CPU time and the clock before the recipe.

    The read sits in an ``if`` condition, so it cannot end a build under
    ``set -e``.

    Args:
        cpu_stat: The CI slice's ``cpu.stat``, :data:`CI_CPU_STAT` on a node.

    Returns:
        The lines; ``ci_usage_started`` is empty on a host with no slice.
    """
    return [
        f"if [ -r '{cpu_stat}' ] && ci_usage_started=\"$({_usage(cpu_stat)})\"; then :; "
        "else ci_usage_started=''; fi",
        'ci_clock_started="$(date +%s)"',
    ]


def sh_ci_load_end_lines(cpu_stat: str, log: str) -> list[str]:
    """The ``sh`` lines that print the CI slice's CPU time over the recipe into the transcript.

    Every read sits in an ``if`` condition, so a slice that vanished during
    the check prints :data:`CI_LOAD_GONE` rather than ending the build under
    ``set -e`` before it writes its status.

    Args:
        cpu_stat: The CI slice's ``cpu.stat``, :data:`CI_CPU_STAT` on a node.
        log: The transcript's path, appended to.

    Returns:
        The lines, run after the recipe's phase and before the status is written.
    """
    used = f"{CI_LOAD_MARKER}: {CI_SLICE_NAME} used %s CPU-seconds in the %s s the check ran\\n"
    return [
        f'if [ -n "$ci_usage_started" ] && ci_usage_ended="$({_usage(cpu_stat)})" '
        '&& [ -n "$ci_usage_ended" ]; then',
        f"  printf '{used}' \"$(( (ci_usage_ended - ci_usage_started) / 1000000 ))\" "
        f"\"$(( $(date +%s) - ci_clock_started ))\" >> '{log}'",
        'elif [ -n "$ci_usage_started" ]; then',
        f"  printf '%s\\n' '{CI_LOAD_GONE}' >> '{log}'",
        "else",
        f"  printf '%s\\n' '{CI_LOAD_NONE}' >> '{log}'",
        "fi",
    ]


def read_ci_load(tail: str) -> str:
    """Read the CI slice's load over a check off its transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<cpu>s/<wall>s`` from the last :data:`CI_LOAD_USED` line, ``none``
        from :data:`CI_LOAD_NONE`, ``gone`` from :data:`CI_LOAD_GONE`, else
        ``unread`` (a build from before the line, or one stopped before it).
    """
    used: re.Match[str] | None = None
    for match in CI_LOAD_USED.finditer(tail):
        used = match
    if used is not None:
        return f"{used.group(1)}s/{used.group(2)}s"
    if CI_LOAD_GONE in tail:
        return "gone"
    if CI_LOAD_NONE in tail:
        return "none"
    return "unread"


__all__ = [
    "CI_CPU_STAT",
    "CI_LOAD_GONE",
    "CI_LOAD_MARKER",
    "CI_LOAD_NONE",
    "CI_LOAD_USED",
    "read_ci_load",
    "sh_ci_load_end_lines",
    "sh_ci_load_start_lines",
]
