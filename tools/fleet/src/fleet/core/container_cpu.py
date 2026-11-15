"""The CPU bound a docker project's containers stated, gathered after its recipe.

MCPs board task 2e2df92a. A docker project's containers run under
execdocker's user manager in the checks slice, beside every other check on
the node, and the fleet charges a check only the workers it granted. On
2026-11-15 an osrs-lab observer build with no CPU bound used 8.15 cores of
lavender's 8-core dispatch quota against its 4-core charge, and the
packages/maketools check beside it ran 416 s of its 300 (fleet job
86f3aa91). Whether a container held its grant was then in a session's
sampler and nowhere a query could read.

So a container states its own bound: a line reading
``FLEET_CONTAINER_CPU: <its cgroup's cpu.max>`` (:data:`CONTAINER_CPU_MARKER`,
for example ``FLEET_CONTAINER_CPU: 400000 100000``), printed from inside
it, where only the container's own cgroup is visible. After the recipe the
build gathers every distinct one from the whole transcript into one line
(:func:`sh_container_cpu_lines`), since a container starts long before the
tail, and the verdict copies it onto the queue row (:func:`read_container_cpus`)
as ``containers=<quota>/<period>[,...]``, or ``none`` when no container said.
"""

from __future__ import annotations

import re
from typing import Final

#: The line a container prints, followed by its ``cpu.max``.
CONTAINER_CPU_MARKER: Final = "FLEET_CONTAINER_CPU"

#: The build's one line gathering every bound its containers stated.
CONTAINER_CPUS_MARKER: Final = "FLEET_CONTAINER_CPUS"


def container_cpus_line(stated: str) -> str:
    """The line naming every distinct bound the containers stated.

    Args:
        stated: The ``cpu.max`` values, comma-joined, or a ``printf`` placeholder.

    Returns:
        The line, without its newline.
    """
    return f"{CONTAINER_CPUS_MARKER}: {stated}"


def container_cpus_none_line() -> str:
    """The line when no container of the recipe stated a bound.

    Returns:
        The line, without its newline.
    """
    return f"{CONTAINER_CPUS_MARKER}: none stated"


def sh_container_cpu_lines(log: str) -> list[str]:
    """The ``sh`` lines that gather the stated bounds into the transcript.

    The read sits in an ``if`` condition, so a transcript it cannot read
    prints :func:`container_cpus_none_line` rather than ending the build
    under ``set -e`` before it writes its status.

    Args:
        log: The transcript's path, read and appended to.

    Returns:
        The lines, run after the recipe's phase.
    """
    gather = f"sed -n 's/^{CONTAINER_CPU_MARKER}: //p' '{log}' | sort -u | paste -s -d , -"
    return [
        f'if container_cpus="$({gather})" && [ -n "$container_cpus" ]; then',
        f"  printf '{container_cpus_line('%s')}\\n' \"$container_cpus\" >> '{log}'",
        "else",
        f"  printf '%s\\n' '{container_cpus_none_line()}' >> '{log}'",
        "fi",
    ]


def read_container_cpus(tail: str) -> str:
    """Read the bounds a recipe's containers stated off its transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        Each stated ``cpu.max`` as ``<quota>/<period>``, comma-joined, from
        the last :func:`container_cpus_line`; ``none`` from
        :func:`container_cpus_none_line`; else ``unread`` (a build that runs
        no container lane, or one stopped before the line).
    """
    last: str | None = None
    for printed in re.finditer(rf"^{CONTAINER_CPUS_MARKER}: .+$", tail, re.MULTILINE):
        last = printed.group(0)
    if last is None:
        return "unread"
    if last == container_cpus_none_line():
        return "none"
    stated = last.removeprefix(container_cpus_line(""))
    return ",".join(value.replace(" ", "/") for value in stated.split(","))


__all__ = [
    "CONTAINER_CPUS_MARKER",
    "CONTAINER_CPU_MARKER",
    "container_cpus_line",
    "container_cpus_none_line",
    "read_container_cpus",
    "sh_container_cpu_lines",
]
