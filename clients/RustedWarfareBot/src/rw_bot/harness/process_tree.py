"""Felling a match so nothing survives, and finding who holds a match port.

A match is not one process. The launcher starts a JVM, the JVM starts nothing
but the planner runs beside it, and under the fleet the whole thing hangs off
a spawn this package made. Killing only the root leaves the engine holding its
channel port, and a zombie engine on a leased port kills the NEXT match at the
bind rather than this one (vhdoom96b, 2026-08-09). So felling has to reach the
whole tree.

THE MECHANISM IS A PROCESS GROUP, decided at spawn time. Every spawn in
:mod:`rw_bot.harness._hook_defaults` starts a new session, which puts the
child and its descendants in one process group whose id is the child's pid,
and signalling the negated pid reaches all of them. The spawn and the kill are
therefore one decision in two places: a spawn that stopped starting a session
would leave this command signalling a group that does not exist.

WHY THIS IS COMPOSITION. Everything here returns data -- an argument vector or
a parsed answer -- and the impls in :mod:`rw_bot.harness._test_hooks` run it,
so each rule is a pure function a test can drive without a process table.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def fell_command(pid: int) -> tuple[str, ...]:
    """Return the command that kills a process and everything under it.

    Args:
        pid: Root process id of the tree, which is also its process group id
            because every spawn here starts a new session.

    Returns:
        The argument vector, program first.

    Raises:
        ValueError: When the pid is not positive. Negating a non-positive pid
            produces a signal target that means something else entirely --
            ``0`` is "every process in my own group", which would fell the
            harness itself, and ``-1`` is "every process I am permitted to
            signal". A bad pid must not be composed into either.
    """
    if pid <= 0:
        raise ValueError(
            f"a process tree is felled by its root pid, got {pid}: negating this "
            "would signal the caller's own group or every reachable process"
        )
    # -s KILL is the portable spelling of the signal; -- stops option parsing
    # so the negative pid is read as a process group rather than a flag.
    return ("kill", "-s", "KILL", "--", f"-{pid}")


#: The command that lists which process is listening where: TCP listeners,
#: numeric, with their owning processes, and no header so every line is a row.
PORT_LISTENER_COMMAND = ("ss", "-Hltnp")

#: The state ``ss`` reports for a socket waiting to accept.
_LISTENING = "LISTEN"

#: Where ``ss`` puts the local address once ``-H`` has removed the header:
#: state, recv queue, send queue, local, peer, users.
_LOCAL_ADDRESS_COLUMN = 3

#: ``ss`` reports the holder as ``users:(("java",pid=1234,fd=3))``. Absent
#: entirely when the socket belongs to another user and this process is not
#: privileged, which reads here as "no holder found" -- correct, because a
#: port this harness cannot see into is not one it may fell.
_PID = re.compile(r"pid=(\d+)")


def parse_port_listener(lines: Sequence[str], port: int) -> int | None:
    """Return the pid listening on a port, or None when nothing is.

    Args:
        lines: Output of :data:`PORT_LISTENER_COMMAND`, one row per line.
        port: The port whose holder is wanted.

    Returns:
        The owning process id, or None when the port is free -- or when the
        holder belongs to another user and is therefore invisible. Both read
        the same here deliberately: a port this harness cannot see into is not
        one it may fell.
    """
    suffix = f":{port}"
    for line in lines:
        parts = line.split()
        if (
            len(parts) > _LOCAL_ADDRESS_COLUMN
            and parts[0] == _LISTENING
            and parts[_LOCAL_ADDRESS_COLUMN].endswith(suffix)
        ):
            found = _PID.search(line)
            if found is not None:
                return int(found.group(1))
    return None


def process_name_command(pid: int) -> tuple[str, ...]:
    """Return the command that names one process.

    Args:
        pid: The process to name.

    Returns:
        The argument vector, program first.

    Raises:
        ValueError: When the pid is not positive, for the same reason
            :func:`fell_command` refuses one.
    """
    if pid <= 0:
        raise ValueError(f"a process is named by a positive pid, got {pid}")
    return ("ps", "-p", str(pid), "-o", "comm=")


def parse_process_name(lines: Sequence[str]) -> str | None:
    """Return the process name one naming command reported.

    Args:
        lines: Output of :func:`process_name_command`.

    Returns:
        The executable's name, or None when the process is gone. ``ps``
        reports absence by printing nothing, so absence is read out of the
        text rather than out of a status.
    """
    for line in lines:
        text = line.strip()
        if text:
            return text
    return None


#: What the bundled runtime's process is called. The only thing that may
#: legitimately hold a match port.
ENGINE_PROCESS_NAME = "java"


def holder_is_an_orphaned_engine(name: str | None) -> bool:
    """Report whether the process holding a match port may be felled.

    A zombie engine from a killed worker still holds its channel port, and
    the next match dies at the bind rather than this one (vhdoom96b,
    2026-08-09). Only a JVM can legitimately hold a match port, so a JVM
    holder is always an orphan -- and anything else is left alone so the bind
    fails loudly instead of this felling a bystander.

    Args:
        name: What :func:`parse_process_name` reported, or None.

    Returns:
        True only when the holder is :data:`ENGINE_PROCESS_NAME`.
    """
    return name == ENGINE_PROCESS_NAME


__all__ = [
    "ENGINE_PROCESS_NAME",
    "PORT_LISTENER_COMMAND",
    "fell_command",
    "holder_is_an_orphaned_engine",
    "parse_port_listener",
    "parse_process_name",
    "process_name_command",
]
