"""Spawning a match so it can be felled, and felling it so nothing survives.

A match is not one process. The launcher starts a JVM, the JVM starts nothing
but the planner runs beside it, and under the fleet the whole thing hangs off
a spawn this package made. Killing only the root leaves the engine holding its
channel port, and a zombie engine on a leased port kills the NEXT match at the
bind rather than this one (vhdoom96b, 2026-08-09). So felling has to reach the
whole tree, and the two platforms reach it in ways that have nothing in common.

WHY THIS IS COMPOSITION AND NOT A BRANCH AT THE CALL SITE. Everything here
returns data -- an argument vector, a pair of spawn flags -- and the impls in
:mod:`rw_bot.harness._test_hooks` apply it without a conditional of their own.
That is not style. Coverage here is 100% of branches with no ``pragma``
available, and a POSIX-only arm inside an implementation is unreachable on the
Windows box the suite runs on and unreachable on Linux for the Windows arm.
Expressed as data, both arms are pure functions of an argument and both are
exercised wherever the suite runs.

THE TWO MECHANISMS, and why each platform needs its own. Windows has no
process groups a child inherits by default, so ``taskkill /T`` walks the
parent/child table and fells the tree from the root down. POSIX has no such
table walk, so the tree is made fellable at SPAWN time instead: a new session
puts the child and its descendants in one process group whose id is the
child's pid, and signalling the negated pid reaches all of them. The spawn and
the kill are therefore one decision in two places, which is the whole reason
they live in one module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TypedDict

from rw_bot.harness.jvm import executable_name
from rw_bot.platform_id import is_windows

#: Windows priority class raising a match above ordinary desktop work.
#:
#: The value of ``subprocess.ABOVE_NORMAL_PRIORITY_CLASS``, written out rather
#: than imported: the name exists only in a Windows Python, so importing it
#: would make this module fail to load on the very platform the port is for.
#: A test pins it to the standard library's value wherever the suite runs on
#: Windows, so the literal cannot drift from the constant it stands in for.
ABOVE_NORMAL_PRIORITY_CLASS = 0x8000

#: What a non-Windows spawn passes for ``creationflags``. The standard library
#: raises on any other value off Windows, so this is the only legal answer
#: rather than a default.
NO_CREATION_FLAGS = 0


class SpawnIsolation(TypedDict):
    """How a spawn makes its child fellable, and how it ranks it.

    Attributes:
        creationflags: Windows creation flags. Carries the priority class on
            Windows and :data:`NO_CREATION_FLAGS` elsewhere, because the
            standard library rejects anything else off Windows.
        start_new_session: Whether to put the child in a new session, making
            it a process-group leader so the group can be signalled as one.
            POSIX only; the Windows implementation takes the argument and
            ignores it, which is why passing it costs nothing there.
    """

    creationflags: int
    start_new_session: bool


def spawn_isolation(platform: str) -> SpawnIsolation:
    """Return the spawn flags a match needs on a platform.

    On Windows the priority class is the load-bearing half. The planner and
    the engine are latency-sensitive tenants on a workstation that also runs
    a container VM and a virus scanner, and without the class a co-tenant
    spike deschedules the batch work rather than the sample stream (log
    2026-08-10). The child inherits it, so the whole match tree is covered.

    On POSIX the session is the load-bearing half, and there is deliberately
    no priority change. Raising priority there means a negative nice value,
    which needs privileges a batch job does not have; and the contention the
    Windows class exists to survive is a workstation condition, not one a
    scheduler-allocated core has.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        The flags, ready to pass straight to ``subprocess.Popen``.
    """
    if is_windows(platform):
        return SpawnIsolation(creationflags=ABOVE_NORMAL_PRIORITY_CLASS, start_new_session=False)
    return SpawnIsolation(creationflags=NO_CREATION_FLAGS, start_new_session=True)


def fell_command(pid: int, platform: str) -> tuple[str, ...]:
    """Return the command that kills a process and everything under it.

    Args:
        pid: Root process id of the tree. On POSIX this is also the process
            group id, because :func:`spawn_isolation` asked for a new session
            and that makes the child its own group leader.
        platform: A ``sys.platform`` value.

    Returns:
        The argument vector, program first.

    Raises:
        ValueError: When the pid is not positive. Negating a non-positive pid
            produces a POSIX signal target that means something else entirely
            -- ``0`` is "every process in my own group", which on POSIX would
            fell the harness itself, and ``-1`` is "every process I am
            permitted to signal". A bad pid must not be composed into either.
    """
    if pid <= 0:
        raise ValueError(
            f"a process tree is felled by its root pid, got {pid}: negating this "
            "would signal the caller's own group or every reachable process"
        )
    if is_windows(platform):
        # /T fells the tree, /F skips the polite close request an unattended
        # engine has nothing to answer with.
        return ("taskkill", "/PID", str(pid), "/T", "/F")
    # -s KILL is the portable spelling of the signal; -- stops option parsing
    # so the negative pid is read as a process group rather than a flag.
    return ("kill", "-s", "KILL", "--", f"-{pid}")


#: Windows netstat's word for a socket waiting to accept.
_WINDOWS_LISTENING = "LISTENING"

#: The same state as ``ss`` spells it.
_POSIX_LISTENING = "LISTEN"

#: How many whitespace-separated columns a Windows listening row has:
#: protocol, local address, foreign address, state, owning pid.
_WINDOWS_ROW_COLUMNS = 5

#: Where ``ss`` puts the local address once ``-H`` has removed the header:
#: state, recv queue, send queue, local, peer, users.
_POSIX_LOCAL_ADDRESS_COLUMN = 3

#: ``ss`` reports the holder as ``users:(("java",pid=1234,fd=3))``. Absent
#: entirely when the socket belongs to another user and this process is not
#: privileged, which reads here as "no holder found" -- correct, because a
#: port this harness cannot see into is not one it may fell.
_POSIX_PID = re.compile(r"pid=(\d+)")

#: What ``tasklist`` prints when nothing matches. It exits 0 either way, so
#: the text is the only signal that the pid is gone.
_WINDOWS_NO_SUCH_TASK = "INFO:"


def port_listener_command(platform: str) -> tuple[str, ...]:
    """Return the command that lists which process is listening where.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        The argument vector, program first. Both forms suppress headers so
        every output line is a row, and both are read by
        :func:`parse_port_listener`.
    """
    if is_windows(platform):
        return ("netstat", "-ano", "-p", "TCP")
    return ("ss", "-Hltnp")


def parse_port_listener(lines: Sequence[str], port: int, platform: str) -> int | None:
    """Return the pid listening on a port, or None when nothing is.

    Args:
        lines: Output of :func:`port_listener_command`, one row per line.
        port: The port whose holder is wanted.
        platform: A ``sys.platform`` value.

    Returns:
        The owning process id, or None when the port is free -- or, on POSIX,
        when the holder belongs to another user and is therefore invisible.
        Both read the same here deliberately: a port this harness cannot see
        into is not one it may fell.
    """
    suffix = f":{port}"
    for line in lines:
        parts = line.split()
        if is_windows(platform):
            if (
                len(parts) == _WINDOWS_ROW_COLUMNS
                and parts[3] == _WINDOWS_LISTENING
                and parts[1].endswith(suffix)
            ):
                return int(parts[-1])
            continue
        if (
            len(parts) > _POSIX_LOCAL_ADDRESS_COLUMN
            and parts[0] == _POSIX_LISTENING
            and parts[_POSIX_LOCAL_ADDRESS_COLUMN].endswith(suffix)
        ):
            found = _POSIX_PID.search(line)
            if found is not None:
                return int(found.group(1))
    return None


def process_name_command(pid: int, platform: str) -> tuple[str, ...]:
    """Return the command that names one process.

    Args:
        pid: The process to name.
        platform: A ``sys.platform`` value.

    Returns:
        The argument vector, program first.

    Raises:
        ValueError: When the pid is not positive, for the same reason
            :func:`fell_command` refuses one.
    """
    if pid <= 0:
        raise ValueError(f"a process is named by a positive pid, got {pid}")
    if is_windows(platform):
        return ("tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH")
    return ("ps", "-p", str(pid), "-o", "comm=")


def parse_process_name(lines: Sequence[str], platform: str) -> str | None:
    """Return the process name one naming command reported.

    Args:
        lines: Output of :func:`process_name_command`.
        platform: A ``sys.platform`` value.

    Returns:
        The executable's name, or None when the process is gone. Both tools
        report absence without failing usefully -- ``tasklist`` exits 0 and
        prints a notice, ``ps`` prints nothing -- so absence is read out of
        the text rather than out of a status.
    """
    for line in lines:
        text = line.strip()
        if not text or text.startswith(_WINDOWS_NO_SUCH_TASK):
            continue
        if is_windows(platform):
            return text.split(",")[0].strip('"')
        return text
    return None


def holder_is_an_orphaned_engine(name: str | None, platform: str) -> bool:
    """Report whether the process holding a match port may be felled.

    A zombie engine from a killed worker still holds its channel port, and
    the next match dies at the bind rather than this one (vhdoom96b,
    2026-08-09). Only a JVM can legitimately hold a match port, so a JVM
    holder is always an orphan -- and anything else is left alone so the bind
    fails loudly instead of this felling a bystander.

    Args:
        name: What :func:`parse_process_name` reported, or None.
        platform: A ``sys.platform`` value.

    Returns:
        True only when the holder is the platform's java executable.
    """
    return name == executable_name("java", platform)


__all__ = [
    "ABOVE_NORMAL_PRIORITY_CLASS",
    "NO_CREATION_FLAGS",
    "SpawnIsolation",
    "fell_command",
    "holder_is_an_orphaned_engine",
    "parse_port_listener",
    "parse_process_name",
    "port_listener_command",
    "process_name_command",
    "spawn_isolation",
]
