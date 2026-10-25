"""A turn held across processes by creating one lock file exclusively.

Lifted out of :mod:`fleet.core.host_claims` (MCPs board task a85ef09e), where
it was written for a host's claim turn, when the lease file needed the same
thing (MCPs board task b1089284): every node runner is its own process on the
hub, and a file several of them read and rewrite needs a turn none of
``threading``'s locks can give.

THE LOCK IS THE FILE'S EXCLUSIVE CREATION. Its text names the turn, so only
the turn that created it removes it; a lock older than the turn's
``stale_seconds`` was left by a process that died inside its turn and is
taken away; and a turn waits ``poll_seconds`` between looks at a lock another
holds. The same on every filesystem the hub has: no ``fcntl``, nothing a
test cannot make the filesystem do.

A LOCK WHOSE HOLDER IS GONE IS TAKEN AWAY BEFORE ITS AGE BOUND (MCPs board
task 86f70457). A mirror's fetch turn may last 1020 s, so a runner killed
inside one held every export of that mirror for 17 minutes. The text names
the holder's pid and host, and a turn waiting on a lock held past
:data:`HOLDER_PROBE_SECONDS` asks this host whether that pid still runs
(:func:`runs`); one this host wrote whose process is gone is taken at once.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import uuid
from collections.abc import Generator
from typing import Final, TypedDict

from platform_core.logging import get_logger

from fleet.core import _test_hooks

_log = get_logger(__name__)

#: How long a lock is held before a waiting turn asks whether its holder
#: still runs. A lease, claim or record turn is done well inside this, so
#: only a long turn, a mirror's fetch, is ever probed.
HOLDER_PROBE_SECONDS: Final = 30

#: The deadline of one probe, in seconds.
PROBE_TIMEOUT_SECONDS: Final = 15

#: The command that lists one process: it prints the pid when it runs and
#: does not when it does not.
_PROBE: Final = ("ps", "-p", "{pid}", "-o", "pid=")

#: What a lock this module wrote says of its holder: the pid and the host.
_HELD_BY: Final = re.compile(r" pid (?P<pid>\d+) on (?P<host>\S+) turn [0-9a-f]{32}$")


class TurnBounds(TypedDict):
    """What one kind of turn is called and how long it may be held.

    Attributes:
        what: The turn's name in a log line, e.g. ``claim turn``.
        waited_for: Who a wait was for, e.g. ``another process claiming for its host``.
        stale_seconds: How old a lock is when its holder is taken to have
            died inside its turn.
        poll_seconds: How long a turn waits between looks at a held lock.
    """

    what: str
    waited_for: str
    stale_seconds: int
    poll_seconds: int


@contextlib.contextmanager
def held(lock: pathlib.Path, *, holder: str, bounds: TurnBounds) -> Generator[None, None, None]:
    """Hold the turn for the span of the ``with`` block.

    Waits while another turn holds the lock, taking away a lock older than
    the bounds allow, and gives the lock back on leaving the block however it
    is left.

    Args:
        lock: The lock file.
        holder: Who takes the turn, written into the lock so a reader of it
            can name the holder.
        bounds: The turn's name and its time bounds.

    Yields:
        Nothing; the turn is held while the block runs.

    Raises:
        OSError: When the lock can be neither created nor read for a reason
            other than another turn holding it.
    """
    token = _take(lock, holder=holder, bounds=bounds)
    try:
        yield
    finally:
        _give_back(lock, token=token, bounds=bounds)


def _take(lock: pathlib.Path, *, holder: str, bounds: TurnBounds) -> str:
    """Create the lock, waiting out or taking away the turn that holds it.

    Args:
        lock: The lock file.
        holder: Who takes the turn.
        bounds: The turn's name and its time bounds.

    Returns:
        The text written into the lock, which only this turn's release
        matches.
    """
    _test_hooks.make_directory(lock.parent)
    token = f"{holder} pid {os.getpid()} on {_test_hooks.hostname()} turn {uuid.uuid4().hex}"
    asked = _test_hooks.now()
    while not _created(lock, token=token):
        age = _age(lock)
        if age is not None and age >= bounds["stale_seconds"]:
            _log.info(
                "%s took away a %s held for %d s, past the %d s a live runner's turn can last, "
                "so its holder died inside it",
                holder,
                bounds["what"],
                age,
                bounds["stale_seconds"],
            )
            lock.unlink(missing_ok=True)
            continue
        if age is not None and age >= HOLDER_PROBE_SECONDS and _holder_gone(lock, holder, bounds):
            continue
        _test_hooks.sleep(bounds["poll_seconds"])
    waited = _test_hooks.now() - asked
    if waited > 0:
        _log.info(
            "%s took its %s after waiting %d s for %s",
            holder,
            bounds["what"],
            waited,
            bounds["waited_for"],
        )
    return token


def _lock_text(lock: pathlib.Path) -> str | None:
    """What a lock says, read once.

    Args:
        lock: The lock file.

    Returns:
        Its text, or None when it was given back since it was found.
    """
    try:
        return lock.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def runs(pid: int) -> bool:
    """Whether a process with this pid runs on this machine.

    Args:
        pid: The process id.

    Returns:
        False only when the probe answered and did not list the pid; a
        probe that failed, timed out or wrote to stderr answers True, so a
        turn whose holder cannot be asked about waits out its age bound as
        it did before the probe existed.
    """
    argv = tuple(part.format(pid=pid) for part in _PROBE)
    probed = _test_hooks.run(argv, timeout_seconds=PROBE_TIMEOUT_SECONDS)
    if probed["returncode"] not in (0, 1) or probed["stderr"].strip():
        _log.info(
            "could not ask whether pid %d runs: %s exited %d: %s",
            pid,
            argv[0],
            probed["returncode"],
            probed["stderr"].strip()[:300],
        )
        return True
    return re.search(rf"(?<!\d){pid}(?!\d)", probed["stdout"]) is not None


def _holder_gone(lock: pathlib.Path, holder: str, bounds: TurnBounds) -> bool:
    """Take away a lock whose holder no longer runs on this machine.

    A turn killed inside its block never gives its lock back: on diphtheria
    on 2026-10-22 systemd stopped a node runner with SIGKILL at 06:18:01Z,
    20 s after it took a mirror's fetch turn, and every export of that
    mirror then waited out the turn's 1020 s bound (MCPs board task
    86f70457). The lock names its holder's pid and host, so a lock this host
    wrote whose process is gone is taken away at once. A lock another host
    wrote, or one written before the host was named, is left to the age
    bound, since this machine cannot see that process.

    Args:
        lock: The lock file.
        holder: Who is waiting for the turn.
        bounds: The turn's name and its time bounds.

    Returns:
        True when the lock was taken away.
    """
    text = _lock_text(lock)
    held = _HELD_BY.search(text) if text is not None else None
    if held is None:
        return False
    pid: str = held.group("pid")
    host: str = held.group("host")
    if host != _test_hooks.hostname() or runs(int(pid)) or _lock_text(lock) != text:
        return False
    _log.info(
        "%s took away a %s whose holder, pid %s on %s, no longer runs: %r",
        holder,
        bounds["what"],
        pid,
        host,
        text,
    )
    lock.unlink(missing_ok=True)
    return True


def _created(lock: pathlib.Path, *, token: str) -> bool:
    """Create the lock exclusively and write the turn's text into it.

    Args:
        lock: The lock file.
        token: The turn's text.

    Returns:
        True when this call created it; False when it already exists, which
        is another turn holding it.
    """
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token)
    return True


def _age(lock: pathlib.Path) -> int | None:
    """How long ago a held lock was created.

    Args:
        lock: The lock file.

    Returns:
        Its age in whole seconds, or None when it has been given back since
        the create that found it.
    """
    now = _test_hooks.now()
    try:
        modified = os.stat(lock).st_mtime
    except FileNotFoundError:
        return None
    return now - int(modified)


def _give_back(lock: pathlib.Path, *, token: str, bounds: TurnBounds) -> None:
    """Remove the lock if this turn still holds it.

    A turn that ran past its stale bound may have been taken away and the
    lock created again by another, so the lock is removed only when its text
    is this turn's.

    Args:
        lock: The lock file.
        token: This turn's text.
        bounds: The turn's name and its time bounds.
    """
    held_now = _test_hooks.read_text(lock) if _test_hooks.file_exists(lock) else None
    if held_now != token:
        _log.info(
            "a %s outlived the %d s bound and was taken away; the lock now reads %r and is "
            "left to its holder",
            bounds["what"],
            bounds["stale_seconds"],
            held_now,
        )
        return
    lock.unlink()


__all__ = ["HOLDER_PROBE_SECONDS", "PROBE_TIMEOUT_SECONDS", "TurnBounds", "held", "runs"]
