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
holds. The same on both platforms and every filesystem the hub has: no
``fcntl``, no ``msvcrt``, nothing a test cannot make the filesystem do.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import uuid
from collections.abc import Generator
from typing import TypedDict

from platform_core.logging import get_logger

from fleet.core import _test_hooks

_log = get_logger(__name__)


class TurnBounds(TypedDict):
    """What one kind of turn is called and how long it may be held.

    Attributes:
        what: The turn's name in a log line, e.g. ``claim turn``.
        waited_for: Who a wait was for, e.g. ``its host's other runner``.
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
    token = f"{holder} pid {os.getpid()} turn {uuid.uuid4().hex}"
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


__all__ = ["TurnBounds", "held"]
