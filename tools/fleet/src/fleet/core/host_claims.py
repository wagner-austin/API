"""A host's claims under way, and the turn its runners take to claim.

MCPs board task a85ef09e. A node's ordinary and elevated runners are two
processes on the hub (:mod:`fleet.cli.node_agent`), and each charged its
launches under way from its own memory only
(:mod:`fleet.cli.node_launch`), so for the 40 to 80 s between a claim and
its run reaching the ledger the other runner read the host as emptier than
it was. On 2026-10-07 serendipity held three runs and 6 workers against a
pool of 4 cores that way, twice in one evening, and fleet job c858b46b ran
321 s of its 300 s budget. Two things now make the host the unit:

THE CLAIMS ARE THE HOST'S. Each claimed job's grant is written to
``<records>/host-claims/<node>/<job>.json``
(:class:`fleet.contracts.host_claim.HostClaim`) when its launch is handed to
the pool, and the file is removed once the launch has ended, by which time
a launched run is on the ledger. Every runner of the host charges every
claim written there, whichever runner wrote it. One file per claim, so
ending a launch is one removal that needs no turn: the launch ends on a
thread of the pool while its runner may be holding the turn for its next
claim. A claim stops counting at its ``until_unix``, the queue job's own
claim lease, so a runner that died mid-launch frees its host when the
queue frees its job, and the turn that finds such a claim removes it.

THE CLAIM IS TAKEN IN TURN. Reading the host's load and claiming against it
are two steps, and two runners between them would both read the same free
room. So a claim pass runs inside :func:`claim_turn`, which holds
``<records>/host-claims/<node>.lock``: the file's exclusive creation is the
lock, its text names the turn so only its holder removes it, and a lock
older than :data:`TURN_STALE_SECONDS` was left by a runner that died inside
its turn and is taken away. Claims are read and written inside a turn.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import uuid
from collections.abc import Generator
from typing import Final

from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.logging import get_logger

from fleet.contracts.host_claim import HostClaim, decode_host_claim, encode_host_claim
from fleet.contracts.node import LiveLoad
from fleet.core import _test_hooks
from fleet.core.remote import SSH_TIMEOUT_SECONDS

_log = get_logger(__name__)

#: The directory beside the ledger that holds each host's claims and lock.
HOST_CLAIMS_DIRECTORY: Final = "host-claims"

#: How old a turn's lock is when its holder is taken to have died inside it.
#: A turn runs the capacity and toolchain probes, each a staged script and
#: its run over ssh at :data:`fleet.core.remote.SSH_TIMEOUT_SECONDS` apiece,
#: and the queue's claim beside them, so five of those deadlines bound the
#: longest turn a live runner can take.
TURN_STALE_SECONDS: Final = 5 * SSH_TIMEOUT_SECONDS

#: How long a runner waits between looks at a lock another turn holds.
TURN_POLL_SECONDS: Final = 1


def claims_directory(directory: pathlib.Path, *, alias: str) -> pathlib.Path:
    """Where a host's claims under way are recorded, one file each.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.

    Returns:
        The host's claims directory.
    """
    return directory / alias


def lock_path(directory: pathlib.Path, *, alias: str) -> pathlib.Path:
    """Where a host's claim turn is held.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.

    Returns:
        The lock file.
    """
    return directory / f"{alias}.lock"


@contextlib.contextmanager
def claim_turn(directory: pathlib.Path, *, alias: str, runner: str) -> Generator[None, None, None]:
    """Hold the host's claim turn for the span of the ``with`` block.

    Waits while another turn holds the lock, taking away a lock older than
    :data:`TURN_STALE_SECONDS`, and gives the lock back on leaving the block
    however it is left.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.
        runner: The label of the runner taking the turn, written into the
            lock so a reader of it can name the holder.

    Yields:
        Nothing; the turn is held while the block runs.

    Raises:
        OSError: When the lock can be neither created nor read for a reason
            other than another turn holding it.
    """
    lock = lock_path(directory, alias=alias)
    token = _take(lock, runner=runner)
    try:
        yield
    finally:
        _give_back(lock, token=token)


def _take(lock: pathlib.Path, *, runner: str) -> str:
    """Create the lock, waiting out or taking away the turn that holds it.

    Args:
        lock: The lock file.
        runner: The label of the runner taking the turn.

    Returns:
        The text written into the lock, which only this turn's release
        matches.
    """
    _test_hooks.make_directory(lock.parent)
    token = f"{runner} pid {os.getpid()} turn {uuid.uuid4().hex}"
    asked = _test_hooks.now()
    while not _created(lock, token=token):
        age = _age(lock)
        if age is not None and age >= TURN_STALE_SECONDS:
            _log.info(
                "%s took away a claim turn held for %d s, past the %d s a live runner's turn "
                "can last, so its holder died inside it",
                runner,
                age,
                TURN_STALE_SECONDS,
            )
            lock.unlink(missing_ok=True)
            continue
        _test_hooks.sleep(TURN_POLL_SECONDS)
    waited = _test_hooks.now() - asked
    if waited > 0:
        _log.info(
            "%s took its claim turn after waiting %d s for its host's other runner", runner, waited
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


def _give_back(lock: pathlib.Path, *, token: str) -> None:
    """Remove the lock if this turn still holds it.

    A turn that ran past :data:`TURN_STALE_SECONDS` may have been taken away
    and the lock created again by another, so the lock is removed only when
    its text is this turn's.

    Args:
        lock: The lock file.
        token: This turn's text.
    """
    held = _test_hooks.read_text(lock) if _test_hooks.file_exists(lock) else None
    if held != token:
        _log.info(
            "a claim turn outlived the %d s bound and was taken away; the lock now reads %r "
            "and is left to its holder",
            TURN_STALE_SECONDS,
            held,
        )
        return
    lock.unlink()


def live_claims(directory: pathlib.Path, *, alias: str) -> tuple[HostClaim, ...]:
    """The host's claims still under way, read inside a claim turn.

    A claim whose ``until_unix`` has passed is removed here, the one place
    a claim its launch never removed is found.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.

    Returns:
        Every recorded claim whose ``until_unix`` has not passed, ordered by
        job id. A host with no claims directory has never claimed.

    Raises:
        JSONTypeError: When a claim file does not hold a valid claim.
    """
    now = _test_hooks.now()
    live: list[HostClaim] = []
    for path in sorted(claims_directory(directory, alias=alias).glob("*.json")):
        text = _read_claim(path)
        if text is None:
            continue
        claim = decode_host_claim(load_json_str(text))
        if claim["until_unix"] <= now:
            _log.info(
                "%s: %s's claim of %s lapsed at %d unlaunched; removing it",
                alias,
                claim["runner"],
                claim["job_id"],
                claim["until_unix"],
            )
            path.unlink(missing_ok=True)
            continue
        live.append(claim)
    return tuple(live)


def _read_claim(path: pathlib.Path) -> str | None:
    """Read one claim file that its launch may be removing as it is read.

    Args:
        path: The claim file.

    Returns:
        Its text, or None when its launch removed it after it was listed:
        that launch has ended, and its run, if it ran, is on the ledger.
    """
    try:
        return _test_hooks.read_text(path)
    except FileNotFoundError:
        return None


def record(directory: pathlib.Path, *, alias: str, claim: HostClaim) -> None:
    """Write a claim for the host, inside the claim turn that made it.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.
        claim: The claimed job's grant.
    """
    _test_hooks.write_text(
        claims_directory(directory, alias=alias) / f"{claim['job_id']}.json",
        dump_json_str(encode_host_claim(claim)),
    )


def discharge(directory: pathlib.Path, *, alias: str, job_id: str) -> None:
    """Remove a job's claim once its launch has ended; no turn is needed.

    On Windows a file another process has open cannot be removed (measured
    on austinpc, Python 3.11.9, 2026-10-08: ``os.unlink`` raises
    ``PermissionError`` while a plain ``open`` of it is held), and the
    host's other runner opens each claim file for the moment it reads it, so
    a removal refused that way is asked again, a poll apart, for as long as a
    turn may last; a claim left behind would charge the host for an hour.

    Args:
        directory: The records' ``host-claims`` directory.
        alias: The node's workspace name.
        job_id: The queue job whose launch ended.

    Raises:
        PermissionError: When the file is still refused after
            :data:`TURN_STALE_SECONDS`, which no read takes.
    """
    path = claims_directory(directory, alias=alias) / f"{job_id}.json"
    asked = _test_hooks.now()
    while True:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            if _test_hooks.now() - asked >= TURN_STALE_SECONDS:
                raise
            _test_hooks.sleep(TURN_POLL_SECONDS)
            continue
        return


def host_load(claims: tuple[HostClaim, ...]) -> LiveLoad:
    """What a host's claims under way hold, counted as live runs are.

    Args:
        claims: The host's live claims.

    Returns:
        One run per claim, with their grants and memory summed.
    """
    return LiveLoad(
        runs=len(claims),
        workers=sum(claim["workers"] for claim in claims),
        ram_gb=sum(claim["ram_gb"] for claim in claims),
    )


__all__ = [
    "HOST_CLAIMS_DIRECTORY",
    "TURN_POLL_SECONDS",
    "TURN_STALE_SECONDS",
    "claim_turn",
    "claims_directory",
    "discharge",
    "host_load",
    "live_claims",
    "lock_path",
    "record",
]
