"""Acquiring and releasing the claim on a project's environment.

THE WHOLE FILE IS THE FIX for the incident in
:mod:`fleet.contracts.lease`: ``poetry sync`` reinstalling a package under a
live interpreter, because two sessions were in one project with no way to know
about each other.

WHY THE FILE IS REWRITTEN RATHER THAN APPENDED TO. The ledger and the feed are
append-only because they are history and history does not change. Leases are
LIVE STATE -- a release must make a claim stop existing, and an append-only
log of "taken" and "released" records would make every reader replay the whole
file to answer one question, and would grow without bound for a set that is
never larger than the number of nodes times projects.

WHY EXPIRY IS CHECKED ON READ AND NOT SWEPT. There is no daemon here and
deliberately so: a package whose correctness depends on a background process
having run is a package that is wrong whenever that process is not running.
An expired lease is simply not returned by :func:`held_leases`, so the
resource frees itself at the moment anybody looks.

EVERY READ AND EVERY REWRITE TAKES THE FILE'S TURN (:func:`_turn`, MCPs
board task b1089284). Every node runner is its own process on the hub and
they all share this one file, and it is rewritten in place: on 2026-10-11 at
17:00:10Z serendipity's runner read it while another process was writing it,
got half a JSON document, and its serving loop ended on ``InvalidJsonError``
(its runner log on the hub). Reading without the turn was a deliberate
choice while one process used the file; with eight, a reader has the reason
to hold one that the choice said it lacked, and a rewrite from what it read
needs one for the same lost-update reason :data:`REWRITING` gives in-process.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import threading
from collections.abc import Generator
from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, load_json_str

from fleet.contracts.lease import (
    Lease,
    claims,
    contends,
    decode_lease,
    describe_contention,
    describe_lease,
    encode_lease,
    is_expired,
)
from fleet.core import _test_hooks, file_turn

#: The lease file's turn as :mod:`fleet.core.file_turn` holds it. Inside it a
#: process reads or rewrites one small local file and nothing else, so a lock
#: half a minute old was left by a process that died holding it.
LEASE_TURN: Final = file_turn.TurnBounds(
    what="lease turn",
    waited_for="another process reading or rewriting the lease file",
    stale_seconds=30,
    poll_seconds=1,
)

#: The lease files this process holds the turn of, guarded by
#: :data:`REWRITING`, so a call inside another's turn does not wait on itself.
_HOLDING: Final[set[pathlib.Path]] = set()

#: Held while the file is read and rewritten. A node runner settles a run on
#: its watch thread while its fill pass takes a lease on the main thread
#: (:mod:`fleet.cli.node_watch`), and two rewrites of one file in one
#: process, each from what it read before the other wrote, would lose one:
#: a release lost brings a finished run's lease back to hold its project.
#: Reentrant because :func:`release_if_held` reads before it releases.
REWRITING: Final = threading.RLock()


def lock_path(path: pathlib.Path) -> pathlib.Path:
    """Where a lease file's turn is held.

    Args:
        path: The lease file.

    Returns:
        The lock file beside it.
    """
    return path.with_name(f"{path.name}.lock")


@contextlib.contextmanager
def _turn(path: pathlib.Path) -> Generator[None, None, None]:
    """Hold the lease file's turn, across processes, for the ``with`` block.

    Reentrant within the process: :data:`REWRITING` is held for the whole
    block, so a nested call is this thread's own and finds its file in
    :data:`_HOLDING` instead of waiting on the lock it created.

    Args:
        path: The lease file.

    Yields:
        Nothing; the turn is held while the block runs.
    """
    with REWRITING:
        if path in _HOLDING:
            yield
            return
        with file_turn.held(
            lock_path(path), holder=f"fleet process {os.getpid()}", bounds=LEASE_TURN
        ):
            _HOLDING.add(path)
            try:
                yield
            finally:
                _HOLDING.discard(path)


def read_leases(path: pathlib.Path) -> tuple[Lease, ...]:
    """Read every lease the file records, expired ones included.

    Args:
        path: The lease file.

    Returns:
        Every recorded lease, in file order. An absent file is an empty set
        rather than an error: no dispatch has ever run in this workspace, and
        refusing the first one because it has no history would make the
        package impossible to start using.

    Raises:
        JSONTypeError: If the file's content is not a JSON list of valid
            leases. Refused rather than skipped -- a lease file that cannot
            be read is a file whose claims cannot be honoured, and treating
            it as empty would hand out a resource somebody holds.
    """
    with _turn(path):
        if not _test_hooks.file_exists(path):
            return ()
        raw = load_json_str(_test_hooks.read_text(path))
    if not isinstance(raw, list):
        raise AppError(
            FleetErrorCode.LEASE_NOT_HELD,
            f"{path} must hold a JSON list of leases, got {type(raw).__name__}; refusing to "
            "read it as empty, which would hand out a resource somebody holds",
        )
    return tuple(decode_lease(entry) for entry in raw)


def held_leases(path: pathlib.Path, *, now_unix: int) -> tuple[Lease, ...]:
    """Read only the leases that are still in force.

    Args:
        path: The lease file.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        The unexpired leases. Expiry is applied here rather than by a sweep,
        so a resource frees itself the moment anybody looks -- see the module
        docstring.
    """
    return tuple(lease for lease in read_leases(path) if not is_expired(lease, now_unix=now_unix))


def find_holder(path: pathlib.Path, *, node: str, project: str, now_unix: int) -> Lease | None:
    """Find the lease standing between a caller and one project on one node.

    Args:
        path: The lease file.
        node: The node's workspace name.
        project: Repo-relative project path.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        The holding lease, or None when the pair is free.
    """
    for lease in held_leases(path, now_unix=now_unix):
        if claims(lease, node=node, project=project):
            return lease
    return None


def find_by_run(path: pathlib.Path, *, run_id: str, now_unix: int) -> Lease | None:
    """Find the live lease one dispatch holds, if it still holds one.

    Distinct from :func:`find_holder`, which asks what stands in the way of a
    NEW dispatch. This asks whether a KNOWN one still holds its claim, and the
    answer None is meaningful rather than an error: a run whose lease lapsed
    while it was still going is the wedge case, and the caller that needs to
    know is the one trying to stop it.

    Args:
        path: The lease file.
        run_id: The dispatch.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        Its lease, or None when it holds none.
    """
    for lease in held_leases(path, now_unix=now_unix):
        if lease["run_id"] == run_id:
            return lease
    return None


def contended_by(
    path: pathlib.Path, *, wanted: tuple[str, ...], now_unix: int
) -> tuple[Lease, tuple[str, ...]] | None:
    """Find the lease holding a fleet-wide resource a dispatch needs.

    ONE FUNCTION WITH TWO CALLERS THAT REACT DIFFERENTLY, deliberately.
    :func:`acquire` calls it and refuses; ``fleet-preflight`` and
    ``fleet-run`` call it BEFORE probing any node and report. Both have to
    answer the same question, and a second implementation of "is this
    contended" is how the advisory answer and the authoritative one come to
    disagree -- with the advisory one being the one people read.

    The early call is not the enforcement. It exists because a fleet-wide
    resource makes every node refuse for the same reason, so probing three
    nodes to collect three identical refusals wastes the round trips and
    produces a message that reads as a capacity problem.

    Args:
        path: The lease file.
        wanted: The resources a dispatch is asking for. Empty asks nothing
            and can never be contended, which is the path every
            self-contained project takes.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        The holding lease and the names it denies, or None when nothing is in
        the way.
    """
    if not wanted:
        return None
    for lease in held_leases(path, now_unix=now_unix):
        names = contends(lease, wanted=wanted)
        if names:
            return lease, names
    return None


def acquire(
    path: pathlib.Path,
    lease: Lease,
    *,
    now_unix: int,
) -> None:
    """Take the claim on one project's environment, or refuse.

    Expired leases are dropped as a side effect of writing, which is the only
    place they are ever removed. That is deliberate: the read path must not
    write, or two concurrent readers would race to rewrite the file. The
    read, the checks and the write are one turn of the file (:func:`_turn`).

    Args:
        path: The lease file.
        lease: The claim being taken.
        now_unix: Current time, whole seconds since the epoch.

    Raises:
        AppError: With ``LEASE_HELD`` when another dispatch holds this node
            and project, naming the holder and how long is left, or with
            ``RESOURCE_HELD`` when one holds a fleet-wide resource this
            dispatch needs. TWO CODES because the fixes differ and a reader
            acts on them differently: an environment is per node, so the
            answer is another node; a fleet-wide resource has no second copy,
            so the answer is to wait. One code would send half its readers
            hunting for capacity that would not have helped.

            Refused rather than queued: a caller that wanted to wait can wait
            on the message, and a queue inside a command-line tool is a
            background process by another name.
    """
    with _turn(path):
        node, project = lease["node"], lease["project"]
        holder = find_holder(path, node=node, project=project, now_unix=now_unix)
        if holder is not None:
            raise AppError(
                FleetErrorCode.LEASE_HELD,
                f"cannot dispatch: {describe_lease(holder, now_unix=now_unix)}",
            )
        contention = contended_by(path, wanted=lease["resources"], now_unix=now_unix)
        if contention is not None:
            blocking, names = contention
            described = describe_contention(blocking, names=names, now_unix=now_unix)
            raise AppError(FleetErrorCode.RESOURCE_HELD, f"cannot dispatch: {described}")
        surviving = held_leases(path, now_unix=now_unix)
        _test_hooks.write_text(
            path,
            dump_json_str([encode_lease(entry) for entry in (*surviving, lease)]),
        )


def release(path: pathlib.Path, *, run_id: str, now_unix: int) -> None:
    """Give up the claim a dispatch holds.

    Args:
        path: The lease file.
        run_id: The dispatch releasing its claim.
        now_unix: Current time, whole seconds since the epoch.

    Raises:
        AppError: With ``LEASE_NOT_HELD`` when no live lease names this run.
            Refused rather than ignored, because the two ways to reach it are
            both faults worth seeing: releasing twice means the caller lost
            track of its own dispatch, and releasing a lease that already
            expired means the run outlived the window it declared and another
            dispatch may already be inside the environment.
    """
    with _turn(path):
        surviving = held_leases(path, now_unix=now_unix)
        remaining = tuple(entry for entry in surviving if entry["run_id"] != run_id)
        if len(remaining) == len(surviving):
            raise AppError(
                FleetErrorCode.LEASE_NOT_HELD,
                f"run {run_id} holds no live lease in {path}; it was either released already, "
                "or it expired while the run was still going and another dispatch may now be "
                "inside the same environment",
            )
        _test_hooks.write_text(path, dump_json_str([encode_lease(entry) for entry in remaining]))


def release_if_held(path: pathlib.Path, *, run_id: str, now_unix: int) -> str:
    """Give a claim back, tolerating one that has already lapsed.

    THE DIFFERENCE FROM :func:`release` IS WHO IS ASKING. ``release`` is for a
    caller that believes it still holds the claim, and a lapsed lease there is
    a fault worth raising. This is for the callers that close a run OUT --
    ``fleet-cancel`` and ``fleet-collect`` -- and for them a lapsed lease is
    ordinary: a run cancelled after its window, or a result collected an hour
    after the build ended, has nothing to give back and refusing would leave
    the only tools that can close it unable to.

    The absence is reported rather than swallowed, so the caller can say so.

    Args:
        path: The lease file.
        run_id: The dispatch.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        What happened, as a sentence for a log line.
    """
    with _turn(path):
        if find_by_run(path, run_id=run_id, now_unix=now_unix) is None:
            return "its lease had already expired, so nothing was held to give back"
        release(path, run_id=run_id, now_unix=now_unix)
        return "lease released"


__all__ = [
    "LEASE_TURN",
    "REWRITING",
    "acquire",
    "contended_by",
    "find_by_run",
    "find_holder",
    "held_leases",
    "lock_path",
    "read_leases",
    "release",
    "release_if_held",
]
