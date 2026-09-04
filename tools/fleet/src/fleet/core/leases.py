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
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, load_json_str

from fleet.contracts.lease import (
    Lease,
    claims,
    decode_lease,
    describe_lease,
    encode_lease,
    is_expired,
)
from fleet.core import _test_hooks


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
    if not path.is_file():
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


def acquire(
    path: pathlib.Path,
    lease: Lease,
    *,
    now_unix: int,
) -> None:
    """Take the claim on one project's environment, or refuse.

    Expired leases are dropped as a side effect of writing, which is the only
    place they are ever removed. That is deliberate: the read path must not
    write, or two concurrent readers would race to rewrite the file and a
    reader would need a lock it has no reason to hold.

    Args:
        path: The lease file.
        lease: The claim being taken.
        now_unix: Current time, whole seconds since the epoch.

    Raises:
        AppError: With ``LEASE_HELD`` when another dispatch holds this node
            and project, naming the holder and how long is left. Refused
            rather than queued: a caller that wanted to wait can wait on the
            message, and a queue inside a command-line tool is a background
            process by another name.
    """
    holder = find_holder(path, node=lease["node"], project=lease["project"], now_unix=now_unix)
    if holder is not None:
        raise AppError(
            FleetErrorCode.LEASE_HELD,
            f"cannot dispatch: {describe_lease(holder, now_unix=now_unix)}",
        )
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
    surviving = held_leases(path, now_unix=now_unix)
    remaining = tuple(entry for entry in surviving if entry["run_id"] != run_id)
    if len(remaining) == len(surviving):
        raise AppError(
            FleetErrorCode.LEASE_NOT_HELD,
            f"run {run_id} holds no live lease in {path}; it was either released already, or "
            "it expired while the run was still going and another dispatch may now be inside "
            "the same environment",
        )
    _test_hooks.write_text(path, dump_json_str([encode_lease(entry) for entry in remaining]))


__all__ = ["acquire", "find_holder", "held_leases", "read_leases", "release"]
