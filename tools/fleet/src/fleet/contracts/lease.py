"""One project's environment, mutated by one dispatch at a time.

THE INCIDENT THIS FILE IS. On 2026-09-04 a GPU measurement run on ``austinpc``
died at 01:32:33 with exit -1, no traceback, no out-of-memory event and 25 GB
of commit still free. Another session had started ``make check`` in the same
project. That recipe runs ``poetry lock; poetry sync --with dev``, and
``poetry sync`` UNINSTALLS AND REINSTALLS -- it rewrote
``model_trainer_server`` into the shared ``.venv`` at 01:32:55, deleting
``site-packages/model_trainer/**`` out from under a live interpreter.

Two properties of that made it invisible until it was measured. The recipe
runs the sync TWICE, once in ``lint`` and once in ``test``, so one ``make
check`` opens the window twice. And 40 of the monorepo's 48 Makefiles do it,
against one ``.venv`` per PROJECT rather than per session, so any two sessions
in one project collide -- not only two running tests.

WHAT WAS BLAMED FIRST AND WAS INNOCENT. ``scripts/reap-test-processes.ps1``
was accused, on timing alone. It could not have done it: its process filter
requires a command line matching ``*pytest*`` or ``*exec(eval*`` and a
benchmark's matches neither, its sweep only considers processes older than
sixty minutes, and an aggregate CPU-idle gate aborts the whole sweep if any
candidate is burning CPU. Recorded here because the wrong diagnosis is the
more expensive mistake: it points at a component that is working.

WHY A LEASE AND NOT A MUTEX AROUND THE SYNC. A mutex would serialise the
mutation and let the readers keep reading, which is precisely wrong -- the
reader is the thing that dies. What has to be excluded is a mutation while
ANY dispatch is using that project's environment, which is a lease on the
pair, held for the run rather than for the sync.

WHY IT EXCLUDES ON (node, project) AND NOT ON THE NODE. Two projects on one
node have two ``.venv`` directories and cannot corrupt each other; serialising
them would give up most of what a 20-core node is for. The shared mutable
thing is the project's environment, so that is the pair :func:`claims` tests.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from fleet.contracts.resources import contended, decode_resources, encode_resources


class Lease(TypedDict):
    """A claim on one project's environment on one node.

    Attributes:
        node: The node's SSH alias.
        project: The project whose environment is claimed, by the repo-
            relative path a Makefile lives at, e.g.
            ``services/Model-Trainer``. A path rather than a bare name
            because two directories may share a leaf name and the venv is per
            directory.
        run_id: The dispatch holding it.
        agent: The label of the session that dispatched, in the board's own
            kebab-case ``<model>-<topic>-<MMDD>`` form. Carried so a refusal
            can name who to talk to rather than only a pid, which is the
            question a blocked session actually has.
        session_id: That session's UUID, for the same reason the board binds
            one: an agent label can be reused across sessions and a UUID
            cannot.
        acquired_unix: When the lease was taken, in whole seconds since the
            epoch.
        expires_unix: When it lapses. REQUIRED, with no "hold forever"
            spelling, because the failure mode being designed against is a
            wedged run -- and a wedge that holds an unexpiring lease converts
            one stuck suite into a project nobody can ever build again. A
            lease that outlives its holder is the thing an expiry is for.
        resources: Fleet-wide exclusive resources this dispatch also holds,
            from its project's declaration. Carried in the SAME record as the
            environment claim rather than in a second kind of lease, so there
            is one thing to expire, one thing to release, and no way for the
            two to disagree about whether a run is still going. Empty for
            every project whose suite is self-contained, which is most of
            them. See :mod:`fleet.contracts.resources`.
    """

    node: str
    project: str
    run_id: str
    agent: str
    session_id: str
    acquired_unix: int
    expires_unix: int
    resources: tuple[str, ...]


def claims(lease: Lease, *, node: str, project: str) -> bool:
    """Whether this lease is the one standing in the way of that pair.

    A field comparison rather than a formatted key, and the difference is not
    style. A ``f"{node}::{project}"`` key was written here first, with a
    docstring asserting that two pairs could not collide because an SSH alias
    holds no colons and a path uses forward slashes. That is an assumption
    about inputs, not a property of the function, and its own test disproved
    it: ``("a::b", "c")`` and ``("a", "b::c")`` both render ``a::b::c``.
    Comparing the two fields makes a collision unrepresentable rather than
    unlikely, and needs no rule about what a node may be called.

    Args:
        lease: The lease to test.
        node: The node's workspace name.
        project: Repo-relative project path.

    Returns:
        True when the lease holds exactly that node and project.
    """
    return lease["node"] == node and lease["project"] == project


def contends(lease: Lease, *, wanted: tuple[str, ...]) -> tuple[str, ...]:
    """Name the exclusive resources this lease would deny a new dispatch.

    Distinct from :func:`claims`, and the distinction is the whole point of
    the resource lease. ``claims`` asks about ONE NODE's copy of a project's
    environment, and the answer to a refusal is "use another node". This asks
    about something there is exactly one of in the fleet, and the answer is
    "no node will help" -- so the two cannot share a code path or a message
    without one of them being misleading.

    Args:
        lease: The lease that may be in the way.
        wanted: The resources a new dispatch is asking for.

    Returns:
        The contended names, empty when there is no overlap.
    """
    return contended(lease["resources"], wanted)


def describe_contention(lease: Lease, *, names: tuple[str, ...], now_unix: int) -> str:
    """Render a resource refusal for the session that just hit it.

    A separate rendering from :func:`describe_lease` because a reader who is
    told only "held by X" will go looking for another node, and for a
    fleet-wide resource that search cannot succeed. The line says so.

    Args:
        lease: The lease holding the resources.
        names: The contended names.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        One line naming the resources, the holder, and the fact that no other
        node is an escape.
    """
    remaining = lease["expires_unix"] - now_unix
    window = f"{remaining}s remaining" if remaining > 0 else f"expired {-remaining}s ago"
    return (
        f"{', '.join(names)} is held fleet-wide by {lease['agent']} "
        f"(run {lease['run_id']}, {lease['project']} on {lease['node']}, session "
        f"{lease['session_id']}), {window}; there is one of it in the fleet, so no other "
        "node is an alternative"
    )


def is_expired(lease: Lease, *, now_unix: int) -> bool:
    """Whether a lease has lapsed and its resource is free again.

    Time is passed in rather than read, because a contract that read the
    clock could not be tested without controlling one, and the monorepo
    routes that through a hook rather than through this layer.

    Args:
        lease: The lease to judge.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        True once ``now_unix`` has reached the expiry. Reaching it counts as
        expired rather than not-yet: the boundary has to fall one way, and
        the direction that frees a resource is the one that cannot deadlock.
    """
    return now_unix >= lease["expires_unix"]


def describe_lease(lease: Lease, *, now_unix: int) -> str:
    """Render a lease for the session that just failed to take it.

    Args:
        lease: The lease that is in the way.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        One line naming the holder and how long is left, because "who is
        holding this and for how long" is the entire question a refused
        caller has.
    """
    remaining = lease["expires_unix"] - now_unix
    window = f"{remaining}s remaining" if remaining > 0 else f"expired {-remaining}s ago"
    return (
        f"{lease['project']} on {lease['node']} is held by {lease['agent']} "
        f"(run {lease['run_id']}, session {lease['session_id']}), {window}"
    )


def encode_lease(lease: Lease) -> JSONObject:
    """Encode a lease.

    Args:
        lease: The lease to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "node": lease["node"],
        "project": lease["project"],
        "run_id": lease["run_id"],
        "agent": lease["agent"],
        "session_id": lease["session_id"],
        "acquired_unix": lease["acquired_unix"],
        "expires_unix": lease["expires_unix"],
        "resources": encode_resources(lease["resources"]),
    }


def decode_lease(value: JSONValue) -> Lease:
    """Decode and validate a lease.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated lease.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the expiry does not follow the acquisition. A lease
            that expires before it was taken is already expired on arrival,
            so every reader would treat the resource as free while the holder
            believed otherwise -- which is the corruption this type prevents,
            reintroduced through its own record.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"lease must be a JSON object, got {type(value).__name__}")
    acquired_unix = require_int(value, "acquired_unix")
    expires_unix = require_int(value, "expires_unix")
    if expires_unix <= acquired_unix:
        raise JSONTypeError(
            f"lease expires at {expires_unix} but was acquired at {acquired_unix}; a lease "
            "that expires before it is taken reads as free to every other session while its "
            "holder believes it is held"
        )
    return Lease(
        node=require_str(value, "node"),
        project=require_str(value, "project"),
        run_id=require_str(value, "run_id"),
        agent=require_str(value, "agent"),
        session_id=require_str(value, "session_id"),
        acquired_unix=acquired_unix,
        expires_unix=expires_unix,
        resources=decode_resources(value.get("resources"), field="lease.resources"),
    )


__all__ = [
    "Lease",
    "claims",
    "contends",
    "decode_lease",
    "describe_contention",
    "describe_lease",
    "encode_lease",
    "is_expired",
]
