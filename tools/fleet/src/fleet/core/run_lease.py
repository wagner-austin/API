"""Taking a dispatch's lease, and giving it back when the dispatch never ran.

THE LEASE IS THE FIRST STEP OF A DISPATCH AND THE ONLY ONE THAT OUTLIVES A
FAILURE. :func:`fleet.core.dispatch.launch` stages, sends and starts the build
after the lease is held, and the ledger row that makes a run count against its
node is written only once the build has launched. So a dispatch that fails in
staging or launch leaves exactly one thing behind: its lease. Before this
module, nothing gave that lease back, so it stood for the project's whole
lease length (measured 2026-09-26, MCPs board task e12affc5): job 4d8b28a9 for
MCPs/packages/session-audit was refused on serendipity with ``NODE_UNREACHABLE``
while its payload was being sent, and the resubmission one minute later, job
7143a25e, was claimed by serendipity again and refused ``LEASE_HELD`` with
793 s of the dead run's lease remaining.

WHY :func:`take` AND :func:`abandon` ARE SEPARATE CALLS AND NOT A ``finally``.
Only a lease this dispatch acquired may be given back. A ``LEASE_HELD``
refusal out of :func:`take` means another dispatch holds the project, and
releasing the project's lease there would hand that run's environment to the
next claim. The caller that holds the :class:`~fleet.contracts.lease.Lease`
this module returned is the one place that knows it acquired it.

The feed gets the terminal ``refused`` event here too, so a subscriber that
read ``leased`` for the run reads its end rather than a run that never
reports again.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.feed import FeedEvent, FeedKind
from fleet.contracts.lease import Lease
from fleet.contracts.project import ProjectConfig, lease_seconds
from fleet.core import _test_hooks, leases, records

#: How much longer than its estimate a dispatch may hold its lease.
#:
#: Two rather than a tighter figure because the estimate comes from whichever
#: machine last ran the suite, and this fleet's nodes differ by more than a
#: factor of two in free memory -- so a run on the smallest node legitimately
#: takes far longer than one on the largest. A lease that expired underneath a
#: healthy run would hand its environment to a second dispatch, which is the
#: corruption the lease exists to prevent, reintroduced by its own timeout.
LEASE_SLACK = 2.0


def run_id_for(project: str, *, started_unix: int) -> str:
    """Name a dispatch.

    Derived rather than random, so the identifier a person reads names the
    thing it identifies. The project's slashes become hyphens because the id
    is used as a directory name on the node.

    Args:
        project: Repo-relative project path.
        started_unix: When the dispatch began.

    Returns:
        The run id.
    """
    return f"{project.replace('/', '-')}-{started_unix}"


def open_lease(
    *,
    node: str,
    project: str,
    run_id: str,
    agent: str,
    session_id: str,
    plan: ProjectConfig,
    now_unix: int,
) -> Lease:
    """Build the claim a dispatch will hold for its run.

    Args:
        node: The node's workspace name.
        project: Repo-relative project path.
        run_id: The dispatch.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.
        plan: The project, whose expected duration sizes the window.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        The lease, sized at :data:`LEASE_SLACK` times the estimate and
        carrying whatever fleet-wide resources the project declared. Read
        from the plan rather than passed separately, so a caller cannot
        dispatch a project while forgetting what it contends for.
    """
    return Lease(
        node=node,
        project=project,
        run_id=run_id,
        agent=agent,
        session_id=session_id,
        acquired_unix=now_unix,
        expires_unix=now_unix + lease_seconds(plan, slack=LEASE_SLACK),
        resources=plan["exclusive_resources"],
    )


def emit(
    feed_path: pathlib.Path,
    *,
    run_id: str,
    node: str,
    project: str,
    kind: FeedKind,
    detail: str,
    now_unix: int,
) -> None:
    """Append one event to the stream subscribers tail.

    It takes the three identifying strings rather than a
    :class:`~fleet.contracts.lease.Lease`, because a lease is not what an
    event is about: the terminal events are emitted when a run is CLOSED, at
    which point the caller may hold a ledger row and no lease at all. Naming
    the fields is what lets both callers use the one function.

    Args:
        feed_path: The feed file.
        run_id: The dispatch the event belongs to.
        node: Its node's workspace name.
        project: Repo-relative project path.
        kind: What happened. Typed as the Literal rather than a string, so a
            kind that does not exist is a type error here rather than a
            decode failure in whoever reads the feed next.
        detail: Human-readable specifics.
        now_unix: Current time, whole seconds since the epoch.
    """
    records.append_feed(
        feed_path,
        FeedEvent(
            at_unix=now_unix,
            run_id=run_id,
            node=node,
            project=project,
            kind=kind,
            detail=detail,
        ),
    )


def take(
    loaded_leases: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    node_name: str,
    project: str,
    plan: ProjectConfig,
    workers: int,
    agent: str,
    session_id: str,
) -> Lease:
    """Name the dispatch, acquire its lease, and announce it on the feed.

    Args:
        loaded_leases: The lease file.
        loaded_feed: The feed file.
        node_name: The node's workspace name.
        project: Repo-relative project path.
        plan: The project's declaration.
        workers: Test workers the capacity check granted, for the feed line.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.

    Returns:
        The lease now held, which names the run.

    Raises:
        AppError: With ``LEASE_HELD`` when another dispatch holds this
            project on this node, or one of its declared resources anywhere.
            Nothing was acquired then, so there is nothing to abandon.
    """
    now_unix = _test_hooks.now()
    lease = open_lease(
        node=node_name,
        project=project,
        run_id=run_id_for(project, started_unix=now_unix),
        agent=agent,
        session_id=session_id,
        plan=plan,
        now_unix=now_unix,
    )
    leases.acquire(loaded_leases, lease, now_unix=now_unix)
    emit(
        loaded_feed,
        run_id=lease["run_id"],
        node=node_name,
        project=project,
        kind="leased",
        detail=f"{workers} worker(s)",
        now_unix=now_unix,
    )
    return lease


def abandon(
    loaded_leases: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    lease: Lease,
    detail: str,
) -> str:
    """End a dispatch that took its lease and never launched.

    The feed line comes first and the lease goes last, the order
    :func:`fleet.core.dispatch.finish` uses and for its reason: a failure
    between the two leaves the lease held, which expires on its own, rather
    than a free environment with no record of why.

    There is no ledger row to close. The ledger records a run once its build
    has launched (:func:`fleet.core.dispatch.launch`), so a dispatch that
    failed before that never counted against its node.

    Args:
        loaded_leases: The lease file.
        loaded_feed: The feed file.
        lease: The lease :func:`take` returned for this dispatch.
        detail: The ``CODE: message`` of the failure that ended it.

    Returns:
        What became of the lease, as a sentence for a log line. It has lapsed
        only when staging outlasted the whole lease window.
    """
    now_unix = _test_hooks.now()
    emit(
        loaded_feed,
        run_id=lease["run_id"],
        node=lease["node"],
        project=lease["project"],
        kind="refused",
        detail=detail,
        now_unix=now_unix,
    )
    return leases.release_if_held(loaded_leases, run_id=lease["run_id"], now_unix=now_unix)


__all__ = ["LEASE_SLACK", "abandon", "emit", "open_lease", "run_id_for", "take"]
