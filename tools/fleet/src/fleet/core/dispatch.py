"""Starting a run on a node, and closing it out afterwards.

THE ORDER IS THE DESIGN. Lease, then stage, then launch, then record. Each
step is the precondition of the next and each has a failure that must not
leave the one before it half-done:

* The lease comes FIRST, before anything is copied. Staging into a project
  another dispatch holds would be pointless work at best, and at worst two
  extractions into one tree.
* The ledger row is written when the run STARTS, not when it ends, because a
  capacity check subtracts live rows from a node's free memory. A ledger that
  only recorded finished work would let the next dispatch onto a node the
  first had already filled.

WHY THE SUITE IS DETACHED FROM THE CONNECTION AND NOT RUN AS AN SSH CHILD.
Windows OpenSSH assigns the session's process tree to a job object precisely
so the tree dies when the connection ends. A suite started as a child of the
dispatching ssh call therefore dies when that call returns -- which is
immediately, since the whole point is not to hold the connection for the
duration of a build. Measured on this fleet's own hardware and written up in
``memory/reference_long_runs_need_task_scheduler.md``: a ten-hour job launched
that way is leashed to a connection, and a process cannot be moved out of a
job object once it is in one. Linux sshd sends the session's process group
SIGHUP for the same effect, so the build is a transient systemd user unit
there.

The scripts that do the launching live in :mod:`fleet.core.dialect`, one set
per platform, split out of here by role when this file crossed the 600-line
ceiling. This module owns the ORDER; the dialect owns what the node is asked
to run, and :mod:`fleet.core.names` owns what each file is called.

NOTHING HERE CATCHES. A failure to stage or launch propagates with its own
code, and :func:`finish` is the explicit act that gives the lease back -- at a
call site that knows the run is over, rather than a ``finally`` that would
also fire on the way out of a success.
"""

from __future__ import annotations

import pathlib
from typing import Protocol

from typing_extensions import TypedDict

from fleet.contracts.feed import FeedEvent, FeedKind
from fleet.contracts.lease import Lease
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import MAKE_TARGET, ProjectConfig, lease_seconds
from fleet.core import _test_hooks, dialect, leases, manifest, names, records, remote, staging

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


def started_row(
    *,
    lease: Lease,
    host: str,
    workers: int,
    detail: str,
) -> LedgerEntry:
    """Build the ledger row a dispatch is recorded as running by.

    Args:
        lease: The claim it holds, which carries its identity.
        host: The SSH alias actually used.
        workers: Test workers granted.
        detail: What to say about it.

    Returns:
        The row, whose ``ended_unix`` equals its start because it has not
        ended. A nullable timestamp would make every reader branch, and the
        outcome field already says whether the run is over.
    """
    return LedgerEntry(
        run_id=lease["run_id"],
        node=lease["node"],
        host=host,
        project=lease["project"],
        agent=lease["agent"],
        session_id=lease["session_id"],
        started_unix=lease["acquired_unix"],
        ended_unix=lease["acquired_unix"],
        outcome="running",
        exit_code=NO_EXIT_CODE,
        workers=workers,
        detail=detail,
    )


def closed_row(
    row: LedgerEntry,
    *,
    outcome: LedgerOutcome,
    exit_code: int,
    ended_unix: int,
    detail: str,
) -> LedgerEntry:
    """Build the row that supersedes a running one.

    Appended rather than replacing, because the ledger is append-only: the
    running row is history, and a reader wanting the current state takes the
    last row for that id.

    Args:
        row: The running row.
        outcome: How it ended.
        exit_code: The recipe's status.
        ended_unix: When it ended.
        detail: What to say about it.

    Returns:
        The closing row.
    """
    return LedgerEntry(
        run_id=row["run_id"],
        node=row["node"],
        host=row["host"],
        project=row["project"],
        agent=row["agent"],
        session_id=row["session_id"],
        started_unix=row["started_unix"],
        ended_unix=ended_unix,
        outcome=outcome,
        exit_code=exit_code,
        workers=row["workers"],
        detail=detail,
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


class Payload(TypedDict):
    """The archive a dispatch stages, and how the feed describes it.

    Attributes:
        data: The gzipped tar's bytes, which the node digests and unpacks.
        description: What the archive holds, for the ``staged`` feed line:
            counted, never listed (a working tree carries about forty-six
            members; an export is one commit).
    """

    data: bytes
    description: str


class PayloadBuilder(Protocol):
    """Builds the archive a dispatch stages, once the lease is held.

    A callable rather than bytes, so the lease is taken BEFORE the archive
    is built whichever tree it comes from: staging into a project another
    dispatch holds would be pointless work at best.
    """

    def __call__(self, run_id: str) -> Payload:
        """Build the archive for one run.

        Args:
            run_id: The dispatch, which names the scratch file.

        Returns:
            The archive and its description.
        """
        ...


class Recipe(TypedDict):
    """What the node runs once the tree is staged.

    Attributes:
        path: The project's directory inside the staged tree, ``""`` for the
            root; the recipe's working directory.
        install: The steps run at the tree's root before the recipe, each an
            argv in the source grammar; empty when the recipe installs its
            own dependencies.
    """

    path: str
    install: tuple[tuple[str, ...], ...]


def working_tree_payload(
    project_root: pathlib.Path,
    *,
    project: str,
    plan: ProjectConfig,
    node_name: str,
    archive_dir: pathlib.Path,
) -> PayloadBuilder:
    """The archive of the hub's working tree, as ``fleet-run`` dispatches it.

    Args:
        project_root: Absolute path to the monorepo root.
        project: Repo-relative project path.
        plan: The project's declaration, for its external paths.
        node_name: The node's workspace name, part of the scratch file's
            name.
        archive_dir: Local directory to build the archive in, which must be
            run output rather than anywhere a build reads --
            :attr:`fleet.cli._config.LoadedWorkspace.archives` is where the
            commands get it. An archive left where a project's tree is staged
            FROM is carried by the next dispatch.

    Returns:
        A builder that computes the manifest and tars it when called.
    """

    def build(run_id: str) -> Payload:
        members = manifest.build_tree(project_root, project, external=plan["external_paths"])
        # Named by run AND node. A run id is the project and the second it
        # started, so two sessions dispatching one project to two nodes in
        # the same second produce the same id -- and with archives in one
        # scratch directory rather than one per workspace, that is two
        # writers on one file. The lease stops them sharing a node, not a
        # filename.
        data = staging.archive(project_root, members, archive_dir / f"{run_id}-{node_name}.tgz")
        return Payload(data=data, description=f"{len(members)} member(s) of the working tree")

    return build


def working_tree_recipe(project: str) -> Recipe:
    """The recipe a working-tree dispatch runs: ``make check`` in the
    project's own directory, whose key IS its repo-relative path, with no
    install steps because every poetry project here installs inside its
    recipe.

    Args:
        project: Repo-relative project path.

    Returns:
        The recipe.
    """
    return Recipe(path=project, install=())


def start(
    loaded_leases: pathlib.Path,
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    node_name: str,
    node: NodeConfig,
    project: str,
    plan: ProjectConfig,
    workers: int,
    agent: str,
    session_id: str,
    build_payload: PayloadBuilder,
    recipe: Recipe,
) -> LedgerEntry:
    """Take the lease, stage the tree, launch the suite, and record it.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        node_name: The node's workspace name.
        node: Its declaration.
        project: Repo-relative project path.
        plan: The project's declaration.
        workers: Test workers the capacity check granted.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.
        build_payload: Builds the archive once the lease is held:
            :func:`working_tree_payload` for ``fleet-run``, the export of a
            commit for the queue's node lane (:mod:`fleet.core.export`).
        recipe: Where in the tree the recipe runs and what readies it.

    Returns:
        The running ledger row.

    Raises:
        AppError: With ``LEASE_HELD`` when another dispatch holds this
            project on this node, the builder's own codes
            (``STAGE_ARCHIVE_UNREADABLE``, ``SHA_NOT_ON_REMOTE``,
            ``EXPORT_FAILED``), ``STAGE_DIGEST_MISMATCH`` from staging, or
            ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the transport.
    """
    now_unix = _test_hooks.now()
    run_id = run_id_for(project, started_unix=now_unix)
    lease = open_lease(
        node=node_name,
        project=project,
        run_id=run_id,
        agent=agent,
        session_id=session_id,
        plan=plan,
        now_unix=now_unix,
    )
    leases.acquire(loaded_leases, lease, now_unix=now_unix)
    emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind="leased",
        detail=f"{workers} worker(s)",
        now_unix=now_unix,
    )

    payload = build_payload(run_id)
    target = staging.stage(
        node["host"],
        platform=node["platform"],
        run_id=run_id,
        stage_root=node["stage_root"],
        payload=payload["data"],
    )
    emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind="staged",
        # Counted, not listed. A real dispatch carries about forty-six members
        # -- the project, its dependencies, and one manifest per package for
        # the guard rules that scan them all -- and naming each turned one feed
        # line into two thousand characters. The members are derivable from
        # `fleet.core.manifest`; the feed's job is to say a run reached this
        # step, with enough to spot a stage that carried the wrong amount.
        detail=f"{len(payload['data'])} bytes, {payload['description']}, to {target}",
        now_unix=_test_hooks.now(),
    )

    spoken = dialect.for_platform(node["platform"])
    remote.send_script(
        node["host"],
        spoken.script_path(target, names.BUILD_STEM),
        spoken.build_script(
            target=target,
            path=recipe["path"],
            workers=workers,
            install=recipe["install"],
            cache_root=names.cache_root(node["stage_root"]),
        ),
        platform=node["platform"],
    )
    remote.run_script(
        node["host"],
        spoken.script_path(target, names.LAUNCH_STEM),
        spoken.launch_script(target=target, run_id=run_id),
        platform=node["platform"],
    )
    row = started_row(lease=lease, host=node["host"], workers=workers, detail=f"staged to {target}")
    records.append_ledger(loaded_ledger, row)
    emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind="started",
        detail=f"make {MAKE_TARGET} at {workers} worker(s)",
        now_unix=_test_hooks.now(),
    )
    return row


def finish(
    loaded_leases: pathlib.Path,
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    row: LedgerEntry,
    outcome: LedgerOutcome,
    exit_code: int,
    detail: str,
) -> LedgerEntry:
    """Close a dispatch out: record the outcome, emit it, release the lease.

    THE LEASE IS RELEASED LAST. If the ledger write failed after the lease had
    already gone, the environment would be free while the record still said
    ``running``, and the next capacity check would count a dispatch that no
    longer exists. In this order a failure leaves the lease held, which
    expires on its own and is the safe direction to fail in.

    A LEASE THAT HAS ALREADY LAPSED IS NOT AN ERROR HERE, and requiring one
    was a defect. Whether a run was PROTECTED is a question about whether its
    lease covered the run; whether a lease is held NOW is a question about how
    promptly somebody came to collect. Conflating them refused a run that had
    finished three minutes inside its window because it was collected twenty
    minutes later. The genuine hazard -- a run that outlived its lease while
    still going -- is decided by the caller against the moment the build
    ended, before it gets here.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        row: The running row this supersedes.
        outcome: How the dispatch ended.
        exit_code: The recipe's status, or
            :data:`~fleet.contracts.ledger.NO_EXIT_CODE` when there was none.
        detail: What to say about it.

    Returns:
        The closing row.
    """
    ended_unix = _test_hooks.now()
    closing = closed_row(
        row, outcome=outcome, exit_code=exit_code, ended_unix=ended_unix, detail=detail
    )
    records.append_ledger(loaded_ledger, closing)
    emit(
        loaded_feed,
        run_id=row["run_id"],
        node=row["node"],
        project=row["project"],
        kind=_OUTCOME_EVENT[outcome],
        detail=detail,
        now_unix=ended_unix,
    )
    leases.release_if_held(loaded_leases, run_id=row["run_id"], now_unix=ended_unix)
    return closing


#: Which feed kind announces each terminal outcome.
#:
#: A mapping rather than a name shared between the two vocabularies, because
#: they are not the same vocabulary: ``running`` is a ledger outcome with no
#: event, and ``leased`` / ``staged`` / ``started`` are events with no outcome.
#: Spelling the overlap once, here, is what stops a reader assuming the two
#: enums are interchangeable.
_OUTCOME_EVENT: dict[LedgerOutcome, FeedKind] = {
    "refused": "refused",
    "passed": "passed",
    "failed": "failed",
    "cancelled": "cancelled",
    "lost": "lost",
    "running": "heartbeat",
}


__all__ = [
    "LEASE_SLACK",
    "Payload",
    "PayloadBuilder",
    "Recipe",
    "closed_row",
    "emit",
    "finish",
    "open_lease",
    "run_id_for",
    "start",
    "started_row",
    "working_tree_payload",
    "working_tree_recipe",
]
