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
sshd sends the session's process group SIGHUP when the connection ends, so a
suite started as a child of the dispatching ssh call dies when that call
returns -- which is immediately, since the whole point is not to hold the
connection for the duration of a build. The Windows nodes taught it first,
whose OpenSSH put the session's tree in a job object for the same effect
(``memory/reference_long_runs_need_task_scheduler.md``). So the build is a
transient systemd unit.

The scripts that do the launching live in :mod:`fleet.core.dialect`, split
out of here by role when this file crossed the 600-line ceiling. This module
owns the ORDER; the dialect owns what the node is asked
to run, :mod:`fleet.core.recipe` what a project's declaration asks of its
build, and :mod:`fleet.core.names` what each file is called.

NOTHING HERE CATCHES. A failure to stage or launch propagates with its own
code, and the lease goes back by an explicit act at a call site that knows the
run is over, rather than a ``finally`` that would also fire on the way out of
a success: :func:`finish` for a run that launched, and
:func:`fleet.core.run_lease.abandon` for one that took its lease and never did
(MCPs board task e12affc5), which is where the lease step itself lives.
"""

from __future__ import annotations

import pathlib
from typing import Protocol

from typing_extensions import TypedDict

from fleet.contracts.feed import FeedKind
from fleet.contracts.lease import Lease
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import MAKE_TARGET, ProjectConfig
from fleet.core import (
    _test_hooks,
    dialect,
    export,
    leases,
    manifest,
    names,
    records,
    remote,
    run_lease,
    staging,
)
from fleet.core.recipe import Recipe, loopback_unit


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
        outcome=LedgerOutcome.RUNNING,
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


class Payload(TypedDict):
    """The archive a dispatch stages, and how the feed describes it.

    Attributes:
        path: The local file the archive was written to, which scp copies
            to the node (:func:`fleet.core.remote.send_file`).
        data: The same archive's bytes, which the node's digest is compared
            against before it unpacks.
        description: What the archive holds, for the ``staged`` feed line:
            counted, never listed (a working tree carries about forty-six
            members; an export is one commit).
    """

    path: pathlib.Path
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


def working_tree_payload(
    project_root: pathlib.Path,
    *,
    project: str,
    plan: ProjectConfig,
    archive_dir: pathlib.Path,
) -> PayloadBuilder:
    """The archive of the hub's working tree, as ``fleet-run`` dispatches it.

    Args:
        project_root: Absolute path to the monorepo root.
        project: Repo-relative project path.
        plan: The project's declaration, for its external paths.
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
        # Named by the run alone: archives share one scratch directory, and a
        # run id names its node (run_lease.run_id_for), so two dispatches of
        # one project to two nodes in one second write two files.
        path = archive_dir / f"{run_id}.tgz"
        data = staging.archive(project_root, members, path)
        return Payload(
            path=path, data=data, description=f"{len(members)} member(s) of the working tree"
        )

    return build


def start(
    loaded_leases: pathlib.Path,
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    loaded_retires: pathlib.Path,
    *,
    node_name: str,
    node: NodeConfig,
    project: str,
    plan: ProjectConfig,
    workers: int,
    agent: str,
    session_id: str,
    node_local: tuple[str, ...],
    build_payload: PayloadBuilder,
    companions: tuple[export.CompanionExport, ...],
    recipe: Recipe,
    registry: pathlib.Path,
) -> LedgerEntry:
    """Take the lease, then :func:`launch` under it: ``fleet-run``'s dispatch.

    A failure after the lease propagates with the lease still held, to a
    person at a terminal who reads the code. The queue's node lane calls
    :func:`fleet.core.run_lease.take` and :func:`launch` itself, because an
    unattended runner must give the lease back (:func:`run_lease.abandon`)
    before a resubmission bounces off it.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        loaded_retires: The retire record file, which owes the run's
            retire from its lease.
        node_name: The node's workspace name.
        node: Its declaration.
        project: Repo-relative project path.
        plan: The project's declaration.
        workers: Test workers the capacity check granted.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.
        node_local: The workspace's node-local resource names, which the
            lease holds as this node's copy.
        build_payload: Builds the archive once the lease is held.
        companions: The repositories staged beside the export.
        recipe: Where in the tree the recipe runs and what readies it.
        registry: The registry directory this command loaded.

    Returns:
        The running ledger row.

    Raises:
        AppError: With ``LEASE_HELD`` from the lease, or any code
            :func:`launch` raises.
    """
    lease = run_lease.take(
        loaded_leases,
        loaded_feed,
        loaded_retires,
        node_name=node_name,
        project=project,
        plan=plan,
        workers=workers,
        agent=agent,
        session_id=session_id,
        node_local=node_local,
    )
    return launch(
        loaded_ledger,
        loaded_feed,
        lease=lease,
        node=node,
        workers=workers,
        build_payload=build_payload,
        companions=companions,
        recipe=recipe,
        registry=registry,
    )


def launch(
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    lease: Lease,
    node: NodeConfig,
    workers: int,
    build_payload: PayloadBuilder,
    companions: tuple[export.CompanionExport, ...],
    recipe: Recipe,
    registry: pathlib.Path,
) -> LedgerEntry:
    """Stage the tree, launch the suite, and record it, under a held lease.

    Args:
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        lease: The lease :func:`fleet.core.run_lease.take` returned, which
            names the run, its node and its project.
        node: The node's declaration.
        workers: Test workers the capacity check granted.
        build_payload: Builds the archive once the lease is held:
            :func:`working_tree_payload` for ``fleet-run``, the export of a
            commit for the queue's node lane (:mod:`fleet.core.export`).
        companions: The repositories staged beside the export, in declared
            order, before the build script is sent. Empty for a project
            whose check reads only its own tree.
        recipe: Where in the tree the recipe runs and what readies it.
        registry: The registry directory staged beside the export.

    Returns:
        The running ledger row.

    Raises:
        AppError: With the builder's own codes (``STAGE_ARCHIVE_UNREADABLE``,
            ``SHA_NOT_ON_REMOTE``, ``REMOTE_CREDENTIAL_REJECTED``,
            ``EXPORT_FAILED``),
            ``STAGE_DIGEST_MISMATCH`` from staging, or ``NODE_UNREACHABLE``
            or ``DISPATCH_FAILED`` from the transport. The lease is still
            held then; giving it back is the caller's act.
    """
    run_id = lease["run_id"]
    node_name = lease["node"]
    project = lease["project"]
    payload = build_payload(run_id)
    # The stage is the archive's one reader, so the file goes however it ends: 245 kept after
    # successes filled the hub (bfca20e6), failures left 3.6 GB in its memory root (fb505198).
    try:
        target = staging.stage(
            node["host"],
            run_id=run_id,
            stage_root=node["stage_root"],
            source=payload["path"],
            payload=payload["data"],
        )
    finally:
        payload["path"].unlink()
    local = payload["path"].with_name(f"{run_id}-{names.REGISTRY_DIRECTORY}.tar.gz")
    staging.stage_registry(node["host"], target=target, registry=registry, local=local)
    run_lease.emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind=FeedKind.STAGED,
        # Counted, not listed. A real dispatch carries about forty-six members
        # -- the project, its dependencies, and one manifest per package for
        # the guard rules that scan them all -- and naming each turned one feed
        # line into two thousand characters. The members are derivable from
        # `fleet.core.manifest`; the feed's job is to say a run reached this
        # step, with enough to spot a stage that carried the wrong amount.
        detail=f"{len(payload['data'])} bytes, {payload['description']}, to {target}",
        now_unix=_test_hooks.now(),
    )
    # BEFORE the build script is sent, because the install steps are part of
    # that script and a project whose install reads its companion would find
    # nothing there. Each one is its own feed line naming the commit it is,
    # so a verdict can be read against the workspace it was measured with --
    # a companion is the tip of a ref, so two runs of one sha can legitimately
    # measure different things, and the line is where that shows.
    for companion in companions:
        beside = staging.stage_companion(
            node["host"],
            stage_root=node["stage_root"],
            directory=companion["directory"],
            ref=companion["ref"],
            sha=companion["sha"],
            source=companion["path"],
            payload=companion["data"],
        )
        run_lease.emit(
            loaded_feed,
            run_id=run_id,
            node=node_name,
            project=project,
            kind=FeedKind.STAGED,
            detail=(
                f"{len(companion['data'])} bytes, the {companion['directory']} companion "
                f"at {companion['sha']}, to {beside}"
            ),
            now_unix=_test_hooks.now(),
        )
        # Verified on the node, so like the archive above it has no reader
        # left; kept, the hub held 169 bundles in 14 GB on 2026-10-15 (MCPs
        # board task f9ee56aa). Its directory is the claimed job's alone.
        companion["path"].unlink()
    for bundles in {companion["path"].parent for companion in companions}:
        bundles.rmdir()

    remote.send_script(
        node["host"],
        dialect.script_path(target, names.BUILD_STEM),
        dialect.build_script(
            target=target,
            path=recipe["path"],
            workers=workers,
            install=recipe["install"],
            cache_root=names.cache_root(node["stage_root"]),
            isolated_docker=recipe["isolated_docker"],
            agent=lease["agent"],
            loopback_unit=loopback_unit(recipe, run_id),
        ),
    )
    remote.run_script(
        node["host"],
        dialect.script_path(target, names.LAUNCH_STEM),
        dialect.launch_script(target=target, run_id=run_id),
    )
    row = started_row(lease=lease, host=node["host"], workers=workers, detail=f"staged to {target}")
    records.append_ledger(loaded_ledger, row)
    run_lease.emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind=FeedKind.STARTED,
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
    run_lease.emit(
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
    LedgerOutcome.REFUSED: FeedKind.REFUSED,
    LedgerOutcome.PASSED: FeedKind.PASSED,
    LedgerOutcome.FAILED: FeedKind.FAILED,
    LedgerOutcome.CANCELLED: FeedKind.CANCELLED,
    LedgerOutcome.LOST: FeedKind.LOST,
    LedgerOutcome.RUNNING: FeedKind.HEARTBEAT,
}


__all__ = [
    "Payload",
    "PayloadBuilder",
    "closed_row",
    "finish",
    "launch",
    "start",
    "started_row",
    "working_tree_payload",
]
