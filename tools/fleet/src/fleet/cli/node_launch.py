"""Launches beside the claiming: a serving runner's claimed jobs start on a pool.

WHY (MCPs board task 8993c306). A node runner's fill pass claimed a job and
then launched it before it claimed again, and a launch is the slow half:
the commit's fetch into the hub's mirror, the archive, the staging over ssh
and the registration take 17 to 35 s for a small project on lavender-wsl
and 90 to 155 s on loki. On 2026-10-05 lavender-wsl claimed another
session's tools/hpc3 job at 08:14:44Z and launched it until 08:15:45Z, and
tools/fleet-execution-linux row 4fb00b3c, submitted at 08:14:52.8Z to a
node with room for it, was claimed only at 08:15:57Z, 64.7 s after it was
submitted. Now the claim hands the launch to :class:`Launcher`, whose pool
(:data:`LAUNCH_WORKERS`, through :data:`fleet.core._test_hooks.executor`)
runs it while the fill pass claims again.

WHAT THE CLAIM STILL DECIDES. Everything the node's room depends on: the
gate (:func:`fleet.cli.node_ready.ready_state`), the claim, and the job's
registry line, tags and grant (:func:`fleet.cli.node_prepare.admit`). A
claimed job is charged to its HOST from that moment: :meth:`Launcher.start`
writes its grant as a host claim (:mod:`fleet.core.host_claims`), which
every runner of the host adds to what the ledger's live runs hold and whose
project every gate leaves out as it leaves out a project a lease holds, so
two launches never share one project's environment and a host is never
granted more than the probe and the owner's reservation allow, whichever of
its runners claimed. Until MCPs board task a85ef09e the charge lived in this
launcher's memory, which the host's other runner, another process, could not
read. The launch removes the claim once it has ended, by which time its run
is on the ledger, which charges it from there on. The launch keeps its order
and its refusals: the commit's fetch before the lease, the lease given back
when the launch fails after it, and every local refusal reported to the
queue verbatim.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from types import TracebackType
from typing import Final, TypedDict

from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli.node_claim import refuse
from fleet.cli.node_prepare import Admitted, Prepared, prepare
from fleet.cli.node_start import report_started
from fleet.contracts.dispatch import DispatchJob
from fleet.contracts.host_claim import HostClaim
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.node import LiveLoad, NodeConfig
from fleet.core import _test_hooks, dispatch, export, host_claims, run_lease
from fleet.core.claim_window import CLAIM_LEASE_SECONDS
from fleet.core.queue_transport import unanswered

_log = get_logger(__name__)

#: How many claimed jobs a runner launches at once: more than any node's
#: room holds runs, so a launch waits for a thread only past that.
LAUNCH_WORKERS: Final = 4


class Launching(TypedDict):
    """What this runner's own launches under way hold on its node.

    The gate charges the host's claims instead, every runner's
    (:mod:`fleet.core.host_claims`); these are what this runner must wait
    for and leave alone.

    Attributes:
        load: Their grants, counted as live runs are; a serve hands over
            only once its count is zero.
        jobs: Their queue jobs, which the collect pass leaves to them.
    """

    load: LiveLoad
    jobs: frozenset[str]


class _InFlight(TypedDict):
    """One claimed job whose launch is under way.

    Attributes:
        project: Its project.
        workers: Its grant.
        ram_gb: What those workers may hold.
    """

    project: str
    workers: int
    ram_gb: float


def launch_claimed(
    loaded: _config.LoadedWorkspace,
    job: DispatchJob,
    *,
    alias: str,
    node: NodeConfig,
    prepared: Prepared,
    build: dispatch.PayloadBuilder,
) -> LedgerEntry | str:
    """Take the project's lease on this node and launch the job under it.

    GUARDED LIKE ``prepare``, because a fault here is otherwise reported
    NOWHERE: there is no failed feed row on this path, so an AppError out of
    staging ended the tick and left the queue row in status claimed until its
    lease ran out, an hour later, with no reason recorded anywhere a waiting
    session could read. Measured 2026-09-24 (board task 1e57ebe5): two jobs
    sat exactly that way for 32 and 14 minutes.

    A FAILURE AFTER THE LEASE GIVES THE LEASE BACK before the refusal, or the
    resubmission the refusal invites bounces off it: MCPs board task
    e12affc5, where job 7143a25e was refused ``LEASE_HELD`` by the dead run
    of job 4d8b28a9 with 793 s of its lease left. A ``LEASE_HELD`` out of the
    lease itself gives nothing back, because that lease is another run's
    (:mod:`fleet.core.run_lease`).

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job.
        alias: This node's workspace name.
        node: Its declaration.
        prepared: What :func:`fleet.cli.node_prepare.prepare` resolved for the job.
        build: Builds the job's export once the lease is held.

    Returns:
        The running ledger row, or the ``CODE: message`` refusal to report.
    """
    try:
        lease = run_lease.take(
            loaded.leases,
            loaded.feed,
            node_name=alias,
            project=job["project"],
            plan=prepared["plan"],
            workers=prepared["workers"],
            agent=job["submitted_by"],
            session_id=job["session_id"],
            node_local=loaded.workspace["node_local_resources"],
        )
    except AppError as refusal:
        return f"{refusal.code}: {refusal.message}"
    try:
        return dispatch.launch(
            loaded.ledger,
            loaded.feed,
            lease=lease,
            node=node,
            workers=prepared["workers"],
            build_payload=build,
            companions=prepared["companions"],
            recipe=dispatch.recipe_for(
                prepared["plan"],
                path=prepared["source"]["path"],
                install=prepared["source"]["install"],
            ),
        )
    except AppError as refusal:
        detail = f"{refusal.code}: {refusal.message}"
        given_back = run_lease.abandon(loaded.leases, loaded.feed, lease=lease, detail=detail)
        _log.info("%s never launched; %s", lease["run_id"], given_back)
        return detail


class Launcher:
    """A serving runner's launches, each on a thread of the pool it opens.

    Used as ``with launcher`` around the serve: its exit waits for every
    launch under way and, when the serve itself raised nothing, raises the
    first launch that did.
    """

    def __init__(
        self,
        loaded: _config.LoadedWorkspace,
        credentials: McpCredentials,
        identity: JSONObject,
        *,
        alias: str,
        node: NodeConfig,
        agent: str,
        hold: Callable[[frozenset[str]], None],
    ) -> None:
        """Bind the launcher to its node and open its pool.

        Args:
            loaded: The workspace and its resolved record paths.
            credentials: The queue's endpoint and headers.
            identity: This runner's identity arguments.
            alias: This node's workspace name.
            node: Its declaration.
            agent: This runner's label.
            hold: Given each run once its start is reported, so the serve's
                watch (:mod:`fleet.cli.node_watch`) reads it.
        """
        self._loaded = loaded
        self._credentials = credentials
        self._identity = identity
        self._alias = alias
        self._node = node
        self._agent = agent
        self._hold = hold
        self._executor = _test_hooks.executor(workers=LAUNCH_WORKERS, name=f"fleet-launch-{alias}")
        self._changed = threading.Lock()
        self._in_flight: dict[str, _InFlight] = {}
        self._failed: Future[str | None] | None = None
        self._unreported = False

    def launching(self) -> Launching:
        """What the launches under way hold on the node now.

        Returns:
            Their grants, projects and jobs.
        """
        with self._changed:
            flights = dict(self._in_flight)
        return Launching(
            load=LiveLoad(
                runs=len(flights),
                workers=sum(flight["workers"] for flight in flights.values()),
                ram_gb=sum(flight["ram_gb"] for flight in flights.values()),
            ),
            jobs=frozenset(flights),
        )

    def start(self, job: DispatchJob, *, admitted: Admitted, sha: str) -> Future[str | None]:
        """Charge a claimed job to its host and hand its launch to the pool.

        Called inside the host's claim turn that claimed the job
        (:func:`fleet.core.host_claims.claim_turn`), so the host's other
        runner reads the claim before it next claims.

        Args:
            job: The claimed job.
            admitted: Its plan and grant.
            sha: Its commit.

        Returns:
            The launch: the run id once its start is reported, or None when
            it was refused.
        """
        flight = _InFlight(
            project=job["project"],
            workers=admitted["workers"],
            ram_gb=admitted["workers"] * admitted["plan"]["worker_ram_gb"],
        )
        host_claims.record(
            self._loaded.host_claims,
            alias=self._alias,
            claim=HostClaim(
                job_id=job["job_id"],
                runner=self._agent,
                project=flight["project"],
                workers=flight["workers"],
                ram_gb=flight["ram_gb"],
                until_unix=_test_hooks.now() + CLAIM_LEASE_SECONDS,
            ),
        )
        with self._changed:
            self._in_flight[job["job_id"]] = flight
        launch = self._executor.submit(self._launch, job, admitted, sha)
        launch.add_done_callback(self._launched)
        return launch

    def take_unreported(self) -> bool:
        """Whether a launch's start report went unanswered since the last ask.

        Returns:
            True once for any number of them, then False until another; the
            serve then runs its collect pass, which adopts each such run
            (:func:`fleet.cli.node_serve.serve_loop`).
        """
        with self._changed:
            unreported = self._unreported
            self._unreported = False
        return unreported

    def raise_failed(self) -> None:
        """Raise the first launch that raised, if one has.

        Raises:
            AppError: That launch's error: a queue or node failure, not a
                refusal, which was reported to the queue.
        """
        with self._changed:
            failed = self._failed
        if failed is not None:
            failed.result()

    def __enter__(self) -> Launcher:
        """Open the span of the serve.

        Returns:
            This launcher.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Wait for every launch under way, then raise the first that raised,
        unless the serve is already raising its own error.

        Args:
            exc_type: The serve's exception type, or None.
            exc: Its exception, or None.
            traceback: Its traceback, or None.
        """
        self._executor.shutdown(wait=True)
        if exc_type is None:
            self.raise_failed()

    def _launched(self, launch: Future[str | None]) -> None:
        """Keep the first launch that raised.

        Args:
            launch: A launch that has finished.
        """
        if launch.exception() is None:
            return
        with self._changed:
            if self._failed is None:
                self._failed = launch

    def _launch(self, job: DispatchJob, admitted: Admitted, sha: str) -> str | None:
        """Fetch, lease, stage and launch one claimed job, and report it started.

        Args:
            job: The claimed job.
            admitted: Its plan and grant.
            sha: Its commit.

        Returns:
            The run id, or None when a local refusal was reported instead.
            A start report the queue did not answer leaves the job claimed
            with its run launched, which is what
            :func:`fleet.cli.node_collect.reconcile_claim` adopts, so it is
            logged and counted for :meth:`take_unreported` instead of
            raised, and the run is left to that adoption rather than held.

        Raises:
            AppError: From the queue's start report or refusal, but for an
                unanswered start report, or the stop of a job cancelled
                while it launched. Not caught: the serve raises it
                (:meth:`raise_failed`).
        """
        try:
            row = self._launch_row(job, admitted, sha)
            if row is None:
                return None
            try:
                report_started(
                    self._loaded,
                    self._credentials,
                    self._identity,
                    job=job,
                    row=row,
                    alias=self._alias,
                    node=self._node,
                    agent=self._agent,
                )
            except AppError as refusal:
                if not unanswered(refusal):
                    raise
                _log.info(
                    "%s: the queue did not answer the start report of %s; the next collect "
                    "pass adopts it: %s",
                    job["job_id"],
                    row["run_id"],
                    refusal.message,
                )
                with self._changed:
                    self._unreported = True
                return row["run_id"]
            self._hold(frozenset({row["run_id"]}))
            return row["run_id"]
        finally:
            host_claims.discharge(self._loaded.host_claims, alias=self._alias, job_id=job["job_id"])
            with self._changed:
                del self._in_flight[job["job_id"]]

    def _launch_row(self, job: DispatchJob, admitted: Admitted, sha: str) -> LedgerEntry | None:
        """Prepare and launch one claimed job, refusing it on a local refusal.

        Args:
            job: The claimed job.
            admitted: Its plan and grant.
            sha: Its commit.

        Returns:
            The running ledger row, or None once the refusal is reported.

        Raises:
            AppError: From the queue's refusal call.
        """
        try:
            prepared = prepare(self._loaded, job, admitted=admitted, sha=sha)
        except AppError as refusal:
            refuse(
                self._credentials, job, self._identity, detail=f"{refusal.code}: {refusal.message}"
            )
            return None

        def build(run_id: str) -> dispatch.Payload:
            path = self._loaded.archives / f"{run_id}.tgz"
            data = export.archive_commit(prepared["mirror"], sha, path, prepared["scope"])
            return dispatch.Payload(path=path, data=data, description=f"git archive of {sha}")

        row = launch_claimed(
            self._loaded, job, alias=self._alias, node=self._node, prepared=prepared, build=build
        )
        if isinstance(row, str):
            refuse(self._credentials, job, self._identity, detail=row)
            return None
        return row


__all__ = [
    "LAUNCH_WORKERS",
    "Launcher",
    "Launching",
    "launch_claimed",
]
