"""Give back the lease of a claim that launched nothing (MCPs board task 31d1627d).

A claiming tick takes the project's lease (:func:`fleet.core.run_lease.take`)
before it stages and launches, so a tick killed between the two leaves a
lease and no run. :func:`fleet.cli.node_collect.reconcile_claim` gives such
a claim back to the queue and discharges its host claim, and until this it
left the lease standing, so the project's lane on the node stayed out of
admission for the rest of the lease. Measured 2026-10-30: job c1e06efc for
MCPs/execution-served-demure was claimed on demure at 23:39:57Z and given
back as DISPATCH_NOT_LAUNCHED at 23:40:46Z, while its lease
MCPs-execution-served-demure-demure-1793403597 held the lane until 23:59:57Z
with no run behind it and two served checks queued, until a session
abandoned it by hand at about 23:50Z.

THE LEASE IS FOUND AS THE RUN IS. A lease carries no job id, so it is the
dead claim's when it holds the job's project on this node, names the job's
submitter and session, as :func:`fleet.cli.node_launch.launch_claimed` takes
it, and was acquired inside the claim's window
(:func:`fleet.core.claim_window.launched_within`), the rule
:func:`fleet.cli.node_collect.launched_by_claim` applies to a ledger row. A
lease whose run the ledger holds live is a launched run's, never a dead
claim's, and is left held: a run's lease is given back when it settles.
"""

from __future__ import annotations

from fleet.cli import _config
from fleet.cli import collect as collect_cli
from fleet.contracts.dispatch import DispatchJob
from fleet.core import _test_hooks, leases, run_lease
from fleet.core.claim_window import launched_within


def give_back_lease(
    loaded: _config.LoadedWorkspace, job: DispatchJob, *, alias: str, detail: str
) -> str:
    """Abandon the lease a claim that launched nothing took, when it still stands.

    Args:
        loaded: The workspace and its resolved record paths.
        job: The claimed job whose tick launched nothing.
        alias: This node's workspace name.
        detail: The ``CODE: message`` the job is given back with, which the
            feed's terminal ``refused`` event for the lease's run carries.

    Returns:
        What became of the project's lease on this node, for the log line.
    """
    lease = leases.find_holder(
        loaded.leases, node=alias, project=job["project"], now_unix=_test_hooks.now()
    )
    if lease is None:
        return f"no lease holds {job['project']} on {alias}"
    claimed = job["claimed_unix"]
    if (
        claimed is None
        or lease["agent"] != job["submitted_by"]
        or lease["session_id"] != job["session_id"]
        or not launched_within(claimed_unix=claimed, started_unix=lease["acquired_unix"])
        or collect_cli.live_rows(loaded, run_id=lease["run_id"])
    ):
        return (
            f"{lease['run_id']} holding {job['project']} on {alias} is not its claim's, "
            "so it stays held"
        )
    given_back = run_lease.abandon(loaded.leases, loaded.feed, lease=lease, detail=detail)
    return f"its lease {lease['run_id']} is abandoned: {given_back}"


__all__ = ["give_back_lease"]
