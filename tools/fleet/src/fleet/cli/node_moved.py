"""Give back a run launched under another stage root (MCPs board task f02b144e).

Every pass reads a node's stage root from the rolled workspace, so a roll
that moves the node's account moves where the runner looks for its runs. A
run launched before that roll is not where the runner now reads, and the
read used to answer only "no result yet": the run read as running until its
lease lapsed and was stopped ``LEASE_NOT_HELD`` with no transcript. Measured
2026-10-30: colossus's tick took the roll of API cf5b3308d, which moved its
stage root from /home/user/fleet/stage to /home/corvis/fleet/stage, at
07:11:58Z. Jobs f492e270 (MCPs/kvm-mcp) and 4c3824c0 (MCPs/document-vault),
launched under /home/user at 07:10:56Z and 07:10:59Z, had both written exit
0 there by 07:12:00Z, and were stopped at 07:32:55Z and 07:32:58Z, 182 s and
187 s past their deadlines, naming logs under /home/corvis that were never
written.

THE RUN IS GIVEN BACK, NOT JUDGED. Its directory under the old root is not
one this runner can name any more, so neither its status nor its transcript
can be read: its job goes back to the queue (:func:`fleet.cli.job_close.give_back`),
which invents no exit status and lets another claim run it, and then its
unit is stopped by name, which does not depend on the stage root, and its
row closed lost (:func:`fleet.core.stop.stop_and_finish`). The queue goes
first, as :mod:`fleet.cli.node_retired` does it, for the same reason.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config, job_close
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.core import stop


def give_back_moved(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    row: LedgerEntry,
    node: NodeConfig,
    unlaunched: str,
) -> Collected:
    """Give a run's job back to the queue, then stop it and close its row lost.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.
        row: The run's live ledger row.
        node: The node it ran on, as the rolled workspace declares it now.
        unlaunched: Why the run is not where that declaration puts it, as
            :func:`fleet.core.collect.attempt_poll_result` said.

    Returns:
        :attr:`~fleet.cli.node_collected.CollectOutcome.SETTLED` with the
        queue's answer.

    Raises:
        AppError: A queue failure from the give-back, with nothing on this
            machine changed, or a node failure from the stop, which leaves
            the row live.
    """
    detail = f"{unlaunched}; its job was given back and the runner ended its unit"
    answered = job_close.give_back(credentials, job, identity, detail=detail)
    stop.stop_and_finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        loaded.retires,
        node=node,
        row=row,
        outcome=LedgerOutcome.LOST,
        exit_code=NO_EXIT_CODE,
        detail=detail,
    )
    return Collected(
        outcome=CollectOutcome.SETTLED,
        line=f"{encode_job_line(job)}: {detail}; the queue left it {answered['status'].value}",
    )


__all__ = ["give_back_moved"]
