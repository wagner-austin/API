"""The window that ties a queue claim to the run its tick launched.

A claiming tick claims a job and launches its run seconds later, so a run
belongs to a claim when it began inside that claim's lease of it. Two
readers ask that question: :func:`fleet.cli.node_collect.launched_by_claim`,
for a claim whose start report never reached the queue (MCPs board task
5a4f9b3e), and :mod:`fleet.cli.node_lost`, for a run whose job has since
left this runner (MCPs board task fd402617). Lifted here from the first
when the second needed it, so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Final

#: How long a claim survives without a report: the same hour the hub runner
#: takes, covering the fetch, the staging and the wait until the next tick's
#: collect renews it (:func:`fleet.cli.node_collect.collect_pass` renews every
#: running job it holds, so the lease is never sized for the slowest suite in
#: advance).
CLAIM_LEASE_SECONDS: Final = 3600

#: How far the queue's clock and this machine's may disagree when a claim is
#: matched to the run it launched: the claim is stamped by the database host
#: and the ledger row by the hub.
CLAIM_CLOCK_SLACK_SECONDS: Final = 60


def launched_within(*, claimed_unix: int, started_unix: int) -> bool:
    """Whether a run that began at one moment could have been launched by a claim.

    Args:
        claimed_unix: When the claim was taken, by the queue's clock.
        started_unix: When the run's ledger row says it began, by the hub's.

    Returns:
        True when the start falls within the claim's lease of the claim,
        allowing :data:`CLAIM_CLOCK_SLACK_SECONDS` of clock disagreement
        before it.
    """
    earliest = claimed_unix - CLAIM_CLOCK_SLACK_SECONDS
    return earliest <= started_unix <= claimed_unix + CLAIM_LEASE_SECONDS


__all__ = ["CLAIM_CLOCK_SLACK_SECONDS", "CLAIM_LEASE_SECONDS", "launched_within"]
