"""What collecting one held job did, and when a running job's lease is renewed.

WHY A COLLECT SAYS WHAT IT DID (MCPs board task c1d48330). Collecting one job
(:func:`fleet.cli.node_collect.collect_one_job`) answered one line of prose,
and the serve's watch counted every answer as a settle: on 2026-10-07 at
02:54:45Z lavender-wsl stopped answering ssh, the watch's collect of row
24c4e934 answered "did not answer the read", and the watch counted the run
closed and stopped reading it, so the row closed only at the next start,
192 s after its check ended. A collect now answers a :class:`Collected`,
whose :class:`CollectOutcome` the watch and the collect pass act on; only
:attr:`CollectOutcome.SETTLED` is a closed row.

WHY A RENEWAL WAITS (MCPs board task c1d48330, A2). The collect pass renews
every running job it holds, and it runs at each fire boundary and again at
every poll after the queue refused a call, until the queue answers. Between
2026-10-06T20:16Z and 2026-10-07T03:48Z that wrote nine renewals under 60 s
after the one before, among them four pairs on serendipity and lavender-wsl
19 to 37 s apart around 01:00:00Z, when mcp-fleet came back from a deploy
and the recovering passes met the fire boundary's: each a pass that found
its run still going and wrote to the queue anyway. A
running job's lease is now renewed only once it was last set
:data:`RENEW_AFTER_SECONDS` or more ago (:func:`lease_age`), read from the
lease the queue reports (``leaseExpiresAt``): a fire boundary's pass, 180 s
after the last, always renews, and a pass seconds after one writes nothing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, TypedDict

from fleet.contracts.dispatch import DispatchJob
from fleet.core.claim_window import CLAIM_LEASE_SECONDS

#: How long ago a running job's lease must have been set before a collect
#: pass renews it. Over 60 s, the spacing the duplicate renewals were counted
#: against, by more than what reading the lease can be off by: the queue's
#: instant is decoded to whole seconds, which floors away up to one, and the
#: hub's clock and the queue's differ by a little more. At exactly 60 s, job
#: d371c8ec's start report at 04:35:00.935Z on 2026-10-07 read as set 60 s
#: before the 04:36:00Z boundary's pass, which renewed it 59.2 s after it.
#: Under 90 s, so the longest gap between two progress entries of a running
#: job, a start just under this before a boundary and the renewal at the
#: next, stays under 270 s with the pass's own seconds.
RENEW_AFTER_SECONDS: Final = 70


class CollectOutcome(StrEnum):
    """What collecting one held job did."""

    #: The node had written its result, or the run outlived its lease and
    #: was stopped, and the job is now closed on the queue.
    SETTLED = "settled"
    #: The run is still going and its lease was renewed.
    RENEWED = "renewed"
    #: The run is still going and its lease was set too recently to renew;
    #: nothing was written.
    RUNNING = "running"
    #: The node did not answer a read; nothing changed, and the run is read
    #: again at the next poll.
    UNREACHABLE = "unreachable"
    #: This machine's ledger no longer calls the run live: another settle
    #: closed it first.
    NOT_LIVE = "not-live"
    #: The queue no longer has this runner holding the job as running: a
    #: cancel or a takeover, which the next fire's collect pass stops.
    NOT_HELD = "not-held"


class Collected(TypedDict):
    """What collecting one held job did, and the line the log gets.

    Attributes:
        outcome: What it did.
        line: One line saying so.
    """

    outcome: CollectOutcome
    line: str


def lease_age(job: DispatchJob, *, now: int) -> int | None:
    """How long ago a held job's lease was set.

    Every lease a runner sets is :data:`fleet.core.claim_window.CLAIM_LEASE_SECONDS`
    long (the claim, the start report and each renewal), so how long ago it
    was set is that length less what remains of it. A running job's lease is
    renewed once this reaches :data:`RENEW_AFTER_SECONDS`.

    Args:
        job: The job, as the queue listed it.
        now: Whole seconds since the epoch.

    Returns:
        The seconds, or None when the job carries no lease, which a renewal
        then sets.
    """
    expires = job["lease_expires_unix"]
    if expires is None:
        return None
    return CLAIM_LEASE_SECONDS - (expires - now)


__all__ = ["RENEW_AFTER_SECONDS", "CollectOutcome", "Collected", "lease_age"]
