"""When a group of endings has settled enough to be worth one post.

THE DEFECT THIS EXISTS TO REMOVE. :func:`hpc_wake.announce.announcements`
groups correctly -- one post per (project, submitter) -- but it groups only
within ONE CYCLE. ``run_cycle`` used to announce every ending in the cycle
that observed it, so the quality of the grouping was decided entirely by how
many jobs happened to finish inside one poll interval. Polling frequency is
a transport concern and notification frequency is a policy concern, and they
were the same number.

Measured on the live board 2026-09-07: ``bridge-hpc-wake-0906`` posted 116
times in 24 hours at a MEDIAN GAP OF 180 SECONDS -- the board's highest-volume
author and its lowest by body size. Over seven days, 47 of ~132 posts carried
exactly ONE job; the rest carried 2 to 82, which is the existing grouping
working whenever jobs happened to end together. So nothing was missing from
the grouping. A 136-member array whose members finish minutes apart defeats
per-cycle grouping by construction.

THE THREE RIPENESS RULES, and why one is not enough:

  quiet  no new member for :data:`SETTLE_SECONDS`. This is the rule that
         does the work, and alone it is unbounded: a group that gains a
         member every four minutes forever is never quiet and never posts.
  full   the group reached :data:`BATCH_CAP` members. Bounds the SIZE of a
         post, so one settle window cannot produce an unreadable wall.
  aged   the oldest member has waited :data:`MAX_HOLD_SECONDS`. Bounds the
         LATENCY, so the quiet rule can never hold news indefinitely. This
         is the rule that makes the other two safe to state.

Every function here is pure. Nothing reads a clock, a file or the board --
the caller supplies ``now_epoch`` -- so the whole policy is exercised over
simulated time without a fake, and the constants below are the only thing a
future tuning argument needs to touch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from platform_core.error_codes_tooling import HpcWakeErrorCode
from platform_core.errors import AppError
from typing_extensions import TypedDict

from hpc_wake.pending import PendingClosure

#: Seconds of quiet before a group is announced.
#:
#: CHOSEN AGAINST A MEASUREMENT, not a preference. The observed median gap
#: between endings in the burst this package was changed for was 180
#: seconds, so a window at or below that would have coalesced nothing --
#: every arrival would have found the group already quiet. 300 gives the
#: next member of a trickling array time to arrive, and costs a genuinely
#: solitary job five minutes of notification latency.
SETTLE_SECONDS: Final = 300

#: Most endings one post may carry before the group is ripe regardless of
#: quiet. ``announce.LINE_CAP`` separately bounds how many are listed
#: individually, so a capped group is still readable; this bounds how long
#: the bridge will keep accumulating before it says anything at all.
BATCH_CAP: Final = 25

#: The longest an ending may wait, however busy its group stays. One hour
#: turns the worst case from "never" into "at most twenty-four posts a day
#: per group", which is the bound the quiet rule cannot provide by itself.
MAX_HOLD_SECONDS: Final = 3600


class PendingGroup(TypedDict):
    """Every ending waiting under one announcement key.

    Attributes:
        key: The ``(project, submitter)`` pair these endings share -- the
            same key :func:`hpc_wake.announce.group_key` produces, because
            a group that settled together must be a group that posts
            together.
        records: The waiting endings, in the order they were observed.
    """

    key: tuple[str, str]
    records: tuple[PendingClosure, ...]


def group_pending(
    pending: Sequence[PendingClosure], keys: Mapping[str, tuple[str, str]]
) -> list[PendingGroup]:
    """Gather waiting endings under their announcement keys.

    Args:
        pending: Every ending currently waiting.
        keys: Announcement key per job id, built by the caller from the
            ledger with :func:`hpc_wake.announce.group_key`.

    Returns:
        One group per key, ordered by key so a cycle is deterministic.

    Raises:
        AppError: ``JOB_UNKNOWN_TO_LEDGER`` when a waiting ending has no
            key. The pending file is written from ledger-backed closures, so
            a missing key means the ledger was rotated or truncated beneath
            a record that is still owed a post. Skipping it would drop that
            post silently, which is this bridge's one unacceptable outcome.
    """
    grouped: dict[tuple[str, str], list[PendingClosure]] = {}
    for record in pending:
        job_id = record["closure"]["job_id"]
        key = keys.get(job_id)
        if key is None:
            raise AppError(
                code=HpcWakeErrorCode.JOB_UNKNOWN_TO_LEDGER,
                message=(
                    f"pending ending {job_id!r} has no ledger entry, so it has no "
                    "announcement key; it is owed a post that can no longer be "
                    "addressed, which is a ledger-integrity defect and not a stray job"
                ),
            )
        grouped.setdefault(key, []).append(record)
    return [
        PendingGroup(key=key, records=tuple(records)) for key, records in sorted(grouped.items())
    ]


def is_ripe(group: PendingGroup, now_epoch: int) -> bool:
    """Decide whether a group should be announced now.

    Args:
        group: The waiting endings under one key. Never empty: groups are
            built by :func:`group_pending`, which creates a key only when
            it has a record to put under it.
        now_epoch: Unix seconds, supplied by the caller so this stays pure.

    Returns:
        True when the group is quiet, full, or aged. See this module's
        docstring for why all three are needed.
    """
    observed = [record["observed_epoch"] for record in group["records"]]
    if len(observed) >= BATCH_CAP:
        return True
    if now_epoch - max(observed) >= SETTLE_SECONDS:
        return True
    return now_epoch - min(observed) >= MAX_HOLD_SECONDS


def partition_ripe(
    pending: Sequence[PendingClosure],
    keys: Mapping[str, tuple[str, str]],
    now_epoch: int,
) -> tuple[list[PendingClosure], list[PendingClosure]]:
    """Split waiting endings into those to announce now and those to hold.

    Args:
        pending: Every ending currently waiting.
        keys: Announcement key per job id.
        now_epoch: Unix seconds.

    Returns:
        ``(ripe, holding)``. A group is announced whole or held whole --
        never split -- because half a group is a post that under-reports a
        sweep, and the member left behind would then arrive as its own post
        later, which is the exact shape this module removes.

    Raises:
        AppError: ``JOB_UNKNOWN_TO_LEDGER`` via :func:`group_pending`.
    """
    ripe: list[PendingClosure] = []
    holding: list[PendingClosure] = []
    for group in group_pending(pending, keys):
        destination = ripe if is_ripe(group, now_epoch) else holding
        destination.extend(group["records"])
    return ripe, holding


__all__ = [
    "BATCH_CAP",
    "MAX_HOLD_SECONDS",
    "SETTLE_SECONDS",
    "PendingGroup",
    "group_pending",
    "is_ripe",
    "partition_ripe",
]
