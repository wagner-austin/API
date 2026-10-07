"""The corvis dispatch queue's wire shapes, decoded strictly.

WHAT THIS PACKAGE IS ON THE OTHER SIDE OF. ``fleet-mcp``'s ``dispatch_*``
tools (MCPs repo, migration 486) hold a queue: a session anywhere enqueues
"run ``make check`` for project X", and a runner process on the hub -- this
package's :mod:`fleet.cli.agent` -- claims it and executes it over the tailnet.
The corvis server has no route to the tailnet and no ssh key, which is the
whole reason the queue is inverted rather than the server reaching out.

THE ANSWERS ARE JSON, AND THAT IS THE TOOL'S DELIBERATE CHOICE. Every other
corvis tool renders prose for a model to read; the dispatch surface does not,
because its primary consumer is this program. ``tools/board-watch`` exists
next door as the counter-example -- it parses ``task_events``' rendered text,
and its error vocabulary has a member per element of that grammar because each
can move independently. Nothing here needs that: a field is a key.

So the decoding still validates every field rather than trusting the shape.
JSON removes the PARSING failure class, not the CONTRACT one -- a tool that
renamed a field would hand back perfectly well-formed JSON with the wrong keys
in it, and a decoder that read ``value["status"]`` without checking would
carry ``None`` into a state machine.
"""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import JSONValue
from platform_core.members import find_member
from typing_extensions import TypedDict

from fleet.contracts.queue_answer import (
    envelope,
    malformed,
    require_optional_instant,
    require_optional_str,
    require_str,
)
from fleet.contracts.tags import NodeTag


class DispatchStatus(StrEnum):
    """Every status a queue job can be in (MCPs migration 486's CHECK)."""

    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    REFUSED = "refused"
    CANCELLED = "cancelled"


class DispatchCommand(StrEnum):
    """The commands a job may ask for. There is no free-command field.

    ``build-bases`` (MCPs mig 497, board 3c9033ff) runs on the hub itself
    rather than on a node -- the R6 rebuild lane; its execution lives in
    :mod:`fleet.core.rebuild`. ``restart-session`` (MCPs mig 507, board
    ccec3417) is the second hub-local verb, the only one that carries a
    target; :mod:`fleet.core.restart` maps it to one session-audit
    invocation.
    """

    CHECK = "check"
    LINT = "lint"
    TEST = "test"
    BUILD_BASES = "build-bases"
    RESTART_SESSION = "restart-session"
    # MCPs mig 525, board task 1fe89973: the twin of restart-session for a
    # session with no pane; the same hub pins and the same session target.
    REVIVE_SESSION = "revive-session"
    # MCPs mig 526, board task 660964d9: ending a live session on purpose,
    # the graceful verb and its explicit hard successor.
    KILL_SESSION = "kill-session"
    KILL_SESSION_HARD = "kill-session-hard"
    # MCPs mig 564, board task 01f31e4a: compacting a working session that
    # got too big; the same hub pins, target and reason as a kill.
    COMPACT_SESSION = "compact-session"
    # MCPs board task 5aa8ed06: a session's own approved exit. The runner
    # performs it exactly as a graceful kill; the verb is what tells the
    # ledger the session ended itself rather than being ended.
    EXIT_SESSION = "exit-session"


class DispatchLane(StrEnum):
    """The two lanes a runner claims from (MCPs mig 532, board task fd5cabfa A5).

    ``hub`` is the rebuild and the five session verbs, run on the hub by
    ``fleet-agent``; ``node`` is the make targets a fleet node runs on a
    checked-out commit, claimed by ``fleet-node-agent``. The queue
    partitions its claims by lane, which is what stops a revive queueing
    behind a check.
    """

    HUB = "hub"
    NODE = "node"


class ClosingStatus(StrEnum):
    """The terminal statuses a runner may report a job closed with."""

    PASSED = "passed"
    FAILED = "failed"
    REFUSED = "refused"


class DispatchJob(TypedDict):
    """One queue row, as this runner needs it.

    A SUBSET OF THE WIRE SHAPE, deliberately. The tool also reports the
    submitting session's id and cwd, every timestamp, and a computed
    ``reclaimable`` flag; a runner acts on none of them, and decoding fields
    nothing reads would make the contract wider than the dependency. What is
    here is what a decision is made from.

    Attributes:
        job_id: The queue row's id, which every later report names.
        project: Repo-relative project path to build.
        command: Which make target.
        status: Where the job is now.
        requested_node: The node the submitter asked for, or None for "any
            node with capacity".
        node: The node a runner committed to, or None before it has.
        run_id: The fleet ledger's run id, empty until the runner mints one.
            This is the join between the queue and this machine's own records.
        claimed_by: The runner holding it, or None.
        submitted_by: The agent label that enqueued it, and
        session_id: that session's UUID. Carried because the LEDGER row this
            runner writes must name who asked for the work -- a dispatch
            whose provenance was the runner's own label would say only that
            the runner ran something, which is the one fact nobody needs.
        session_target: The session a ``restart-session`` job acts on, and
            None for every other command. The tool renders it as an explicit
            ``null`` on those, so a missing key is a changed contract, not
            an absent target.
        sha: The commit a make-target job checks (MCPs mig 532), forty
            lowercase hex, and None on the hub verbs. The export runner
            fetches exactly this and a closure cites it.
        required_tags: The node capabilities the job requires, from
            :class:`~fleet.contracts.tags.NodeTag`; the queue only hands a
            node-lane runner a job whose tags its node carries, and the
            runner re-checks the value against the registry's declaration
            for the project before it exports.
        task_id: The board task whose thread receives the verdict, or None
            when the submitter named none, in which case the verdict is a
            note addressed to the submitting label.
        claimed_unix: When the claim was taken, whole seconds since the
            epoch, or None while nothing holds the job. The one timestamp
            decoded, because the collect pass decides from it (MCPs board
            task 5a4f9b3e): a claim whose start never reached the queue is
            matched to the run its tick launched by when that run began.
        lease_expires_unix: When the holder's lease runs out, whole seconds
            since the epoch, or None while no lease is set. Decoded because
            the collect pass renews a running job only once its lease was
            last set long enough ago (MCPs board task c1d48330,
            :func:`fleet.cli.node_collected.lease_age`), so a second pass
            seconds after the first writes nothing to the queue.
    """

    job_id: str
    project: str
    command: DispatchCommand
    status: DispatchStatus
    requested_node: str | None
    node: str | None
    run_id: str
    claimed_by: str | None
    submitted_by: str
    session_id: str
    session_target: str | None
    sha: str | None
    required_tags: tuple[NodeTag, ...]
    task_id: str | None
    claimed_unix: int | None
    lease_expires_unix: int | None


def _require_status(row: dict[str, JSONValue], *, answer: str) -> DispatchStatus:
    """Read the status field against the closed vocabulary.

    Args:
        row: The decoded object.
        answer: The whole answer, for the error message.

    Returns:
        The narrowed status.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not one of them.
    """
    value = require_str(row, "status", answer=answer)
    status = find_member(value, DispatchStatus)
    if status is not None:
        return status
    raise malformed(f"status {value!r} is not one of {', '.join(DispatchStatus)}", answer=answer)


def _require_command(row: dict[str, JSONValue], *, answer: str) -> DispatchCommand:
    """Read the command field against the closed vocabulary.

    Args:
        row: The decoded object.
        answer: The whole answer, for the error message.

    Returns:
        The narrowed command.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when it is not one of them.
    """
    value = require_str(row, "command", answer=answer)
    command = find_member(value, DispatchCommand)
    if command is not None:
        return command
    raise malformed(f"command {value!r} is not one of {', '.join(DispatchCommand)}", answer=answer)


def _require_tags(row: dict[str, JSONValue], *, answer: str) -> tuple[NodeTag, ...]:
    """Read the ``requiredTags`` field against the tag vocabulary.

    Args:
        row: The decoded object.
        answer: The whole answer, for the error message.

    Returns:
        The tags, in the queue's order.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the field is absent, not
            an array, or names a tag this fleet does not derive: the queue's
            CHECK and this vocabulary are twins, so a stranger here means
            one of them moved without the other.
    """
    value = row.get("requiredTags")
    if not isinstance(value, list):
        raise malformed(
            f"field 'requiredTags' is {type(value).__name__}, not an array", answer=answer
        )
    tags: list[NodeTag] = []
    for index, entry in enumerate(value):
        matched = find_member(entry, NodeTag) if isinstance(entry, str) else None
        if matched is None:
            raise malformed(
                f"requiredTags[{index}] {entry!r} is not one of {', '.join(NodeTag)}",
                answer=answer,
            )
        tags.append(matched)
    return tuple(tags)


def decode_job(value: JSONValue, *, answer: str) -> DispatchJob:
    """Decode one job object.

    Args:
        value: The object from the answer.
        answer: The whole answer, for the error message.

    Returns:
        The validated job.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when any field is missing or the
            wrong type.
    """
    if not isinstance(value, dict):
        raise malformed(f"a job is {type(value).__name__}, not an object", answer=answer)
    return DispatchJob(
        job_id=require_str(value, "id", answer=answer),
        project=require_str(value, "project", answer=answer),
        command=_require_command(value, answer=answer),
        status=_require_status(value, answer=answer),
        requested_node=require_optional_str(value, "requestedNode", answer=answer),
        node=require_optional_str(value, "node", answer=answer),
        run_id=require_str(value, "runId", answer=answer),
        claimed_by=require_optional_str(value, "claimedBy", answer=answer),
        submitted_by=require_str(value, "submittedBy", answer=answer),
        session_id=require_str(value, "sessionId", answer=answer),
        session_target=require_optional_str(value, "sessionTarget", answer=answer),
        sha=require_optional_str(value, "sha", answer=answer),
        required_tags=_require_tags(value, answer=answer),
        task_id=require_optional_str(value, "taskId", answer=answer),
        claimed_unix=require_optional_instant(value, "claimedAt", answer=answer),
        lease_expires_unix=require_optional_instant(value, "leaseExpiresAt", answer=answer),
    )


def decode_claim(answer: str) -> DispatchJob | None:
    """Decode a ``dispatch_claim`` answer.

    Args:
        answer: The tool's text.

    Returns:
        The claimed job, or None when the queue was empty. An empty queue is
        the OUTCOME OF MOST POLLS and is not an error -- treating it as one
        would make the normal case indistinguishable from a fault in every
        log this runner writes.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` on a shape this cannot read.
    """
    claimed = envelope(answer, "claimed")
    if claimed is None:
        return None
    return decode_job(claimed, answer=answer)


def decode_reported(answer: str) -> DispatchJob:
    """Decode a ``dispatch_report`` or ``dispatch_get`` answer's job.

    Both answer the job under ``job``; ``dispatch_get``'s trail beside it is
    not read.

    Args:
        answer: The tool's text.

    Returns:
        The updated job.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` on a shape this cannot read.
    """
    return decode_job(envelope(answer, "job"), answer=answer)


def decode_listing(answer: str) -> tuple[DispatchJob, ...]:
    """Decode a ``dispatch_list`` answer's jobs.

    The pagination block is deliberately ignored: this runner lists only its
    own held work, which is bounded by how many jobs one runner can hold, and
    a page boundary there would mean the queue had already gone wrong in a
    way a second page would not fix.

    Args:
        answer: The tool's text.

    Returns:
        The jobs, newest first as the tool returns them.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` on a shape this cannot read.
    """
    jobs = envelope(answer, "jobs")
    if not isinstance(jobs, list):
        raise malformed(f"'jobs' is {type(jobs).__name__}, not an array", answer=answer)
    return tuple(decode_job(row, answer=answer) for row in jobs)


class ListingPage(TypedDict):
    """One page of a ``dispatch_list`` answer, with where the next begins.

    Attributes:
        jobs: The page's jobs, newest first as the tool returns them.
        next_offset: The ``offset`` that asks for the following page, or
            None on the last one.
    """

    jobs: tuple[DispatchJob, ...]
    next_offset: int | None


def decode_listing_page(answer: str) -> ListingPage:
    """Decode a ``dispatch_list`` answer that may run past one page.

    :func:`decode_listing` ignores the pagination block because what it
    reads is bounded by what one runner holds. The jobs cancelled under a
    runner are not bounded that way, they accumulate for as long as it
    runs, so a reader of those needs the block, and a page boundary there is
    ordinary rather than a sign the queue went wrong.

    Args:
        answer: The tool's text.

    Returns:
        The page.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` on a shape this cannot read,
            including a ``nextOffset`` that is neither a whole number nor
            null.
    """
    pagination = envelope(answer, "pagination")
    if not isinstance(pagination, dict):
        raise malformed(
            f"'pagination' is {type(pagination).__name__}, not an object", answer=answer
        )
    next_offset = pagination.get("nextOffset", False)
    if next_offset is None:
        return ListingPage(jobs=decode_listing(answer), next_offset=None)
    if isinstance(next_offset, bool) or not isinstance(next_offset, int):
        raise malformed(
            f"'nextOffset' is {type(next_offset).__name__}, not a whole number or null",
            answer=answer,
        )
    return ListingPage(jobs=decode_listing(answer), next_offset=next_offset)


class TrailClaim(TypedDict):
    """One ``claimed`` entry on a job's trail.

    Attributes:
        actor: The runner that took the claim.
        claimed_unix: When, whole seconds since the epoch.
    """

    actor: str
    claimed_unix: int


def decode_trail_claims(answer: str) -> tuple[TrailClaim, ...]:
    """Decode the claims on a ``dispatch_get`` answer's trail: who, and when.

    A claim taken over by another runner overwrites the job's ``claimedBy``
    and ``claimedAt``, so the trail is the only record of every runner that
    ever held it (board task fd402617). Only ``claimed`` entries are read,
    and of them only the two fields a decision is made from.

    Args:
        answer: The tool's text.

    Returns:
        Every claim, oldest first as the trail runs.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the trail is not an array, an
            entry is not an object, or a claim lacks its actor or instant.
    """
    trail = envelope(answer, "trail")
    if not isinstance(trail, list):
        raise malformed(f"'trail' is {type(trail).__name__}, not an array", answer=answer)
    claims: list[TrailClaim] = []
    for entry in trail:
        if not isinstance(entry, dict):
            raise malformed(f"a trail entry is {type(entry).__name__}", answer=answer)
        if require_str(entry, "kind", answer=answer) != "claimed":
            continue
        instant = require_optional_instant(entry, "createdAt", answer=answer)
        if instant is None:
            raise malformed("a claim on the trail has a null 'createdAt'", answer=answer)
        claims.append(
            TrailClaim(actor=require_str(entry, "actor", answer=answer), claimed_unix=instant)
        )
    return tuple(claims)


def encode_job_line(job: DispatchJob) -> str:
    """Render one job as the single line the agent logs.

    Args:
        job: The job.

    Returns:
        The line, without a trailing newline.
    """
    where = job["node"] if job["node"] is not None else (job["requested_node"] or "any node")
    run = f" run={job['run_id']}" if job["run_id"] != "" else ""
    # A restart is not a make; naming a target that does not exist would
    # send the reader to a Makefile for a rule they will not find.
    target = job["session_target"]
    # A check names the commit it checks, abbreviated as git abbreviates,
    # because the line is read beside git log; a hub verb has none.
    at = f" at {job['sha'][:12]}" if job["sha"] is not None else ""
    verb = (
        f"restart session {target}"
        if target is not None
        else f"make {job['command']} {job['project']}{at}"
    )
    return f"{job['job_id']} {job['status']} {verb} @{where}{run}"


__all__ = [
    "ClosingStatus",
    "DispatchCommand",
    "DispatchJob",
    "DispatchLane",
    "DispatchStatus",
    "ListingPage",
    "TrailClaim",
    "decode_claim",
    "decode_job",
    "decode_listing",
    "decode_listing_page",
    "decode_reported",
    "decode_trail_claims",
    "encode_job_line",
]
