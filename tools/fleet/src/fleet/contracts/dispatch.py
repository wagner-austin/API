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

from typing import Final, Literal

from platform_core.error_codes_tooling import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, load_json_str
from typing_extensions import TypedDict

from fleet.contracts.tags import NODE_TAGS, NodeTag

#: Every status a queue job can be in (MCPs migration 486's CHECK).
DISPATCH_STATUSES: Final = (
    "queued",
    "claimed",
    "running",
    "passed",
    "failed",
    "refused",
    "cancelled",
)

#: Narrow type for a queue job's status.
DispatchStatus = Literal["queued", "claimed", "running", "passed", "failed", "refused", "cancelled"]

#: The commands a job may ask for. There is no free-command field.
#: ``build-bases`` (MCPs mig 497, board 3c9033ff) runs on the hub itself
#: rather than on a node -- the R6 rebuild lane; its execution lives in
#: :mod:`fleet.core.rebuild`. ``restart-session`` (MCPs mig 507, board
#: ccec3417) is the second hub-local verb, the only one that carries a
#: target; :mod:`fleet.core.restart` maps it to one session-audit
#: invocation.
DISPATCH_COMMANDS: Final = (
    "check",
    "lint",
    "test",
    "build-bases",
    "restart-session",
    # MCPs mig 525, board task 1fe89973: the twin of restart-session for a
    # session with no pane; the same hub pins and the same session target.
    "revive-session",
    # MCPs mig 526, board task 660964d9: ending a live session on purpose,
    # the graceful verb and its explicit hard successor.
    "kill-session",
    "kill-session-hard",
)

#: Narrow type for a queue job's command.
DispatchCommand = Literal[
    "check",
    "lint",
    "test",
    "build-bases",
    "restart-session",
    "revive-session",
    "kill-session",
    "kill-session-hard",
]

#: The two lanes a runner claims from (MCPs mig 532, board task fd5cabfa
#: A5). ``hub`` is the rebuild and the four session verbs, run on the hub
#: by ``fleet-agent``; ``node`` is the make targets a fleet node runs on a
#: checked-out commit, claimed by ``fleet-node-agent``. The queue partitions
#: its claims by lane, which is what stops a revive queueing behind a check.
DISPATCH_LANES: Final = ("hub", "node")

#: Narrow type for a runner's lane.
DispatchLane = Literal["hub", "node"]

#: The terminal statuses a runner may report.
CLOSING_STATUSES: Final = ("passed", "failed", "refused")

#: Narrow type for what a runner closes a job with.
ClosingStatus = Literal["passed", "failed", "refused"]


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
            :data:`~fleet.contracts.tags.NODE_TAGS`; the queue only hands a
            node-lane runner a job whose tags its node carries, and the
            runner re-checks the value against the registry's declaration
            for the project before it exports.
        task_id: The board task whose thread receives the verdict, or None
            when the submitter named none, in which case the verdict is a
            note addressed to the submitting label.
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


def _malformed(detail: str, *, answer: str) -> AppError[FleetErrorCode]:
    """Build the refusal for an answer that is not the documented shape.

    Args:
        detail: What was wrong, specifically.
        answer: The whole answer, echoed so the reader sees what arrived.

    Returns:
        The error to raise.
    """
    return AppError(
        code=FleetErrorCode.QUEUE_ANSWER_MALFORMED,
        message=(
            f"the dispatch queue answered a shape this runner cannot read: "
            f"{detail}. The tool's contract is JSON with named keys, so this "
            f"means the contract moved and the fix is in the MCPs repo, not "
            f"here. Received: {answer[:400]}"
        ),
    )


def _require_str(row: dict[str, JSONValue], key: str, *, answer: str) -> str:
    """Read one string field.

    Args:
        row: The decoded object.
        key: The field name.
        answer: The whole answer, for the error message.

    Returns:
        The value.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when absent or not a string.
    """
    value = row.get(key)
    if not isinstance(value, str):
        raise _malformed(f"field {key!r} is {type(value).__name__}, not a string", answer=answer)
    return value


def _require_optional_str(row: dict[str, JSONValue], key: str, *, answer: str) -> str | None:
    """Read one nullable string field.

    ``null`` and a missing key are NOT the same here, and the difference is
    checked: the tool renders every absent value as an explicit ``null``
    precisely so a consumer can tell "no node yet" from "the field is gone".

    Args:
        row: The decoded object.
        key: The field name.
        answer: The whole answer, for the error message.

    Returns:
        The value, or None when the field is present and null.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when absent, or present and
            neither a string nor null.
    """
    if key not in row:
        raise _malformed(f"field {key!r} is missing", answer=answer)
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise _malformed(
            f"field {key!r} is {type(value).__name__}, not a string or null", answer=answer
        )
    return value


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
    value = _require_str(row, "status", answer=answer)
    for status in DISPATCH_STATUSES:
        if value == status:
            return status
    raise _malformed(
        f"status {value!r} is not one of {', '.join(DISPATCH_STATUSES)}", answer=answer
    )


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
    value = _require_str(row, "command", answer=answer)
    for command in DISPATCH_COMMANDS:
        if value == command:
            return command
    raise _malformed(
        f"command {value!r} is not one of {', '.join(DISPATCH_COMMANDS)}", answer=answer
    )


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
        raise _malformed(
            f"field 'requiredTags' is {type(value).__name__}, not an array", answer=answer
        )
    tags: list[NodeTag] = []
    for index, entry in enumerate(value):
        matched: NodeTag | None = None
        for tag in NODE_TAGS:
            if entry == tag:
                matched = tag
        if matched is None:
            raise _malformed(
                f"requiredTags[{index}] {entry!r} is not one of {', '.join(NODE_TAGS)}",
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
        raise _malformed(f"a job is {type(value).__name__}, not an object", answer=answer)
    return DispatchJob(
        job_id=_require_str(value, "id", answer=answer),
        project=_require_str(value, "project", answer=answer),
        command=_require_command(value, answer=answer),
        status=_require_status(value, answer=answer),
        requested_node=_require_optional_str(value, "requestedNode", answer=answer),
        node=_require_optional_str(value, "node", answer=answer),
        run_id=_require_str(value, "runId", answer=answer),
        claimed_by=_require_optional_str(value, "claimedBy", answer=answer),
        submitted_by=_require_str(value, "submittedBy", answer=answer),
        session_id=_require_str(value, "sessionId", answer=answer),
        session_target=_require_optional_str(value, "sessionTarget", answer=answer),
        sha=_require_optional_str(value, "sha", answer=answer),
        required_tags=_require_tags(value, answer=answer),
        task_id=_require_optional_str(value, "taskId", answer=answer),
    )


def _envelope(answer: str, key: str) -> JSONValue:
    """Pull one named member out of a tool answer.

    Args:
        answer: The tool's whole text.
        key: The member to read.

    Returns:
        Its value.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the answer is not a JSON
            object or does not carry the member. Not an ``InvalidJsonError``:
            a caller here cannot act on "the JSON was bad" any differently
            than on "the JSON was fine and had the wrong keys", and both mean
            the same thing -- the tool changed.
    """
    body = load_json_str(answer)
    if not isinstance(body, dict):
        raise _malformed(f"the answer is {type(body).__name__}, not an object", answer=answer)
    if key not in body:
        raise _malformed(f"the answer has no {key!r} member", answer=answer)
    return body[key]


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
    claimed = _envelope(answer, "claimed")
    if claimed is None:
        return None
    return decode_job(claimed, answer=answer)


def decode_reported(answer: str) -> DispatchJob:
    """Decode a ``dispatch_report`` answer.

    Args:
        answer: The tool's text.

    Returns:
        The updated job.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` on a shape this cannot read.
    """
    return decode_job(_envelope(answer, "job"), answer=answer)


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
    jobs = _envelope(answer, "jobs")
    if not isinstance(jobs, list):
        raise _malformed(f"'jobs' is {type(jobs).__name__}, not an array", answer=answer)
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
    pagination = _envelope(answer, "pagination")
    if not isinstance(pagination, dict):
        raise _malformed(
            f"'pagination' is {type(pagination).__name__}, not an object", answer=answer
        )
    next_offset = pagination.get("nextOffset", False)
    if next_offset is None:
        return ListingPage(jobs=decode_listing(answer), next_offset=None)
    if isinstance(next_offset, bool) or not isinstance(next_offset, int):
        raise _malformed(
            f"'nextOffset' is {type(next_offset).__name__}, not a whole number or null",
            answer=answer,
        )
    return ListingPage(jobs=decode_listing(answer), next_offset=next_offset)


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
    "CLOSING_STATUSES",
    "DISPATCH_COMMANDS",
    "DISPATCH_LANES",
    "DISPATCH_STATUSES",
    "ClosingStatus",
    "DispatchCommand",
    "DispatchJob",
    "DispatchLane",
    "DispatchStatus",
    "ListingPage",
    "decode_claim",
    "decode_job",
    "decode_listing",
    "decode_listing_page",
    "decode_reported",
    "encode_job_line",
]
