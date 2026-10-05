"""Fakes for the corvis dispatch queue, split out of ``conftest``.

Kept beside it rather than in it because ``conftest`` reached the monorepo's
600-line ceiling, and it reached it by holding fakes for TWO different
boundaries: the fleet's own ssh-and-clock seams, and the queue's HTTP one.
Those are two roles, and only one of them grows every time the queue learns a
tool.

Everything here is a FAKE, not a mock. Each implements the Protocol its
production counterpart does and records what it was asked for, so an
assertion is about the request this package actually builds rather than about
a patching library's call-recording API.
"""

from __future__ import annotations

import urllib.error
from collections.abc import Sequence
from datetime import UTC, datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
)
from platform_core.mcp_client import McpCredentials, McpHttpResponse
from platform_core.mcp_testing import DECLARED_FLEET_URL

from fleet.core import queue
from tests.conftest import DEMO_PROJECT


class ToolRefusal:
    """A scripted :class:`FakeQueue` reply that the tool threw, not answered.

    Rendered the way :class:`FakeRefusingQueue` renders every reply, as a
    successful result carrying ``isError: true``, so one script can hold a
    refusal between ordinary answers: a start report refused after a claim
    that succeeded, say (MCPs board task 88b8fe61).

    Attributes:
        message: What the tool would have raised.
    """

    message: str

    def __init__(self, message: str) -> None:
        """Hold the refusal's message.

        Args:
            message: What the tool would have raised.
        """
        self.message = message


class Unanswered:
    """A scripted :class:`FakeQueue` reply that nothing answered at all.

    The call raises what ``urllib`` raised at 11:54:58Z on 2026-10-05 while a
    deploy recreated mcp-fleet, so a test that binds the queue through
    :func:`fleet.core.queue_transport.answering` sees the production
    ``QUEUE_UNANSWERED`` for that one call and answers for the rest (MCPs
    board task 8993c306).

    Attributes:
        error: What the transport raises.
    """

    error: OSError

    def __init__(self) -> None:
        """Hold the refusal."""
        self.error = urllib.error.URLError(
            ConnectionRefusedError(
                10061, "No connection could be made because the target machine actively refused it"
            )
        )


class FakeQueue:
    """A dispatch-queue endpoint that answers from a script and records calls.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`. A FAKE, not
    a mock: it speaks the real JSON-RPC-over-SSE shape the live endpoint
    speaks, so what is asserted is the request this package actually builds
    and the answer it can actually read.

    ``dispatch_tick`` is answered on its own and kept apart, in ``ticks``:
    every node runner tick records itself (MCPs board task 939ec5c7, A4), so
    scripting it into every claim test would bury each one's subject under a
    call it does not assert on; the tests that do read ``ticks``.

    Attributes:
        tools: Every tool name it was asked for, ``dispatch_tick`` aside, in
            order.
        arguments: Every arguments object it was sent, ``dispatch_tick``'s
            aside, in order.
        ticks: Every ``dispatch_tick`` arguments object, in order.
    """

    tools: list[str]
    arguments: list[JSONObject]
    ticks: list[JSONObject]
    _replies: list[str | ToolRefusal | Unanswered]

    def __init__(self, replies: Sequence[str | ToolRefusal | Unanswered]) -> None:
        """Build a queue that will answer with these tool texts in order.

        Args:
            replies: One rendered tool answer, a :class:`ToolRefusal`, or an
                :class:`Unanswered`, per expected call; an unanswered call is
                still recorded. Running out is an error rather than a default:
                a test that made more calls than it declared has changed
                behaviour it did not mean to assert on.
        """
        self.tools = []
        self.arguments = []
        self.ticks = []
        self._replies = list(replies)

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Record the call and answer the next scripted tool result.

        Args:
            url: Absolute URL posted to.
            headers: Every request header.
            body: The encoded JSON-RPC body.
            timeout_seconds: The caller's timeout.

        Returns:
            The next scripted answer, wrapped in the SSE framing.

        Raises:
            AssertionError: If more calls are made than replies were given.
            OSError: The transport's own error, for an :class:`Unanswered` reply.
        """
        envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
        params = narrow_json_to_dict(envelope["params"])
        tool = narrow_json_to_str(params["name"])
        arguments = narrow_json_to_dict(params["arguments"])
        if tool == "dispatch_tick":
            self.ticks.append(arguments)
            return _framed({"content": [{"text": TICK_RECORDED}]})
        self.tools.append(tool)
        self.arguments.append(arguments)
        assert self._replies, f"unscripted queue call: {tool}"
        reply = self._replies.pop(0)
        if isinstance(reply, Unanswered):
            raise reply.error
        result: JSONObject = (
            {"isError": True, "content": [{"text": reply.message}]}
            if isinstance(reply, ToolRefusal)
            else {"content": [{"text": reply}]}
        )
        return _framed(result)


#: When :class:`FakeQueue` says it stored a tick.
TICKED_AT = "2026-10-02T05:00:00.000Z"

#: ``dispatch_tick``'s answer in :class:`FakeQueue`, trimmed to what the
#: runner reads.
TICK_RECORDED = dump_json_str({"tick": {"tickedAt": TICKED_AT}})


#: The identity keys every runner call carries beside its own arguments.
IDENTITY_KEYS = frozenset({"agent", "sessionId", "cwd"})


def tick_body(arguments: JSONObject) -> JSONObject:
    """A recorded ``dispatch_tick`` call without its identity arguments.

    Args:
        arguments: One entry of :attr:`FakeQueue.ticks`.

    Returns:
        The tick as :func:`fleet.contracts.runner_tick.encode_runner_tick`
        wrote it.
    """
    return {key: value for key, value in arguments.items() if key not in IDENTITY_KEYS}


def _framed(result: JSONObject) -> McpHttpResponse:
    """Wrap one tool result in the JSON-RPC envelope and SSE framing.

    Args:
        result: The tool result.

    Returns:
        The response the live endpoint would send.
    """
    payload: JSONObject = {"jsonrpc": "2.0", "id": 1, "result": result}
    return McpHttpResponse(
        status=200,
        content_type="text/event-stream",
        body=f"event: message\ndata: {dump_json_str(payload)}\n\n",
    )


#: Credentials every queue test posts with.
QUEUE_CREDENTIALS = McpCredentials(
    url="http://127.0.0.1:8035/mcp",
    api_key="test-key",
    tenant_id="2e137b5f-0000-4000-8000-000000000000",
)

#: The runner identity every queue test acts under.
RUNNER_IDENTITY: JSONObject = {
    "agent": "fleet-runner-austinpc",
    "sessionId": "33333333-cccc-4ccc-8ccc-333333333333",
    "cwd": "C:/fleet",
}


#: The job id every :func:`queue_job` row carries unless overridden — hoisted
#: so the agent suites can assert on run ids derived from it without
#: restating the literal.
DEFAULT_JOB_ID = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"

#: The commit every :func:`queue_job` check names unless overridden (MCPs
#: mig 532): forty lowercase hex, the shape the queue's pin admits.
DEFAULT_SHA = "4e3c6bc1d9f0a7b2c3e4f5061728394a5b6c7d8e"


def queue_job(**overrides: JSONValue) -> JSONObject:
    """Build one wire-shape job object, as ``dispatch_*`` renders it.

    Every field the decoder reads is present by default, so a test that omits
    one is deliberately testing its absence. The default row is a check, so it
    carries :data:`DEFAULT_SHA`; a hub verb's row passes ``sha=None``.

    Args:
        **overrides: Fields to vary from the defaults.

    Returns:
        The wire object.
    """
    row: JSONObject = {
        "id": DEFAULT_JOB_ID,
        "project": DEMO_PROJECT,
        "command": "check",
        "status": "queued",
        "requestedNode": None,
        "node": None,
        "runId": "",
        "claimedBy": None,
        "submittedBy": "opus-dispatch-0905",
        "sessionId": "11111111-aaaa-4aaa-8aaa-111111111111",
        "sessionTarget": None,
        "sha": DEFAULT_SHA,
        "requiredTags": [],
        "taskId": None,
        "claimedAt": None,
    }
    row.update(overrides)
    return row


def queue_instant(unix: int) -> str:
    """An instant as the queue renders one: JavaScript's ``toISOString``.

    Args:
        unix: Whole seconds since the epoch.

    Returns:
        The rendered instant, in UTC with milliseconds.
    """
    return datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def listing_page(jobs: list[JSONObject], next_offset: int | None) -> str:
    """One page of a ``dispatch_list`` answer, with its pagination block.

    Args:
        jobs: The page's wire rows.
        next_offset: Where the next page begins, or None on the last.

    Returns:
        The rendered answer.
    """
    return dump_json_str({"jobs": jobs, "pagination": {"nextOffset": next_offset}})


def trail_answer(job: JSONObject, claims: list[tuple[str, int]]) -> str:
    """A ``dispatch_get`` answer: the job, and a trail holding these claims.

    Each claim is preceded by the submission and followed by a progress
    entry, as the real trail interleaves them, so a decoder that read any
    entry but a claim as one would be caught.

    Args:
        job: The job's wire row.
        claims: ``(actor, claimed_unix)`` per claim, oldest first.

    Returns:
        The rendered answer.
    """
    trail: list[JSONValue] = [
        {"kind": "submitted", "actor": "opus-dispatch-0905", "createdAt": queue_instant(0)}
    ]
    for actor, claimed_unix in claims:
        trail.append({"kind": "claimed", "actor": actor, "createdAt": queue_instant(claimed_unix)})
        trail.append({"kind": "progress", "actor": actor, "createdAt": queue_instant(claimed_unix)})
    return dump_json_str({"job": job, "trail": trail})


class FakeRefusingQueue:
    """An endpoint whose every answer is a tool that threw.

    A DIFFERENT SHAPE FROM A JSON-RPC ERROR, and that is the point. An MCP
    tool that raises is caught by the SDK and rendered as an ordinary
    successful result carrying ``isError: true`` and the message in a text
    block. A client reading only the protocol-level ``error`` member would
    hand that prose to a caller expecting JSON, which then reports a JSON
    fault for something that was entirely the tool's -- and discards the
    message that said what was wrong.

    Attributes:
        tools: Every tool name it was asked for, in order.
    """

    tools: list[str]

    def __init__(self, message: str) -> None:
        """Build an endpoint that refuses every call with this message.

        Args:
            message: What the tool would have raised.
        """
        self.message = message
        self.tools = []

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Answer the refusal.

        Args:
            url: Absolute URL posted to, unused.
            headers: Every request header, unused.
            body: The encoded JSON-RPC body, read only for the tool name.
            timeout_seconds: The caller's timeout, unused.

        Returns:
            The SSE-framed thrown-tool result.
        """
        envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
        params = narrow_json_to_dict(envelope["params"])
        self.tools.append(narrow_json_to_str(params["name"]))
        payload: JSONObject = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"isError": True, "content": [{"text": self.message}]},
        }
        return McpHttpResponse(
            status=200,
            content_type="text/event-stream",
            body=f"event: message\ndata: {dump_json_str(payload)}\n\n",
        )


class FakeEnv:
    """An environment backed by a dictionary.

    Satisfies :class:`~fleet.core._test_hooks.EnvProtocol`.

    Attributes:
        values: The variables that are set.
    """

    values: dict[str, str]

    def __init__(self, values: dict[str, str]) -> None:
        """Build the environment.

        Args:
            values: The variables that are set.
        """
        self.values = dict(values)

    def __call__(self, name: str) -> str | None:
        """Read a variable, normalising blank to unset.

        The normalisation is part of the Protocol, not a convenience. A fake
        that returned ``""`` where the real reader returns None would share
        the blind spot with the code under test, so the blank-credential case
        would pass here and fail against the live queue.

        Args:
            name: The variable name.

        Returns:
            Its trimmed value, or None when unset or blank.
        """
        raw = self.values.get(name)
        if raw is None:
            return None
        trimmed = raw.strip()
        return trimmed if trimmed != "" else None


def queue_env() -> FakeEnv:
    """The environment every agent test runs under: both secrets, and the
    endpoint named outright.

    Naming the endpoint keeps an agent test off the real stack declaration,
    which :func:`fleet.core.queue.load_credentials` reads when
    ``FLEET_DISPATCH_URL`` is unset; that default has its own tests in
    ``test_queue.py``.

    Returns:
        A fresh environment, so no test sees another's changes.
    """
    return FakeEnv(
        {
            queue.API_KEY_VARIABLE: "test-key",
            queue.TENANT_ID_VARIABLE: "tenant",
            queue.URL_VARIABLE: DECLARED_FLEET_URL,
        }
    )
