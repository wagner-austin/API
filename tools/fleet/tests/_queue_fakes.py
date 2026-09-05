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

from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
)
from platform_core.mcp_client import McpCredentials, McpHttpResponse

from tests.conftest import DEMO_PROJECT


class FakeQueue:
    """A dispatch-queue endpoint that answers from a script and records calls.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`. A FAKE, not
    a mock: it speaks the real JSON-RPC-over-SSE shape the live endpoint
    speaks, so what is asserted is the request this package actually builds
    and the answer it can actually read.

    Attributes:
        tools: Every tool name it was asked for, in order.
        arguments: Every arguments object it was sent, in order.
    """

    tools: list[str]
    arguments: list[JSONObject]
    _replies: list[str]

    def __init__(self, replies: Sequence[str]) -> None:
        """Build a queue that will answer with these tool texts in order.

        Args:
            replies: One rendered tool answer per expected call. Running out
                is an error rather than a default: a test that made more
                calls than it declared has changed behaviour it did not mean
                to assert on.
        """
        self.tools = []
        self.arguments = []
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
        """
        envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
        params = narrow_json_to_dict(envelope["params"])
        self.tools.append(narrow_json_to_str(params["name"]))
        self.arguments.append(narrow_json_to_dict(params["arguments"]))
        assert self._replies, f"unscripted queue call: {self.tools[-1]}"
        payload: JSONObject = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"text": self._replies.pop(0)}]},
        }
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


def queue_job(**overrides: JSONValue) -> JSONObject:
    """Build one wire-shape job object, as ``dispatch_*`` renders it.

    Every field the decoder reads is present by default, so a test that omits
    one is deliberately testing its absence.

    Args:
        **overrides: Fields to vary from the defaults.

    Returns:
        The wire object.
    """
    row: JSONObject = {
        "id": "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
        "project": DEMO_PROJECT,
        "command": "check",
        "status": "queued",
        "requestedNode": None,
        "node": None,
        "runId": "",
        "claimedBy": None,
        "submittedBy": "opus-dispatch-0905",
        "sessionId": "11111111-aaaa-4aaa-8aaa-111111111111",
    }
    row.update(overrides)
    return row


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
