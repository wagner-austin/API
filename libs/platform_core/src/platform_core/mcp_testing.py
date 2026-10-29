"""Test doubles for the MCP transport, for every package that speaks it.

LIFTED ON 2026-09-06, when a THIRD copy was about to be written. The same
poster fake and the same three response builders already existed twice --
in ``libs/platform_core/tests/test_mcp_client.py`` and in
``tools/hpc-wake/tests/conftest.py`` -- with the same fields, the same
scripted-replies design and two slightly different docstrings. ``tools/
fleet-wake`` needed them as well, and a third copy of a fake that encodes the
server's SSE framing is a third place for that framing to go stale.

A SEPARATE MODULE FROM :mod:`platform_core.testing`, which is 533 lines
against this repository's 600-line ceiling, and which holds environment and
httpx fakes that have nothing to do with this transport. Same split
``oauth_testing`` already makes, and for the same reason.

THESE ARE FAKES, NOT MOCKS. :class:`FakeHttpPost` implements
:class:`~platform_core.mcp_client.McpPostProtocol` and records what it was
handed, so an assertion is about the request the client actually built rather
than about a patching library's call-recording API. Running out of scripted
replies RAISES: a caller that made more requests than the test declared has
changed behaviour the test did not mean to assert on, and answering it again
would hide exactly that.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.stack_endpoints import (
    FLEET_SERVICE,
    ROSTER_MACHINES_PATH,
    STACK_ENDPOINTS_PATH,
    STACK_PLACEMENT_PATH,
    TASKBOARD_SERVICE,
)


def sse_body(payload: str) -> str:
    """Wrap a JSON-RPC payload in the framing the server actually sends.

    Args:
        payload: The serialised JSON-RPC body.

    Returns:
        The whole response body, with the ``event:`` and ``data:`` lines.
    """
    return f"event: message\ndata: {payload}\n\n"


def tool_text_body(text: str) -> str:
    """Build a successful ``tools/call`` response carrying one text block.

    Args:
        text: The rendered text the tool returns.

    Returns:
        The whole response body, framed.
    """
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": text}]},
    }
    return sse_body(dump_json_str(payload))


def posted_ok() -> McpHttpResponse:
    """A 200 whose payload is a successful tool call.

    Returns:
        The response, with the content type the server sends.
    """
    return McpHttpResponse(
        status=200,
        body=tool_text_body("posted"),
        content_type=EVENT_STREAM_MEDIA_TYPE,
    )


def sent_arguments(body: bytes) -> JSONObject:
    """Read the tool arguments back out of a recorded request body.

    Args:
        body: The bytes :class:`FakeHttpPost` recorded.

    Returns:
        The ``params.arguments`` object that was sent.

    Raises:
        JSONTypeError: If the body is not the envelope shape, which means the
            client built something other than a ``tools/call``.
    """
    envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
    return narrow_json_to_dict(narrow_json_to_dict(envelope["params"])["arguments"])


class FakeHttpPost:
    """An MCP poster that answers from a script and records every call.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol` exactly --
    same parameter names, same keyword-only split, same return type -- so a
    test exercises the real client against it rather than against a signature
    that merely resembles the real one.

    Attributes:
        urls: Every URL it was given, in order.
        headers: Every header mapping it was given, in order.
        bodies: Every request body it was given, in order.
        timeouts: Every timeout it was given, in order.
    """

    urls: list[str]
    headers: list[dict[str, str]]
    bodies: list[bytes]
    timeouts: list[int]

    def __init__(self, replies: Sequence[McpHttpResponse]) -> None:
        """Build a poster that answers with these responses in order.

        Args:
            replies: One response per expected call, oldest first. Running
                out is an error rather than a default -- see the module
                docstring.
        """
        self.urls = []
        self.headers = []
        self.bodies = []
        self.timeouts = []
        self._replies = list(replies)

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Record one call and answer with the next scripted response.

        Args:
            url: The URL posted to.
            headers: The request headers.
            body: The encoded request body.
            timeout_seconds: The timeout the caller chose.

        Returns:
            The next scripted response.

        Raises:
            AssertionError: When more calls were made than replies scripted,
                naming the URL of the unscripted one.
        """
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.bodies.append(body)
        self.timeouts.append(timeout_seconds)
        if not self._replies:
            raise AssertionError(f"unscripted POST to {url}")
        return self._replies.pop(0)


def announcing_poster(notes: int = 1) -> FakeHttpPost:
    """A poster scripted for a bridge cycle that HAS something to announce.

    ``notes + 1`` replies, not ``notes``, since 2026-09-21: every bridge
    registers itself on the session ledger with a checkin before its first
    announcement, because the board refuses a write from a session no
    ledger surface knows (MCPs mig 514) and a service session is one.
    Shared here rather than copied into each bridge's suite, since all
    three learned the same lesson on the same day.

    Args:
        notes: How many announcements the cycle will make.

    Returns:
        The poster.
    """
    return FakeHttpPost([posted_ok() for _ in range(notes + 1)])


def notes_sent(poster: FakeHttpPost) -> list[JSONObject]:
    """The announcements a cycle posted, without its registering checkin.

    Named rather than spelled ``bodies[1:]`` at each call site: the offset
    is the ledger checkin, and an index nobody explained is how a later
    edit puts an assertion back on the wrong post.

    Args:
        poster: The cycle's scripted poster.

    Returns:
        One decoded arguments object per announcement, in order.
    """
    return [sent_arguments(body) for body in poster.bodies[1:]]


#: The machine :func:`stack_endpoint_files` places every service on, and the
#: roster address it gives that machine: deliberately not any address the
#: real stack uses, so an assertion that finds it proves the url was formed
#: from the files the test supplied.
DECLARED_STACK_HOST = "stack-host"
DECLARED_STACK_ADDRESS = "stack.declared.test"

#: The ports :func:`stack_endpoint_files` declares for the taskboard and fleet-mcp.
DECLARED_TASKBOARD_PORT = 8033
DECLARED_FLEET_PORT = 8035

#: The urls those files form (MCPs board task 50381dd9).
DECLARED_TASKBOARD_URL = f"http://{DECLARED_STACK_ADDRESS}:{DECLARED_TASKBOARD_PORT}/mcp"
DECLARED_FLEET_URL = f"http://{DECLARED_STACK_ADDRESS}:{DECLARED_FLEET_PORT}/mcp"


def stack_endpoint_files() -> dict[pathlib.Path, str]:
    """The files a service's url is formed from, each path to its text.

    Shared rather than written into each suite because several packages
    (``platform_core``, ``ci-wake``, ``hpc3``, ``board-watch``, ``fleet``)
    form the same urls, and the files' shapes are the MCPs repository's
    ``scripts/fleet/stack-endpoints.json`` and ``placement.json`` and this
    repository's roster, one place to follow them when they change.

    Returns:
        The declaration giving the taskboard :data:`DECLARED_TASKBOARD_PORT`
        and fleet-mcp :data:`DECLARED_FLEET_PORT`, the placement putting both
        on :data:`DECLARED_STACK_HOST`, and that machine's roster file giving
        it :data:`DECLARED_STACK_ADDRESS`.
    """
    services = {TASKBOARD_SERVICE: DECLARED_TASKBOARD_PORT, FLEET_SERVICE: DECLARED_FLEET_PORT}
    declaration: JSONObject = {
        "services": {service: {"port": port} for service, port in services.items()}
    }
    placement: JSONObject = {"services": {service: [DECLARED_STACK_HOST] for service in services}}
    machine: JSONObject = {"tailnetIp": DECLARED_STACK_ADDRESS}
    return {
        STACK_ENDPOINTS_PATH: dump_json_str(declaration),
        STACK_PLACEMENT_PATH: dump_json_str(placement),
        ROSTER_MACHINES_PATH / f"{DECLARED_STACK_HOST}.json": dump_json_str(machine),
    }


__all__ = [
    "DECLARED_FLEET_PORT",
    "DECLARED_FLEET_URL",
    "DECLARED_STACK_ADDRESS",
    "DECLARED_STACK_HOST",
    "DECLARED_TASKBOARD_PORT",
    "DECLARED_TASKBOARD_URL",
    "FakeHttpPost",
    "announcing_poster",
    "notes_sent",
    "posted_ok",
    "sent_arguments",
    "sse_body",
    "stack_endpoint_files",
    "tool_text_body",
]
