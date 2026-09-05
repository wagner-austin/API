"""Calling one MCP tool over HTTP, for every package in the monorepo that does.

LIFTED OUT OF ``tools/board-watch`` ON 2026-09-05, when a second package needed
the same thing. It was never board-specific: the endpoint speaks JSON-RPC 2.0,
answers with a Server-Sent Events stream even for a single result, and requires
two headers. None of that is a property of the taskboard, and a copy of it in
each caller would be a copy of a transport quirk that has already cost this
workspace one silent failure -- the watcher whose cursor regex matched nothing,
forever, while its whole suite passed.

THREE FACTS ABOUT THE TRANSPORT, all verified against the live server rather
than read off a spec:

1. The response body is ``event: message`` followed by one ``data:`` line
   carrying the JSON. That is not an option a caller chose; it is what the
   server sends when ``Accept`` allows it, and ``Accept`` MUST allow it or the
   request is rejected outright.
2. No handshake is required. ``tools/call`` works on a fresh connection with no
   prior ``initialize`` -- verified 2026-09-05.
3. The server is STATELESS PER POST (``mcp-shared/src/session.ts``: "Creates a
   new server and transport per POST request"). There is no connection to hold
   open, so there is no long-poll and no server-initiated push to wait for. A
   caller that wants to know when something changes must poll, and this
   function is one poll.

THE HTTP SEAM IS A PARAMETER, NOT A MODULE HOOK. Each consuming package already
owns a ``_test_hooks`` module with its own poster, and a lib that reached for a
hook of its own would give every caller two seams to keep in agreement. Passing
the poster in keeps the injection where the package's other injection already
is, and keeps this module pure.
"""

from __future__ import annotations

import http.client
import urllib.request
from typing import Final, Protocol, TypedDict

from platform_core.error_codes import McpClientErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    narrow_json_to_str,
)

#: The SSE field carrying the JSON-RPC payload.
_DATA_PREFIX: Final = "data: "

#: What the endpoint answers with when it accepts the request.
EVENT_STREAM_MEDIA_TYPE: Final = "text/event-stream"

#: How long a single call may take before it is abandoned.
DEFAULT_MCP_TIMEOUT_SECONDS: Final = 20


class McpHttpResponse(TypedDict):
    """What a POST to an MCP endpoint answered.

    Attributes:
        status: The HTTP status line's code.
        content_type: The ``Content-Type`` header, lowercased, or the empty
            string when the response carried none.
        body: The decoded response body.
    """

    status: int
    content_type: str
    body: str


class McpPostProtocol(Protocol):
    """Post a JSON body to an MCP endpoint and read the whole response.

    Implementations MUST return a non-2xx response rather than raising: a 401
    from a rotated key is the ordinary way this fails, and the caller's job is
    to attach a code to it, not to catch an exception.
    """

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Perform the request.

        Args:
            url: Absolute URL to post to.
            headers: Every request header, already complete.
            body: The encoded request body.
            timeout_seconds: How long to wait for the whole exchange.

        Returns:
            The response.
        """
        ...


class McpCredentials(TypedDict):
    """Everything one MCP call needs.

    Attributes:
        url: Absolute URL of the MCP endpoint.
        api_key: The value for the ``x-api-key`` header.
        tenant_id: The value for the ``X-Tenant-Id`` header. Without it the
            server answers ``No tenant context`` rather than refusing the
            request, so a missing tenant looks like a tool failure.
    """

    url: str
    api_key: str
    tenant_id: str


class _ResponseHeaders(Protocol):
    """The one header lookup this module makes."""

    def get(self, name: str, failobj: str) -> str:
        """Read a header.

        Args:
            name: The header name.
            failobj: What to answer when the header is absent.

        Returns:
            The header value, or ``failobj``.
        """
        ...


class _UrlResponse(Protocol):
    """The four members this module uses from an opened URL.

    A Protocol rather than :class:`http.client.HTTPResponse` because
    ``OpenerDirector.open`` is typed loosely enough that everything read off
    its result would otherwise be untyped. Annotating the assignment with this
    narrows it at the boundary, which is where parsing belongs.
    """

    status: int
    headers: _ResponseHeaders

    def read(self) -> bytes:
        """Read the whole body.

        Returns:
            The body bytes.
        """
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class _PassThroughErrorProcessor(urllib.request.HTTPErrorProcessor):
    """An error processor that hands back non-2xx responses instead of raising.

    ``urllib``'s default turns a 401 into a raised
    :class:`urllib.error.HTTPError`, which would force :func:`call_mcp_tool` to
    catch an exception in order to read a status code it needs. The status is
    not an exceptional condition here -- a rotated API key is the ordinary way
    this fails, and the caller's job is to attach
    :attr:`McpClientErrorCode.HTTP_STATUS` to it.

    Replacing the processor removes the exception at its source rather than
    handling it downstream, so no ``except`` appears anywhere in this path.
    """

    def http_response(
        self, request: urllib.request.Request, response: http.client.HTTPResponse
    ) -> http.client.HTTPResponse:
        """Return the response unchanged, whatever its status.

        Args:
            request: The request that produced it, unused.
            response: The response as received.

        Returns:
            That same response.
        """
        return response

    https_response = http_response


def urllib_mcp_post(
    url: str,
    *,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: int,
) -> McpHttpResponse:
    """Post with the standard library -- the production :class:`McpPostProtocol`.

    ``urllib`` rather than ``httpx`` because an MCP call from a CLI is a single
    short-lived POST, usually to loopback. :mod:`platform_core.http_client`
    exists for services that hold a client across many requests and need
    transports injected; borrowing it here would add a dependency to buy
    nothing.

    Lives beside the client rather than in each caller's ``_test_hooks``: the
    SEAM belongs to the package (production binds this, a test binds a fake),
    but the implementation behind it is the same everywhere, down to the
    error-processor replacement above.

    Args:
        url: Absolute URL to post to.
        headers: Every request header, already complete.
        body: The encoded request body.
        timeout_seconds: How long to wait for the whole exchange.

    Returns:
        The response, whatever its status.
    """
    request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_PassThroughErrorProcessor)
    # Annotated at the assignment so the Protocol, not ``open``'s loose return
    # type, is what the rest of this function is checked against.
    response: _UrlResponse = opener.open(request, timeout=timeout_seconds)
    answer = McpHttpResponse(
        status=response.status,
        content_type=response.headers.get("Content-Type", "").lower(),
        body=response.read().decode("utf-8"),
    )
    response.close()
    return answer


def rpc_envelope(tool: str, arguments: JSONObject) -> bytes:
    """Build the JSON-RPC request body for one tool call.

    Args:
        tool: The MCP tool name.
        arguments: Its arguments object.

    Returns:
        The encoded body.
    """
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    return dump_json_str(payload).encode("utf-8")


def payload_from_stream(body: str) -> JSONObject:
    """Pull the JSON-RPC payload out of an SSE body.

    Args:
        body: The whole response body.

    Returns:
        The decoded payload object.

    Raises:
        AppError: ``RESPONSE_NOT_EVENT_STREAM`` when no ``data:`` line is
            present, which means the endpoint answered in a shape this client
            does not know how to read.
    """
    for line in body.splitlines():
        if line.startswith(_DATA_PREFIX):
            return narrow_json_to_dict(load_json_str(line[len(_DATA_PREFIX) :]))
    raise AppError(
        code=McpClientErrorCode.RESPONSE_NOT_EVENT_STREAM,
        message=(
            "MCP response carried no 'data:' line; expected a "
            f"{EVENT_STREAM_MEDIA_TYPE} body from tools/call"
        ),
    )


def text_from_result(payload: JSONObject) -> str:
    """Join every text block of a tool result.

    TWO WAYS A TOOL CALL FAILS, AND BOTH ARRIVE HERE. A JSON-RPC ``error``
    member is the protocol layer refusing -- an unknown tool, a schema
    violation. But a tool that THROWS is not a protocol error at all: the MCP
    SDK catches it and answers a perfectly ordinary result with
    ``isError: true`` and the message in a text block. Both are the tool
    failing, both raise ``RPC_ERROR``, and the message carries the tool's own
    text either way.

    Reading only the ``error`` member is the mistake that hides the second
    kind: a caller expecting JSON would receive the refusal's PROSE, try to
    parse it, and report "invalid JSON payload" -- which names the transport
    for a fault that was entirely the tool's, and throws away the message
    that said what was actually wrong.

    Args:
        payload: The decoded JSON-RPC payload.

    Returns:
        The concatenated text content.

    Raises:
        AppError: ``RPC_ERROR`` when the payload carries an ``error`` member,
            or a result flagged ``isError``.
    """
    error: JSONValue | None = payload.get("error")
    if error is not None:
        raise AppError(
            code=McpClientErrorCode.RPC_ERROR,
            message=f"MCP tool call returned an error: {dump_json_str(error)}",
        )
    result = narrow_json_to_dict(payload["result"])
    blocks = narrow_json_to_list(result["content"])
    text = "".join(narrow_json_to_str(narrow_json_to_dict(block)["text"]) for block in blocks)
    if result.get("isError") is True:
        raise AppError(
            code=McpClientErrorCode.RPC_ERROR,
            message=f"MCP tool reported a failure: {text}",
        )
    return text


def call_mcp_tool(
    post: McpPostProtocol,
    credentials: McpCredentials,
    tool: str,
    arguments: JSONObject,
    *,
    timeout_seconds: int = DEFAULT_MCP_TIMEOUT_SECONDS,
) -> str:
    """Call one MCP tool and return its rendered text.

    Args:
        post: The caller's HTTP seam, from its own ``_test_hooks``.
        credentials: Endpoint and both headers.
        tool: The MCP tool name.
        arguments: Its arguments object.
        timeout_seconds: How long to wait for the whole exchange.

    Returns:
        The tool's text content, with every block concatenated. Whether that
        text is prose or JSON is the TOOL's contract, not this function's --
        ``task_events`` answers in prose and ``dispatch_claim`` answers in
        JSON, and both arrive here the same way.

    Raises:
        AppError: ``HTTP_STATUS`` when the endpoint refused, which for a
            rotated key is a 401 and is the ordinary failure here;
            ``RESPONSE_NOT_EVENT_STREAM`` or ``RPC_ERROR`` from the decoders.
    """
    response = post(
        credentials["url"],
        headers={
            "x-api-key": credentials["api_key"],
            "X-Tenant-Id": credentials["tenant_id"],
            "Content-Type": "application/json",
            "Accept": f"application/json, {EVENT_STREAM_MEDIA_TYPE}",
        },
        body=rpc_envelope(tool, arguments),
        timeout_seconds=timeout_seconds,
    )
    if response["status"] != 200:
        raise AppError(
            code=McpClientErrorCode.HTTP_STATUS,
            message=(
                f"MCP endpoint answered HTTP {response['status']} for tool "
                f"{tool!r}: {response['body'].strip()}"
            ),
        )
    return text_from_result(payload_from_stream(response["body"]))
