"""Call one MCP tool over the board's HTTP surface and return its text.

The endpoint speaks JSON-RPC 2.0 and answers with a Server-Sent Events
stream even for a single result, so the body is ``event: message`` followed
by one ``data:`` line carrying the JSON. That is not an option the caller
chose; it is what the server sends when ``Accept`` allows it, and the
``Accept`` header must allow it or the request is rejected outright.

No handshake is required. ``tools/call`` works on a fresh connection with no
prior ``initialize``, verified against the live board on 2026-09-05.
"""

from __future__ import annotations

from typing import Final

from platform_core.error_codes import BoardWatchErrorCode
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

from board_watch import _test_hooks
from board_watch.config import BoardCredentials

#: The SSE field carrying the JSON-RPC payload.
_DATA_PREFIX: Final = "data: "

#: What the endpoint answers with when it accepts the request.
_EVENT_STREAM_MEDIA_TYPE: Final = "text/event-stream"


def _rpc_envelope(tool: str, arguments: JSONObject) -> bytes:
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


def _payload_from_stream(body: str) -> JSONObject:
    """Pull the JSON-RPC payload out of an SSE body.

    Args:
        body: The whole response body.

    Returns:
        The decoded payload object.

    Raises:
        AppError: ``RESPONSE_NOT_EVENT_STREAM`` when no ``data:`` line is
            present, which means the endpoint answered in a shape this
            client does not know how to read.
    """
    for line in body.splitlines():
        if line.startswith(_DATA_PREFIX):
            return narrow_json_to_dict(load_json_str(line[len(_DATA_PREFIX) :]))
    raise AppError(
        code=BoardWatchErrorCode.RESPONSE_NOT_EVENT_STREAM,
        message=(
            "board response carried no 'data:' line; expected a "
            f"{_EVENT_STREAM_MEDIA_TYPE} body from tools/call"
        ),
    )


def _text_from_result(payload: JSONObject) -> str:
    """Join every text block of a tool result.

    Args:
        payload: The decoded JSON-RPC payload.

    Returns:
        The concatenated text content.

    Raises:
        AppError: ``RPC_ERROR`` when the payload carries an ``error`` member
            instead of a result, which is the tool itself failing rather than
            the transport.
    """
    error: JSONValue | None = payload.get("error")
    if error is not None:
        raise AppError(
            code=BoardWatchErrorCode.RPC_ERROR,
            message=f"board tool call returned an error: {dump_json_str(error)}",
        )
    result = narrow_json_to_dict(payload["result"])
    blocks = narrow_json_to_list(result["content"])
    return "".join(narrow_json_to_str(narrow_json_to_dict(block)["text"]) for block in blocks)


def call_tool(
    credentials: BoardCredentials,
    tool: str,
    arguments: JSONObject,
    *,
    timeout_seconds: int = _test_hooks.DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Call one MCP tool and return its rendered text.

    Args:
        credentials: Endpoint and both headers.
        tool: The MCP tool name.
        arguments: Its arguments object.
        timeout_seconds: How long to wait for the whole exchange.

    Returns:
        The tool's text content, with every block concatenated.

    Raises:
        AppError: ``HTTP_STATUS`` when the endpoint refused, which for a
            rotated key is a 401 and is the ordinary failure here;
            ``RESPONSE_NOT_EVENT_STREAM`` or ``RPC_ERROR`` from the decoders.
    """
    response = _test_hooks.http_post(
        credentials["url"],
        headers={
            "x-api-key": credentials["api_key"],
            "X-Tenant-Id": credentials["tenant_id"],
            "Content-Type": "application/json",
            "Accept": f"application/json, {_EVENT_STREAM_MEDIA_TYPE}",
        },
        body=_rpc_envelope(tool, arguments),
        timeout_seconds=timeout_seconds,
    )
    if response["status"] != 200:
        raise AppError(
            code=BoardWatchErrorCode.HTTP_STATUS,
            message=(
                f"board endpoint answered HTTP {response['status']} for tool "
                f"{tool!r}: {response['body'].strip()}"
            ),
        )
    return _text_from_result(_payload_from_stream(response["body"]))
