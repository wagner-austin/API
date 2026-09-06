"""One MCP call: what goes on the wire, and every way the answer can fail.

LIFTED FROM ``tools/board-watch`` ON 2026-09-05 with the module it tests. The
fake-poster tests assert what the CLIENT asks for; the two at the end run the
real poster against a real socket, because nothing a fake answers would notice
if it sent no headers or raised on a 401.

THE 401 CASE IS WHY THE SOCKET TESTS EXIST. ``urllib`` raises on a non-2xx by
default, and a poster that raised would turn the ordinary rotated-key failure
into a traceback with no status in it. The pass-through error processor is what
prevents that, and only a real server answering 401 proves it works.

The server writes HTTP by hand on a :class:`socketserver.StreamRequestHandler`
rather than subclassing :class:`http.server.BaseHTTPRequestHandler`. That base
class requires a method named ``do_POST``, which no lowercase-name rule will
accept on a definition, and suppressing a rule to satisfy a base class is not a
trade this repository makes. Writing the status line directly needs no
suppression -- and since these tests are about status codes, an explicit status
line is a gain rather than a cost.
"""

from __future__ import annotations

import socketserver
import threading
from collections.abc import Generator
from typing import Final

import pytest

from platform_core.error_codes_tooling import McpClientErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.mcp_client import (
    DEFAULT_MCP_TIMEOUT_SECONDS,
    McpCredentials,
    McpHttpResponse,
    call_mcp_tool,
    urllib_mcp_post,
)
from platform_core.mcp_testing import (
    FakeHttpPost,
    sent_arguments,
    sse_body,
    tool_text_body,
)

#: The credentials every client test posts with.
TEST_CREDENTIALS: Final = McpCredentials(
    url="http://127.0.0.1:8033/mcp",
    api_key="test-key",
    tenant_id="2e137b5f-0000-4000-8000-000000000000",
)

#: Body the socket handler answers with, echoing what it was sent.
_TEMPLATE = '{{"seen":"{body}","key":"{key}"}}'


def rpc_body(payload: JSONObject) -> str:
    """Serialise a JSON-RPC payload and frame it as the server sends it.

    Composes with :func:`~platform_core.mcp_testing.sse_body` rather than
    repeating the framing: that function owns what the wire looks like, this
    one owns turning an object into the string it frames. The tests below
    build malformed payloads deliberately, which is why this takes an object
    the caller can misshape rather than a finished body.

    Args:
        payload: The payload object.

    Returns:
        The whole response body.
    """
    return sse_body(dump_json_str(payload))


def ok(body: str) -> McpHttpResponse:
    """A 200 carrying an event-stream body.

    Args:
        body: The response body.

    Returns:
        The response.
    """
    return McpHttpResponse(status=200, content_type="text/event-stream", body=body)


def refused(status: int, body: str) -> McpHttpResponse:
    """A non-200 answer.

    Args:
        status: The status code.
        body: The response body.

    Returns:
        The response.
    """
    return McpHttpResponse(status=status, content_type="application/json", body=body)


def test_posts_a_jsonrpc_tool_call_with_both_headers() -> None:
    """The request shape is the contract the endpoint accepts.

    Both headers matter and neither is optional: without ``x-api-key`` the
    endpoint answers 401, and without ``X-Tenant-Id`` it answers a tenant
    error from inside the tool. Both were found by making the call.
    """
    poster = FakeHttpPost([ok(tool_text_body("rendered"))])

    assert call_mcp_tool(poster, TEST_CREDENTIALS, "task_events", {"limit": 5}) == "rendered"
    assert poster.urls == [TEST_CREDENTIALS["url"]]
    headers = poster.headers[0]
    assert headers["x-api-key"] == TEST_CREDENTIALS["api_key"]
    assert headers["X-Tenant-Id"] == TEST_CREDENTIALS["tenant_id"]
    assert "text/event-stream" in headers["Accept"]
    envelope = narrow_json_to_dict(load_json_str(poster.bodies[0].decode("utf-8")))
    assert envelope["method"] == "tools/call"
    assert narrow_json_to_dict(envelope["params"])["name"] == "task_events"
    assert sent_arguments(poster.bodies[0]) == {"limit": 5}


def test_joins_every_text_block() -> None:
    """A result may carry more than one block and all of it is the answer."""
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"text": "first "}, {"text": "second"}]},
    }

    assert (
        call_mcp_tool(FakeHttpPost([ok(rpc_body(payload))]), TEST_CREDENTIALS, "task_events", {})
        == "first second"
    )


def test_carries_a_json_answer_through_unchanged() -> None:
    """The tool decides prose or JSON; this client only concatenates blocks.

    ``task_events`` answers in prose and ``dispatch_claim`` answers in JSON,
    and both arrive the same way. A client that "helpfully" parsed would
    break one of them.
    """
    body = tool_text_body('{"claimed": null}')

    answer = call_mcp_tool(FakeHttpPost([ok(body)]), TEST_CREDENTIALS, "dispatch_claim", {})

    assert narrow_json_to_dict(load_json_str(answer)) == {"claimed": None}


def test_a_refused_status_raises_with_the_status_in_the_message() -> None:
    """A rotated key is a 401 and is the ordinary way this fails."""
    poster = FakeHttpPost([refused(401, '{"error":"Unauthorized"}')])

    with pytest.raises(AppError) as raised:
        call_mcp_tool(poster, TEST_CREDENTIALS, "task_events", {})

    assert raised.value.code is McpClientErrorCode.HTTP_STATUS
    assert "401" in raised.value.message
    assert "Unauthorized" in raised.value.message


def test_a_body_with_no_data_line_raises_its_own_code() -> None:
    """An endpoint answering in a shape this client cannot read is distinct
    from one answering an error, and the two send a reader different ways."""
    poster = FakeHttpPost([ok("event: message\n\n")])

    with pytest.raises(AppError) as raised:
        call_mcp_tool(poster, TEST_CREDENTIALS, "task_events", {})

    assert raised.value.code is McpClientErrorCode.RESPONSE_NOT_EVENT_STREAM


def test_a_jsonrpc_error_raises_its_own_code() -> None:
    """The transport succeeded and the tool failed; that is not an HTTP fault."""
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32602, "message": "bad cursor"},
    }
    poster = FakeHttpPost([ok(rpc_body(payload))])

    with pytest.raises(AppError) as raised:
        call_mcp_tool(poster, TEST_CREDENTIALS, "task_events", {})

    assert raised.value.code is McpClientErrorCode.RPC_ERROR
    assert "bad cursor" in raised.value.message


def test_a_tool_that_threw_raises_rather_than_returning_its_prose() -> None:
    """The failure mode a caller expecting JSON would otherwise misread.

    An MCP tool that throws is NOT a JSON-RPC error: the SDK catches it and
    answers an ordinary result with ``isError: true`` and the message in a
    text block. A client that read only the ``error`` member would hand that
    prose back to a caller expecting JSON, which would report "invalid JSON
    payload" -- naming the transport for a fault that was the tool's, and
    discarding the message that said what was wrong.
    """
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "isError": True,
            "content": [{"text": "dispatch_report failed: DISPATCH_NOT_CLAIMANT: job aaaa"}],
        },
    }
    poster = FakeHttpPost([ok(rpc_body(payload))])

    with pytest.raises(AppError) as raised:
        call_mcp_tool(poster, TEST_CREDENTIALS, "dispatch_report", {})

    assert raised.value.code is McpClientErrorCode.RPC_ERROR
    assert "DISPATCH_NOT_CLAIMANT" in raised.value.message


def test_a_result_not_flagged_is_returned_normally() -> None:
    """The flag is read as an explicit True, never as truthiness: a result
    carrying ``isError: false`` is a success and must not be raised on."""
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"isError": False, "content": [{"text": "fine"}]},
    }

    assert (
        call_mcp_tool(FakeHttpPost([ok(rpc_body(payload))]), TEST_CREDENTIALS, "task_events", {})
        == "fine"
    )


def test_the_timeout_reaches_the_transport() -> None:
    """A caller-set timeout has to arrive at the socket to mean anything."""
    poster = FakeHttpPost([ok(tool_text_body("x")), ok(tool_text_body("x"))])

    call_mcp_tool(poster, TEST_CREDENTIALS, "task_events", {}, timeout_seconds=3)
    call_mcp_tool(poster, TEST_CREDENTIALS, "task_events", {})

    assert poster.timeouts == [3, DEFAULT_MCP_TIMEOUT_SECONDS]


class _Handler(socketserver.StreamRequestHandler):
    """Answers one HTTP request, echoing the body and the api-key header."""

    def handle(self) -> None:
        """Read one request and write one response.

        The path decides the status: ``/ok`` answers 200 and anything else
        answers 401, so a single server covers both branches of the poster.
        """
        request_line = self.rfile.readline().decode("utf-8").strip()
        parts = request_line.split(" ")
        path = parts[1] if len(parts) > 1 else ""
        length = 0
        api_key = ""
        while True:
            raw = self.rfile.readline().decode("utf-8").strip()
            if raw == "":
                break
            name, _, value = raw.partition(":")
            if name.lower() == "content-length":
                length = int(value.strip())
            if name.lower() == "x-api-key":
                api_key = value.strip()
        body = self.rfile.read(length).decode("utf-8")
        payload = _TEMPLATE.format(body=body, key=api_key).encode("utf-8")
        status = "200 OK" if path == "/ok" else "401 Unauthorized"
        self.wfile.write(f"HTTP/1.1 {status}\r\n".encode())
        self.wfile.write(b"Content-Type: application/json; charset=utf-8\r\n")
        self.wfile.write(f"Content-Length: {len(payload)}\r\n".encode())
        self.wfile.write(b"Connection: close\r\n\r\n")
        self.wfile.write(payload)


@pytest.fixture(name="server_url")
def _server_url() -> Generator[str, None, None]:
    """Run a real HTTP server on a loopback port for the duration of a test.

    Yields:
        The base URL, with no trailing slash.
    """
    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port: int = server.socket.getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_the_real_poster_sends_headers_and_body(server_url: str) -> None:
    """A poster that dropped either would fail only against a live server."""
    response = urllib_mcp_post(
        f"{server_url}/ok",
        headers={"x-api-key": "secret-value", "Content-Type": "application/json"},
        body=b"payload-bytes",
        timeout_seconds=5,
    )

    assert response["status"] == 200
    assert "payload-bytes" in response["body"]
    assert "secret-value" in response["body"]
    assert response["content_type"] == "application/json; charset=utf-8"


def test_the_real_poster_returns_a_401_instead_of_raising(server_url: str) -> None:
    """The rotated-key case, which is how this fails in practice.

    Without the pass-through error processor ``urllib`` raises here, and the
    caller could not attach ``HTTP_STATUS`` to a status it never saw.
    """
    response = urllib_mcp_post(
        f"{server_url}/refused",
        headers={"x-api-key": "stale"},
        body=b"{}",
        timeout_seconds=5,
    )

    assert response["status"] == 401


__all__ = [
    "TEST_CREDENTIALS",
    "ok",
    "refused",
    "rpc_body",
    "test_a_body_with_no_data_line_raises_its_own_code",
    "test_a_jsonrpc_error_raises_its_own_code",
    "test_a_refused_status_raises_with_the_status_in_the_message",
    "test_a_result_not_flagged_is_returned_normally",
    "test_a_tool_that_threw_raises_rather_than_returning_its_prose",
    "test_carries_a_json_answer_through_unchanged",
    "test_joins_every_text_block",
    "test_posts_a_jsonrpc_tool_call_with_both_headers",
    "test_the_real_poster_returns_a_401_instead_of_raising",
    "test_the_real_poster_sends_headers_and_body",
    "test_the_timeout_reaches_the_transport",
]
