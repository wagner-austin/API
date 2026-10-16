"""A queue call nothing answered, given its code (MCPs board task 8993c306).

:func:`fleet.core.queue_transport.answering` wraps a transport given here as
a function that answers or raises what ``urllib`` raised on 2026-10-05, so
what is asserted is the error a serve sees for each way a call goes
unanswered, and that an ANSWER, a 503 included, passes through untouched.
The production binding is then asked of a real loopback port nothing
listens on, the shape of mcp-fleet while a deploy recreates it.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
import urllib.error
from typing import Final

import pytest
from platform_core.error_codes_tooling import McpClientErrorCode
from platform_core.errors import AppError, FleetErrorCode
from platform_core.mcp_client import (
    McpCredentials,
    McpHttpResponse,
    McpPostProtocol,
    call_mcp_tool,
)

from fleet.core import _test_hooks, queue
from fleet.core.queue_transport import answering, unanswered
from tests._queue_fakes import QUEUE_CREDENTIALS

#: Where every case here posts.
URL = "http://127.0.0.1:8035/mcp"

#: What the endpoint refused every serving runner with on 2026-10-05.
REFUSED = "No connection could be made because the target machine actively refused it"

#: The head mcp-fleet answered an unserved tool with, byte for byte as read
#: on 2026-10-16T01:27Z, less its Date line.
KEEPALIVE_HEAD: Final = (
    b"HTTP/1.1 200 OK\r\nX-Powered-By: Express\r\ncache-control: no-cache, no-transform\r\n"
    b"connection: keep-alive\r\ncontent-type: text/event-stream\r\nx-accel-buffering: no\r\n"
    b"Transfer-Encoding: chunked\r\n\r\n"
)

#: The chunk it then sent every 15 s: an event-stream comment, never a result.
KEEPALIVE_CHUNK: Final = b"d\r\n: keepalive\n\n\r\n"

#: The peer's keepalive interval, far inside any per-read timeout.
KEEPALIVE_INTERVAL_SECONDS: Final = 0.25

#: Enough keepalives to outlast the deadline several times over.
KEEPALIVE_SENDS: Final = 24

#: The call's whole-exchange deadline in the keepalive case.
DEADLINE_SECONDS: Final = 1

#: Slack for thread scheduling between the deadline and the raise.
SLACK_SECONDS: Final = 1.0

#: What a peer's send meets once the client shut the connection: Windows
#: reports an abort or a reset, Linux a broken pipe or a reset.
HUNG_UP: Final = frozenset({"ConnectionAbortedError", "ConnectionResetError", "BrokenPipeError"})


def _post(answer: McpHttpResponse | OSError) -> McpPostProtocol:
    """A transport that answers, or raises what ``urllib`` raised.

    Args:
        answer: The response, or the error to raise.

    Returns:
        The transport.
    """

    def post(
        url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
    ) -> McpHttpResponse:
        if isinstance(answer, OSError):
            raise answer
        return answer

    return post


def _call(post: McpPostProtocol) -> McpHttpResponse:
    """Post one empty body through :func:`answering`.

    Args:
        post: The transport.

    Returns:
        What it answered.
    """
    return answering(post)(URL, headers={}, body=b"{}", timeout_seconds=5)


class TestACallThatIsAnswered:
    def test_passes_a_503_through_untouched_for_the_client_to_code(self) -> None:
        busy = McpHttpResponse(status=503, content_type="text/plain", body="starting")

        assert _call(_post(busy)) == busy


class TestACallNothingAnswered:
    @pytest.mark.parametrize(
        ("silence", "named"),
        [
            (urllib.error.URLError(ConnectionRefusedError(10061, REFUSED)), "URLError"),
            (TimeoutError("timed out"), "TimeoutError"),
            (
                http.client.RemoteDisconnected("Remote end closed connection without response"),
                "RemoteDisconnected",
            ),
        ],
    )
    def test_raises_queue_unanswered_naming_the_endpoint_with_the_cause_chained(
        self, silence: OSError, named: str
    ) -> None:
        with pytest.raises(AppError) as raised:
            _call(_post(silence))

        assert raised.value.code is FleetErrorCode.QUEUE_UNANSWERED
        assert raised.value.message == f"{URL} did not answer: {named}: {silence}"
        assert raised.value.__cause__ is silence


class TestUnanswered:
    def test_is_true_for_queue_unanswered_and_false_for_an_answer_refused_by_status(
        self,
    ) -> None:
        silent = AppError(code=FleetErrorCode.QUEUE_UNANSWERED, message="no answer")
        refused = AppError(code=McpClientErrorCode.HTTP_STATUS, message="HTTP 503")
        malformed = AppError(code=FleetErrorCode.QUEUE_ANSWER_MALFORMED, message="no id")

        assert [unanswered(silent), unanswered(refused), unanswered(malformed)] == [
            True,
            False,
            False,
        ]


class TestTheProductionSeam:
    def test_a_queue_call_to_a_port_nothing_listens_on_raises_queue_unanswered(self) -> None:
        """The real transport against a real closed loopback port, as
        mcp-fleet is between a deploy's stop and its start."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            bound: tuple[str, int] = probe.getsockname()
        closed = f"http://127.0.0.1:{bound[1]}/mcp"
        assert _test_hooks.http_post is _test_hooks._default_http_post

        credentials = McpCredentials(
            url=closed,
            api_key=QUEUE_CREDENTIALS["api_key"],
            tenant_id=QUEUE_CREDENTIALS["tenant_id"],
        )

        with pytest.raises(AppError) as raised:
            queue.held_by(credentials, agent="fleet-node-lavender")

        assert raised.value.code is FleetErrorCode.QUEUE_UNANSWERED
        assert (
            raised.value.message == f"{closed} did not answer: URLError: {raised.value.__cause__}"
        )
        assert str(raised.value.__cause__).startswith("<urlopen error [")

    def test_a_queue_call_answered_by_keepalives_alone_ends_at_its_deadline(self) -> None:
        """The real transport against a real loopback peer sending what
        mcp-fleet sent the hub tick that held the hub lane 290 s on
        2026-10-16 (MCPs board task e14815d3): a 200 event stream for a tool
        it no longer served, then a keepalive comment and never a result.
        A byte arrives well inside every per-read timeout, so only the
        whole-exchange deadline can end the call."""
        listener = socket.create_server(("127.0.0.1", 0))
        port: int = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}/mcp"
        hung_up: list[OSError] = []

        def keep_alive() -> None:
            conn, _ = listener.accept()
            with conn:
                conn.recv(65536)
                try:
                    conn.sendall(KEEPALIVE_HEAD)
                    for _ in range(KEEPALIVE_SENDS):
                        conn.sendall(KEEPALIVE_CHUNK)
                        time.sleep(KEEPALIVE_INTERVAL_SECONDS)
                except OSError as gone:
                    # The send the client's hang-up refused is the evidence.
                    hung_up.append(gone)

        peer = threading.Thread(target=keep_alive, daemon=True)
        peer.start()
        assert _test_hooks.http_post is _test_hooks._default_http_post
        credentials = McpCredentials(
            url=url,
            api_key=QUEUE_CREDENTIALS["api_key"],
            tenant_id=QUEUE_CREDENTIALS["tenant_id"],
        )
        started = time.monotonic()

        with pytest.raises(AppError) as raised:
            call_mcp_tool(
                _test_hooks.http_post,
                credentials,
                "dispatch_refuse_undeclared",
                {},
                timeout_seconds=DEADLINE_SECONDS,
            )

        elapsed = time.monotonic() - started
        peer.join(timeout=KEEPALIVE_SENDS * KEEPALIVE_INTERVAL_SECONDS + 2)
        listener.close()
        cut = f"{url} did not finish the exchange within its {DEADLINE_SECONDS} s deadline"
        assert raised.value.code is FleetErrorCode.QUEUE_UNANSWERED
        assert raised.value.message == f"{url} did not answer: TimeoutError: {cut}"
        assert elapsed < DEADLINE_SECONDS + SLACK_SECONDS
        assert [type(gone).__name__ in HUNG_UP for gone in hung_up] == [True]
