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
import urllib.error

import pytest
from platform_core.error_codes_tooling import McpClientErrorCode
from platform_core.errors import AppError, FleetErrorCode
from platform_core.mcp_client import McpCredentials, McpHttpResponse, McpPostProtocol

from fleet.core import _test_hooks, queue
from fleet.core.queue_transport import answering, unanswered
from tests._queue_fakes import QUEUE_CREDENTIALS

#: Where every case here posts.
URL = "http://127.0.0.1:8035/mcp"

#: What the endpoint refused every serving runner with on 2026-10-05.
REFUSED = "No connection could be made because the target machine actively refused it"


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
