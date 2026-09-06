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

from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse


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


__all__ = [
    "FakeHttpPost",
    "posted_ok",
    "sent_arguments",
    "sse_body",
    "tool_text_body",
]
