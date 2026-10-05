"""The queue's post seam, with a call the queue never answered given a code.

WHY (MCPs board task 8993c306). Every call this package makes to the
dispatch queue and the board goes through one post
(:data:`fleet.core._test_hooks.http_post`), and the standard library's raises
a bare ``URLError`` when nothing answers: the connection refused, reset or
timed out before any HTTP status came back. A serving node runner cannot
tell that from any other fault, so one refusal ended its serve. Deploys
recreate mcp-fleet on diphtheria, which refuses every call for a minute or
two: on 2026-10-05 the serving loop's listing was refused at 07:24:56Z and
again at 11:54:58Z (mcp-fleet created 11:54:37Z), and the second time the
settle of a run that ended at 11:55:29Z met the same refusal, ended the
serve, and row bde22e57 closed at the next fire, 105 s after its check.

WHAT IT DOES. :func:`answering` gives a post that raises
``QUEUE_UNANSWERED`` for exactly that case, the cause chained, so a caller
that can ask again later (the serve, :mod:`fleet.cli.node_serve`) names it
by code, and every other caller fails as before with a code instead of a
bare ``URLError``. A status the queue DID answer, a 401 or a 503 among
them, is untouched: :func:`platform_core.mcp_client.call_mcp_tool` turns it
into its own ``HTTP_STATUS``. THIS IS THE ONE PLACE THE PACKAGE CATCHES A
TRANSPORT ERROR, and it re-raises: nothing here retries or softens.
"""

from __future__ import annotations

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.errors import AppError, ErrorCodeType
from platform_core.mcp_client import McpHttpResponse, McpPostProtocol


def unanswered(error: AppError[ErrorCodeType]) -> bool:
    """Whether an error is a queue call that got no answer at all.

    Args:
        error: Any error a queue or board call raised, whatever its code's family.

    Returns:
        True for ``QUEUE_UNANSWERED``.
    """
    return error.code is FleetErrorCode.QUEUE_UNANSWERED


def answering(post: McpPostProtocol) -> McpPostProtocol:
    """Give a post whose unanswered call raises ``QUEUE_UNANSWERED``.

    Args:
        post: The transport, :func:`platform_core.mcp_client.urllib_mcp_post`
            in production.

    Returns:
        The same post, with an ``OSError`` out of it (``URLError``,
        ``TimeoutError`` and ``ConnectionResetError`` are all one) raised as
        ``QUEUE_UNANSWERED`` naming the endpoint and the error.
    """

    def answered(
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Post, raising a call nothing answered as ``QUEUE_UNANSWERED``.

        Args:
            url: Absolute URL to post to.
            headers: Every request header, already complete.
            body: The encoded request body.
            timeout_seconds: How long to wait for the whole exchange.

        Returns:
            The response, whatever its status.

        Raises:
            AppError: ``QUEUE_UNANSWERED`` when no answer came back.
        """
        try:
            return post(url, headers=headers, body=body, timeout_seconds=timeout_seconds)
        except OSError as silence:
            raise AppError(
                code=FleetErrorCode.QUEUE_UNANSWERED,
                message=f"{url} did not answer: {type(silence).__name__}: {silence}",
            ) from silence

    return answered


__all__ = ["answering", "unanswered"]
