"""The one board write this package makes.

Everything mechanical about it -- the argument shape, the identity, the
decision not to read the board back -- lives in :mod:`platform_core.board`,
shared with ``tools/hpc-wake``. What is here is the BINDING: this package's
HTTP seam and this package's identity, joined to the shared call in one place,
so the cycle does not have to know about either.
"""

from __future__ import annotations

from platform_core.board import post_to_task
from platform_core.mcp_client import McpCredentials

from fleet_wake import _test_hooks
from fleet_wake.announce import Announcement
from fleet_wake.identity import IDENTITY


def post_announcement(
    credentials: McpCredentials, task_id: str, announcement: Announcement
) -> None:
    """Append one announcement to the standing task's thread.

    Args:
        credentials: Endpoint and both board secrets.
        task_id: The standing task.
        announcement: What to post.

    Raises:
        AppError: Through :func:`platform_core.board.post_to_task` --
            ``HTTP_STATUS`` when the endpoint refused (a rotated key is the
            ordinary case), ``RPC_ERROR`` when the board did (an identity
            mismatch would land here, naming the established label). Not
            caught: see :func:`fleet_wake.cycle.run_cycle` on why a failed
            post must end the cycle before anything is recorded.
    """
    post_to_task(
        _test_hooks.http_post,
        credentials,
        IDENTITY,
        task_id=task_id,
        kind="note",
        body=announcement["body"],
    )


__all__ = ["post_announcement"]
