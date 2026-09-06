"""The one board write this package makes.

``task_post`` into the standing task's thread, as the bridge's own service
identity. No claim is needed -- posting into a visible task's thread is open
to any identity -- and no reply is parsed: the board echoes the appended
line and the only contract this package relies on is that a non-error
response means the post landed. Reading the board back is deliberately not
done here; the closure file is the bridge's position, and the post's
delivery to a waiting session is ``board-watch``'s job.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from hpc_wake import _test_hooks
from hpc_wake.announce import Announcement
from hpc_wake.identity import BRIDGE_AGENT, BRIDGE_SESSION_ID, ROOM


def post_announcement(
    credentials: McpCredentials, task_id: str, announcement: Announcement
) -> None:
    """Append one announcement to the standing task's thread.

    Args:
        credentials: Endpoint and both board secrets.
        task_id: The standing task.
        announcement: What to post.

    Raises:
        AppError: Through :func:`platform_core.mcp_client.call_mcp_tool` --
            ``HTTP_STATUS`` when the endpoint refused (a rotated key is the
            ordinary case), ``RPC_ERROR`` when the board did (an identity
            mismatch would land here, naming the established label).
    """
    arguments: dict[str, JSONValue] = {
        "room": ROOM,
        "taskId": task_id,
        "kind": "note",
        "body": announcement["body"],
        "agent": BRIDGE_AGENT,
        "sessionId": BRIDGE_SESSION_ID,
        "cwd": "service://hpc-wake",
    }
    call_mcp_tool(_test_hooks.http_post, credentials, "task_post", arguments)


__all__ = ["post_announcement"]
