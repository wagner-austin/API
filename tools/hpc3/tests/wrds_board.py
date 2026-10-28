"""A taskboard that accepts a WRDS inventory's report, for the CLI tests.

The stack's two files are answered through the core's byte seam and every
other path is read for real, so a command that also reads its own source --
the filesystem inventory packages its scanner -- still does.
"""

from __future__ import annotations

import pathlib

from platform_core.mcp_testing import FakeHttpPost, posted_ok, stack_endpoints_text
from platform_core.session_label import API_KEY_NAME, STACK_ENV_PATH, TENANT_ID_NAME
from platform_core.stack_endpoints import STACK_ENDPOINTS_PATH

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.core import _test_hooks as core_hooks

TASK = "c4522545-5bfc-449d-8863-87ed1d87e5b1"
TENANT = "2e137b5f-0000-4000-8000-0000aa"
ENV_TEXT = f"{API_KEY_NAME}=internal-key\n{TENANT_ID_NAME}={TENANT}\n"


def bind_stack_files(env_text: str) -> None:
    """Answer the stack's ``.env`` and endpoint declaration.

    Args:
        env_text: What the stack's ``.env`` holds.
    """
    stack = {
        STACK_ENV_PATH: env_text.encode(),
        STACK_ENDPOINTS_PATH: stack_endpoints_text().encode(),
    }
    real_read_bytes = core_hooks.read_bytes

    def _read_bytes(path: pathlib.Path) -> bytes:
        return stack[path] if path in stack else real_read_bytes(path)

    core_hooks.read_bytes = _read_bytes


def accepting_board(posts: int) -> FakeHttpPost:
    """Bind the stack's files and a board accepting ``posts`` writes.

    Args:
        posts: How many writes the board answers ``posted``.

    Returns:
        The poster, recording every request.
    """
    bind_stack_files(ENV_TEXT)
    post = FakeHttpPost([posted_ok() for _ in range(posts)])
    cli_hooks.http_post = post
    return post


__all__ = ["ENV_TEXT", "TASK", "TENANT", "accepting_board", "bind_stack_files"]
