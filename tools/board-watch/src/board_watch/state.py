"""Where a watcher's position lives between invocations.

The command reads once and exits, so the cursor cannot live in memory. One
small JSON document per agent label holds it.

KEYED BY AGENT, NOT BY PROCESS. Two sessions watching the same board are
subscribed to different labels and are at different positions, and a shared
file would give whichever ran last the other's place in the feed. The label
is already the thing that distinguishes them everywhere else on the board.
"""

from __future__ import annotations

import pathlib
from typing import Final, TypedDict

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    optional_str,
    require_str,
)

from board_watch import _test_hooks

#: Default directory for cursor documents, under the user's home so a
#: watcher survives a working-directory change between polls.
DEFAULT_STATE_DIRECTORY: Final = pathlib.Path.home() / ".board-watch"


class WatchState(TypedDict):
    """One watcher's persisted position.

    Attributes:
        agent: The label this position belongs to. Stored as well as being
            in the filename so a document moved by hand still says whose it
            is.
        cursor: The board cursor to read forward from, or None when the
            watcher has primed against an empty board.
    """

    agent: str
    cursor: str | None


def state_path(agent: str, directory: pathlib.Path) -> pathlib.Path:
    """Locate one agent's cursor document.

    Args:
        agent: The agent label.
        directory: The directory holding cursor documents.

    Returns:
        The path, which may not exist yet.
    """
    return directory / f"{agent}.json"


def encode_state(state: WatchState) -> str:
    """Render a position as the document written to disk.

    Args:
        state: The position.

    Returns:
        The JSON text.
    """
    document: JSONObject = {"agent": state["agent"], "cursor": state["cursor"]}
    return dump_json_str(document)


def decode_state(raw: str) -> WatchState:
    """Read a position back from its document.

    Args:
        raw: The JSON text.

    Returns:
        The position.

    Raises:
        JSONTypeError: When the document is not an object, or ``agent`` is
            absent or not a string. Propagated from the ``require_*``
            validators rather than softened: a cursor document that cannot
            be read is not a watcher that should quietly start over, because
            starting over means replaying the whole feed.
    """
    document = narrow_json_to_dict(load_json_str(raw))
    return WatchState(agent=require_str(document, "agent"), cursor=optional_str(document, "cursor"))


def load_state(agent: str, directory: pathlib.Path) -> WatchState | None:
    """Read one agent's position if it has ever been written.

    Args:
        agent: The agent label.
        directory: The directory holding cursor documents.

    Returns:
        The position, or None when this agent has no document yet, which is
        the signal to prime.
    """
    path = state_path(agent, directory)
    if not _test_hooks.file_exists(path):
        return None
    return decode_state(_test_hooks.read_text(path))


def save_state(state: WatchState, directory: pathlib.Path) -> None:
    """Write one agent's position.

    Args:
        state: The position.
        directory: The directory holding cursor documents.
    """
    _test_hooks.write_text(state_path(state["agent"], directory), encode_state(state))
