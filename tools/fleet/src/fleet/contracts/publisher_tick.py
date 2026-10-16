"""What the hub's pump records about one wake bridge publisher's tick.

The pump (``tools/hpc-wake/scripts/run_cycle.py``) runs every wake bridge
from a checkout on the hub every three minutes, and which commit each one
ran was written only to files on the hub: ``runs/pump-health.tsv`` kept each
publisher's exit and the checkout's HEAD said the rest, and no check the
audit runs can read either (MCPs board task 9f1367d5). The pump now sends
this record through MCPs ``dispatch_publisher_tick`` for every publisher on
every tick, and the queue keeps one row per state it ran in (MCPs mig 782),
so "the hub's hpc-wake ran this commit, clean, with exit 0" is a row a query
reads.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    require_dict,
    require_int,
)
from typing_extensions import TypedDict


class PublisherTick(TypedDict):
    """One publisher's tick, as the state it ran in.

    Attributes:
        host: The node alias of the host the pump runs on.
        publisher: The publisher's name in the pump's table, e.g. ``hpc-wake``.
        repo: The repository its code came from, the last segment of its
            checkout's ``origin`` remote, e.g. ``API``.
        head_sha: That checkout's HEAD when it ran, forty lowercase hex.
        dirty: Whether the checkout's tracked files differed from HEAD.
        exit_code: The publisher's exit code on this tick.
    """

    host: str
    publisher: str
    repo: str
    head_sha: str
    dirty: bool
    exit_code: int


def encode_publisher_tick(tick: PublisherTick) -> JSONObject:
    """Encode a tick as ``dispatch_publisher_tick``'s arguments.

    Args:
        tick: The tick.

    Returns:
        The arguments, identity aside.
    """
    return {
        "host": tick["host"],
        "publisher": tick["publisher"],
        "repo": tick["repo"],
        "headSha": tick["head_sha"],
        "dirty": tick["dirty"],
        "exitCode": tick["exit_code"],
    }


def decode_ticks_counted(answer: str) -> int:
    """Read how many ticks the queue has counted in the stored state.

    Args:
        answer: ``dispatch_publisher_tick``'s answer text.

    Returns:
        The ticks seen in this state, this one included.

    Raises:
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If it is not an object carrying the stored state's
            tick count.
    """
    body = load_json_str(answer)
    if not isinstance(body, dict):
        raise JSONTypeError(
            f"dispatch_publisher_tick must answer an object, got {type(body).__name__}"
        )
    return require_int(require_dict(body, "tick"), "ticks")


__all__ = ["PublisherTick", "decode_ticks_counted", "encode_publisher_tick"]
