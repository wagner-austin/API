"""Telling the dispatch queue what a node runner's tick decided.

A module of its own beside :mod:`fleet.core.queue`, whose one-call-one-decode
shape it keeps, because that file is at the 600-line ceiling and this is a
separate concern: ``queue`` moves jobs, this records the runner itself
(MCPs board task 939ec5c7, A4). ``dispatch_tick`` keeps one row per runner,
replaced every tick, so ``fleet_status`` can show each node's tags, load,
fitting projects and verdict in one read instead of seven runner logs.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.runner_tick import RunnerTick, decode_recorded_at, encode_runner_tick
from fleet.core import _test_hooks


def record_tick(credentials: McpCredentials, tick: RunnerTick, *, identity: JSONObject) -> str:
    """Record one tick as this runner's row.

    Args:
        credentials: The queue's endpoint and headers.
        tick: What the tick found and decided.
        identity: From :func:`fleet.core.queue.identity_arguments`; its
            ``agent`` is the row's runner.

    Returns:
        When the queue stored it.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            a refusal of the tick's shape among them.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer does not carry the stored row's instant.
    """
    arguments: JSONObject = {**encode_runner_tick(tick), **identity}
    return decode_recorded_at(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_tick", arguments)
    )


__all__ = ["record_tick"]
