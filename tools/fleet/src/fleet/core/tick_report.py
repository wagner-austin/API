"""Telling the dispatch queue what a runner's tick did.

A module of its own beside :mod:`fleet.core.queue`, whose one-call-one-decode
shape it keeps, because that file is at the 600-line ceiling and this is a
separate concern: ``queue`` moves jobs, this records the runners themselves.
``dispatch_tick`` keeps one row per node runner, replaced every tick, so
``fleet_status`` can show each node's tags, load, fitting projects and
verdict in one read instead of seven runner logs (MCPs board task 939ec5c7,
A4). ``dispatch_publisher_tick`` keeps one row per state each of the hub
pump's wake bridges ran in, so an audit's query reads which commit the hub
ran (MCPs board task 9f1367d5).
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.contracts.publisher_tick import (
    PublisherTick,
    decode_ticks_counted,
    encode_publisher_tick,
)
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


def record_publisher_tick(
    credentials: McpCredentials, tick: PublisherTick, *, identity: JSONObject
) -> int:
    """Record one hub publisher's tick in the state it ran in.

    Args:
        credentials: The queue's endpoint and headers.
        tick: The publisher, the checkout it ran from and its exit.
        identity: From :func:`fleet.core.queue.identity_arguments`, the pump's.

    Returns:
        How many ticks the queue has counted in that state, this one included.

    Raises:
        AppError: Any transport or contract failure from the underlying call,
            a refusal of the tick's shape among them.
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If the answer does not carry the stored state's count.
    """
    arguments: JSONObject = {**encode_publisher_tick(tick), **identity}
    return decode_ticks_counted(
        call_mcp_tool(_test_hooks.http_post, credentials, "dispatch_publisher_tick", arguments)
    )


__all__ = ["record_publisher_tick", "record_tick"]
