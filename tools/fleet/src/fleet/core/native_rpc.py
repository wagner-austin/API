"""Read-only Codex JSON-RPC over its existing WebSocket stdio proxy.

The proxy copies WebSocket bytes, not newline JSON. A socket pair attaches
the SSH child's standard streams to the maintained websockets client without
another network listener, remote Python dependency, or framing implementation.
The caller owns the target identities; this module never starts model turns.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)
from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection, connect


class RuntimeReadError(ValueError):
    """The runtime explicitly refused one JSON-RPC request."""


def _request(
    connection: ClientConnection,
    request_id: int,
    method: str,
    params: JSONObject,
    deadline: float,
) -> JSONObject:
    """Receive a correlated response within the pass's shared deadline.

    Args:
        connection: Already initialized WebSocket connection.
        request_id: Unique request number on this connection.
        method: Read-only protocol method.
        params: Validated request parameters.
        deadline: Monotonic absolute deadline for the entire observation.

    Returns:
        The response result object.

    Raises:
        ValueError: Unexpected response, RPC error, or excessive notifications.
        TimeoutError: The shared deadline expired.
    """
    connection.send(dump_json_str({"id": request_id, "method": method, "params": params}))
    for _ in range(32):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("CODEX_OBSERVE_TIMEOUT: runtime response deadline expired")
        raw = connection.recv(timeout=remaining)
        if not isinstance(raw, str):
            raise ValueError("CODEX_OBSERVE_RESPONSE: expected a text JSON-RPC response")
        response = narrow_json_to_dict(load_json_str(raw))
        if "id" not in response:
            require_str(response, "method")
            continue
        if type(response["id"]) is not int or response["id"] != request_id:
            raise ValueError("CODEX_OBSERVE_RESPONSE: runtime response identity mismatch")
        if "error" in response:
            raise RuntimeReadError(f"CODEX_OBSERVE_RPC: {method} returned an error")
        return require_dict(response, "result")
    raise ValueError("CODEX_OBSERVE_RESPONSE: notification limit exceeded")


def read_threads(
    argv: Sequence[str], thread_ids: tuple[str, ...], *, timeout_seconds: int
) -> tuple[JSONObject | str, ...]:
    """Read thread metadata through a bounded, disposable proxy child.

    Args:
        argv: Local proxy or SSH invocation; no shell interpolation.
        thread_ids: Registered Codex identities on the target host.
        timeout_seconds: Entire pass deadline, including the handshake.

    Returns:
        Thread objects in request order. No conversation turns are requested.

    Raises:
        OSError: Process or socket cannot be opened.
        TimeoutError: Handshake or request deadline expired.
        WebSocketException: The runtime refused or lost its connection.
        ValueError: A response cannot be validated.
    """
    deadline = time.monotonic() + timeout_seconds
    parent, child = socket.socketpair()
    with parent, child, tempfile.TemporaryFile() as diagnostics:
        process = subprocess.Popen(
            argv,
            stdin=child.fileno(),
            stdout=child.fileno(),
            stderr=diagnostics,
        )
        child.close()
        try:
            with connect(
                "ws://localhost/",
                sock=parent,
                proxy=None,
                compression=None,
                open_timeout=timeout_seconds,
                close_timeout=1,
                ping_interval=None,
                max_size=1024 * 1024,
            ) as connection:
                _request(
                    connection,
                    0,
                    "initialize",
                    {
                        "clientInfo": {"name": "corvis_fleet_observer", "version": "1.0"},
                        "capabilities": {"experimentalApi": True},
                    },
                    deadline,
                )
                connection.send(dump_json_str({"method": "initialized"}))
                return tuple(
                    _read_thread(connection, index, thread_id, deadline)
                    for index, thread_id in enumerate(thread_ids, start=1)
                )
        finally:
            # Only the disposable proxy child is ours. Never stop the daemon.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def attempt_read_threads(
    argv: Sequence[str], thread_ids: tuple[str, ...], *, timeout_seconds: int
) -> tuple[tuple[JSONObject | str, ...], str | None]:
    """Represent a runtime transport outage explicitly, never as idle.

    Args:
        argv: Disposable proxy command.
        thread_ids: Registered thread identities to read.
        timeout_seconds: Bound for the entire observation.

    Returns:
        Thread objects and no failure, or no objects and a named failure.
        Protocol contract faults propagate instead of fabricating observations.
    """
    try:
        return read_threads(argv, thread_ids, timeout_seconds=timeout_seconds), None
    except (OSError, TimeoutError, WebSocketException) as error:
        return (), f"CODEX_OBSERVE_UNAVAILABLE: {type(error).__name__}"


def _read_thread(
    connection: ClientConnection, request_id: int, thread_id: str, deadline: float
) -> JSONObject | str:
    """Read one target; an explicit RPC refusal is unavailable, never departed."""
    try:
        return require_dict(
            _request(
                connection,
                request_id,
                "thread/read",
                {
                    "threadId": thread_id,
                    "includeTurns": False,
                },
                deadline,
            ),
            "thread",
        )
    except RuntimeReadError as error:
        # Only an explicit RPC refusal is represented here. The caller records
        # it as unavailable for this identity and still observes other targets.
        return str(error)


__all__ = ["RuntimeReadError", "attempt_read_threads", "read_threads"]
