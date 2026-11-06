"""Codex JSON-RPC over its existing WebSocket stdio proxy.

The proxy copies WebSocket bytes, not newline JSON. A socket pair attaches
the SSH child's standard streams to the maintained websockets client without
another network listener, remote Python dependency, or framing implementation.
The caller owns the target identities.

Reading is :func:`read_threads`. The one write is :func:`deliver_wake`
(board task 562cfa7a), which starts a turn in an existing thread only after
re-reading it idle on the same connection, and only when no turn already
carries the wake's key; the methods and fields it sends were read from the
schema ``codex app-server generate-json-schema`` prints for the release the
session host's app-server runs (0.154.0, the pin MCPs
scripts/host/session-host/codex-cli.sh installs, generated on ziggurat at
2026-11-06T18:57:46Z and read at 19:01Z: all 24 of the methods and fields
read_threads and deliver_wake send are in it; MCPs board task 59b6684d).
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager

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


def rpc_request(
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
        method: Protocol method.
        params: Validated request parameters.
        deadline: Monotonic absolute deadline for the entire pass.

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
    with runtime_connection(argv, "corvis_fleet_observer", timeout_seconds) as (
        connection,
        deadline,
    ):
        return tuple(
            _read_thread(connection, index, thread_id, deadline)
            for index, thread_id in enumerate(thread_ids, start=1)
        )


@contextmanager
def runtime_connection(
    argv: Sequence[str], client: str, timeout_seconds: int
) -> Generator[tuple[ClientConnection, float], None, None]:
    """Open one initialized connection through a bounded, disposable proxy child.

    Args:
        argv: Local proxy or SSH invocation; no shell interpolation.
        client: The ``clientInfo`` name the runtime records for this caller.
        timeout_seconds: Entire pass deadline, including the handshake.

    Yields:
        The initialized connection and the pass's monotonic deadline.

    Raises:
        OSError: Process or socket cannot be opened.
        TimeoutError: Handshake deadline expired.
        WebSocketException: The runtime refused or lost its connection.
        ValueError: The handshake response cannot be validated.
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
                max_size=MAX_MESSAGE_BYTES,
            ) as connection:
                rpc_request(
                    connection,
                    0,
                    "initialize",
                    {
                        "clientInfo": {"name": client, "version": "1.0"},
                        "capabilities": {"experimentalApi": True},
                    },
                    deadline,
                )
                connection.send(dump_json_str({"method": "initialized"}))
                yield connection, deadline
        finally:
            # Only the disposable proxy child is ours. Never stop the
            # app-server it relays to, the session host's declared
            # corvis-codex-app-server.service (MCPs board task 59b6684d).
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


#: The largest message one connection accepts. The biggest answer read is a
#: page of 50 summary turns, each its user message and final reply:
#: measured 2026-11-03 on ziggurat, 75,894 bytes for the longest-running
#: registered thread (01a17515), so 1 MiB is over thirteen times that.
MAX_MESSAGE_BYTES = 1024 * 1024


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
            rpc_request(
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


__all__ = [
    "MAX_MESSAGE_BYTES",
    "RuntimeReadError",
    "attempt_read_threads",
    "read_threads",
    "rpc_request",
    "runtime_connection",
]
