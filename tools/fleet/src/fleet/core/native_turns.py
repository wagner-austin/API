"""Start, find and read turns of an existing Codex thread (MCPs board task 562cfa7a).

One concern: the protocol a wake speaks to the Codex runtime, on one
connection that :func:`fleet.core.native_rpc.runtime_connection` opens. The
methods and fields were read from the schema ``codex app-server
generate-json-schema`` prints for the running daemon's release, 0.166.0, on
2026-11-03:

* ``thread/read`` (``includeTurns`` false) gives the thread's status. Only
  ``idle`` is woken: ``active`` (with or without ``waitingOnApproval`` or
  ``waitingOnUserInput``), ``notLoaded`` and ``systemError`` are left alone.
  ``notLoaded`` matters most: it is a thread this daemon does not host (a
  standalone TUI runs its own), and a turn started here would run a copy of
  its history the operator never sees.
* ``thread/turns/list`` (newest first, ``summary`` items) gives recent turns,
  each with its status and its user message, whose ``clientId`` is the
  ``clientUserMessageId`` the turn was started with.
* ``turn/start`` with ``threadId``, one text ``input`` and
  ``clientUserMessageId``. No model, effort, sandbox, approval or cwd field
  is sent, so the thread keeps every setting it has. Its answer is the new
  turn, whose ``id`` is the evidence.

NO SECOND TURN FOR ONE KEY. A key already on a recent turn's user message is
FOUND, never started again, so a pass whose commit was lost cannot start a
duplicate. The search covers :data:`TURN_PAGE` turns; the next pass after a
lost commit is three minutes later, and an idle thread runs no turns of its
own in between.

THE WINDOW IT CANNOT CLOSE. Between reading the thread idle and starting the
turn the operator could type; the runtime would then treat ``turn/start`` as
a steer of the operator's turn. The read and the start are consecutive
requests on one connection, so the window is one round trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    optional_str,
    require_dict,
    require_list,
    require_str,
    require_str_list,
)
from websockets.exceptions import WebSocketException

from fleet.core.native_rpc import RuntimeReadError, rpc_request, runtime_connection

#: How many recent turns are searched for a key or an open turn's end.
#: Measured 2026-11-03: 50 summary turns of the longest-running thread on
#: ziggurat answered in 75,894 bytes.
TURN_PAGE: Final = 50

#: The turn statuses that end a turn; ``inProgress`` does not.
FINAL_STATUSES: Final = ("completed", "failed", "interrupted")


class WakeRequest(TypedDict):
    """The wake to deliver.

    Attributes:
        key: The key the session host minted; the turn's ``clientId``.
        text: The turn's input.
    """

    key: str
    text: str


class WakeDelivery(TypedDict):
    """What one connection did for one thread.

    Attributes:
        runtime: The thread's status as read before acting: ``idle``,
            ``notLoaded``, ``systemError``, or ``active`` with its flags
            after a colon.
        outcome: ``started`` when this call started the turn, ``found``
            when a recent turn already carries the key, ``left`` when the
            thread was not idle, ``none`` when no wake was requested.
        turn_id: The turn started or found, else ``None``.
        ended: Each requested open turn that has ended, with its status.
    """

    runtime: str
    outcome: Literal["started", "found", "left", "none"]
    turn_id: str | None
    ended: dict[str, str]


def runtime_word(status: JSONObject) -> str:
    """Name a thread status the way the hub's log line spells it.

    Args:
        status: The thread's ``status`` object.

    Returns:
        Its ``type``, with an active thread's flags after a colon.

    Raises:
        AppError: A malformed status.
    """
    kind = require_str(status, "type")
    if kind != "active":
        return kind
    flags = require_str_list(status, "activeFlags")
    return f"active:{','.join(flags)}" if flags else "active"


def deliver_wake(
    argv: Sequence[str],
    thread_id: str,
    wake: WakeRequest | None,
    open_turns: tuple[str, ...],
    *,
    timeout_seconds: int,
) -> WakeDelivery:
    """Read one thread, read how its open turns ended, and deliver its wake when idle.

    Args:
        argv: Local proxy or SSH invocation of ``codex app-server proxy``.
        thread_id: The registered thread.
        wake: The wake to deliver, or ``None`` to only read open turns.
        open_turns: Turns earlier wakes started whose end is not recorded.
        timeout_seconds: Entire pass deadline, including the handshake.

    Returns:
        What was read and done.

    Raises:
        OSError: Process or socket cannot be opened.
        TimeoutError: The deadline expired.
        WebSocketException: The runtime refused or lost its connection.
        RuntimeReadError: The runtime answered a request with an error.
        ValueError: The runtime answered for another thread.
        AppError: A response cannot be validated.
    """
    with runtime_connection(argv, "corvis_fleet_wake", timeout_seconds) as (
        connection,
        deadline,
    ):
        thread = require_dict(
            rpc_request(
                connection,
                1,
                "thread/read",
                {"threadId": thread_id, "includeTurns": False},
                deadline,
            ),
            "thread",
        )
        if require_str(thread, "id") != thread_id:
            raise ValueError("CODEX_WAKE_IDENTITY: runtime returned another thread")
        runtime = runtime_word(require_dict(thread, "status"))
        turns = require_list(
            rpc_request(
                connection,
                2,
                "thread/turns/list",
                {
                    "threadId": thread_id,
                    "limit": TURN_PAGE,
                    "sortDirection": "desc",
                    "itemsView": "summary",
                },
                deadline,
            ),
            "data",
        )
        statuses: dict[str, str] = {}
        keyed: dict[str, str] = {}
        for value in turns:
            turn = narrow_json_to_dict(value)
            turn_id = require_str(turn, "id")
            statuses[turn_id] = require_str(turn, "status")
            for entry in require_list(turn, "items"):
                item = narrow_json_to_dict(entry)
                if require_str(item, "type") != "userMessage":
                    continue
                client = optional_str(item, "clientId")
                if client is not None:
                    keyed[client] = turn_id
        ended = {
            turn_id: statuses[turn_id]
            for turn_id in open_turns
            if statuses.get(turn_id) in FINAL_STATUSES
        }
        if wake is None:
            return WakeDelivery(runtime=runtime, outcome="none", turn_id=None, ended=ended)
        found = keyed.get(wake["key"])
        if found is not None:
            return WakeDelivery(runtime=runtime, outcome="found", turn_id=found, ended=ended)
        if runtime != "idle":
            return WakeDelivery(runtime=runtime, outcome="left", turn_id=None, ended=ended)
        started = require_dict(
            rpc_request(
                connection,
                3,
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": wake["text"], "text_elements": []}],
                    "clientUserMessageId": wake["key"],
                },
                deadline,
            ),
            "turn",
        )
        return WakeDelivery(
            runtime=runtime, outcome="started", turn_id=require_str(started, "id"), ended=ended
        )


def attempt_deliver_wake(
    argv: Sequence[str],
    thread_id: str,
    wake: WakeRequest | None,
    open_turns: tuple[str, ...],
    *,
    timeout_seconds: int,
) -> tuple[WakeDelivery | None, str | None]:
    """Represent a transport outage or a runtime refusal as a value, never as delivered.

    Args:
        argv: Disposable proxy command.
        thread_id: The registered thread.
        wake: The wake, or ``None``.
        open_turns: Open turns to read.
        timeout_seconds: Bound for the entire pass.

    Returns:
        The delivery and no failure, or no delivery and a named failure;
        the caller then leaves the wake pending for the next pass. Protocol
        contract faults propagate.
    """
    try:
        return (
            deliver_wake(argv, thread_id, wake, open_turns, timeout_seconds=timeout_seconds),
            None,
        )
    except RuntimeReadError as error:
        return None, f"CODEX_WAKE_REFUSED: {error}"
    except (OSError, TimeoutError, WebSocketException) as error:
        return None, f"CODEX_WAKE_UNAVAILABLE: {type(error).__name__}"


__all__ = [
    "FINAL_STATUSES",
    "TURN_PAGE",
    "WakeDelivery",
    "WakeRequest",
    "attempt_deliver_wake",
    "deliver_wake",
    "runtime_word",
]
