"""Teleport outcome emitters + per-attempt dispatch-context tracking.

Absorbs the former ``diagnostics/teleport_attempts.py``: the executor
records the dispatch context (live-client state plus captured-message
index) when it sends the wire teleport; the HFSM completion gates
resolve the attempt into exactly one recorded outcome carrying the
sent/received wire windows -- so live stalls have discriminating data
(5 stalls in run 20260609-233736 had none).

Eight resolutions: exact/inexact landing, 0x52 rejection, stall
timeout, and the four executor discard classes from the
rejection-loop audit (hostile-mine tile, stale combat target, stale
resource target, invalid resource-target source).
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.contracts.enforcement import enforce_contract, require
from tankpit_bot.ledger.outcome._emit import emit_action_outcome
from tankpit_bot.ledger.ring import ActionOutcomeRecordDict
from tankpit_bot.types.message import CapturedMessage

_WINDOW_MESSAGE_LIMIT = 12
_WINDOW_PAYLOAD_HEAD = 24

_NO_WINDOW = "(none)"


class _PendingTeleportDispatchDict(TypedDict):
    """Dispatch context held until a completion gate resolves the attempt.

    Attributes:
        target_x: Requested landing X coordinate.
        target_y: Requested landing Y coordinate.
        started_ms: Wall-clock dispatch time.
        message_index: Length of the captured-message list at dispatch;
            everything after this index is the attempt's wire window.
        sent_window: Compact live-client context at dispatch time.
    """

    target_x: int
    target_y: int
    started_ms: int
    message_index: int
    sent_window: str


_pending: _PendingTeleportDispatchDict | None = None


def reset_teleport_dispatch_tracking() -> None:
    """Clear the pending dispatch. Called from test-isolation fixtures."""
    global _pending
    _pending = None


class TeleportDispatchContract:
    """Structural invariants on a recorded teleport dispatch."""

    @property
    def name(self) -> str:
        """Name of the contract."""
        return "teleport_dispatch"

    def check(
        self,
        *,
        target_x: int,
        target_y: int,
        message_index: int,
        sent_window: str,
    ) -> None:
        """Validate a dispatch context before it enters the ledger.

        Args:
            target_x: Requested landing X coordinate.
            target_y: Requested landing Y coordinate.
            message_index: Captured-message count at dispatch time.
            sent_window: Compact live-client context at dispatch time.

        Raises:
            LedgerInvariantError: If the coordinates are off-map or
                the message index is negative.
        """
        require(
            0 <= target_x <= 255 and 0 <= target_y <= 255,
            LedgerInvariantError,
            target_x=repr(target_x),
            target_y=repr(target_y),
        )
        require(
            message_index >= 0,
            LedgerInvariantError,
            message_index=repr(message_index),
            sent_window=sent_window,
        )


@enforce_contract(TeleportDispatchContract())
def record_teleport_dispatch(
    *,
    target_x: int,
    target_y: int,
    message_index: int,
    sent_window: str,
) -> None:
    """Record the dispatch context for the teleport just sent.

    Args:
        target_x: Requested landing X coordinate.
        target_y: Requested landing Y coordinate.
        message_index: Captured-message count at dispatch time.
        sent_window: Compact live-client context at dispatch time
            (formatted by the executor from this tick's snapshot).
    """
    global _pending
    _pending = _PendingTeleportDispatchDict(
        target_x=target_x,
        target_y=target_y,
        started_ms=get_current_time_ms(),
        message_index=message_index,
        sent_window=sent_window,
    )


def _format_message_window(messages: list[CapturedMessage]) -> str:
    """Render the attempt's wire window as a compact one-line summary.

    Args:
        messages: Captured messages exchanged since the dispatch.

    Returns:
        Pipe-joined ``direction:length:payload-head`` entries for the
        last :data:`_WINDOW_MESSAGE_LIMIT` messages, or ``(none)``.
    """
    if not messages:
        return _NO_WINDOW
    parts: list[str] = []
    for message in messages[-_WINDOW_MESSAGE_LIMIT:]:
        payload = message["payload"]
        parts.append(f"{message['direction']}:{len(payload)}:{payload[:_WINDOW_PAYLOAD_HEAD]}")
    return " | ".join(parts)


def _consume_pending_windows(messages: list[CapturedMessage]) -> tuple[str, str]:
    """Resolve and clear the pending dispatch's wire windows.

    Args:
        messages: The bot's full captured-message list.

    Returns:
        ``(sent_window, received_window)``; both ``(none)`` when the
        gate fired without a recorded dispatch (e.g. tracking reset
        between dispatch and completion).
    """
    global _pending
    if _pending is None:
        return (_NO_WINDOW, _NO_WINDOW)
    pending = _pending
    _pending = None
    return (
        pending["sent_window"],
        _format_message_window(messages[pending["message_index"] :]),
    )


def emit_teleport_landed(
    *,
    duration_ms: int,
    target_x: int,
    target_y: int,
    landed_x: int,
    landed_y: int,
    messages: list[CapturedMessage],
) -> ActionOutcomeRecordDict:
    """Record a confirmed teleport landing (exact or displaced).

    Args:
        duration_ms: Dispatch-to-landing wall-clock ms.
        target_x: Requested landing X.
        target_y: Requested landing Y.
        landed_x: Actual landing X per the server.
        landed_y: Actual landing Y.
        messages: The bot's captured-message list, for the wire window.

    Returns:
        The recorded outcome (``landed_exact`` or ``landed_inexact``).
    """
    sent_window, received_window = _consume_pending_windows(messages)
    exact = landed_x == target_x and landed_y == target_y
    return emit_action_outcome(
        action_kind="teleport",
        outcome="landed_exact" if exact else "landed_inexact",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        landed_x=landed_x,
        landed_y=landed_y,
        sent_window=sent_window,
        received_window=received_window,
    )


def emit_teleport_stall_timeout(
    *,
    duration_ms: int,
    target_x: int,
    target_y: int,
    timeout_ms: int,
    messages: list[CapturedMessage],
) -> ActionOutcomeRecordDict:
    """Record a teleport that stalled past its timeout.

    Args:
        duration_ms: Dispatch-to-stall wall-clock ms.
        target_x: Requested landing X.
        target_y: Requested landing Y.
        timeout_ms: The stall threshold that fired.
        messages: The bot's captured-message list, for the wire window.

    Returns:
        The recorded outcome.
    """
    sent_window, received_window = _consume_pending_windows(messages)
    return emit_action_outcome(
        action_kind="teleport",
        outcome="stall_timeout",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        timeout_ms=timeout_ms,
        sent_window=sent_window,
        received_window=received_window,
    )


def emit_teleport_command_rejected(
    *,
    duration_ms: int,
    target_x: int,
    target_y: int,
    error_code: int,
    messages: list[CapturedMessage],
) -> ActionOutcomeRecordDict:
    """Record a teleport the server refused with a 0x52 Supervisor error.

    Args:
        duration_ms: Dispatch-to-rejection wall-clock ms.
        target_x: Requested landing X.
        target_y: Requested landing Y.
        error_code: The 0x52 error code.
        messages: The bot's captured-message list, for the wire window.

    Returns:
        The recorded outcome.
    """
    sent_window, received_window = _consume_pending_windows(messages)
    return emit_action_outcome(
        action_kind="teleport",
        outcome="command_rejected",
        duration_ms=duration_ms,
        target_x=target_x,
        target_y=target_y,
        error_code=error_code,
        sent_window=sent_window,
        received_window=received_window,
    )


def emit_teleport_discarded_combat_target_stale(
    *, target_x: int, target_y: int, target_id: int
) -> ActionOutcomeRecordDict:
    """Record an executor discard: the locked combat target went stale.

    Args:
        target_x: Requested landing X.
        target_y: Requested landing Y.
        target_id: The stale combat target's tank id.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="teleport",
        outcome="discarded_combat_target_stale",
        duration_ms=0,
        target_x=target_x,
        target_y=target_y,
        target_id=target_id,
    )


def emit_teleport_discarded_resource_target_stale(
    *, target_x: int, target_y: int, resource_kind: str
) -> ActionOutcomeRecordDict:
    """Record an executor discard: the locked resource target went stale.

    Args:
        target_x: Requested landing X.
        target_y: Requested landing Y.
        resource_kind: The locked resource kind (``fuel``/``equipment``).

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="teleport",
        outcome="discarded_resource_target_stale",
        duration_ms=0,
        target_x=target_x,
        target_y=target_y,
        resource_kind=resource_kind,
    )


def emit_teleport_discarded_resource_target_invalid(
    *, target_x: int, target_y: int, source: str
) -> ActionOutcomeRecordDict:
    """Record an executor discard: resource target's source is untrusted.

    Args:
        target_x: Requested landing X.
        target_y: Requested landing Y.
        source: The container's entity source that failed the trust check.

    Returns:
        The recorded outcome.
    """
    return emit_action_outcome(
        action_kind="teleport",
        outcome="discarded_resource_target_invalid",
        duration_ms=0,
        target_x=target_x,
        target_y=target_y,
        source=source,
    )


__all__ = [
    "TeleportDispatchContract",
    "emit_teleport_command_rejected",
    "emit_teleport_discarded_combat_target_stale",
    "emit_teleport_discarded_resource_target_invalid",
    "emit_teleport_discarded_resource_target_stale",
    "emit_teleport_landed",
    "emit_teleport_stall_timeout",
    "record_teleport_dispatch",
    "reset_teleport_dispatch_tracking",
]
