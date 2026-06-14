"""Per-attempt teleport diagnostics for the live HFSM bot.

The action lab records sent/received wire windows around every teleport
attempt; the bot executor historically recorded nothing, so live stalls
(5 in run 20260609-233736, 6 in run 20260610-011x, all against passable
ground with affordable fuel) had no discriminating data. This module
lifts the lab's attempt-window idea into the bot's asynchronous flow:

* :func:`record_teleport_dispatch` snapshots the dispatch context
  (live-client state plus the captured-message index) when the executor
  sends the wire teleport.
* :func:`emit_teleport_attempt_outcome` runs at the HFSM completion
  gates (landed or stall timeout) and emits the SAME
  ``teleport_attempt`` diagnostic shape the action lab emits -- so
  ``tankpit-issue-report`` covers bot teleports with no extra wiring.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.types.message import CapturedMessage

_WINDOW_MESSAGE_LIMIT = 12
_WINDOW_PAYLOAD_HEAD = 24


class _PendingTeleportAttemptDict(TypedDict):
    """Dispatch context held until the HFSM resolves the attempt.

    Attributes:
        target_x: Requested landing X coordinate.
        target_y: Requested landing Y coordinate.
        started_ms: Wall-clock dispatch time.
        message_index: Length of the captured-message list at dispatch;
            everything after this index is the attempt's wire window.
        cycle_id: Monotonic attempt counter for this process.
        sent_window: Compact live-client context at dispatch time.
    """

    target_x: int
    target_y: int
    started_ms: int
    message_index: int
    cycle_id: int
    sent_window: str


_pending: _PendingTeleportAttemptDict | None = None
_cycle_counter = 0


def reset_teleport_attempt_tracking() -> None:
    """Reset the pending attempt and cycle counter.

    Called from test isolation fixtures; a fresh bot process starts
    clear.
    """
    global _pending, _cycle_counter
    _pending = None
    _cycle_counter = 0


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
            (formatted by the executor from this tick's snapshot; kept
            as a plain string so this module never imports the
            action_lab snapshot machinery the bot package depends on).
    """
    global _pending, _cycle_counter
    _cycle_counter += 1
    _pending = _PendingTeleportAttemptDict(
        target_x=target_x,
        target_y=target_y,
        started_ms=get_current_time_ms(),
        message_index=message_index,
        cycle_id=_cycle_counter,
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
        return "(none)"
    parts: list[str] = []
    for message in messages[-_WINDOW_MESSAGE_LIMIT:]:
        payload = message["payload"]
        parts.append(f"{message['direction']}:{len(payload)}:{payload[:_WINDOW_PAYLOAD_HEAD]}")
    return " | ".join(parts)


def emit_teleport_attempt_outcome(
    *,
    status: str,
    messages: list[CapturedMessage],
) -> bool:
    """Emit the ``teleport_attempt`` diagnostic for the pending dispatch.

    Args:
        status: Outcome label (``landed_exact``, ``landed_inexact``, or
            ``stall_timeout``) matching the action lab's status values so
            the issue report classifies success/failure identically.
        messages: The bot's full captured-message list; the window since
            the recorded dispatch index is summarized into the
            diagnostic.

    Returns:
        True when a pending dispatch was resolved and emitted; False
        when no dispatch was pending (completion observed without a
        recorded dispatch, e.g. the gate fired for an action issued
        before tracking was reset).
    """
    global _pending
    if _pending is None:
        return False
    pending = _pending
    _pending = None
    emit_diagnostic(
        diagnostic_kind="teleport_attempt",
        target_x=pending["target_x"],
        target_y=pending["target_y"],
        teleport_cycle_id=pending["cycle_id"],
        status=status,
        duration_ms=get_current_time_ms() - pending["started_ms"],
        sent_window=pending["sent_window"],
        received_window=_format_message_window(messages[pending["message_index"] :]),
        page_snapshots="(dispatch context in sent_window)",
        page_snapshot_count=0,
    )
    return True


__all__ = [
    "emit_teleport_attempt_outcome",
    "record_teleport_dispatch",
    "reset_teleport_attempt_tracking",
]
