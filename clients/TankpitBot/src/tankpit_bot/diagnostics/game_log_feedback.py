"""Consume world-truth feedback lines from the in-game text log.

Live run 20260610 proved two truth signals the client renders that
never arrive on the wire, and that the bot previously discarded:

* ``Empty container`` -- a pickup against a drained container. The wire
  is silent on failed pickups (``container_pickup`` only fires on
  success: 42 ``pickup_fuel`` sends produced 26 confirmations), so the
  stale volume belief survived and the bot retried the same empty
  container every tick until killed.
* ``You can't go there!`` -- a rejected move. Previously learned only
  through the 12s stall timeout instead of instantly.

This module is the consumer: the executor records the last dispatched
pickup/move target, and each tick's new game-log entries delete or
correct the contradicted belief at that target.

The ``Tank full`` line is intentionally NOT consumed here: fuel
capacity is now derived from ``self_state["rank"]`` via
:func:`tankpit_bot.state.rank_formulas.fuel_capacity` (see
``wiki/pages/game-economy.md``), so a tank-full signal teaches nothing
new. Callers that need to know when a pickup is wasteful gate on the
rank-derived capacity, not on a learned watermark.
"""

from __future__ import annotations

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.dom_scraper import GameLogEntry
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state import get_world_service, mark_move_target_failed
from tankpit_bot.sniffer.world_state_containers import remove_container_at

_EMPTY_CONTAINER_TEXT = "Empty container"
_BLOCKED_MOVE_TEXT = "You can't go there!"

_last_pickup_target: tuple[int, int] | None = None
_last_move_target: tuple[int, int] | None = None


def reset_game_log_feedback() -> None:
    """Reset the recorded dispatch targets.

    Called from test isolation fixtures; a fresh bot process starts
    clear.
    """
    global _last_pickup_target, _last_move_target
    _last_pickup_target = None
    _last_move_target = None


def record_pickup_dispatch(x: int, y: int) -> None:
    """Record the container target of the pickup command just sent.

    Args:
        x: Pickup target X coordinate.
        y: Pickup target Y coordinate.
    """
    global _last_pickup_target
    _last_pickup_target = (x, y)


def record_move_dispatch(x: int, y: int) -> None:
    """Record the tile target of the move command just sent.

    Args:
        x: Move target X coordinate.
        y: Move target Y coordinate.
    """
    global _last_move_target
    _last_move_target = (x, y)


def _consume_empty_container() -> None:
    """Delete the contradicted container belief at the last pickup target."""
    if _last_pickup_target is None:
        return
    x, y = _last_pickup_target
    remove_container_at(get_world_service(), x, y)
    emit_diagnostic(
        diagnostic_kind="game_log_feedback",
        feedback="empty_container",
        target_x=x,
        target_y=y,
    )


def _consume_blocked_move() -> None:
    """Mark the last move target as failed without waiting for a stall."""
    if _last_move_target is None:
        return
    x, y = _last_move_target
    mark_move_target_failed(x, y, get_current_time_ms())
    emit_diagnostic(
        diagnostic_kind="game_log_feedback",
        feedback="blocked_move",
        target_x=x,
        target_y=y,
    )


def register_world_feedback_from_game_log(entries: list[GameLogEntry]) -> int:
    """Apply every world-truth feedback line found in new log entries.

    Args:
        entries: New game-log entries from this tick's poll, in order.

    Returns:
        Number of feedback lines consumed.
    """
    consumed = 0
    for entry in entries:
        text = entry["text"]
        if text == _EMPTY_CONTAINER_TEXT:
            _consume_empty_container()
        elif text == _BLOCKED_MOVE_TEXT:
            _consume_blocked_move()
        else:
            continue
        consumed += 1
    return consumed


__all__ = [
    "record_move_dispatch",
    "record_pickup_dispatch",
    "register_world_feedback_from_game_log",
    "reset_game_log_feedback",
]
