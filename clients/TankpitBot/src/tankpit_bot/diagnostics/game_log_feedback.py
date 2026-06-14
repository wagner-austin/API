"""Consume world-truth feedback lines from the in-game text log.

Live run 20260610 proved three truth signals the client renders that
never arrive on the wire, and that the bot previously discarded:

* ``Empty container`` -- a pickup against a drained container. The wire
  is silent on failed pickups (``container_pickup`` only fires on
  success: 42 ``pickup_fuel`` sends produced 26 confirmations), so the
  stale volume belief survived and the bot retried the same empty
  container every tick until killed.
* ``Tank full`` -- a fuel pickup at capacity. No capacity model existed
  anywhere, so the bot kept walking to fuel it could not hold.
* ``You can't go there!`` -- a rejected move. Previously learned only
  through the 12s stall timeout instead of instantly.

This module is the consumer: the executor records the last dispatched
pickup/move target, and each tick's new game-log entries delete or
correct the contradicted belief at that target.
"""

from __future__ import annotations

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.dom_scraper import GameLogEntry
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state import mark_move_target_failed
from tankpit_bot.sniffer.world_state_containers import remove_container_at
from tankpit_bot.state.types import WorldStateDict

_EMPTY_CONTAINER_TEXT = "Empty container"
_TANK_FULL_TEXT = "Tank full"
_BLOCKED_MOVE_TEXT = "You can't go there!"

_last_pickup_target: tuple[int, int] | None = None
_last_move_target: tuple[int, int] | None = None
_learned_fuel_capacity = 0


def reset_game_log_feedback() -> None:
    """Reset dispatch targets and learned capacity.

    Called from test isolation fixtures; a fresh bot process starts
    clear.
    """
    global _last_pickup_target, _last_move_target, _learned_fuel_capacity
    _last_pickup_target = None
    _last_move_target = None
    _learned_fuel_capacity = 0


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


def get_learned_fuel_capacity() -> int:
    """Return the fuel capacity learned from ``Tank full`` feedback.

    Returns:
        Learned capacity, or ``0`` while unknown.
    """
    return _learned_fuel_capacity


def is_fuel_at_learned_capacity(fuel: int) -> bool:
    """Return True when capacity is known and fuel has reached it.

    Args:
        fuel: Current fuel total.

    Returns:
        True when picking up more fuel would be a wasted action.
    """
    return 0 < _learned_fuel_capacity <= fuel


def _consume_empty_container() -> None:
    """Delete the contradicted container belief at the last pickup target."""
    if _last_pickup_target is None:
        return
    x, y = _last_pickup_target
    remove_container_at(x, y)
    emit_diagnostic(
        diagnostic_kind="game_log_feedback",
        feedback="empty_container",
        target_x=x,
        target_y=y,
    )


def _consume_tank_full(world: WorldStateDict) -> None:
    """Learn fuel capacity from a pickup that found the tank full.

    The fuel read happens at scrape time, not when the game generated
    the line -- a teleport in between can spend hundreds of fuel. Run
    20260611-004505: capacity raised to 2010 on observation, then a
    ``Tank full`` line scraped after a teleport read fuel at 1100 and
    overwrote the watermark downward. Observed fuel never exceeds
    capacity, so 2010 was proof of capacity >= 2010 and the 1100 read
    was stale. A tank-full line may therefore establish or raise the
    learned capacity, never lower it.

    Args:
        world: Current world state providing the authoritative fuel total.
    """
    global _learned_fuel_capacity
    self_state = world["self_state"]
    if self_state is None or self_state["fuel"] <= 0:
        return
    if self_state["fuel"] < _learned_fuel_capacity:
        emit_diagnostic(
            diagnostic_kind="game_log_feedback",
            feedback="tank_full_stale_read",
            observed_fuel=self_state["fuel"],
            learned_fuel_capacity=_learned_fuel_capacity,
        )
        return
    _learned_fuel_capacity = self_state["fuel"]
    emit_diagnostic(
        diagnostic_kind="game_log_feedback",
        feedback="tank_full",
        learned_fuel_capacity=_learned_fuel_capacity,
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


def _raise_outgrown_capacity(world: WorldStateDict) -> None:
    """Raise the learned capacity when observed fuel exceeds it.

    Observed fuel is always a lower bound of true capacity, so a
    contradicting observation tightens the belief instead of discarding
    it. Run 20260610-223x proved the discard version oscillated: the
    ``Tank full`` line is scraped a beat after the fuel total updates,
    so capacity learned at 1080 was invalidated when fuel read 1100,
    then re-learned -- three times in eight minutes. Raising to the
    observed value keeps the bound tight through the same lag and still
    self-corrects after a rank-up.

    Args:
        world: Current world state providing the authoritative fuel total.
    """
    global _learned_fuel_capacity
    if _learned_fuel_capacity <= 0:
        return
    self_state = world["self_state"]
    if self_state is None or self_state["fuel"] <= _learned_fuel_capacity:
        return
    emit_diagnostic(
        diagnostic_kind="game_log_feedback",
        feedback="capacity_raised",
        previous_capacity=_learned_fuel_capacity,
        observed_fuel=self_state["fuel"],
    )
    _learned_fuel_capacity = self_state["fuel"]


def register_world_feedback_from_game_log(
    entries: list[GameLogEntry],
    world: WorldStateDict,
) -> int:
    """Apply every world-truth feedback line found in new log entries.

    Args:
        entries: New game-log entries from this tick's poll, in order.
        world: Current world state for fuel reads.

    Returns:
        Number of feedback lines consumed.
    """
    _raise_outgrown_capacity(world)
    consumed = 0
    for entry in entries:
        text = entry["text"]
        if text == _EMPTY_CONTAINER_TEXT:
            _consume_empty_container()
        elif text == _TANK_FULL_TEXT:
            _consume_tank_full(world)
        elif text == _BLOCKED_MOVE_TEXT:
            _consume_blocked_move()
        else:
            continue
        consumed += 1
    return consumed


__all__ = [
    "get_learned_fuel_capacity",
    "is_fuel_at_learned_capacity",
    "record_move_dispatch",
    "record_pickup_dispatch",
    "register_world_feedback_from_game_log",
    "reset_game_log_feedback",
]
