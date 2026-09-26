"""Shared bot-action helpers for the state-machine and command-error tests."""

from __future__ import annotations

from tankpit_bot.bot.states import (
    ActionKind,
    BotState,
    BotStateDataDict,
    InFlightActionDict,
    make_in_flight_action,
)


def _set_bot_action(
    state_data: BotStateDataDict,
    state: BotState,
    kind: ActionKind,
    tx: int,
    ty: int,
    started_ms: int = -1,
) -> BotStateDataDict:
    """Build new state data with state and in-flight action set."""
    from tankpit_bot.browser import get_current_time_ms

    ts = get_current_time_ms() if started_ms < 0 else started_ms
    return BotStateDataDict(
        state=state,
        fuel_threshold=state_data["fuel_threshold"],
        in_flight_action=make_in_flight_action(kind, tx, ty, ts),
    )


def _pending_action(
    kind: ActionKind,
    *,
    target_x: int = 100,
    target_y: int = 100,
) -> InFlightActionDict:
    """Build a pending in-flight action of the requested kind, dispatched now.

    Args:
        kind: Action kind.
        target_x: Target X coordinate.
        target_y: Target Y coordinate.

    Returns:
        A PENDING in-flight action stamped with the current time.
    """
    from tankpit_bot.browser import get_current_time_ms

    return make_in_flight_action(kind, target_x, target_y, get_current_time_ms())
