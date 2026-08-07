"""Shared bot-action helper for the state-machine tests."""

from __future__ import annotations

from tankpit_bot.bot.states import (
    ActionKind,
    BotStateDataDict,
    StateName,
    make_in_flight_action,
)


def _set_bot_action(
    state_data: BotStateDataDict,
    state: StateName,
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
