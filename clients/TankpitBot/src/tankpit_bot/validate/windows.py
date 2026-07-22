"""Fuel-window slicing shared by the archive validators.

One window spans two consecutive absolute fuel readings; its end is
timestamp-INCLUSIVE (a debit and its closing sync share a millisecond
on this wire — see ``wire_timeline``). Predicates classify windows for
the walk-episode validator in ``archive``.
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.protocol.commands import (
    CMD_MAP_TELEPORT,
    CMD_MINE,
    CMD_MOVE,
    CMD_PICKUP_EQUIPMENT,
    CMD_PICKUP_FUEL,
    CMD_RADAR,
)
from tankpit_bot.validate.wire_timeline import WireTimelineDict

_SPENDING_COMMANDS = frozenset(
    {CMD_MOVE, CMD_PICKUP_FUEL, CMD_PICKUP_EQUIPMENT, CMD_MAP_TELEPORT, CMD_RADAR, CMD_MINE}
)


class FuelWindowDict(TypedDict):
    """Events between two consecutive absolute fuel readings.

    ``spending_commands`` counts every fuel-spending sent command
    and ``move_commands`` the CMD_MOVE subset — the walk-episode
    validator allows move sends inside an episode; the firing/damage
    validators allow no spending at all.
    """

    delta: int
    own_shot_weapons: list[int]
    enemy_shot_weapons: list[int]
    walked_tiles: int
    move_echoes: int
    spending_commands: int
    move_commands: int
    pickups: int
    detonations: int


def _events_in(timestamps: list[int], start_ms: int, end_ms: int) -> int:
    """Count timestamps inside the half-open window (start, end].

    The end is INCLUSIVE: a debit's cause and its closing fuel sync
    share a millisecond on this wire (see ``wire_timeline`` docstring).

    Args:
        timestamps: Event timestamps.
        start_ms: Window start (exclusive).
        end_ms: Window end (inclusive).

    Returns:
        Number of events inside the window.
    """
    return sum(1 for ts in timestamps if start_ms < ts <= end_ms)


def build_fuel_windows(timeline: WireTimelineDict) -> list[FuelWindowDict]:
    """Slice a timeline into fuel windows between consecutive readings.

    Args:
        timeline: Extracted wire timeline for one session.

    Returns:
        One window per consecutive fuel-reading pair.
    """
    readings = timeline["fuel_readings"]
    windows: list[FuelWindowDict] = []
    for index in range(1, len(readings)):
        start_ms = readings[index - 1]["timestamp_ms"]
        end_ms = readings[index]["timestamp_ms"]
        windows.append(
            FuelWindowDict(
                delta=readings[index]["fuel"] - readings[index - 1]["fuel"],
                own_shot_weapons=[
                    shot["weapon"]
                    for shot in timeline["own_shots"]
                    if start_ms < shot["timestamp_ms"] <= end_ms
                ],
                enemy_shot_weapons=[
                    shot["weapon"]
                    for shot in timeline["enemy_shots"]
                    if start_ms < shot["timestamp_ms"] <= end_ms
                ],
                walked_tiles=sum(
                    move["tiles"]
                    for move in timeline["self_moves"]
                    if start_ms < move["timestamp_ms"] <= end_ms
                ),
                move_echoes=_events_in(
                    [move["timestamp_ms"] for move in timeline["self_moves"]],
                    start_ms,
                    end_ms,
                ),
                spending_commands=_events_in(
                    [
                        action["timestamp_ms"]
                        for action in timeline["sent_actions"]
                        if action["command"] in _SPENDING_COMMANDS
                    ],
                    start_ms,
                    end_ms,
                ),
                move_commands=_events_in(
                    [
                        action["timestamp_ms"]
                        for action in timeline["sent_actions"]
                        if action["command"] == CMD_MOVE
                    ],
                    start_ms,
                    end_ms,
                ),
                pickups=_events_in(timeline["pickup_timestamps"], start_ms, end_ms),
                detonations=_events_in(timeline["detonation_timestamps"], start_ms, end_ms),
            )
        )
    return windows


def _is_walk_window(window: FuelWindowDict) -> bool:
    """Report whether a window contains walking and nothing else.

    Args:
        window: One fuel window.

    Returns:
        True when the window has movement echoes and every sent spend
        in it is a move command.
    """
    return (
        window["walked_tiles"] > 0
        and not window["own_shot_weapons"]
        and not window["enemy_shot_weapons"]
        and window["spending_commands"] == window["move_commands"]
        and not window["pickups"]
        and not window["detonations"]
    )


def _is_silent_window(window: FuelWindowDict) -> bool:
    """Report whether a window contains no events at all.

    Args:
        window: One fuel window.

    Returns:
        True when nothing happened between the two readings (the fuel
        delta may still be nonzero — a debit can land one sync late).
    """
    return (
        window["walked_tiles"] == 0
        and not window["own_shot_weapons"]
        and not window["enemy_shot_weapons"]
        and window["spending_commands"] == 0
        and not window["pickups"]
        and not window["detonations"]
    )


__all__ = [
    "FuelWindowDict",
    "build_fuel_windows",
]
