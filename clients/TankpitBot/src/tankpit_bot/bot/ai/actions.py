"""Behavior execution — translates chosen behaviors into bot commands.

Each action function takes a BehaviorScoreDict and AI state, returning
updated AI state and the bot command to execute.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    BehaviorMode,
    BehaviorScoreDict,
)
from tankpit_bot.bot.types import (
    BotCommand,
    make_move_command,
    make_pickup_move_command,
    make_shoot_command,
)
from tankpit_bot.state.types import SelfStateDict

# Max move distance per command (viewport is 17x17, centered on player = 8 cells radius)
_MAX_MOVE_STEP = 8

# Max shoot distance — targets must be within the 18x18 viewport to be hit.
# The server rejects shoot commands beyond this range ("You can't do this").
_MAX_SHOOT_RANGE = 8


def _step_toward(
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    terrain: TerrainMapProtocol | None = None,
) -> tuple[int, int]:
    """Calculate a move target clamped to viewport range, terrain-aware.

    The game's move command accepts targets up to ~8 cells away.
    This clamps long-range moves to a single step in the right direction,
    then checks terrain passability on the destination.

    Args:
        sx: Current X position.
        sy: Current Y position.
        tx: Desired target X.
        ty: Desired target Y.
        terrain: Optional terrain map for passability checks.

    Returns:
        (x, y) clamped to at most _MAX_MOVE_STEP cells from current position.
        Returns (sx, sy) if the destination is impassable terrain.
    """
    dx = tx - sx
    dy = ty - sy
    if abs(dx) > _MAX_MOVE_STEP:
        dx = _MAX_MOVE_STEP if dx > 0 else -_MAX_MOVE_STEP
    if abs(dy) > _MAX_MOVE_STEP:
        dy = _MAX_MOVE_STEP if dy > 0 else -_MAX_MOVE_STEP
    dest_x, dest_y = sx + dx, sy + dy
    if terrain is not None and not terrain.is_passable(dest_x, dest_y):
        return (sx, sy)
    return (dest_x, dest_y)


def execute_behavior(
    behavior: BehaviorScoreDict,
    ai_state: AIStateDict,
    self_state: SelfStateDict,
    timestamp_ms: int,
    terrain: TerrainMapProtocol | None = None,
) -> tuple[AIStateDict, BotCommand]:
    """Execute the chosen behavior and return updated state + command.

    Dispatches to the appropriate action based on behavior mode.
    Updates AI state tracking (mode, ticks, combat target, waypoint index).

    Args:
        behavior: The chosen behavior to execute.
        ai_state: Current AI state.
        self_state: Player's own state.
        timestamp_ms: Current game timestamp in milliseconds.
        terrain: Optional terrain map for movement passability checks.

    Returns:
        Tuple of (updated AIStateDict, BotCommand to execute).
    """
    mode = behavior["mode"]

    # Track mode transitions
    ticks_in_mode = 0 if mode != ai_state["active_mode"] else ai_state["ticks_in_mode"] + 1

    if mode == "HUNT":
        return _execute_hunt(behavior, ai_state, self_state, timestamp_ms, ticks_in_mode, terrain)
    if mode == "COLLECT_FUEL":
        return _execute_collect_fuel(behavior, ai_state, ticks_in_mode)
    # mode == "COLLECT_EQUIPMENT"
    return _execute_collect_equipment(behavior, ai_state, ticks_in_mode)


def _update_ai_state(
    ai_state: AIStateDict,
    mode: BehaviorMode,
    ticks_in_mode: int,
    combat_target_id: int = -1,
    combat_target_x: int = 0,
    combat_target_y: int = 0,
    patrol_waypoint_index: int = -1,
    last_scan_ms: int = -1,
    last_shoot_ms: int = -1,
) -> AIStateDict:
    """Create an updated AI state with the given changes.

    Args:
        ai_state: Previous AI state.
        mode: New active mode.
        ticks_in_mode: Updated tick counter.
        combat_target_id: Combat target (-1 to keep current, or new value).
        combat_target_x: Combat target X.
        combat_target_y: Combat target Y.
        patrol_waypoint_index: Waypoint index (-1 to keep current).
        last_scan_ms: Last scan timestamp (-1 to keep current).
        last_shoot_ms: Last shoot timestamp (-1 to keep current).

    Returns:
        New AIStateDict with updates applied.
    """
    return AIStateDict(
        config=ai_state["config"],
        active_mode=mode,
        patrol_waypoint_index=(
            ai_state["patrol_waypoint_index"]
            if patrol_waypoint_index == -1
            else patrol_waypoint_index
        ),
        last_scan_ms=(ai_state["last_scan_ms"] if last_scan_ms == -1 else last_scan_ms),
        last_shoot_ms=(ai_state["last_shoot_ms"] if last_shoot_ms == -1 else last_shoot_ms),
        last_map_open_ms=ai_state["last_map_open_ms"],
        combat_target_id=(
            combat_target_id if combat_target_id != -1 else ai_state["combat_target_id"]
        ),
        combat_target_x=combat_target_x,
        combat_target_y=combat_target_y,
        ticks_in_mode=ticks_in_mode,
        killed_tank_ids=ai_state["killed_tank_ids"],
        last_shot_target_id=ai_state["last_shot_target_id"],
        last_shot_target_name=ai_state["last_shot_target_name"],
    )


def _execute_hunt(
    behavior: BehaviorScoreDict,
    ai_state: AIStateDict,
    self_state: SelfStateDict,
    timestamp_ms: int,
    ticks_in_mode: int,
    terrain: TerrainMapProtocol | None = None,
) -> tuple[AIStateDict, BotCommand]:
    """Execute HUNT behavior.

    Shoots when cooldown allows, otherwise moves toward the target.

    Args:
        behavior: HUNT behavior with target coordinates.
        ai_state: Current AI state.
        self_state: Player's own state.
        timestamp_ms: Current timestamp.
        ticks_in_mode: Ticks spent in current mode.

    Returns:
        Updated state and command.
    """
    config = ai_state["config"]
    tx, ty = behavior["target_x"], behavior["target_y"]

    new_state = _update_ai_state(
        ai_state,
        "HUNT",
        ticks_in_mode,
        combat_target_x=tx,
        combat_target_y=ty,
    )

    sx, sy = self_state["x"], self_state["y"]
    dist = abs(tx - sx) + abs(ty - sy)

    # Target beyond viewport — can only move, not shoot
    if dist > _MAX_SHOOT_RANGE:
        step_x, step_y = _step_toward(sx, sy, tx, ty, terrain)
        return new_state, make_move_command(step_x, step_y)

    # Check if we should shoot
    if timestamp_ms - ai_state["last_shoot_ms"] >= config["shoot_cooldown_ms"]:
        return (
            AIStateDict(**{**new_state, "last_shoot_ms": timestamp_ms}),
            make_shoot_command(tx, ty, behavior["target_id"]),
        )

    # Move toward target
    step_x, step_y = _step_toward(sx, sy, tx, ty, terrain)
    return new_state, make_move_command(step_x, step_y)


def _execute_collect_fuel(
    behavior: BehaviorScoreDict,
    ai_state: AIStateDict,
    ticks_in_mode: int,
) -> tuple[AIStateDict, BotCommand]:
    """Execute COLLECT_FUEL behavior — pickup move to fuel container.

    Args:
        behavior: COLLECT_FUEL behavior with target coordinates.
        ai_state: Current AI state.
        ticks_in_mode: Ticks spent in current mode.

    Returns:
        Updated state and pickup move command.
    """
    new_state = _update_ai_state(ai_state, "COLLECT_FUEL", ticks_in_mode)
    return new_state, make_pickup_move_command(behavior["target_x"], behavior["target_y"])


def _execute_collect_equipment(
    behavior: BehaviorScoreDict,
    ai_state: AIStateDict,
    ticks_in_mode: int,
) -> tuple[AIStateDict, BotCommand]:
    """Execute COLLECT_EQUIPMENT behavior — pickup move to equipment container.

    Args:
        behavior: COLLECT_EQUIPMENT behavior with target coordinates.
        ai_state: Current AI state.
        ticks_in_mode: Ticks spent in current mode.

    Returns:
        Updated state and pickup move command.
    """
    new_state = _update_ai_state(ai_state, "COLLECT_EQUIPMENT", ticks_in_mode)
    return new_state, make_pickup_move_command(behavior["target_x"], behavior["target_y"])


__all__ = [
    "execute_behavior",
]
