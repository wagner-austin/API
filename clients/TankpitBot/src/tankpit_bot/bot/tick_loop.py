"""Tick loop orchestrator for the bot.

Runs the sync-decide-execute cycle on each server tick. The game uses
a command queue — move/pickup commands are queued and the tank walks
there automatically. No need to wait for arrival.

Cycle: SYNC → DECIDE → EXECUTE → WAIT
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.bot import ai_strategy, executor, world_sync
from tankpit_bot.bot.ai.equipment import is_reachable
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.states import InFlightActionDict, make_no_action, transition_to
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sniffer.world_state import (
    check_and_clear_combat_hit,
    check_and_clear_our_shot_response,
    drain_killed_tank_ids,
    get_inventory_state,
    get_terrain_map,
    increment_container_failed_pickups,
    mark_move_target_failed,
    peek_combat_hit,
    peek_our_shot_response,
)

log = get_logger(__name__)


def run_tick_loop(bot: Bot, page: PageProtocol) -> None:
    """Run the main tick loop.

    Args:
        bot: Bot instance.
        page: Playwright page for waiting between ticks.
    """
    while True:
        _tick_once(bot)
        page.wait_for_timeout(TICK_RATE_MS)


def _tick_once(bot: Bot) -> None:
    """Execute one sync-decide-execute cycle.

    Args:
        bot: Bot instance.
    """
    # 1. SYNC — drain CDP message buffer
    world_sync.drain_messages(bot)

    # 2. Read state
    world = bot.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        return

    bot._update_state_from_world()
    if not _is_ready_for_decision(bot):
        return
    if _has_in_flight_action(bot):
        return

    # 4. Merge kills from protocol
    bot._ai_state = _merge_protocol_kills(bot._ai_state)
    now = get_current_time_ms()
    if _has_pending_shot_feedback(bot, now):
        return

    # 5. Combat feedback
    combat_feedback = _get_combat_feedback(bot)

    # 6. DECIDE
    inventory = get_inventory_state()
    terrain = get_terrain_map()

    decision = ai_strategy.decide(
        world,
        self_state,
        bot._ai_state,
        inventory,
        now,
        terrain,
        combat_feedback,
    )

    # 7. Persist AI state
    bot._ai_state = decision["updated_ai_state"]

    # 8. EXECUTE — game queues commands
    executor.execute(bot, decision)


def _has_in_flight_action(bot: Bot) -> bool:
    """Return True when a previously issued action is still resolving.

    Reads the authoritative InFlightActionDict from bot state. Every
    action kind is handled explicitly — move, collect, teleport, scan,
    shoot, and map_open. There is no implicit fallback; the action
    record is the single source of truth for command lifecycle.

    Args:
        bot: Bot instance.

    Returns:
        True if the bot should wait for an in-flight action to resolve.
    """
    action = bot._state_data["in_flight_action"]
    if action["kind"] == "none" or action["outcome"] != "pending":
        return False

    kind = action["kind"]
    if kind in ("move", "collect", "teleport"):
        return _wait_for_movement_action(bot, action)

    # Scan: waits for radar completion signal, subject to stall timeout
    if kind == "scan":
        return _wait_for_scan_action(bot, action)

    # map_open: wait for at least one fresh server sync before replanning.
    # The game does not expose a reliable map-open flag, so we only use the
    # recorded action timestamp plus fresh world sync as the sequencing guard.
    if kind == "map_open":
        return _wait_for_map_open_action(bot, action)

    # Shoot: fire-and-forget, do not block replanning.
    # The action record is authoritative (records what was sent) but
    # this action resolves immediately — no server confirmation needed.
    return False


def _wait_for_movement_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a move/collect/teleport action is still resolving."""
    kind = action["kind"]
    tx, ty = action["target_x"], action["target_y"]
    if _clear_stalled_action(bot, action):
        return False
    if kind == "move" and _clear_blocked_walk(bot, action):
        return False
    if kind == "collect" and _clear_blocked_collection(bot, action):
        return False
    if kind == "move":
        log.info("SYNC: waiting for movement to (%d,%d)", tx, ty)
    elif kind == "teleport":
        log.info("SYNC: waiting for teleport to (%d,%d)", tx, ty)
    else:
        log.info("SYNC: waiting for collection at (%d,%d)", tx, ty)
    return True


def _wait_for_scan_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a radar scan is still pending."""
    if _clear_stalled_action(bot, action):
        return False
    log.info("SYNC: waiting for radar results")
    return True


def _wait_for_map_open_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a map-open action is waiting on fresh server sync."""
    if _clear_stalled_action(bot, action):
        return False
    if _clear_completed_map_open(bot, action):
        return False
    log.info("SYNC: waiting for map open sync")
    return True


def _clear_stalled_action(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a stalled action using its authoritative timing.

    Args:
        bot: Bot instance.
        action: The in-flight action record to check.

    Returns:
        True if the stalled action was cleared and the tick should replan.
    """
    started_ms = action["started_ms"]
    if started_ms <= 0:
        return False
    elapsed_ms = get_current_time_ms() - started_ms
    timeout_ms = bot._ai_state["config"]["action_stall_timeout_ms"]
    if elapsed_ms < timeout_ms:
        return False
    tx, ty = action["target_x"], action["target_y"]
    log.info(
        "SYNC: %s to (%d,%d) stalled for %d ms, replanning",
        action["kind"],
        tx,
        ty,
        elapsed_ms,
    )
    if action["kind"] == "collect":
        increment_container_failed_pickups(tx, ty)
        log.info("SYNC: marked container at (%d,%d) as failed pickup", tx, ty)
    if action["kind"] in ("move", "teleport"):
        now = get_current_time_ms()
        mark_move_target_failed(tx, ty, now)
        log.info("SYNC: marked (%d,%d) as failed %s target", tx, ty, action["kind"])
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_completed_map_open(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a pending map_open after a fresh server sync arrives.

    map_open has no explicit protocol confirmation. The safest available
    signal is that the client received at least one newer world-state sync
    after the command was issued, so downstream decisions use fresh data.

    Args:
        bot: Bot instance.
        action: The pending map_open action record.

    Returns:
        True if the map_open action was cleared.
    """
    world = bot.get_world_state()
    if world["timestamp_ms"] <= action["started_ms"]:
        return False
    bot._state_data = transition_to(
        bot._state_data,
        bot.get_state(),
        in_flight_action=make_no_action(),
    )
    return True


def _clear_blocked_walk(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a walk when terrain shows the destination is unreachable.

    Args:
        bot: Bot instance.
        action: The in-flight move action record.

    Returns:
        True if the blocked walk was cleared and the tick should replan.
    """
    self_state = bot.get_world_state()["self_state"]
    terrain = get_terrain_map()
    if self_state is None or terrain is None:
        return False
    tx, ty = action["target_x"], action["target_y"]
    if is_reachable(terrain, self_state["x"], self_state["y"], tx, ty):
        return False
    log.info("SYNC: movement to (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_blocked_collection(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a collection when the target is terrain-blocked.

    pickup_move can legitimately complete from an adjacent tile, so
    adjacency is treated as still viable. Otherwise, if terrain says
    there is no path, the queued collection is abandoned.

    Args:
        bot: Bot instance.
        action: The in-flight collect action record.

    Returns:
        True if the blocked collection was cleared and the tick should replan.
    """
    world = bot.get_world_state()
    self_state = world["self_state"]
    terrain = get_terrain_map()
    if self_state is None or terrain is None:
        return False
    tx, ty = action["target_x"], action["target_y"]
    if abs(self_state["x"] - tx) <= 1 and abs(self_state["y"] - ty) <= 1:
        return False
    if is_reachable(terrain, self_state["x"], self_state["y"], tx, ty):
        return False
    log.info(
        "SYNC: collection target (%d,%d) is terrain-blocked, replanning",
        tx,
        ty,
    )
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _is_ready_for_decision(bot: Bot) -> bool:
    """Return True when the bot state machine is ready to execute AI plans.

    The planner may only dispatch commands from executable states. During
    startup and disconnect handling, world state may already exist while the
    bot state machine is still converging to ``IDLE``. Replanning in those
    transitional states can produce invalid command transitions.

    Args:
        bot: Bot instance.

    Returns:
        True if the bot can safely plan and execute a command this tick.
    """
    state = bot.get_state()
    if state in ("INITIALIZING", "WAITING_FOR_POSITION", "DISCONNECTED"):
        log.info("SYNC: deferring decisions while state=%s", state)
        return False
    return True


def _merge_protocol_kills(ai_state: AIStateDict) -> AIStateDict:
    """Merge Deactivation kills from protocol into AI killed_tank_ids.

    Args:
        ai_state: Current AI state.

    Returns:
        Updated AI state with new kills merged.
    """
    new_kills = drain_killed_tank_ids()
    if not new_kills:
        return ai_state
    now = get_current_time_ms()
    merged = dict(ai_state["killed_tank_ids"])
    for tank_id in new_kills:
        merged[str(tank_id)] = now
        log.info("AI: kill registered (tank_id=%d)", tank_id)
    return AIStateDict(**{**ai_state, "killed_tank_ids": merged})


def _has_pending_shot_feedback(bot: Bot, timestamp_ms: int) -> bool:
    """Return True while a fired shot is still inside its feedback window.

    Args:
        bot: Bot instance.
        timestamp_ms: Current timestamp in milliseconds.

    Returns:
        True if a shot outcome is still pending and the tick should wait.
    """
    target_id = bot._ai_state["last_shot_target_id"]
    if target_id == -1:
        return False
    if peek_combat_hit():
        return False
    if peek_our_shot_response():
        return False
    if str(target_id) in bot._ai_state["killed_tank_ids"]:
        return False
    elapsed_ms = timestamp_ms - bot._ai_state["last_shoot_ms"]
    if elapsed_ms >= bot._ai_state["config"]["shot_feedback_timeout_ms"]:
        return False
    log.info(
        "SYNC: waiting for shot feedback for %s (id=%d)",
        bot._ai_state["last_shot_target_name"],
        target_id,
    )
    return True


def _get_combat_feedback(bot: Bot) -> CombatFeedback:
    """Get combat feedback from protocol weapon byte.

    Hit detection relies on the weapon byte in CombatHit responses:
    - weapon_byte > 0: special ammo used = confirmed hit
    - weapon_byte == 0 with dual enabled + stocked: miss (target empty)
    - weapon_byte == 0 without dual: normal single shot, can't tell
    - no response (timeout) with dual enabled + stocked: miss
    - no response without dual: can't tell

    Ammo count is decremented on each confirmed hit by mark_combat_hit.
    Combat feedback never rewrites inventory counts — protocol messages
    (0x49, 0x67, 0x74) are the sole authority for item counts.

    Args:
        bot: Bot instance.

    Returns:
        "hit" if weapon byte > 0 or kill confirmed, "miss" if dual was
        available but no hit detected, "" if feedback is indeterminate.
    """
    if bot._ai_state["last_shot_target_id"] == -1:
        log.info("FEEDBACK: no shot pending (last_shot_target_id=-1)")
        return ""
    got_hit = check_and_clear_combat_hit()
    got_response = check_and_clear_our_shot_response()
    if got_hit:
        log.info("FEEDBACK: hit confirmed")
        return "hit"
    if str(bot._ai_state["last_shot_target_id"]) in bot._ai_state["killed_tank_ids"]:
        log.info("FEEDBACK: kill confirmed")
        return "hit"
    inventory = get_inventory_state()
    dual_available = inventory["dual_shots"]["enabled"] and inventory["dual_shots"]["count"] > 0
    if got_response:
        if dual_available:
            log.info("FEEDBACK: miss (dual active, server used single)")
            return "miss"
        dual_count = inventory["dual_shots"]["count"]
        log.info(
            "FEEDBACK: single shot (no dual, count=%d)",
            dual_count,
        )
        return ""
    # No CombatHit response at all (timeout).
    if dual_available:
        log.info("FEEDBACK: miss (dual active, no combat hit)")
        return "miss"
    log.info(
        "FEEDBACK: no dual, ignoring (count=%d)",
        inventory["dual_shots"]["count"],
    )
    return ""


__all__ = [
    "run_tick_loop",
]
